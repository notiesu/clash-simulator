import argparse
import json
import re
import time
from pathlib import Path
from urllib.parse import urljoin, urlencode

from playwright.sync_api import TimeoutError as PlaywrightTimeoutError
from playwright.sync_api import sync_playwright

BASE = "https://royaleapi.com"
LEADERBOARD_URL = f"{BASE}/players/leaderboard"


# ----------------------------
# Replay scraping helpers
# ----------------------------
def build_data_replay_url(ds: dict, referrer_url: str) -> str:
    params = {
        "tag": ds["replay"],
        "team_tags": ds.get("teamTags", ""),
        "opponent_tags": ds.get("opponentTags", ""),
        "team_crowns": ds.get("teamCrowns", ""),
        "opponent_crowns": ds.get("opponentCrowns", ""),
        "referrer_path": referrer_url,
    }
    return f"{BASE}/data/replay?{urlencode(params)}"


def collect_replay_datasets(page):
    buttons = page.query_selector_all("button.replay_button[data-replay]")
    dsets = []
    for b in buttons:
        ds = b.evaluate("b => b.dataset")
        if ds and ds.get("replay"):
            dsets.append(ds)
    return dsets


def get_next_page_url(page) -> str | None:
    """
    RoyaleAPI icon-only "next" button:
      <a class="item" href="/player/<TAG>/battles/history?before=...">
          <i class="angle right icon"></i>
      </a>
    """
    el = page.query_selector("a.item:has(i.angle.right.icon)")
    if not el:
        return None
    href = el.get_attribute("href")
    if not href:
        return None
    return urljoin(page.url, href)


def reveal_next_pagination(page, *, max_scrolls: int = 35, sleep_s: float = 0.25) -> None:
    """
    Pagination is often lazy-rendered. Scroll to trigger it.
    Avoid 'networkidle' because RoyaleAPI may keep connections busy indefinitely.
    """
    for _ in range(max_scrolls):
        if get_next_page_url(page):
            return
        page.evaluate("window.scrollBy(0, Math.floor(window.innerHeight * 1.8))")
        time.sleep(sleep_s)


def click_next_if_present(page) -> bool:
    reveal_next_pagination(page, max_scrolls=35, sleep_s=0.25)

    btn = page.locator("a.item:has(i.angle.right.icon)").first
    if btn.count() == 0:
        return False

    before_url = page.url
    try:
        btn.scroll_into_view_if_needed()
        btn.click(timeout=5_000)
    except Exception:
        # fallback: JS click
        try:
            page.evaluate("(el) => el.click()", btn.element_handle())
        except Exception:
            return False

    # verify it actually moved
    t0 = time.time()
    while time.time() - t0 < 5.0:
        time.sleep(0.25)
        if page.url != before_url:
            return True
    return False


# ----------------------------
# Leaderboard scraping
# ----------------------------
PLAYER_LINK_RE = re.compile(r"^/player/([A-Z0-9]+)$")


def scrape_leaderboard_top_players(page, num_players: int) -> list[dict]:
    """
    Returns list of dicts: [{"tag": "...", "url": "..."}]
    Uses the leaderboard page and extracts /player/<TAG> links.
    """
    page.goto(LEADERBOARD_URL, wait_until="domcontentloaded", timeout=120_000)

    # Scroll a bit to ensure table content loads
    page.evaluate("window.scrollBy(0, Math.floor(window.innerHeight * 1.5))")
    time.sleep(0.5)

    anchors = page.query_selector_all("a[href^='/player/']")
    seen = set()
    players = []

    for a in anchors:
        href = a.get_attribute("href") or ""
        m = PLAYER_LINK_RE.match(href)
        if not m:
            continue
        tag = m.group(1)
        if tag in seen:
            continue
        seen.add(tag)
        players.append({"tag": tag, "url": urljoin(BASE, href)})
        if len(players) >= num_players:
            break

    return players


# ----------------------------
# Core per-player pipeline
# ----------------------------
def collect_replays_for_player(page, player_tag: str, target: int, *, max_pages: int = 200):
    """
    Navigates player battles pages and collects replay datasets up to target.
    Returns:
      ordered_replay_ids, replay_ds_by_id, referrer_by_id
    """
    battles_url = f"{BASE}/player/{player_tag}/battles"
    page.goto(battles_url, wait_until="domcontentloaded", timeout=120_000)

    if "/login" in page.url:
        raise RuntimeError(f"Not logged in (redirected to /login) while loading player {player_tag}")

    # Sometimes replay buttons appear only after a tiny scroll
    page.evaluate("window.scrollBy(0, Math.floor(window.innerHeight * 0.8))")
    time.sleep(0.3)

    seen_replays: set[str] = set()
    ordered_replay_ids: list[str] = []
    replay_ds_by_id: dict[str, dict] = {}
    referrer_by_id: dict[str, str] = {}

    stuck_rounds = 0
    max_stuck_rounds = 10
    page_hops = 0

    # initial count print
    print(f"[{player_tag}] Replay buttons found (initial):", page.locator("button.replay_button[data-replay]").count())

    while len(seen_replays) < target and page_hops < max_pages:
        current_url = page.url

        dsets = collect_replay_datasets(page)
        added = 0
        for ds in dsets:
            rid = ds.get("replay")
            if not rid or rid in seen_replays:
                continue
            seen_replays.add(rid)
            ordered_replay_ids.append(rid)
            replay_ds_by_id[rid] = ds
            referrer_by_id[rid] = current_url
            added += 1
            if len(seen_replays) >= target:
                break

        print(f"[{player_tag}] Collected {len(seen_replays)}/{target} (added {added} from {current_url})")
        if len(seen_replays) >= target:
            break

        prev_btn_count = page.locator("button.replay_button[data-replay]").count()

        # try next page
        if click_next_if_present(page):
            page_hops += 1
            # small settle
            try:
                page.wait_for_load_state("domcontentloaded", timeout=15_000)
            except PlaywrightTimeoutError:
                pass
            time.sleep(0.35)
            stuck_rounds = 0
            continue

        # if no next page exists / cannot click it, attempt a deeper scroll then re-check once
        reveal_next_pagination(page, max_scrolls=25, sleep_s=0.2)
        if click_next_if_present(page):
            page_hops += 1
            time.sleep(0.35)
            stuck_rounds = 0
            continue

        # no progress
        new_btn_count = page.locator("button.replay_button[data-replay]").count()
        if new_btn_count <= prev_btn_count:
            stuck_rounds += 1
            print(f"[{player_tag}] No new replay buttons loaded (stuck_rounds={stuck_rounds}/{max_stuck_rounds}).")
            if stuck_rounds >= max_stuck_rounds:
                break
        else:
            stuck_rounds = 0

    return ordered_replay_ids[:target], replay_ds_by_id, referrer_by_id


def download_replays(page, player_tag: str, out_dir: Path, ordered_replay_ids, replay_ds_by_id, referrer_by_id, battles_url: str, sleep_s: float):
    out_dir.mkdir(parents=True, exist_ok=True)
    saved = 0
    for replay_id in ordered_replay_ids:
        ds = replay_ds_by_id[replay_id]
        out_path = out_dir / f"replay_{replay_id}.json"
        if out_path.exists():
            continue
        data_url = build_data_replay_url(ds, referrer_by_id.get(replay_id, battles_url))
        resp = page.request.get(data_url)
        if not resp.ok:
            print(f"[{player_tag}] FAIL replay {replay_id} status={resp.status}")
            continue
        out_path.write_text(json.dumps(resp.json(), indent=2), encoding="utf-8")
        saved += 1
        time.sleep(max(0.0, float(sleep_s)))
    return saved


# ----------------------------
# CLI + main
# ----------------------------
def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Download RoyaleAPI replay JSONs: single-player mode OR leaderboard mode."
    )
    ap.add_argument("--player_tag", default=None, help="Single player tag (no leading #). Ignored if --num_players > 0.")
    ap.add_argument("--num_battles", type=int, default=25, help="Replay count to fetch per player.")
    ap.add_argument("--num_players", type=int, default=0, help="If > 0, scrape top N players from leaderboard.")
    ap.add_argument("--out_dir", default="replays_out", help="Output directory root.")
    ap.add_argument("--profile_dir", default="royaleapi_profile", help="Chromium user-data dir (keeps login cookies).")
    ap.add_argument("--headless", action="store_true", help="Run headless (not recommended for first login).")
    ap.add_argument("--max_pages", type=int, default=200, help="Safety cap on page traversals per player.")
    ap.add_argument("--sleep", type=float, default=0.35, help="Sleep between replay downloads (seconds).")
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    out_root = Path(args.out_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    per_player_target = max(0, int(args.num_battles))
    if per_player_target == 0:
        print("--num_battles=0, nothing to do.")
        return 0

    leaderboard_mode = int(args.num_players) > 0
    if not leaderboard_mode and not args.player_tag:
        raise SystemExit("Provide --player_tag for single-player mode, or use --num_players for leaderboard mode.")

    with sync_playwright() as p:
        context = p.chromium.launch_persistent_context(
            user_data_dir=args.profile_dir,
            headless=bool(args.headless),
        )
        page = context.new_page()

        # Login flow (only needs to happen once)
        page.goto(f"{BASE}/login?lang=en", wait_until="domcontentloaded")
        if not args.headless:
            print("\nLogin page opened in the browser.")
            print("1) Log in in that browser window (if needed).")
            print("2) When you are fully logged in, come back here and press Enter.")
            input("Press Enter to continue...")

        if leaderboard_mode:
            num_players = int(args.num_players)
            players = scrape_leaderboard_top_players(page, num_players)
            print(f"Leaderboard scraped: {len(players)}/{num_players} players")

            batch_dir = out_root / f"leaderboard_top_{len(players)}"
            batch_dir.mkdir(parents=True, exist_ok=True)

            manifest = {
                "source": LEADERBOARD_URL,
                "num_players_requested": num_players,
                "num_players_found": len(players),
                "num_battles_per_player": per_player_target,
                "players": [],
            }

            total_saved = 0
            for idx, pl in enumerate(players, start=1):
                tag = pl["tag"]
                print(f"\n=== [{idx}/{len(players)}] Player {tag} ===")

                battles_url = f"{BASE}/player/{tag}/battles"
                try:
                    ordered_ids, ds_by_id, ref_by_id = collect_replays_for_player(
                        page, tag, per_player_target, max_pages=int(args.max_pages)
                    )
                except Exception as e:
                    print(f"[{tag}] ERROR collecting replays: {e}")
                    manifest["players"].append({"tag": tag, "error": str(e), "saved": 0})
                    continue

                player_out = batch_dir / tag
                saved = download_replays(
                    page,
                    tag,
                    player_out,
                    ordered_ids,
                    ds_by_id,
                    ref_by_id,
                    battles_url=battles_url,
                    sleep_s=float(args.sleep),
                )
                total_saved += saved
                manifest["players"].append(
                    {"tag": tag, "saved": saved, "collected": len(ordered_ids), "out_dir": str(player_out)}
                )
                print(f"[{tag}] Saved {saved} replay files into {player_out}")

                # a little pause between players to be polite
                time.sleep(0.6)

            (batch_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
            print(f"\nDone. Total saved: {total_saved}. Manifest: {batch_dir / 'manifest.json'}")

        else:
            tag = args.player_tag
            battles_url = f"{BASE}/player/{tag}/battles"
            ordered_ids, ds_by_id, ref_by_id = collect_replays_for_player(
                page, tag, per_player_target, max_pages=int(args.max_pages)
            )
            player_out = out_root / tag
            saved = download_replays(
                page,
                tag,
                player_out,
                ordered_ids,
                ds_by_id,
                ref_by_id,
                battles_url=battles_url,
                sleep_s=float(args.sleep),
            )
            print(f"Done. Saved {saved} replay JSON files into: {player_out.resolve()}")

        context.close()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
