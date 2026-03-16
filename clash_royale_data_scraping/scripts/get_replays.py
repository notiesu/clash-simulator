#!/usr/bin/env python3
"""
RoyaleAPI replay downloader (Hog 2.6 focused)

Key fixes vs the broken behavior you showed:
- DO NOT click replay buttons and scrape rendered HTML (often stays "Loading" or container is empty).
- Instead, use the same approach as your older working script: build the /data/replay URL from the
  replay button's dataset and download via Playwright's authenticated request context.
- Keep function name: scrape_players_from_hog26_ratings (per your request).
"""

import argparse
import json
import re
import time
from pathlib import Path
from urllib.parse import urljoin, urlencode, urlparse

from playwright.sync_api import TimeoutError as PlaywrightTimeoutError
from playwright.sync_api import sync_playwright


BASE = "https://royaleapi.com"

# Normalized Hog 2.6 deck (canonical card ids)
HOG26_DECK = [
    "cannon",
    "fireball",
    "hog-rider",
    "ice-golem",
    "ice-spirit",
    "musketeer",
    "skeletons",
    "the-log",
]


# ----------------------------
# Small utils
# ----------------------------
def _sleep(s: float) -> None:
    try:
        time.sleep(max(0.0, float(s)))
    except Exception:
        time.sleep(0.0)


def normalize_card_id(card: str) -> str:
    """
    RoyaleAPI sometimes adds suffixes in deck keys:
      - evolution:  musketeer-ev1, cannon-ev2
      - "hero":     ice-golem-hero, musketeer-hero  (seen in your logs)

    We normalize to the base card id for deck comparisons.
    """
    if not card:
        return card
    c = str(card).strip()
    c = re.sub(r"-ev\d+$", "", c)
    c = re.sub(r"-hero$", "", c)
    return c


def _normalize_deck_key(deck_key: str) -> list[str]:
    """
    deck_key example:
      cannon-ev1,fireball,hog-rider,ice-golem-hero,ice-spirit,musketeer-ev1,skeletons,the-log
    """
    if not deck_key:
        return []
    cards = [normalize_card_id(x.strip()) for x in str(deck_key).split(",") if x.strip()]
    return cards[:8]


def _same_deck(a: list[str], b: list[str]) -> bool:
    return sorted(a) == sorted(b)


def _referrer_path(referrer_url: str) -> str:
    """
    RoyaleAPI expects referrer_path like:
      /player/<TAG>/battles
      /player/<TAG>/battles/history?before=...
    """
    try:
        u = urlparse(referrer_url)
        path = u.path or "/"
        if u.query:
            path = f"{path}?{u.query}"
        return path
    except Exception:
        # fall back: if caller passed a path already
        return referrer_url if referrer_url.startswith("/") else "/"


# ----------------------------
# Replay download: "old working way"
# ----------------------------
def build_data_replay_url(ds: dict, referrer_url: str) -> str:
    params = {
        "tag": ds["replay"],
        "team_tags": ds.get("teamTags", ""),
        "opponent_tags": ds.get("opponentTags", ""),
        "team_crowns": ds.get("teamCrowns", ""),
        "opponent_crowns": ds.get("opponentCrowns", ""),
        "referrer_path": _referrer_path(referrer_url),
    }
    return f"{BASE}/data/replay?{urlencode(params)}"


def _normalize_tag_list(tag_field: str) -> list[str]:
    if not tag_field:
        return []
    out: list[str] = []
    for raw in str(tag_field).split(","):
        t = raw.strip()
        if t.startswith("#"):
            t = t[1:]
        if t:
            out.append(t)
    return out


def _parse_deck_id(deck_div_id: str) -> list[str]:
    # deck div ids look like: deck_card1,card2,...
    if not deck_div_id or not str(deck_div_id).startswith("deck_"):
        return []
    deck_csv = str(deck_div_id)[len("deck_") :]
    cards = [normalize_card_id(c.strip()) for c in deck_csv.split(",") if c.strip()]
    return cards[:8]


def extract_decks_near_replay_button(button) -> list[dict]:
    return button.evaluate(
        """(b) => {
            let node = b;
            for (let i = 0; i < 12; i++) {
              if (!node) break;
              const deckDivs = node.querySelectorAll ? node.querySelectorAll('div[id^="deck_"]') : [];
              if (deckDivs && deckDivs.length >= 1) break;
              node = node.parentElement;
            }
            if (!node || !node.querySelectorAll) return [];

            const segs = Array.from(node.querySelectorAll('div.team-segment'));
            const out = [];

            if (segs.length > 0) {
              for (const seg of segs) {
                const a = seg.querySelector('a[href^="/player/"]');
                const href = a ? a.getAttribute('href') : '';
                // Handles: /player/TAG, /player/TAG/battles, /player/TAG/battles/history?...
                let playerTag = null;
                if (href) {
                  const parts = href.split('?')[0].split('/').filter(Boolean); // remove query, split path
                  if (parts.length >= 2 && parts[0] === 'player') {
                    playerTag = parts[1];
                  }
                }

                const deckDiv = seg.querySelector('div[id^="deck_"]');
                const deckId = deckDiv ? deckDiv.getAttribute('id') : null;

                if (playerTag && deckId && deckId.startsWith('deck_')) {
                  out.push({player_tag: playerTag, deck_id: deckId});
                }
              }
              return out;
            }

            // fallback: deck divs only (no player mapping)
            const deckDivs = Array.from(node.querySelectorAll('div[id^="deck_"]'));
            for (const d of deckDivs) {
              const deckId = d.getAttribute('id');
              if (deckId && deckId.startsWith('deck_')) {
                out.push({player_tag: null, deck_id: deckId});
              }
            }
            return out;
        }"""
    )


def collect_replay_datasets(page) -> list[dict]:
    buttons = page.query_selector_all("button.replay_button[data-replay]")
    dsets: list[dict] = []
    for b in buttons:
        ds = b.evaluate("b => b.dataset") or {}
        if ds.get("replay"):
            try:
                ds["_decks"] = extract_decks_near_replay_button(b)
            except Exception:
                ds["_decks"] = []
            dsets.append(ds)
    return dsets


def _ds_has_hog26_deck(ds: dict, player_tag: str) -> bool:
    """Return True if this battle's deck (for the player OR either side) matches Hog 2.6.

    RoyaleAPI HTML changes a lot. Sometimes we can map deck->player_tag, sometimes we only
    see raw deck divs with no player mapping. Be permissive: if ANY detected deck for this
    battle matches Hog 2.6, accept it.
    """
    decks = ds.get("_decks") or []

    # 1) Try exact player match if mapping exists
    if isinstance(decks, list):
        for item in decks:
            try:
                if item.get("player_tag") == player_tag:
                    deck = _parse_deck_id(item.get("deck_id"))
                    if deck and set(deck) == set(HOG26_DECK):
                        return True
            except Exception:
                pass

        # 2) Fallback: accept if any detected deck matches (mapping missing / broken)
        for item in decks:
            try:
                deck = _parse_deck_id(item.get("deck_id"))
                if deck and set(deck) == set(HOG26_DECK):
                    return True
            except Exception:
                pass

    # 3) Extra fallback: sometimes the deck is exposed directly as a CSV string in the dataset
    for k in ("teamDeck", "team_deck", "deck", "deck_key", "team-deck"):
        raw = ds.get(k)
        if not raw:
            continue
        try:
            cards = [normalize_card_id(c.strip()) for c in str(raw).split(",") if c.strip()]
            if len(cards) >= 8 and set(cards[:8]) == set(HOG26_DECK):
                return True
        except Exception:
            continue

    return False


def _next_page_href(page) -> str | None:
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


def reveal_next_pagination(page, *, max_scrolls: int = 30, sleep_s: float = 0.25) -> None:
    for _ in range(max_scrolls):
        if _next_page_href(page):
            return
        page.evaluate("window.scrollBy(0, Math.floor(window.innerHeight * 1.8))")
        _sleep(sleep_s)


def click_next_if_present(page) -> bool:
    reveal_next_pagination(page, max_scrolls=35, sleep_s=0.25)

    loc = page.locator("a.item:has(i.angle.right.icon)").first
    if loc.count() == 0:
        # maybe rel="next"
        rel = page.locator('a[rel="next"]').first
        if rel.count() == 0:
            return False
        loc = rel

    before_url = page.url
    try:
        loc.scroll_into_view_if_needed()
        loc.click(timeout=5_000)
    except Exception:
        # JS click fallback
        try:
            page.evaluate("(el) => el.click()", loc.element_handle())
        except Exception:
            return False

    t0 = time.time()
    while time.time() - t0 < 6.0:
        _sleep(0.25)
        if page.url != before_url:
            return True
    return False


# ----------------------------
# Hog2.6 player scraper (KEEP NAME)
# ----------------------------
PLAYER_LINK_RE = re.compile(r"^/player/([A-Z0-9]+)$")


def scrape_players_from_hog26_ratings(page, max_players: int = 200) -> list[str]:
    """
    Scrape player tags from the Hog 2.6 deck ratings page.

    URL format (matches what you printed in pdb):
      https://royaleapi.com/decks/stats/<deck_key>/players/ratings
    """
    deck_key = "cannon-ev1,fireball,hog-rider,ice-golem-hero,ice-spirit,musketeer-ev1,skeletons,the-log"
    url = f"{BASE}/decks/stats/{deck_key}/players/ratings"

    page.goto(url, wait_until="domcontentloaded", timeout=120_000)
    _sleep(0.8)

    # accept cookies if present
    try:
        btn = page.query_selector('button:has-text("Accept")')
        if btn:
            btn.click()
            _sleep(0.2)
    except Exception:
        pass

    tags: list[str] = []
    seen: set[str] = set()
    page_hops = 0

    while len(tags) < int(max_players) and page_hops < 50:
        # Use the player_container structure you pasted:
        # <div class="player_container"> ... <a href="/player/9GYYGP0U">Name</a> ...
        anchors = page.query_selector_all("div.player_container a[href^='/player/']")
        for a in anchors:
            href = (a.get_attribute("href") or "").strip()
            m = PLAYER_LINK_RE.match(href)
            if not m:
                continue
            tag = m.group(1)
            if tag in seen:
                continue
            seen.add(tag)
            tags.append(tag)
            if len(tags) >= int(max_players):
                break

        if len(tags) >= int(max_players):
            break

        # paginate if possible
        if not click_next_if_present(page):
            # some pages use rel="next" href without needing a click target in viewport
            href = _next_page_href(page)
            if not href:
                break
            next_url = urljoin(page.url, href)
            if next_url == page.url:
                break
            page.goto(next_url, wait_until="domcontentloaded", timeout=120_000)
            _sleep(0.6)
        else:
            try:
                page.wait_for_load_state("domcontentloaded", timeout=15_000)
            except PlaywrightTimeoutError:
                pass
            _sleep(0.4)

        page_hops += 1

    return tags


# ----------------------------
# Core: collect + download hog2.6 replays for a player
# ----------------------------
def collect_hog26_replays_for_player(page, player_tag: str, out_dir: str | Path, need: int, *, max_pages: int = 200) -> int:
    """
    - Walk /player/<tag>/battles pages (and history pages)
    - Identify replay buttons that correspond to Hog 2.6 deck
    - Download /data/replay payload with authenticated cookies
    - Save as replay_<id>.json with {"meta":..., "data":...}
    """
    import json
    from urllib.parse import urljoin

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    battles_url = f"{BASE}/player/{player_tag}/battles"
    page.goto(battles_url, wait_until="domcontentloaded", timeout=120_000)

    if "/login" in page.url:
        raise RuntimeError(f"Not logged in (redirected to /login) while loading player {player_tag}")

    _sleep(0.4)

    saved = 0
    seen: set[str] = set()
    page_hops = 0
    stuck_rounds = 0

    while saved < int(need) and page_hops < int(max_pages):
        current_url = page.url

        dsets = collect_replay_datasets(page)
        print(f"[{player_tag}] Replay items found: {len(dsets)}")

        # quick deck frequency log
        deck_counts: dict[str, int] = {}
        for ds in dsets:
            dk = None
            for item in (ds.get("_decks") or []):
                if item.get("player_tag") == player_tag:
                    dk = item.get("deck_id")
                    break
            if dk:
                deck_counts[dk] = deck_counts.get(dk, 0) + 1
        if deck_counts:
            print(f"[{player_tag}] Top decks on this page (raw deck_key):")
            for k, v in sorted(deck_counts.items(), key=lambda kv: -kv[1])[:5]:
                pretty = k[len("deck_"):] if k.startswith("deck_") else k
                print(f"   {v:2d}x  {pretty}")

        added_this_page = 0

        for ds in dsets:
            rid = ds.get("replay")
            if not rid or rid in seen:
                continue

            # filter: hog2.6 only
            if not _ds_has_hog26_deck(ds, player_tag):
                continue

            seen.add(rid)

            # build replay URL (referrer_path should be a path, but your helper accepts referrer_url;
            # we're keeping your helper as-is for now and passing the current_url like you did.)
            data_url = build_data_replay_url(ds, current_url)

            # Fetch replay JSON INSIDE the page context (most robust against 403/CF)
            fetch_result = page.evaluate(
                """async ({url, ref}) => {
                    const res = await fetch(url, {
                        method: "GET",
                        credentials: "include",
                        headers: {
                            "accept": "application/json, text/plain, */*",
                            "x-requested-with": "XMLHttpRequest",
                            "referer": ref,
                        },
                    });
                    return { status: res.status, text: await res.text() };
                }""",
                {"url": data_url, "ref": current_url},
            )

            status = int(fetch_result.get("status") or 0)
            if status != 200:
                print(f"[replay {rid}] GET {status} -> {data_url}")
                # small backoff on blocks
                if status in (401, 403, 429):
                    _sleep(1.25)
                continue

            try:
                replay_json = json.loads(fetch_result.get("text") or "")
            except Exception as e:
                print(f"[replay {rid}] JSON parse failed: {e}")
                continue

            # Build meta like your older files
            team_tags = _normalize_tag_list(ds.get("teamTags", ""))
            opp_tags = _normalize_tag_list(ds.get("opponentTags", ""))

            # decks from scraped HTML near replay button (best-effort)
            deck_by_player: dict[str, list[str]] = {}
            for item in (ds.get("_decks") or []):
                try:
                    ptag = item.get("player_tag")
                    deck = _parse_deck_id(item.get("deck_id"))
                    if ptag and deck:
                        deck_by_player[ptag] = deck
                except Exception:
                    pass

            meta = {
                "replay_id": rid,
                "player_tag": player_tag,
                "team_tags": team_tags,
                "opponent_tags": opp_tags,
                "team_decks": [deck_by_player.get(t) for t in team_tags],
                "opponent_decks": [deck_by_player.get(t) for t in opp_tags],
                "source_battles_url": current_url,
            }

            out_path = out_dir / f"replay_{rid}.json"
            out_path.write_text(json.dumps({"meta": meta, "data": replay_json}, indent=2), encoding="utf-8")
            saved += 1
            added_this_page += 1
            print(f"[replay {rid}] saved -> {out_path}")

            if saved >= int(need):
                break

            _sleep(0.35)

        print(f"[{player_tag}] Collected {saved}/{need} (added {added_this_page})")

        if saved >= int(need):
            break

        # paginate
        prev_url = page.url
        if click_next_if_present(page):
            page_hops += 1
            stuck_rounds = 0
            try:
                page.wait_for_load_state("domcontentloaded", timeout=15_000)
            except PlaywrightTimeoutError:
                pass
            _sleep(0.35)
            continue

        # no next
        if page.url == prev_url:
            stuck_rounds += 1
            print(f"[{player_tag}] Stuck round {stuck_rounds}/8")
            if stuck_rounds >= 8:
                break
        else:
            stuck_rounds = 0

        # try direct href nav fallback
        href = _next_page_href(page)
        if href:
            page.goto(urljoin(page.url, href), wait_until="domcontentloaded", timeout=120_000)
            page_hops += 1
            stuck_rounds = 0
            _sleep(0.4)
        else:
            print(f"[{player_tag}] No more history pages.")
            break

    return saved


# ----------------------------
# Login helper
# ----------------------------
def ensure_logged_in(page) -> None:
    page.goto(f"{BASE}/login?lang=en", wait_until="domcontentloaded", timeout=120_000)
    print("\nLogin page opened in the browser.")
    print("1) Log in in that browser window (if needed).")
    print("2) When you are fully logged in, come back here and press Enter.")
    input("Press Enter to continue...")

    # quick sanity check: go to homepage; if it bounces to /login, user isn't logged in
    page.goto(BASE, wait_until="domcontentloaded", timeout=120_000)
    if "/login" in page.url:
        raise RuntimeError("Still not logged in after prompt. (Page redirected to /login)")


# ----------------------------
# CLI
# ----------------------------
def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hog2.6", dest="hog26", action="store_true", help="Scrape hog 2.6 players then download hog 2.6 replays.")
    ap.add_argument("--player_tag", default=None, help="Single player tag (no leading #). If set, ignores --hog2.6 players scraping.")
    ap.add_argument("--num_battles", type=int, default=10, help="How many replays to download per player (or total for single player).")
    ap.add_argument("--out_dir", default="hog26_replays", help="Output folder.")
    ap.add_argument("--profile_dir", default="royaleapi_profile", help="Playwright persistent profile dir (keeps cookies).")
    ap.add_argument("--headless", action="store_true", help="Run headless (NOT recommended for first login).")
    ap.add_argument("--max_players", type=int, default=100, help="How many hog2.6 players to scrape.")
    ap.add_argument("--max_pages", type=int, default=200, help="Max history pages to traverse per player.")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    out_root = Path(args.out_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    with sync_playwright() as p:
        # IMPORTANT: persistent context so page.request shares login cookies
        context = p.chromium.launch_persistent_context(
            user_data_dir=args.profile_dir,
            headless=bool(args.headless),
        )
        page = context.new_page()

        if not args.headless:
            ensure_logged_in(page)
        else:
            # headless: assume cookies already exist in profile_dir
            page.goto(BASE, wait_until="domcontentloaded", timeout=120_000)
            if "/login" in page.url:
                raise RuntimeError("Headless mode but not logged in. Run once without --headless to store cookies.")

        # Single-player override
        if args.player_tag:
            tag = args.player_tag.strip().lstrip("#")
            print(f"\n=== [single] Player {tag} (need={args.num_battles}) ===")
            saved = collect_hog26_replays_for_player(page, tag, out_root / tag, need=int(args.num_battles), max_pages=int(args.max_pages))
            print(f"\nDone. Saved {saved} replays in: {(out_root / tag).resolve()}")
            context.close()
            return

        if not args.hog26:
            raise SystemExit("Provide --player_tag for single-player mode OR pass --hog2.6 to scrape hog2.6 players.")

        players = scrape_players_from_hog26_ratings(page, max_players=int(args.max_players))
        print(f"\nHog2.6 players scraped: {len(players)}")

        remaining = int(args.num_battles)

        for i, tag in enumerate(players, start=1):
            if remaining <= 0:
                break
            print(f"\n=== [hog26 {i}/{len(players)}] Player {tag} (remaining={remaining}) ===")
            try:
                got = collect_hog26_replays_for_player(
                    page,
                    tag,
                    out_root / tag,
                    need=remaining,
                    max_pages=int(args.max_pages),
                )
            except Exception as e:
                print(f"[{tag}] ERROR: {e}")
                continue

            remaining -= got
            if got == 0:
                print(f"[{tag}] No hog2.6 replays found for this player (skipping).")

        print("\nDone.")
        context.close()


if __name__ == "__main__":
    main()
