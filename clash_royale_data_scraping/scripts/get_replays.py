import json
import time
from pathlib import Path
from urllib.parse import urlencode
from playwright.sync_api import sync_playwright

BASE = "https://royaleapi.com"
PLAYER_TAG = "CQ882QP8G"
OUT_DIR = Path("replays_out")
OUT_DIR.mkdir(exist_ok=True)

def build_data_replay_url(ds: dict, referrer_url: str) -> str:
    params = {
        "tag": ds["replay"],
        "team_tags": ds["teamTags"],
        "opponent_tags": ds["opponentTags"],
        "team_crowns": ds["teamCrowns"],
        "opponent_crowns": ds["opponentCrowns"],
        "referrer_path": referrer_url,
    }
    return f"{BASE}/data/replay?{urlencode(params)}"

def collect_replay_datasets(page):
    buttons = page.query_selector_all("button.replay_button[data-replay]")
    dsets = []
    for b in buttons:
        ds = b.evaluate("b => b.dataset")
        if ds.get("replay"):
            dsets.append(ds)
    return dsets

with sync_playwright() as p:
    context = p.chromium.launch_persistent_context(
        user_data_dir="royaleapi_profile",
        headless=False,
    )
    page = context.new_page()

    # 1) Go to login and STOP
    page.goto(f"{BASE}/login?lang=en", wait_until="domcontentloaded")
    print("\nLogin page opened in the browser.")
    print("1) Log in in that browser window.")
    print("2) When you are fully logged in, come back here and press Enter.")
    input("Press Enter to continue...")

    # 2) Now go to battles
    battles_url = f"{BASE}/player/{PLAYER_TAG}/battles"
    page.goto(battles_url, wait_until="domcontentloaded", timeout=120_000)
    page.wait_for_selector("button.replay_button[data-replay]", timeout=60_000)

    # If it still redirects to login, login didn't stick
    if "/login" in page.url:
        raise RuntimeError("Still not logged in (redirected to /login). Try logging in again.")

    replay_ds_list = collect_replay_datasets(page)
    print("Replay buttons found:", len(replay_ds_list))

    # 3) Download replay JSON for each button
    for ds in replay_ds_list:
        replay_id = ds["replay"]
        out_path = OUT_DIR / f"replay_{replay_id}.json"
        if out_path.exists():
            print("Skip:", replay_id)
            continue

        data_url = build_data_replay_url(ds, battles_url)
        resp = page.request.get(data_url)

        if not resp.ok:
            print("FAIL", replay_id, resp.status)
            continue

        out_path.write_text(json.dumps(resp.json(), indent=2), encoding="utf-8")
        print("Saved", out_path.name)
        time.sleep(0.5)

    context.close()
