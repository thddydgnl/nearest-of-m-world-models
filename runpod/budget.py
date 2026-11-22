#!/usr/bin/env python3
"""Spend tracking and enforcement for this study's pods only (the RunPod account is shared).

Spend is estimated from our ledger (uptime x $/hr), not from the account balance, so the user's own
concurrent pods never count against this study or trigger enforcement.

Usage:
  budget.py status          per-pod and total estimated spend of our pods, plus the account balance delta
  budget.py enforce         delete all of our live pods if estimated spend >= LIMIT or now >= HARD_DEADLINE,
                            and delete any single pilot's pods once that pilot's spend reaches its cap in caps.json
  budget.py watch [secs]    run `enforce` every <secs> (default 300) until HARD_DEADLINE has passed and none of our pods is alive
"""
import json
import re
import subprocess
import sys
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

INFRA = Path(__file__).resolve().parent
LEDGER = INFRA / "pods.ledger"   # JSON lines written by pod_up.sh / pod_down.sh
SEEN = INFRA / "pods.seen.json"  # last time each of our pods was seen alive
CAPS = INFRA / "caps.json"       # {"<pilot tag>": usd}; a tag's strays ("<tag>-stray") count toward it
START_BALANCE = 26.62           # account balance before the IMLE-WM revision runs (2026-11-22T17:07Z; shared account)
LIMIT = 24.0                     # USD for revision r1; the user's cap is 30 (2026-11-23) but the account balance is 26.62
HARD_DEADLINE = datetime(2026, 11, 27, 0, 0, tzinfo=timezone.utc)
STORAGE_MARGIN = 1.03            # container disk and volume storage are billed on top of GPU time


def now():
    return datetime.now(timezone.utc)


def stamp(t):
    return t.strftime("%Y-%m-%dT%H:%M:%SZ")


def parse_time(s):
    return datetime.strptime(s, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)


def api_key():
    text = (Path.home() / ".runpod" / "config.toml").read_text()
    m = re.search(r"^\s*apikey\s*=\s*['\"]?([^'\"\n]+)", text, re.I | re.M)
    if not m:
        sys.exit("no api key in ~/.runpod/config.toml")
    return m.group(1).strip()


def account():
    q = {"query": "query { myself { clientBalance pods { id name desiredStatus costPerHr } } }"}
    req = urllib.request.Request(
        "https://api.runpod.io/graphql",
        data=json.dumps(q).encode(),
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {api_key()}"},
    )
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.load(r)["data"]["myself"]


def ledger():
    pods = {}
    if LEDGER.exists():
        for line in LEDGER.read_text().splitlines():
            if not line.strip():
                continue
            e = json.loads(line)
            p = pods.setdefault(e["id"], {"id": e["id"], "tag": e.get("tag", "?")})
            if e["event"] == "up":
                p.update(gpu=e["gpu"], cost=float(e["cost"]), created=parse_time(e["time"]))
            elif e["event"] == "deleted":
                p["deleted"] = parse_time(e["time"])
    return pods


def record_deleted(pid, tag):
    with LEDGER.open("a") as f:
        f.write(json.dumps({"id": pid, "tag": tag, "event": "deleted", "time": stamp(now())}) + "\n")


def pilot(tag):
    """Cap key of a pod tag: "<exp>.<n>" and "<exp>.<n>-stray" both count toward caps.json["<exp>"]."""
    if tag.endswith("-stray"):
        tag = tag[:-len("-stray")]
    return tag.split(".")[0]


PREFIX = "nom-"


def adopt_untracked(alive, pods):
    """Pods named nom-* that are alive but missing from the ledger (pod_up.sh lost its read-back during the
    rest.runpod.io outage) are appended as "<tag>-stray" so that spend and caps count them."""
    added = False
    for pid, a in alive.items():
        if a["name"].startswith(PREFIX) and pid not in pods:
            tag = a["name"][len(PREFIX):] + "-stray"
            with LEDGER.open("a") as f:
                f.write(json.dumps({"id": pid, "tag": tag, "gpu": "?", "cost": str(a["costPerHr"]),
                                    "event": "up", "time": stamp(now())}) + "\n")
            print(f"WARNING: adopted untracked pod {a['name']} {pid} as {tag}")
            added = True
    return ledger() if added else pods


def snapshot():
    acct = account()
    alive = {p["id"]: p for p in acct["pods"]}
    pods = adopt_untracked(alive, ledger())
    seen = json.loads(SEEN.read_text()) if SEEN.exists() else {}
    t = now()
    for pid in pods:
        if pid in alive:
            seen[pid] = stamp(t)
    SEEN.write_text(json.dumps(seen, indent=1))
    total, rows = 0.0, []
    for pid, p in pods.items():
        if "created" not in p:
            continue
        if pid in alive:
            end = t
        elif "deleted" in p:
            end = p["deleted"]
        else:
            end = parse_time(seen[pid]) if pid in seen else p["created"]
        cost = max((end - p["created"]).total_seconds(), 0) / 3600 * p["cost"] * STORAGE_MARGIN
        total += cost
        state = alive[pid]["desiredStatus"] if pid in alive else "gone"
        rows.append((p["tag"], pid, p["gpu"], p["cost"], state, cost))
    return acct, alive, pods, total, rows


def status():
    acct, alive, pods, total, rows = snapshot()
    bal = float(acct["clientBalance"])
    print(f"{stamp(now())} ours_est=${total:.2f} limit=${LIMIT:.0f} account_delta=${START_BALANCE - bal:.2f} balance=${bal:.2f}")
    by_pilot = {}
    for tag, pid, gpu, rate, state, cost in sorted(rows):
        print(f"  {tag:<14} {pid} {gpu:<30} ${rate:.2f}/h {state:<8} ${cost:.2f}")
        by_pilot[pilot(tag)] = by_pilot.get(pilot(tag), 0.0) + cost
    return alive, pods, total, by_pilot


def delete(pid, tag, reason):
    print(f"ENFORCE({reason}): deleting {tag} {pid}")
    r = subprocess.run(["runpodctl", "pod", "delete", pid], check=False, capture_output=True, text=True)
    ok = r.returncode == 0
    if not ok:  # rest.runpod.io outage (2026-11-21): fall back to GraphQL podTerminate
        try:
            import rp_gql
            ok = rp_gql.terminate(pid)
        except Exception as e:
            print(f"ENFORCE: GraphQL terminate of {pid} failed: {e}")
    if not ok:
        print(f"ENFORCE: {tag} {pid} could NOT be deleted; it stays counted and will be retried")
        return
    record_deleted(pid, tag)
    env = INFRA / "pods" / f"{tag.replace('-stray', '')}.env"
    if env.exists():
        env.rename(env.with_name(f"{env.stem}.env.deleted-enforced-{now().strftime('%H%M%S')}"))


def enforce():
    alive, pods, total, by_pilot = status()
    caps = json.loads(CAPS.read_text()) if CAPS.exists() else {}
    live = {pid: p for pid, p in pods.items() if pid in alive and "deleted" not in p}
    if total >= LIMIT or now() >= HARD_DEADLINE:
        reason = "budget" if total >= LIMIT else "deadline"
        for pid, p in live.items():
            delete(pid, p["tag"], reason)
        return {}, pods
    for pid, p in live.items():
        cap = caps.get(pilot(p["tag"]))
        if cap is not None and by_pilot.get(pilot(p["tag"]), 0.0) >= cap:
            delete(pid, p["tag"], f"pilot cap ${cap}")
    return alive, pods


def watch(period):
    while True:
        try:
            alive, pods = enforce()
            if now() >= HARD_DEADLINE and not any(pid in alive for pid in pods):
                print("deadline passed and no pod of ours is alive; watchdog exits")
                return
        except Exception as e:  # keep watching through transient API errors
            print(f"{stamp(now())} watchdog error: {e}")
        sys.stdout.flush()
        time.sleep(period)


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "status"
    if cmd == "status":
        status()
    elif cmd == "enforce":
        enforce()
    elif cmd == "watch":
        watch(int(sys.argv[2]) if len(sys.argv) > 2 else 300)
    else:
        sys.exit(__doc__)
