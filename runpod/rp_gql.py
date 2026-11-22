#!/usr/bin/env python3
"""GraphQL fallbacks for RunPod pod queries and deletion.

On 2026-11-21 around 18:20Z the REST host used by runpodctl 2.9 (rest.runpod.io) stopped resolving (NXDOMAIN on public
resolvers) while https://api.runpod.io/graphql kept working: `runpodctl pod create` still created pods, but its read-back,
`pod get`, `pod list` and `pod delete` failed. These helpers use only the GraphQL endpoint.

Usage:
  rp_gql.py pods [prefix]          one JSON line per live pod (id, name, desiredStatus, costPerHr, uptime, gpu)
  rp_gql.py find <name>            JSON lines of the live pods with exactly this name
  rp_gql.py sshinfo <pod_id>       "ip port" of the pod's public ssh port (exit 1 while it is not up yet)
  rp_gql.py terminate <pod_id>     terminate one pod; exit 0 once it is no longer listed (or already gone)
"""
import json
import re
import sys
import time
import urllib.request
from pathlib import Path

URL = "https://api.runpod.io/graphql"
POD_FIELDS = ("id name desiredStatus costPerHr lastStatusChange machine { gpuDisplayName } "
              "runtime { uptimeInSeconds ports { ip isIpPublic privatePort publicPort type } }")


def api_key():
    text = (Path.home() / ".runpod" / "config.toml").read_text()
    m = re.search(r"^\s*apikey\s*=\s*['\"]?([^'\"\n]+)", text, re.I | re.M)
    if not m:
        sys.exit("no api key in ~/.runpod/config.toml")
    return m.group(1).strip()


def gql(query, timeout=30):
    req = urllib.request.Request(URL, data=json.dumps({"query": query}).encode(),
                                 headers={"Content-Type": "application/json", "Authorization": f"Bearer {api_key()}"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        d = json.load(r)
    if d.get("errors"):
        raise RuntimeError(json.dumps(d["errors"])[:500])
    return d["data"]


def pods():
    return gql("query { myself { pods { %s } } }" % POD_FIELDS)["myself"]["pods"]


def row(p):
    rt = p.get("runtime") or {}
    return {"id": p["id"], "name": p["name"], "desiredStatus": p["desiredStatus"], "costPerHr": p["costPerHr"],
            "uptimeInSeconds": rt.get("uptimeInSeconds"), "gpu": (p.get("machine") or {}).get("gpuDisplayName")}


def sshinfo(pid):
    for p in pods():
        if p["id"] != pid:
            continue
        for port in ((p.get("runtime") or {}).get("ports") or []):
            if port.get("privatePort") == 22 and port.get("isIpPublic") and port.get("publicPort"):
                return port["ip"], int(port["publicPort"])
    return None


def terminate(pid, wait_s=120):
    if not any(p["id"] == pid for p in pods()):
        return True
    gql('mutation { podTerminate(input: {podId: "%s"}) }' % pid)
    t0 = time.time()
    while time.time() - t0 < wait_s:
        if not any(p["id"] == pid for p in pods()):
            return True
        time.sleep(5)
    return False


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    if cmd == "pods":
        pre = sys.argv[2] if len(sys.argv) > 2 else ""
        for p in pods():
            if p["name"].startswith(pre):
                print(json.dumps(row(p)))
    elif cmd == "find":
        for p in pods():
            if p["name"] == sys.argv[2]:
                print(json.dumps(row(p)))
    elif cmd == "sshinfo":
        s = sshinfo(sys.argv[2])
        if not s:
            sys.exit(1)
        print(s[0], s[1])
    elif cmd == "terminate":
        ok = terminate(sys.argv[2])
        print(("terminated " if ok else "STILL LISTED ") + sys.argv[2])
        sys.exit(0 if ok else 1)
    else:
        sys.exit(__doc__)
