"""
Check the Ray Serve app end to end: deploy it, load the leaderboard, run one
live Boltz-2 prediction, then repeat it to confirm the cache.

    python check_serve.py                       # deploy, check, shut down
    python check_serve.py --keep                # leave the app running afterwards

Needs one L4 GPU for the live prediction replica.
"""

import argparse
import time

import requests
from ray import serve

from serve_viewer import app

URL = "http://localhost:8000"
SMILES = "CNC(=O)c1nn(C)c2ccccc12"   # a small EV-A71 2A binder from the OpenBind release


def timed_predict(smiles: str):
    t0 = time.time()
    r = requests.post(f"{URL}/predict", json={"smiles": smiles}, timeout=900)
    return r, time.time() - t0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--keep", action="store_true", help="keep the app running after the checks")
    ap.add_argument("--smiles", default=SMILES)
    args = ap.parse_args()

    t0 = time.time()
    serve.run(app, blocking=False)
    print(f"[ok] app deployed in {time.time() - t0:.0f} s (includes GPU node startup and model load)")

    page = requests.get(URL, timeout=60)
    assert page.status_code == 200 and "<table>" in page.text, page.text[:300]
    print(f"[ok] leaderboard page: {page.text.count('data-cid=')} complexes listed")

    r, dt = timed_predict(args.smiles)
    body = r.json()
    assert r.status_code == 200, body
    assert body["cif"].startswith("data_"), "prediction did not return an mmCIF structure"
    m = body["metrics"]
    print(f"[ok] live prediction in {dt:.0f} s: status={body['status']} "
          f"ligand_iptm={m.get('ligand_iptm', float('nan')):.2f} "
          f"affinity={m.get('affinity_pred_value', float('nan')):.2f} "
          f"p_binder={m.get('affinity_probability_binary', float('nan')):.2f}")

    r2, dt2 = timed_predict(args.smiles)
    assert r2.status_code == 200 and r2.json()["status"] == "cached", r2.text[:300]
    print(f"[ok] repeat prediction served from cache in {dt2:.1f} s")

    bad = requests.post(f"{URL}/predict", json={"smiles": "not-a-molecule"}, timeout=900)
    assert bad.status_code in (400, 422), bad.status_code
    print(f"[ok] invalid SMILES rejected with HTTP {bad.status_code}")

    print("\nAll checks passed.")
    if args.keep:
        print(f"App is running at {URL}. Press Ctrl+C to stop.")
        while True:
            time.sleep(3600)
    serve.shutdown()


if __name__ == "__main__":
    main()
