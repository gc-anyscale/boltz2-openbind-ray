"""
Stage 3: Ray Serve app with a leaderboard and a 3D overlay viewer.

    serve run serve_viewer:app

Then open the Serve URL from the Anyscale Workspace "Ports" tab (port 8000).
Reads aiddd_pipeline.py output (one row per crystal complex). Pick a model at
the top, click a complex to see the predicted pose (green) over the X-ray
ligand (orange). Uses 3Dmol.js from a CDN in the browser.

Live prediction: paste a SMILES at the top and Boltz-2 predicts pose and
affinity on an L4 replica (Boltz2Live), reusing the pipeline's Boltz-2 stage.
Repeat SMILES come back from the cache under predictions/live/.

pip install "ray[serve]" fastapi
"""

import hashlib
import json
import os
from pathlib import Path

import pandas as pd
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse, PlainTextResponse
from ray import serve
from ray.serve.handle import DeploymentHandle

SHARED = Path(os.environ.get("AIDDD_SHARED", "/mnt/user_storage/aiddd"))
RESULTS = Path(os.environ.get("AIDDD_RESULTS", SHARED / "results" / "ev71-2a"))
STRUCTURES = SHARED / "raw" / "OpenBind_EV-A71_2A" / "structures"
MODELS = {"boltz2": "Boltz-2", "openbind0": "OpenBind-0", "of3ft": "OF3 fine-tuned",
          "of3ray": "OF3 Ray Train fine-tune"}

api = FastAPI()

PAGE = """<!doctype html><html><head><meta charset="utf-8">
<title>Co-folding benchmark on Ray</title>
<script src="https://cdnjs.cloudflare.com/ajax/libs/3Dmol/2.0.4/3Dmol-min.js"></script>
<style>
 body{font-family:system-ui,sans-serif;margin:0;display:grid;grid-template-columns:380px 1fr;height:100vh}
 #list{overflow:auto;border-right:1px solid #ddd;padding:12px}
 #view{position:relative} #v{width:100%;height:100%}
 table{border-collapse:collapse;width:100%;font-size:13px}
 td,th{padding:4px 6px;border-bottom:1px solid #eee;text-align:left}
 tr:hover{background:#f4f6ff;cursor:pointer} .good{color:#1a7f37;font-weight:600}
 h2{font-size:16px;margin:4px 0 10px} #stats{font-size:13px;color:#444;margin-bottom:10px}
 .tab{display:inline-block;margin:0 6px 8px 0;padding:3px 8px;border:1px solid #ccd;border-radius:12px;font-size:12px;color:#334;text-decoration:none} .tab.on{background:#334;color:#fff}
 #live{padding-bottom:10px;margin-bottom:10px;border-bottom:1px solid #ddd}
 #legend{position:absolute;top:10px;left:10px;background:#fff9;padding:6px 10px;border-radius:6px;font-size:13px}
</style></head><body>
<div id="list">
<form id="live" onsubmit="predict(event)"><h2>Live Boltz-2 prediction</h2>
 <input id="smi" placeholder="SMILES, e.g. CNC(=O)c1nn(C)c2ccccc12" style="width:100%;box-sizing:border-box;padding:5px">
 <button style="margin:6px 0">Predict</button> <span id="lstat"></span></form>
<h2>__MODEL__: __N__ crystal complexes</h2><div>__TABS__</div><div id="stats">__STATS__</div>
<table><tr><th>Complex</th><th>Lig RMSD</th><th>Stage</th><th>pKD</th></tr>__ROWS__</table></div>
<div id="view"><div id="v"></div>
<div id="legend"><b>green</b> __MODEL__ &nbsp; <b>orange</b> X-ray &nbsp; <span id="cur"></span></div></div>
<script>
let viewer = $3Dmol.createViewer("v",{backgroundColor:"white"});
async function show(cid){
  document.getElementById("cur").textContent = cid;
  viewer.clear();
  const p = await (await fetch("/structure/pred/__KEY__/"+cid)).text();
  viewer.addModel(p,"cif");
  viewer.setStyle({model:0},{cartoon:{color:"#cfd8dc",opacity:0.6}});
  viewer.setStyle({model:0,chain:"B"},{stick:{color:"#2e9e4f",radius:0.25}});
  const g = await fetch("/structure/gt/"+cid);
  if (g.ok){ viewer.addModel(await g.text(), "pdb");
    viewer.setStyle({model:1},{});
    viewer.setStyle({model:1,resn:"LIG"},{stick:{color:"#f28c28",radius:0.25}}); }
  viewer.zoomTo({chain:"B"}); viewer.render();
}
async function predict(ev){
  ev.preventDefault();
  const smi = document.getElementById("smi").value.trim(), st = document.getElementById("lstat");
  if(!smi) return;
  st.textContent = "predicting on GPU (about a minute; cached repeats are instant)...";
  const r = await fetch("/predict", {method:"POST", headers:{"Content-Type":"application/json"},
                                     body: JSON.stringify({smiles: smi})});
  const j = await r.json();
  if(!r.ok || j.status === "failed" || j.status === "skipped"){ st.textContent = "failed: " + (j.error || r.status); return; }
  const m = j.metrics, f = v => (typeof v === "number" ? v.toFixed(2) : "n/a");
  st.innerHTML = `ligand ipTM <b>${f(m.ligand_iptm)}</b> &middot; affinity <b>${f(m.affinity_pred_value)}</b> (log10 IC50, &micro;M) &middot; P(binder) <b>${f(m.affinity_probability_binary)}</b> &middot; ${j.seconds.toFixed(0)} s`;
  document.getElementById("cur").textContent = "live: " + smi;
  viewer.clear(); viewer.addModel(j.cif, "cif");
  viewer.setStyle({model:0},{cartoon:{color:"#cfd8dc",opacity:0.6}});
  viewer.setStyle({model:0,chain:"B"},{stick:{color:"#2e9e4f",radius:0.25}});
  viewer.zoomTo({chain:"B"}); viewer.render();
}
const first = document.querySelector("tr[data-cid]"); if(first) show(first.dataset.cid);
</script></body></html>"""


@serve.deployment(ray_actor_options={"num_gpus": 1, "accelerator_type": "L4"},
                  autoscaling_config={"min_replicas": 1, "max_replicas": 4,
                                      "target_ongoing_requests": 1})
class Boltz2Live:
    """Boltz-2 on one L4 per replica. Same predictor class as the batch pipeline."""

    def __init__(self):
        from aiddd_pipeline import Boltz2Predictor

        class LivePredictor(Boltz2Predictor):
            key = "live"   # cache under predictions/live/, separate from the benchmark

        self.pred = LivePredictor()

    def __call__(self, smiles: str) -> dict:
        from aiddd_pipeline import validate
        lig_id = "smi_" + hashlib.sha1(smiles.encode()).hexdigest()[:12]
        batch = validate(pd.DataFrame({"ligand_id": [lig_id], "smiles": [smiles]}))
        r = self.pred(batch).iloc[0]
        out = {"ligand_id": lig_id, "status": r.live_status, "error": r.live_error,
               "metrics": json.loads(r.live_metrics), "seconds": float(r.live_batch_seconds)}
        if r.live_cif:
            out["cif"] = Path(r.live_cif).read_text()
        return out


@serve.deployment(num_replicas=1, ray_actor_options={"num_cpus": 1})
@serve.ingress(api)
class Viewer:
    def __init__(self, live: DeploymentHandle):
        self.live = live
        self.df = pd.read_parquet(RESULTS)
        self.models = [k for k in MODELS if f"{k}_lig_rmsd" in self.df]

    @api.get("/", response_class=HTMLResponse)
    def index(self, model: str = ""):
        key = model if model in self.models else self.models[0]
        df = self.df.dropna(subset=[f"{key}_cif"]).sort_values(f"{key}_lig_rmsd", na_position="last")
        rows = []
        for _, r in df.iterrows():
            rmsd = r[f"{key}_lig_rmsd"]
            cls = "good" if pd.notna(rmsd) and rmsd < 2 else ""
            pkd = "" if pd.isna(r.experimental_pKD) else f"{r.experimental_pKD:.1f}"
            rows.append(
                f'<tr data-cid="{r.complex_name}" onclick="show(\'{r.complex_name}\')">'
                f"<td>{r.complex_name.split('-', 1)[1]}</td><td class='{cls}'>{'' if pd.isna(rmsd) else f'{rmsd:.2f}'}</td>"
                f"<td>{'fragment' if r.fragment_screen else 'follow-on'}</td><td>{pkd}</td></tr>")
        ok = df[f"{key}_lig_rmsd"].dropna()
        stats = (f"median ligand RMSD {ok.median():.2f} &Aring; &middot; {(ok < 2).mean():.0%} under 2 &Aring;"
                 if len(ok) else "not scored yet")
        tabs = "".join(f'<a class="tab{" on" if k == key else ""}" href="/?model={k}">{MODELS[k]}</a>'
                       for k in self.models)
        return (PAGE.replace("__MODEL__", MODELS[key]).replace("__KEY__", key).replace("__N__", str(len(df)))
                .replace("__TABS__", tabs).replace("__STATS__", stats).replace("__ROWS__", "".join(rows)))

    @api.post("/predict")
    async def predict(self, body: dict):
        smiles = str(body.get("smiles", "")).strip()
        if not smiles or len(smiles) > 300:
            return JSONResponse({"status": "failed", "error": "need a SMILES under 300 chars"}, 400)
        out = await self.live.remote(smiles)
        return JSONResponse(out, 200 if out["status"] in ("ok", "cached") else 422)

    @api.get("/structure/pred/{key}/{cid}", response_class=PlainTextResponse)
    def pred(self, key: str, cid: str):
        row = self.df[self.df.complex_name == cid]
        if key not in self.models or row.empty or not row.iloc[0][f"{key}_cif"]:
            return PlainTextResponse("not found", status_code=404)
        return Path(row.iloc[0][f"{key}_cif"]).read_text()

    @api.get("/structure/gt/{cid}", response_class=PlainTextResponse)
    def gt(self, cid: str):
        row = self.df[self.df.complex_name == cid]
        if "ref_pdb" in self.df and not row.empty and Path(row.iloc[0].ref_pdb).exists():
            return Path(row.iloc[0].ref_pdb).read_text()
        hits = list(STRUCTURES.glob(f"*/{cid}/{cid}_complex_ref.pdb"))
        if hits:
            return hits[0].read_text()
        return PlainTextResponse("no ground truth", status_code=404)


app = Viewer.bind(Boltz2Live.bind())
