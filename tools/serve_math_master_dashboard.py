"""Serve a read-only LAN dashboard for the local math master experiment."""

from __future__ import annotations

import argparse
import json
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from cftn_text.config import load_config
from tools.run_math_master_experiment import build_contract, build_v7_merged_contract


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else {}
    except (OSError, ValueError, json.JSONDecodeError):
        return {}


def _read_metrics(path: Path, limit: int = 120) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                value = json.loads(line)
                if isinstance(value, dict):
                    rows.append(value)
                    if len(rows) > limit:
                        rows.pop(0)
    except (OSError, ValueError, json.JSONDecodeError):
        return []
    return rows


def _compact_metric(row: dict[str, Any]) -> dict[str, Any]:
    validation = row.get("validation") or {}
    # curriculum_gate is populated only for an advance, completion, or
    # fail-closed terminal decision. curriculum_acceptance contains the actual
    # per-epoch measurements that the trend must display.
    gate = row.get("curriculum_acceptance") or row.get("curriculum_gate") or {}
    transition = row.get("curriculum_transition") or {}
    checks = gate.get("checks") or {}
    if not isinstance(checks, dict):
        checks = {}
    else:
        checks = {
            str(name): check for name, check in checks.items() if isinstance(check, dict)
        }
    return {
        "epoch": row.get("epoch"),
        "step": row.get("global_step"),
        "phase": transition.get("phase"),
        "phase_epoch": transition.get("phase_epoch"),
        "train_loss": row.get("train_loss"),
        "validation_loss": validation.get("loss"),
        "token_accuracy": validation.get("teacher_forced_token_accuracy"),
        "sequence_accuracy": validation.get("teacher_forced_sequence_accuracy"),
        "generation_accuracy": gate.get("generation_accuracy"),
        "valid_rate": gate.get("valid_rate"),
        "gate_pass": gate.get("pass"),
        "acceptance_checks": checks,
        "streak": transition.get("consecutive_passes"),
        "advance": transition.get("advance"),
        "failed": transition.get("failed"),
        "checkpoint_eligible": row.get("checkpoint_eligible"),
        "learning_rate": row.get("learning_rate"),
        "epoch_seconds": row.get("timing", {}).get("epoch_seconds"),
    }


def collect_snapshot(
    artifact: Path,
    manifest_path: Path,
    config_path: Path,
    contract_profile: str = "v5",
) -> dict[str, Any]:
    status = _read_json(artifact / "status.json")
    summary = _read_json(artifact / "summary.json")
    metrics = _read_metrics(artifact / "metrics.jsonl")
    config = load_config(config_path)
    curriculum = config.get("data", {}).get("curriculum", {})
    manifest = _read_json(manifest_path)
    contract = (
        build_contract(
            manifest,
            minimum_epochs_per_phase=int(curriculum.get("minimum_epochs_per_phase", 10)),
            maximum_epochs_per_phase=int(curriculum.get("maximum_epochs_per_phase", 60)),
            consecutive_passes=int(curriculum.get("advance_after_consecutive_passes", 2)),
            examples_per_epoch=int(curriculum.get("examples_per_epoch", 512)),
        )
        if manifest.get("phases")
        else {"phases": []}
    )
    if contract_profile == "v7_merged" and contract["phases"]:
        contract = build_v7_merged_contract(contract)
    compact = [_compact_metric(row) for row in metrics]
    latest = compact[-1] if compact else {}
    active_name = latest.get("phase")
    phase_targets = manifest.get("phase_train_targets", {})
    phase_audits = manifest.get("audit", {}).get("phase_views", {})
    phases = []
    for index, phase in enumerate(contract["phases"]):
        if active_name == phase["name"]:
            phase_state = "active"
        elif compact and any(
            row.get("phase") == phase["name"] and row.get("advance") for row in compact
        ):
            phase_state = "passed"
        else:
            phase_state = "scheduled"
        phases.append(
            {
                "index": index + 1,
                "name": phase["name"],
                "minimum_epochs": phase["minimum_epochs"],
                "maximum_epochs": phase["maximum_epochs"],
                "required_passes": phase["advance_after_consecutive_passes"],
                "state": phase_state,
                "stored_active_records": phase_targets.get(phase["name"]),
                "stored_replay_records": phase_audits.get(
                    phase["name"], {}
                ).get("replay"),
            }
        )
    checkpoints = []
    try:
        checkpoints = [
            {
                "name": path.name,
                "size_mib": round(path.stat().st_size / (1024 * 1024), 1),
                "modified_unix": path.stat().st_mtime,
            }
            for path in sorted(
                artifact.glob("*.pth"), key=lambda item: item.stat().st_mtime, reverse=True
            )[:5]
        ]
    except OSError:
        checkpoints = []
    return {
        "artifact": str(artifact),
        "status": status,
        "summary": summary,
        "latest": latest,
        "trend": compact[-30:],
        "phases": phases,
        "checkpoints": checkpoints,
        "contract": {
            "examples_per_epoch": curriculum.get("examples_per_epoch", 96),
            "minimum_epochs_per_phase": curriculum.get("minimum_epochs_per_phase", 10),
            "maximum_epochs_per_phase": curriculum.get("maximum_epochs_per_phase", 60),
            "required_passes": curriculum.get("advance_after_consecutive_passes", 2),
            "total_phase_budget": sum(
                int(phase["maximum_epochs"]) for phase in contract["phases"]
            ),
            "active_fraction": manifest.get("replay_policy", {}).get(
                "active_fraction"
            ),
            "prior_fraction": manifest.get("replay_policy", {}).get(
                "prior_fraction"
            ),
            "minimum_replay_rows_per_prior_criterion": manifest.get(
                "replay_policy", {}
            ).get("minimum_rows_per_prior_criterion"),
        },
    }


PAGE = r"""<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>CFTN math curriculum</title><style>
:root{color-scheme:dark;--bg:#0d1117;--card:#161d27;--line:#2b3544;--text:#e7edf5;--muted:#9ba9ba;--ok:#3ddc97;--warn:#ffbd59;--bad:#ff6b6b;--accent:#68a8ff}*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--text);font:15px system-ui,sans-serif}main{width:min(1100px,100%);margin:auto;padding:18px}.stack{display:grid;grid-template-columns:1fr;gap:14px}.card{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:15px;overflow:auto}h1{font-size:22px;margin:0 0 5px}h2{font-size:17px;margin:0 0 12px}.muted{color:var(--muted)}.metrics{display:grid;grid-template-columns:repeat(auto-fit,minmax(145px,1fr));gap:10px}.metric{border-left:3px solid var(--accent);padding:7px 10px;background:#111821}.metric b{display:block;font-size:19px}.ok{color:var(--ok)}.warn{color:var(--warn)}.bad{color:var(--bad)}.acceptance-note{margin:14px 0 8px}table{width:100%;border-collapse:collapse;font-size:13px}th,td{text-align:left;padding:7px;border-bottom:1px solid var(--line);white-space:nowrap}th{color:var(--muted)}.bar{height:9px;background:#283241;border-radius:9px;overflow:hidden}.bar i{display:block;height:100%;background:var(--accent)}code{white-space:normal} @media(max-width:600px){main{padding:10px}.card{padding:11px}th,td{padding:6px 5px;font-size:12px}}
</style></head><body><main><h1>CFTN math master experiment</h1><p id="stamp" class="muted">Loading…</p><div class="stack"><section id="overview" class="card"></section><section id="acceptance" class="card"></section><section id="trend" class="card"></section><section id="phases" class="card"></section><section id="checkpoints" class="card"></section></div></main><script>
const e=s=>String(s??'—').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c])),n=v=>Number.isFinite(Number(v))?Number(v).toFixed(4):'—',pct=v=>Number.isFinite(Number(v))?(100*Number(v)).toFixed(2)+'%':'—',dur=v=>Number.isFinite(Number(v))?Number(v).toFixed(1)+'s':'—';
function table(rows,heads){return '<table><thead><tr>'+heads.map(x=>'<th>'+e(x)+'</th>').join('')+'</tr></thead><tbody>'+rows.map(r=>'<tr>'+r.map(x=>'<td>'+x+'</td>').join('')+'</tr>').join('')+'</tbody></table>'}
function words(v){return String(v??'').replaceAll('_',' ')}
function gateName(key){let p=String(key).split(':');if(key==='primary_generation_accuracy')return 'Overall generated-answer accuracy';if(key==='primary_valid_rate')return 'Overall valid-answer rate';if(key==='teacher_forced_token_accuracy')return 'Teacher-forced token accuracy';if(key==='teacher_forced_sequence_accuracy')return 'Teacher-forced exact sequence';if(key==='validation_loss')return 'Validation loss';if(p[0]==='primary_family')return 'Active criterion · '+words(p.slice(1).join(':'))+' · generation accuracy';if(p[0]==='primary_operation')return 'Active operation · '+words(p.slice(1).join(':'))+' · generation accuracy';if(p[0]==='primary_trace_semantic')return 'Active criterion · '+words(p.slice(1).join(':'))+' · semantic trace';if(p[0]==='primary_trace')return 'Active criterion · '+words(p.slice(1).join(':'))+' · exact trace';if(p[0]==='panel')return 'Panel · '+words(p[1])+' · '+words(p.slice(2).join(' '));if(p[0]==='panel_family')return 'Retention panel · '+words(p[1])+' · '+words(p.slice(2).join(':'));return words(key)}
function gateValue(key,v){return v!==null&&v!==undefined&&Number.isFinite(Number(v))?(String(key).includes('loss')?n(v):pct(v)):'—'}
function gateRequirement(key,x){if(Number.isFinite(Number(x.maximum)))return '≤ '+gateValue(key,x.maximum);if(Number.isFinite(Number(x.minimum)))return '≥ '+gateValue(key,x.minimum);return '—'}
function resultBadge(pass){return '<b class="'+(pass?'ok':'bad')+'">'+(pass?'PASS':'MISS')+'</b>'}
function draw(d){let s=d.status||{},m=d.latest||{},c=d.contract||{},pe=Number(m.phase_epoch||0),mx=Number(c.maximum_epochs_per_phase||1),progress=Math.min(1,pe/mx),state=s.state||'waiting';document.querySelector('#stamp').textContent='Updated '+new Date().toLocaleTimeString()+' · refreshes every 30 seconds · '+d.artifact;document.querySelector('#overview').innerHTML='<h2>Current state</h2><div class="metrics"><div class="metric"><span>State</span><b class="'+(state==='running'?'ok':state==='failed_acceptance'?'bad':'warn')+'">'+e(state)+'</b></div><div class="metric"><span>Phase</span><b>'+e(m.phase)+'</b></div><div class="metric"><span>Phase epoch</span><b>'+e(pe)+' / '+e(mx)+'</b></div><div class="metric"><span>Global epoch / step</span><b>'+e(s.epoch)+' / '+e(s.global_step)+'</b></div><div class="metric"><span>Epoch time</span><b>'+dur(m.epoch_seconds)+'</b></div><div class="metric"><span>Learning rate</span><b>'+n(m.learning_rate)+'</b></div></div><p>Each phase may train for <b>'+e(c.minimum_epochs_per_phase)+'–'+e(mx)+' epochs</b> and needs <b>'+e(c.required_passes)+' consecutive complete passes</b>. The full fail-closed maximum is '+e(c.total_phase_budget)+' epochs.</p><div class="bar"><i style="width:'+(100*progress)+'%"></i></div>';
let gate=m.gate_pass===true?'PASS':m.gate_pass===false?'MISS':'interim',checks=Object.entries(m.acceptance_checks||{}),gateRows=[[e('Minimum phase epoch'),e(pe),'≥ '+e(c.minimum_epochs_per_phase),resultBadge(pe>=Number(c.minimum_epochs_per_phase)),'—'],[e('Consecutive complete passes'),e(m.streak),'≥ '+e(c.required_passes),resultBadge(Number(m.streak)>=Number(c.required_passes)),'—']].concat(checks.map(([key,x])=>[e(gateName(key)),gateValue(key,x.observed),gateRequirement(key,x),resultBadge(x.pass===true),Number.isFinite(Number(x.examples))?e(x.examples):'—']));document.querySelector('#acceptance').innerHTML='<h2>Latest validation and acceptance</h2><div class="metrics"><div class="metric"><span>Training loss</span><b>'+n(m.train_loss)+'</b></div><div class="metric"><span>Validation loss</span><b>'+n(m.validation_loss)+'</b></div><div class="metric"><span>Token accuracy</span><b>'+pct(m.token_accuracy)+'</b></div><div class="metric"><span>Exact sequence</span><b>'+pct(m.sequence_accuracy)+'</b></div><div class="metric"><span>Generated answer</span><b>'+pct(m.generation_accuracy)+'</b></div><div class="metric"><span>Valid answer</span><b>'+pct(m.valid_rate)+'</b></div><div class="metric"><span>Gate / streak</span><b class="'+(m.gate_pass?'ok':'warn')+'">'+gate+' · '+e(m.streak)+'/'+e(c.required_passes)+'</b></div><div class="metric"><span>Checkpoint</span><b>'+e(m.checkpoint_eligible?'eligible':'not eligible')+'</b></div></div><p class="muted acceptance-note">Every scientific criterion below must pass in one completed validation. Phase advancement also requires the minimum phase epoch and the consecutive-pass streak.</p>'+table(gateRows,['Acceptance criterion','Observed','Requirement','Result','N']);
let tr=(d.trend||[]).slice().reverse().map(x=>[e(x.epoch),e(x.phase),e(x.phase_epoch),n(x.train_loss),n(x.validation_loss),pct(x.token_accuracy),pct(x.generation_accuracy),pct(x.valid_rate),e(x.gate_pass===true?'PASS':x.gate_pass===false?'MISS':'interim'),e(x.streak)]);document.querySelector('#trend').innerHTML='<h2>Validation trend — newest first</h2>'+table(tr,['Epoch','Phase','Phase epoch','Train loss','Val loss','Token','Generation','Valid','Gate','Streak']);
let ph=(d.phases||[]).map(x=>[e(x.index),e(x.name),e(x.stored_active_records),e(x.stored_replay_records),e(x.minimum_epochs)+'–'+e(x.maximum_epochs),e(x.required_passes),e(x.state)]);document.querySelector('#phases').innerHTML='<h2>Curriculum phases and stored exposure</h2><p class="muted">Epoch sampling: '+pct(c.active_fraction)+' active / '+pct(c.prior_fraction)+' cumulative replay · stored replay floor '+e(c.minimum_replay_rows_per_prior_criterion)+' rows per prior criterion.</p>'+table(ph,['#','Phase','Active rows','Replay rows','Epoch budget','Passes','State']);let cp=(d.checkpoints||[]).map(x=>[e(x.name),e(x.size_mib)+' MiB',new Date(1000*x.modified_unix).toLocaleString()]);document.querySelector('#checkpoints').innerHTML='<h2>Recent checkpoints</h2>'+(cp.length?table(cp,['Checkpoint','Size','Modified']):'<p class="muted">No checkpoint yet.</p>')}
async function refresh(){try{let r=await fetch('/api/status',{cache:'no-store'});draw(await r.json())}catch(err){document.querySelector('#stamp').textContent='Dashboard error: '+err}}refresh();setInterval(refresh,30000);
</script></body></html>"""


def make_handler(artifact: Path, manifest: Path, config: Path, contract_profile: str):
    class Handler(BaseHTTPRequestHandler):
        def send_body(
            self, body: bytes, content_type: str, status: HTTPStatus = HTTPStatus.OK
        ) -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:  # noqa: N802
            path = urlparse(self.path).path
            if path == "/":
                self.send_body(PAGE.encode(), "text/html; charset=utf-8")
            elif path == "/api/status":
                payload = collect_snapshot(artifact, manifest, config, contract_profile)
                self.send_body(json.dumps(payload).encode(), "application/json; charset=utf-8")
            elif path == "/healthz":
                self.send_body(b'{"ok":true}\n', "application/json")
            else:
                self.send_body(b"not found\n", "text/plain", HTTPStatus.NOT_FOUND)

        def log_message(self, _format: str, *_args: object) -> None:
            return

    return Handler


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", default="C:/CFTN/artifacts/math_master_experiment_100k_v6/run")
    parser.add_argument("--manifest", default="C:/CFTN/.datasets/math_master_experiment_100k_v6/manifest.json")
    parser.add_argument("--config", default="config/math_master_experiment_local_v6.yaml")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8789)
    parser.add_argument("--contract-profile", choices=("v5", "v7_merged"), default="v5")
    args = parser.parse_args()
    server = ThreadingHTTPServer(
        (args.host, args.port),
        make_handler(
            Path(args.artifact).resolve(),
            Path(args.manifest).resolve(),
            Path(args.config).resolve(),
            args.contract_profile,
        ),
    )
    print(f"Math master dashboard listening on http://{args.host}:{args.port}", flush=True)
    try:
        server.serve_forever(0.5)
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
