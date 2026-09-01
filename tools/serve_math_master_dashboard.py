"""Serve a read-only LAN dashboard for the local math master experiment."""

from __future__ import annotations

import argparse
import json
import subprocess
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

from cftn_text.config import load_config
from tools.run_math_master_experiment import (
    build_contract,
    build_v7_merged_contract,
    build_v8_cumulative_contract,
    build_v9_cumulative_balanced_contract,
    build_v10_multiplication_contract,
)


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


def _gpu_snapshot() -> dict[str, Any]:
    """Read lightweight live GPU telemetry without coupling it to the trainer."""

    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            check=True,
            text=True,
            timeout=3,
        )
        fields = [item.strip() for item in result.stdout.splitlines()[0].split(",")]
        if len(fields) != 4:
            return {"available": False}
        utilization, memory_used, memory_total, temperature = (
            float(value) for value in fields
        )
        return {
            "available": True,
            "utilization_percent": utilization,
            "memory_used_mib": memory_used,
            "memory_total_mib": memory_total,
            "temperature_c": temperature,
        }
    except (FileNotFoundError, IndexError, OSError, ValueError, subprocess.SubprocessError):
        return {"available": False}


def _generation_cap_watch(generation_panels: Any) -> dict[str, Any]:
    """Detect incomplete byte-level generations that end at a panel's budget."""

    if not isinstance(generation_panels, dict):
        return {"available": False, "suspected": False, "panels": []}
    panels: list[dict[str, Any]] = []
    for name, panel in generation_panels.items():
        if not isinstance(panel, dict):
            continue
        budget = int(panel.get("max_new_tokens") or 0)
        rows_path = panel.get("rows_path")
        if budget <= 0 or not isinstance(rows_path, str):
            continue
        total = invalid = invalid_at_cap = 0
        try:
            with Path(rows_path).open("r", encoding="utf-8") as handle:
                for line in handle:
                    row = json.loads(line)
                    if not isinstance(row, dict):
                        continue
                    total += 1
                    generation = str(row.get("generation") or "")
                    has_answer = row.get("parsed_answer") not in (None, "")
                    if not has_answer:
                        invalid += 1
                        # The math tokenizer is byte-level; encoded byte length
                        # therefore identifies outputs that consumed the budget.
                        if len(generation.encode("utf-8")) >= budget:
                            invalid_at_cap += 1
        except (OSError, ValueError, json.JSONDecodeError):
            continue
        panels.append(
            {
                "name": str(name),
                "budget": budget,
                "examples": total,
                "invalid": invalid,
                "invalid_at_cap": invalid_at_cap,
                "suspected": invalid_at_cap > 0,
            }
        )
    return {
        "available": bool(panels),
        "suspected": any(panel["suspected"] for panel in panels),
        "panels": panels,
    }


def _latest_raw_metric(artifact: Path) -> dict[str, Any]:
    """Return the latest raw metric without exposing training data to the UI."""

    metrics = _read_metrics(artifact / "metrics.jsonl", limit=1)
    return metrics[-1] if metrics else {}


def _criterion_examples(artifact: Path, key: str) -> dict[str, Any]:
    """Load literal held-out generation rows that correspond to one gate.

    The paths come only from the latest trainer-written metric. The endpoint
    accepts a gate key, never a filesystem path, so the LAN dashboard cannot
    use it to read arbitrary files.
    """

    raw = _latest_raw_metric(artifact)
    validation = raw.get("validation") if isinstance(raw.get("validation"), dict) else {}
    panels = (
        validation.get("generation_panels")
        if isinstance(validation.get("generation_panels"), dict)
        else {}
    )
    parts = key.split(":")
    panel_name: str | None = None
    family: str | None = None
    operation: str | None = None
    if parts[0] in {"panel", "panel_family"} and len(parts) >= 2:
        panel_name = parts[1]
        if parts[0] == "panel_family" and len(parts) >= 3:
            family = ":".join(parts[2:])
    elif parts[0] == "primary_family" and len(parts) >= 2:
        family = ":".join(parts[1:])
    elif parts[0] == "primary_operation" and len(parts) >= 2:
        operation = ":".join(parts[1:])
    elif parts[0] == "primary_trace_semantic" and len(parts) >= 2:
        family = ":".join(parts[1:])
    elif parts[0] == "primary_trace" and len(parts) >= 2:
        family = ":".join(parts[1:])

    if panel_name is None and parts[0].startswith("primary"):
        panel_name = next(
            (name for name in panels if str(name).startswith("active_")), None
        )
    panel = panels.get(panel_name) if panel_name else None
    rows_path = panel.get("rows_path") if isinstance(panel, dict) else None
    if not isinstance(rows_path, str):
        return {
            "available": False,
            "key": key,
            "message": "No generation rows are available for this criterion.",
            "rows": [],
        }

    rows: list[dict[str, Any]] = []
    try:
        with Path(rows_path).open("r", encoding="utf-8") as handle:
            for line in handle:
                row = json.loads(line)
                if not isinstance(row, dict):
                    continue
                if family is not None and row.get("family") != family:
                    continue
                if operation is not None and row.get("operation") != operation:
                    continue
                rows.append(
                    {
                        "problem": row.get("problem"),
                        "expected_answer": row.get("expected_answer"),
                        "generation": row.get("generation"),
                        "parsed_answer": row.get("parsed_answer"),
                        "correct": row.get("correct"),
                    }
                )
    except (OSError, ValueError, json.JSONDecodeError):
        return {
            "available": False,
            "key": key,
            "message": "The current validation-row file could not be read.",
            "rows": [],
        }
    return {
        "available": True,
        "key": key,
        "panel": panel_name,
        "family": family,
        "operation": operation,
        "rows": rows,
    }


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
    elif contract_profile == "v8_cumulative" and contract["phases"]:
        contract = build_v8_cumulative_contract(contract, manifest)
    elif contract_profile == "v9_cumulative_balanced" and contract["phases"]:
        contract = build_v9_cumulative_balanced_contract(contract, manifest)
    elif contract_profile == "v10_multiplication" and contract["phases"]:
        contract = build_v10_multiplication_contract(contract, manifest)
    compact = [_compact_metric(row) for row in metrics]
    latest = compact[-1] if compact else {}
    latest_raw = metrics[-1] if metrics else {}
    latest_validation = latest_raw.get("validation") or {}
    live_metrics = status.get("metrics") if isinstance(status.get("metrics"), dict) else {}
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
        "live": {
            "learning_rate": live_metrics.get("learning_rate", latest.get("learning_rate")),
            "gpu": _gpu_snapshot(),
        },
        "generation_cap_watch": _generation_cap_watch(
            latest_validation.get("generation_panels")
        ),
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
:root{color-scheme:dark;--bg:#0d1117;--card:#161d27;--line:#2b3544;--text:#e7edf5;--muted:#9ba9ba;--ok:#3ddc97;--warn:#ffbd59;--bad:#ff6b6b;--accent:#68a8ff}*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--text);font:15px system-ui,sans-serif}main{width:min(1100px,100%);margin:auto;padding:18px}.stack{display:grid;grid-template-columns:1fr;gap:14px}.card{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:15px;overflow:auto}h1{font-size:22px;margin:0 0 5px}h2{font-size:17px;margin:0 0 12px}.muted{color:var(--muted)}.metrics{display:grid;grid-template-columns:repeat(auto-fit,minmax(145px,1fr));gap:10px}.metric{border-left:3px solid var(--accent);padding:7px 10px;background:#111821}.metric b{display:block;font-size:19px}.ok{color:var(--ok)}.warn{color:var(--warn)}.bad{color:var(--bad)}.acceptance-note{margin:14px 0 8px}table{width:100%;border-collapse:collapse;font-size:13px}th,td{text-align:left;padding:7px;border-bottom:1px solid var(--line);white-space:nowrap}th{color:var(--muted)}.bar{height:9px;background:#283241;border-radius:9px;overflow:hidden}.bar i{display:block;height:100%;background:var(--accent)}code{white-space:normal}.inspect{appearance:none;border:0;padding:0;background:transparent;color:var(--text);font:inherit;text-align:left;cursor:pointer;text-decoration:underline;text-decoration-color:var(--bad);text-underline-offset:3px}.inspect:hover{color:var(--accent)}dialog{width:min(980px,96vw);max-height:88vh;color:var(--text);background:var(--card);border:1px solid var(--line);border-radius:10px;padding:0}dialog::backdrop{background:#000b}.modal-head{position:sticky;top:0;display:flex;justify-content:space-between;align-items:center;gap:12px;padding:14px 16px;background:var(--card);border-bottom:1px solid var(--line)}.modal-head h2{margin:0}.modal-close{border:1px solid var(--line);border-radius:6px;background:#222d3b;color:var(--text);padding:6px 10px;cursor:pointer}.examples{padding:14px 16px}.example{border:1px solid var(--line);border-radius:8px;margin:0 0 12px;padding:12px}.example.good{border-left:4px solid var(--ok)}.example.bad{border-left:4px solid var(--bad)}.literal{display:block;max-height:300px;overflow:auto;white-space:pre-wrap;overflow-wrap:anywhere;margin:8px 0 0;padding:9px;background:#0b1017;border-radius:6px;font:12px ui-monospace,Consolas,monospace} @media(max-width:600px){main{padding:10px}.card{padding:11px}th,td{padding:6px 5px;font-size:12px}}
</style></head><body><main><h1>CFTN math master experiment</h1><p id="stamp" class="muted">Loading…</p><div class="stack"><section id="overview" class="card"></section><section id="acceptance" class="card"></section><section id="trend" class="card"></section><section id="phases" class="card"></section><section id="checkpoints" class="card"></section></div></main><dialog id="examples-modal"><div class="modal-head"><h2 id="examples-title">Validation examples</h2><button class="modal-close" id="examples-close">Close</button></div><div id="examples-body" class="examples"></div></dialog><script>
const e=s=>String(s??'—').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c])),n=v=>Number.isFinite(Number(v))?Number(v).toFixed(4):'—',lr=v=>Number.isFinite(Number(v))?Number(v).toExponential(5):'—',pct=v=>Number.isFinite(Number(v))?(100*Number(v)).toFixed(2)+'%':'—',dur=v=>Number.isFinite(Number(v))?Number(v).toFixed(1)+'s':'—';
function table(rows,heads){return '<table><thead><tr>'+heads.map(x=>'<th>'+e(x)+'</th>').join('')+'</tr></thead><tbody>'+rows.map(r=>'<tr>'+r.map(x=>'<td>'+x+'</td>').join('')+'</tr>').join('')+'</tbody></table>'}
function words(v){return String(v??'').replaceAll('_',' ')}
function gateName(key){let p=String(key).split(':');if(key==='primary_generation_accuracy')return 'Overall generated-answer accuracy';if(key==='primary_valid_rate')return 'Overall valid-answer rate';if(key==='teacher_forced_token_accuracy')return 'Teacher-forced token accuracy';if(key==='teacher_forced_sequence_accuracy')return 'Teacher-forced exact sequence';if(key==='validation_loss')return 'Validation loss';if(p[0]==='primary_family')return 'Active criterion · '+words(p.slice(1).join(':'))+' · generation accuracy';if(p[0]==='primary_operation')return 'Active operation · '+words(p.slice(1).join(':'))+' · generation accuracy';if(p[0]==='primary_trace_semantic')return 'Active criterion · '+words(p.slice(1).join(':'))+' · semantic trace';if(p[0]==='primary_trace')return 'Active criterion · '+words(p.slice(1).join(':'))+' · exact trace';if(p[0]==='panel')return 'Panel · '+words(p[1])+' · '+words(p.slice(2).join(' '));if(p[0]==='panel_family')return 'Retention panel · '+words(p[1])+' · '+words(p.slice(2).join(':'));return words(key)}
function gateValue(key,v){return v!==null&&v!==undefined&&Number.isFinite(Number(v))?(String(key).includes('loss')?n(v):pct(v)):'—'}
function gateRequirement(key,x){if(Number.isFinite(Number(x.maximum)))return '≤ '+gateValue(key,x.maximum);if(Number.isFinite(Number(x.minimum)))return '≥ '+gateValue(key,x.minimum);return '—'}
function resultBadge(pass){return '<b class="'+(pass?'ok':'bad')+'">'+(pass?'PASS':'MISS')+'</b>'}
function capBadgeForCheck(key,panels){let parts=String(key).split(':'),panelName=(parts[0]==='panel'||parts[0]==='panel_family')?parts[1]:null,active=panels.find(x=>String(x.name).startsWith('active_'));let panel=panelName?panels.find(x=>x.name===panelName):(String(key).startsWith('primary_')?active:null);return panel&&panel.suspected?' <span class="bad">⚠ '+e(panel.invalid_at_cap)+' invalid at '+e(panel.budget)+' cap</span>':''}
function inspectableCriterion(key,label,check,panels){let display=e(label)+capBadgeForCheck(key,panels);return check&&check.pass===false&&Number(check.examples)>0?'<button class="inspect" data-criterion="'+e(key)+'" title="Inspect literal held-out validation outputs">'+display+'</button>':display}
function draw(d){let s=d.status||{},m=d.latest||{},live=d.live||{},g=live.gpu||{},cap=d.generation_cap_watch||{},capPanels=cap.panels||[],capText=!cap.available?'waiting':cap.suspected?'ALERT: '+capPanels.filter(x=>x.suspected).map(x=>x.name+' '+x.invalid_at_cap+'/'+x.examples+' at '+x.budget).join(' · '):'clear',c=d.contract||{},pe=Number(m.phase_epoch||0),mx=Number(c.maximum_epochs_per_phase||1),progress=Math.min(1,pe/mx),state=s.state||'waiting',gpuText=g.available?(n(g.utilization_percent)+'% · '+n(g.temperature_c)+'°C'):'unavailable',vramText=g.available?(n(g.memory_used_mib)+' / '+n(g.memory_total_mib)+' MiB'):'unavailable';document.querySelector('#stamp').textContent='Updated '+new Date().toLocaleTimeString()+' · refreshes every 30 seconds · '+d.artifact;document.querySelector('#overview').innerHTML='<h2>Current state</h2><div class="metrics"><div class="metric"><span>State</span><b class="'+(state==='running'?'ok':state==='failed_acceptance'?'bad':'warn')+'">'+e(state)+'</b></div><div class="metric"><span>Phase</span><b>'+e(m.phase)+'</b></div><div class="metric"><span>Phase epoch</span><b>'+e(pe)+' / '+e(mx)+'</b></div><div class="metric"><span>Global epoch / step</span><b>'+e(s.epoch)+' / '+e(s.global_step)+'</b></div><div class="metric"><span>Epoch time</span><b>'+dur(m.epoch_seconds)+'</b></div><div class="metric"><span>Live learning rate</span><b>'+lr(live.learning_rate)+'</b></div><div class="metric"><span>GPU utilization / temp</span><b>'+e(gpuText)+'</b></div><div class="metric"><span>VRAM used / total</span><b>'+e(vramText)+'</b></div><div class="metric"><span>Generation cap check</span><b class="'+(cap.suspected?'bad':'ok')+'">'+e(capText)+'</b></div></div><p>Each phase may train for <b>'+e(c.minimum_epochs_per_phase)+'–'+e(mx)+' epochs</b> and needs <b>'+e(c.required_passes)+' consecutive complete passes</b>. The full fail-closed maximum is '+e(c.total_phase_budget)+' epochs.</p><div class="bar"><i style="width:'+(100*progress)+'%"></i></div>';
let gate=m.gate_pass===true?'PASS':m.gate_pass===false?'MISS':'interim',checks=Object.entries(m.acceptance_checks||{}),gateRows=[[e('Minimum phase epoch'),e(pe),'≥ '+e(c.minimum_epochs_per_phase),resultBadge(pe>=Number(c.minimum_epochs_per_phase)),'—'],[e('Consecutive complete passes'),e(m.streak),'≥ '+e(c.required_passes),resultBadge(Number(m.streak)>=Number(c.required_passes)),'—']].concat(checks.map(([key,x])=>[inspectableCriterion(key,gateName(key),x,capPanels),gateValue(key,x.observed),gateRequirement(key,x),resultBadge(x.pass===true),Number.isFinite(Number(x.examples))?e(x.examples):'—']));document.querySelector('#acceptance').innerHTML='<h2>Latest validation and acceptance</h2><div class="metrics"><div class="metric"><span>Training loss</span><b>'+n(m.train_loss)+'</b></div><div class="metric"><span>Validation loss</span><b>'+n(m.validation_loss)+'</b></div><div class="metric"><span>Token accuracy</span><b>'+pct(m.token_accuracy)+'</b></div><div class="metric"><span>Exact sequence</span><b>'+pct(m.sequence_accuracy)+'</b></div><div class="metric"><span>Generated answer</span><b>'+pct(m.generation_accuracy)+'</b></div><div class="metric"><span>Valid answer</span><b>'+pct(m.valid_rate)+'</b></div><div class="metric"><span>Gate / streak</span><b class="'+(m.gate_pass?'ok':'warn')+'">'+gate+' · '+e(m.streak)+'/'+e(c.required_passes)+'</b></div><div class="metric"><span>Checkpoint</span><b>'+e(m.checkpoint_eligible?'eligible':'not eligible')+'</b></div></div><p class="muted acceptance-note">Every scientific criterion below must pass in one completed validation. Click a failed scientific criterion to inspect its literal held-out questions and complete model responses. A red cap notice means invalid generations exhausted the panel output allowance.</p>'+table(gateRows,['Acceptance criterion','Observed','Requirement','Result','N']);
let tr=(d.trend||[]).slice().reverse().map(x=>[e(x.epoch),e(x.phase),e(x.phase_epoch),n(x.train_loss),n(x.validation_loss),pct(x.token_accuracy),pct(x.generation_accuracy),pct(x.valid_rate),e(x.gate_pass===true?'PASS':x.gate_pass===false?'MISS':'interim'),e(x.streak)]);document.querySelector('#trend').innerHTML='<h2>Validation trend — newest first</h2>'+table(tr,['Epoch','Phase','Phase epoch','Train loss','Val loss','Token','Generation','Valid','Gate','Streak']);
let ph=(d.phases||[]).map(x=>[e(x.index),e(x.name),e(x.stored_active_records),e(x.stored_replay_records),e(x.minimum_epochs)+'–'+e(x.maximum_epochs),e(x.required_passes),e(x.state)]);document.querySelector('#phases').innerHTML='<h2>Curriculum phases and stored exposure</h2><p class="muted">Epoch sampling: '+pct(c.active_fraction)+' active / '+pct(c.prior_fraction)+' cumulative replay · stored replay floor '+e(c.minimum_replay_rows_per_prior_criterion)+' rows per prior criterion.</p>'+table(ph,['#','Phase','Active rows','Replay rows','Epoch budget','Passes','State']);let cp=(d.checkpoints||[]).map(x=>[e(x.name),e(x.size_mib)+' MiB',new Date(1000*x.modified_unix).toLocaleString()]);document.querySelector('#checkpoints').innerHTML='<h2>Recent checkpoints</h2>'+(cp.length?table(cp,['Checkpoint','Size','Modified']):'<p class="muted">No checkpoint yet.</p>')}
async function showExamples(key){let modal=document.querySelector('#examples-modal'),title=document.querySelector('#examples-title'),body=document.querySelector('#examples-body');title.textContent='Validation examples';body.innerHTML='<p class="muted">Loading literal validation rows…</p>';modal.showModal();try{let response=await fetch('/api/criterion?key='+encodeURIComponent(key),{cache:'no-store'}),data=await response.json();title.textContent=gateName(key)+' · '+((data.rows||[]).length)+' examples';if(!data.available){body.innerHTML='<p class="bad">'+e(data.message||'Examples are unavailable.')+'</p>';return}body.innerHTML=(data.rows||[]).map((row,index)=>'<article class="example '+(row.correct?'good':'bad')+'"><b>Example '+(index+1)+' · '+(row.correct?'correct':'incorrect')+'</b><p><b>Question</b><code class="literal">'+e(row.problem)+'</code></p><p><b>Expected answer:</b> '+e(row.expected_answer)+'</p><p><b>Parsed model answer:</b> '+e(row.parsed_answer)+'</p><p><b>Full model answer</b><code class="literal">'+e(row.generation)+'</code></p></article>').join('')||'<p class="muted">No held-out generation rows matched this criterion.</p>'}catch(err){body.innerHTML='<p class="bad">Could not load validation examples: '+e(err)+'</p>'}}
document.addEventListener('click',event=>{let button=event.target.closest('[data-criterion]');if(button)showExamples(button.dataset.criterion)});document.querySelector('#examples-close').addEventListener('click',()=>document.querySelector('#examples-modal').close());
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
            parsed_url = urlparse(self.path)
            path = parsed_url.path
            if path == "/":
                self.send_body(PAGE.encode(), "text/html; charset=utf-8")
            elif path == "/api/status":
                payload = collect_snapshot(artifact, manifest, config, contract_profile)
                self.send_body(json.dumps(payload).encode(), "application/json; charset=utf-8")
            elif path == "/api/criterion":
                key = parse_qs(parsed_url.query).get("key", [""])[0]
                payload = _criterion_examples(artifact, str(key))
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
    parser.add_argument(
        "--contract-profile",
        choices=("v5", "v7_merged", "v8_cumulative", "v9_cumulative_balanced", "v10_multiplication"),
        default="v5",
    )
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
