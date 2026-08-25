"""Serve a local read-only V2 dashboard backed by SSH snapshots from RunPod."""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlparse


REMOTE_PROBE = r'''python3 - <<'PY'
import json
import os
import subprocess
import time
from pathlib import Path

try:
    import yaml
except ImportError:
    yaml = None

artifact_root = Path("/workspace/cftn-text/artifacts/v2_broad_math_400k_r4")
data_root = Path("/workspace/cftn-text/data/v2_broad_math_400k_r4")
markers = ("run_v2.py", "prepare_v2_data", "train_math_tower", "recover_v2_math", "train_v2_dispatcher", "train_v1_3_integration", "evaluate_v2", "evaluate_v1_3")
sensitive_names = {
    "api_key",
    "authorization",
    "password",
    "secret",
    "token",
    "wandb_api_key",
}
sensitive_suffixes = ("_api_key", "_access_token", "_auth_token", "_password", "_secret")
stage_directories = {
    "train_math": "math",
    "math_answer_recovery": "math_answer_recovery",
    "select_math_checkpoint": "math_checkpoint_selection",
    "evaluate_math": "evaluation_math_v2",
    "train_learned_dispatcher": "learned_dispatcher_v2",
    "calibrate_frozen_gpt_language": "gpt_language_calibration",
    "train_exact_string_specialist": "string_specialist",
    "seal_native_specialists": "native_specialist_evaluation",
    "train_single_specialist_capacity": "single_specialist_capacity",
    "train_dense_mixed_messages": "dense_mixed_messages",
    "train_dense_recurrent": "dense_recurrent",
    "train_supervised_soft_wake": "supervised_soft_wake",
    "evaluate_zero_update_hard_baseline": "hard_transition_baseline",
    "train_hardened_wake": "hardened_wake",
    "evaluate_native_typed_dispatch": "native_dispatch_evaluation",
    "evaluate_sealed_causal_suite": "sealed_evaluation",
    "assemble_v2_evidence": ".",
}

def read_json(path):
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else {}
    except (OSError, ValueError, json.JSONDecodeError):
        return {}

def read_yaml(path):
    if yaml is None:
        return {}
    try:
        value = yaml.safe_load(path.read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else {}
    except (OSError, ValueError, yaml.YAMLError):
        return {}

def redact(value):
    if isinstance(value, dict):
        result = {}
        for key, item in value.items():
            name = str(key).lower()
            result[str(key)] = (
                "[redacted]"
                if name in sensitive_names or name.endswith(sensitive_suffixes)
                else redact(item)
            )
        return result
    if isinstance(value, list):
        return [redact(item) for item in value]
    return value

def read_jsonl(path, limit=250):
    rows = []
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                parsed = json.loads(line)
                if isinstance(parsed, dict):
                    rows.append(redact(parsed))
                    if len(rows) > limit:
                        rows.pop(0)
    except (OSError, ValueError, json.JSONDecodeError):
        return []
    return rows

def command(arguments):
    try:
        return subprocess.run(arguments, check=True, capture_output=True, text=True, timeout=8).stdout
    except (OSError, subprocess.SubprocessError):
        return ""

def tail(path, limit=12000):
    try:
        with path.open("rb") as handle:
            handle.seek(0, os.SEEK_END)
            size = handle.tell()
            handle.seek(max(0, size - limit))
            text = handle.read().replace(b"\\x00", b"").decode("utf-8", errors="replace")
        return {"path": str(path), "bytes": size, "modified_unix": path.stat().st_mtime, "text": text[-limit:]}
    except OSError:
        return {"path": str(path), "bytes": 0, "text": ""}

pipeline = redact(read_json(artifact_root / "pipeline_state.json"))
config_path = Path("/workspace/CFTN-MATHS/config/v2_broad_math.yaml")
for stage_details in pipeline.get("stages", {}).values():
    words = str(stage_details.get("command") or "").split()
    if "--config" in words:
        index = words.index("--config")
        if index + 1 < len(words):
            config_path = Path(words[index + 1])
            break
config = read_yaml(config_path)
math_training = config.get("math_training", {})
training_contract = redact({
    "config_path": str(config_path),
    "curriculum": config.get("data", {}).get("curriculum", {}),
    "validation_examples": config.get("data", {}).get("validation_examples"),
    "math_training": {
        key: math_training.get(key)
        for key in (
            "batch_size",
            "max_epochs",
            "minimum_epochs",
            "early_stop_patience",
            "early_stopping_enabled",
            "learning_rate",
            "minimum_learning_rate",
            "warmup_fraction",
        )
    },
})
stage = str(pipeline.get("current_stage") or "")
recovery_root = artifact_root / "math_answer_recovery"
recovery_contract = read_json(recovery_root / "recovery_contract.json")
recovery_status = read_json(recovery_root / "status.json")
recovery_terminal_states = {"completed", "failed_acceptance", "error"}
if recovery_contract and str(recovery_status.get("state") or "starting") not in recovery_terminal_states:
    recovery_curriculum = dict(recovery_contract.get("curriculum") or {})
    recovery_curriculum["phases"] = list(recovery_contract.get("phases") or [])
    recovery_math_training = dict(recovery_contract.get("math_training") or {})
    training_contract = redact({
        "config_path": str(config_path),
        "recovery_contract_path": str(recovery_root / "recovery_contract.json"),
        "source_checkpoint": recovery_contract.get("source_checkpoint"),
        "source_checkpoint_sha256": recovery_contract.get("source_checkpoint_sha256"),
        "curriculum": recovery_curriculum,
        "validation_examples": config.get("data", {}).get("validation_examples"),
        "math_training": recovery_math_training,
    })
    stage = "math_answer_recovery"
    pipeline = dict(pipeline)
    pipeline["state"] = str(recovery_status.get("state") or "starting")
    pipeline["current_stage"] = stage
    pipeline["repository_revision"] = recovery_contract.get("repository_revision") or pipeline.get("repository_revision")
    pipeline_stages = dict(pipeline.get("stages") or {})
    pipeline_stages[stage] = {
        "state": str(recovery_status.get("state") or "starting"),
        "source_checkpoint_sha256": recovery_contract.get("source_checkpoint_sha256"),
    }
    pipeline["stages"] = pipeline_stages
stage_artifacts = {}
for stage_name, directory in stage_directories.items():
    stage_root = artifact_root if directory == "." else artifact_root / directory
    status = redact(read_json(stage_root / "status.json"))
    metrics = read_jsonl(stage_root / "metrics.jsonl")
    summary = redact(read_json(stage_root / "summary.json"))
    report = redact(read_json(stage_root / "report.json"))
    if stage_name == "assemble_v2_evidence" and not report:
        report = redact(read_json(stage_root / "v2_final_report.json"))
    if status or metrics or summary or report or stage_name in pipeline.get("stages", {}):
        stage_artifacts[stage_name] = {
            "directory": str(stage_root),
            "pipeline": pipeline.get("stages", {}).get(stage_name, {}),
            "status": status,
            "metrics": metrics,
            "summary": summary,
            "report": report,
        }
processes = []
for line in command(["ps", "-eo", "pid=,ppid=,etime=,stat=,%cpu=,%mem=,args="]).splitlines():
    if any(marker in line for marker in markers):
        parts = line.split(None, 6)
        if len(parts) == 7:
            processes.append({"pid": parts[0], "ppid": parts[1], "elapsed": parts[2], "state": parts[3], "cpu_percent": parts[4], "memory_percent": parts[5], "command": parts[6]})
gpus = []
for line in command(["nvidia-smi", "--query-gpu=index,name,utilization.gpu,memory.used,memory.total,temperature.gpu", "--format=csv,noheader,nounits"]).splitlines():
    columns = [value.strip() for value in line.split(",")]
    if len(columns) == 6:
        gpus.append({"index": columns[0], "name": columns[1], "utilization_percent": columns[2], "memory_used_mib": columns[3], "memory_total_mib": columns[4], "temperature_c": columns[5]})
checkpoints = []
for path in sorted(artifact_root.rglob("checkpoint_epoch_*.pth"), key=lambda item: item.stat().st_mtime if item.exists() else 0, reverse=True)[:20]:
    try:
        details = path.stat()
        checkpoints.append({"path": str(path), "size_bytes": details.st_size, "modified_unix": details.st_mtime})
    except OSError:
        pass
wandb_runs = []
for path in artifact_root.rglob("wandb_run.json"):
    parsed = read_json(path)
    if parsed:
        wandb_runs.append({"path": str(path), **redact(parsed)})
if stage == "math_answer_recovery":
    stdout = tail(artifact_root / "math_answer_recovery.stdout.log")
    stderr = tail(artifact_root / "math_answer_recovery.stderr.log")
else:
    stdout = tail(artifact_root / "pipeline_logs" / f"{stage}.stdout.log") if stage else {}
    stderr = tail(artifact_root / "pipeline_logs" / f"{stage}.stderr.log") if stage else {}
disk = command(["df", "-h", "/workspace"]).splitlines()[-1:] 
print(json.dumps({"format": "cftn_text_remote_dashboard_v2", "updated_unix": time.time(), "artifact_root": str(artifact_root), "data_root": str(data_root), "pipeline": pipeline, "training_contract": training_contract, "data_preparation": redact(read_json(data_root / "prepare_status.json")), "data_manifest": redact(read_json(data_root / "manifest.json")), "stage_artifacts": stage_artifacts, "current_stage_artifact": stage_artifacts.get(stage, {}), "gpu": {"gpus": gpus}, "processes": processes, "checkpoints": checkpoints, "wandb": {"preflight": redact(read_json(artifact_root / "startup_preflight.json")).get("wandb", {}), "runs": wandb_runs}, "logs": {"stdout": stdout, "stderr": stderr}, "disk": disk}, separators=(",", ":")))
PY
'''


class RemoteProbe:
    def __init__(self, host: str, port: int, identity_file: str, timeout: int) -> None:
        self.host = host
        self.port = int(port)
        self.identity_file = identity_file
        self.timeout = int(timeout)

    def fetch(self) -> dict[str, Any]:
        command = [
            "ssh",
            "-p",
            str(self.port),
            "-i",
            self.identity_file,
            "-o",
            "BatchMode=yes",
            "-o",
            f"ConnectTimeout={self.timeout}",
            f"root@{self.host}",
            "bash",
            "-s",
        ]
        try:
            result = subprocess.run(
                command,
                input=REMOTE_PROBE,
                capture_output=True,
                text=True,
                timeout=self.timeout + 15,
                check=True,
            )
            return json.loads(result.stdout)
        except subprocess.CalledProcessError as exc:
            detail = (exc.stderr or exc.stdout or "remote probe returned no detail").strip()
            return {
                "format": "cftn_text_remote_dashboard_v2",
                "error": f"remote SSH probe failed: {detail[-1000:]}",
                "updated_unix": time.time(),
            }
        except (OSError, subprocess.SubprocessError, ValueError, json.JSONDecodeError) as exc:
            return {"format": "cftn_text_remote_dashboard_v2", "error": str(exc), "updated_unix": time.time()}


PAGE = """<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>CFTN V2 progress</title><style>
:root{color-scheme:dark;--bg:#0d1218;--card:#17212b;--line:#2b3b49;--text:#e7edf3;--muted:#9cabb9;--ok:#5bd190;--warn:#ffc857;--bad:#ff7d74;--blue:#60a5fa;--purple:#c084fc}*{box-sizing:border-box}body{margin:0;padding:18px;background:var(--bg);color:var(--text);font:14px system-ui,sans-serif}h1{margin:0 0 4px}h2{font-size:17px;margin:0 0 10px}h3{font-size:14px;margin:12px 0 7px}.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(235px,1fr));gap:11px;margin:12px 0}.card{background:var(--card);border:1px solid var(--line);border-radius:9px;padding:13px;overflow:auto}.wide{margin:12px 0}.muted{color:var(--muted)}.ok{color:var(--ok)}.warn{color:var(--warn)}.bad{color:var(--bad)}.metric{font-size:25px;font-weight:650;margin:5px 0}.bar{height:9px;border-radius:9px;background:#293846;overflow:hidden}.bar i{display:block;height:100%;background:var(--ok)}.bar.warn i{background:var(--warn)}pre{margin:0;max-height:360px;overflow:auto;white-space:pre-wrap;overflow-wrap:anywhere;background:#090e13;border-radius:7px;padding:10px;font-size:12px}table{border-collapse:collapse;width:100%;font-size:12px}th,td{padding:6px;border-bottom:1px solid var(--line);text-align:left;vertical-align:top;white-space:nowrap}th{position:sticky;top:0;background:var(--card);color:var(--muted)}code{font-size:12px}a{color:#7cc4ff}.chart{width:100%;height:180px;background:#101820;border-radius:7px}.legend{display:flex;gap:14px;flex-wrap:wrap;margin:5px 0 9px}.dot{display:inline-block;width:9px;height:9px;border-radius:50%;margin-right:5px}details{margin-top:8px}summary{cursor:pointer;color:#b9c7d4}.health{font-weight:650}.wrap{white-space:normal;min-width:240px}.pill{display:inline-block;border:1px solid currentColor;border-radius:999px;padding:3px 8px;font-size:12px;font-weight:650}.callout{border-left:4px solid var(--warn);background:#121b23;border-radius:5px;padding:10px 12px;margin:10px 0;line-height:1.45}.callout.okay{border-color:var(--ok)}@media(max-width:600px){body{padding:10px}.metric{font-size:21px}th,td{padding:5px}.chart{height:150px}}</style></head><body>
<h1>CFTN V2 progress</h1><div class="muted" id="stamp">Connecting through SSH…</div>
<div id="summary" class="grid"></div>
<div class="card wide"><h2>Current step</h2><div id="current"></div></div>
<div class="card wide"><h2>Math curriculum</h2><div id="curriculum"></div></div>
<div class="card wide"><h2>Training assessment</h2><div id="assessment"></div></div>
<div class="card wide"><h2>Validation trend</h2><div id="trend"></div></div>
<div class="grid"><div class="card"><h2>Latest validation breakdown</h2><div id="breakdowns"></div></div><div class="card"><h2>Generation validation</h2><div id="generation"></div></div></div>
<div class="card wide"><h2>Pipeline stages</h2><div id="stages"></div><details><summary>Raw artifacts for every stage</summary><pre id="artifacts"></pre></details></div>
<div class="grid"><div class="card"><h2>GPU</h2><div id="gpu"></div></div><div class="card"><h2>Processes</h2><div id="processes"></div></div></div>
<div class="grid"><div class="card"><h2>Checkpoints</h2><div id="checkpoints"></div></div><div class="card"><h2>W&amp;B</h2><div id="wandb"></div></div></div>
<div class="grid"><div class="card"><h2>Current stdout</h2><pre id="stdout"></pre></div><div class="card"><h2>Current stderr</h2><pre id="stderr"></pre></div></div>
<script>
const $=x=>document.getElementById(x), esc=x=>String(x??'').replace(/[&<>"]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));
const numeric=x=>x!==null&&x!==undefined&&x!==''&&Number.isFinite(Number(x));
const n=x=>numeric(x)?Number(x).toLocaleString(undefined,{maximumFractionDigits:4}):'—';
const pct=x=>numeric(x)?(100*Number(x)).toFixed(2)+'%':'—';
function dur(x){if(!numeric(x))return '—';x=Number(x);let h=Math.floor(x/3600),m=Math.floor((x%3600)/60),s=Math.floor(x%60);return (h?h+'h ':'')+(m?m+'m ':'')+s+'s'}
function tbl(rows,heads,raw=false){return rows?.length?'<div style="overflow:auto"><table><tr>'+heads.map(x=>'<th>'+esc(x)+'</th>').join('')+'</tr>'+rows.map(r=>'<tr>'+r.map(x=>'<td'+(String(x).length>45?' class="wrap"':'')+'>'+(raw?x:esc(x))+'</td>').join('')+'</tr>').join('')+'</table></div>':'<span class="muted">None yet.</span>'}
function get(o,path,fallback=null){for(const k of path.split('.')){if(o==null||!(k in o))return fallback;o=o[k]}return o}
function latest(a){return a?.length?a[a.length-1]:{}}
function median(values){let v=values.filter(Number.isFinite).sort((a,b)=>a-b);if(!v.length)return NaN;let m=Math.floor(v.length/2);return v.length%2?v[m]:(v[m-1]+v[m])/2}
function signed(x,digits=1,suffix='%'){x=Number(x);return Number.isFinite(x)?(x>0?'+':'')+x.toFixed(digits)+suffix:'—'}
function finiteTree(o){if(typeof o==='number'&&!Number.isFinite(o))return false;if(Array.isArray(o))return o.every(finiteTree);if(o&&typeof o==='object')return Object.values(o).every(finiteTree);return true}
function lineChart(series,{zeroOne=false}={}){let points=series.flatMap(s=>s.values.filter(v=>Number.isFinite(v.y)));if(!points.length)return '<span class="muted">No completed epochs yet.</span>';let xs=points.map(p=>p.x),ys=points.map(p=>p.y),xmin=Math.min(...xs),xmax=Math.max(...xs),ymin=zeroOne?0:Math.min(...ys),ymax=zeroOne?1:Math.max(...ys);if(ymax===ymin)ymax=ymin+1;let X=x=>35+(x-xmin)/Math.max(1,xmax-xmin)*650,Y=y=>145-(y-ymin)/(ymax-ymin)*120;let paths=series.map(s=>{let v=s.values.filter(p=>Number.isFinite(p.y));return `<polyline fill="none" stroke="${s.color}" stroke-width="2" points="${v.map(p=>X(p.x)+','+Y(p.y)).join(' ')}"/>`}).join('');let legend='<div class="legend">'+series.map(s=>`<span><i class=dot style="background:${s.color}"></i>${esc(s.label)}</span>`).join('')+'</div>';return legend+`<svg class=chart viewBox="0 0 700 165" preserveAspectRatio="none"><line x1="35" y1="25" x2="35" y2="145" stroke="#52616e"/><line x1="35" y1="145" x2="685" y2="145" stroke="#52616e"/><text x="2" y="30" fill="#9cabb9" font-size="11">${esc(zeroOne?'100%':n(ymax))}</text><text x="2" y="148" fill="#9cabb9" font-size="11">${esc(zeroOne?'0%':n(ymin))}</text><text x="35" y="160" fill="#9cabb9" font-size="11">E${xmin}</text><text x="665" y="160" fill="#9cabb9" font-size="11">E${xmax}</text>${paths}</svg>`}
function health(d,p,a){let issues=[],stderr=d.logs?.stderr?.text||'',status=a.status||{};if(p.state==='running'&&!(d.processes||[]).length)issues.push('No expected process found');if(status.updated_unix&&Date.now()/1000-status.updated_unix>1800)issues.push('Stage status has not moved for over 30 minutes');if(/traceback|cuda error|out of memory|nan|infinity/i.test(stderr))issues.push('Error or non-finite marker in stderr');if(!finiteTree(a.metrics||[]))issues.push('Non-finite metric recorded');return issues}
function curriculumState(d,status,live,rows,epochProgress){let c=d.training_contract?.curriculum||{},phases=c.phases||[],epoch=Number(status.epoch??live.epoch??latest(rows).epoch??0),activeIndex=phases.findIndex(x=>epoch<=Number(x.through_epoch));if(activeIndex<0&&phases.length)activeIndex=phases.length-1;let active=phases[activeIndex]||{},start=activeIndex>0?Number(phases[activeIndex-1].through_epoch)+1:1,count=Math.max(1,Number(active.through_epoch)-start+1),phaseProgress=Math.max(0,Math.min(1,(epoch-start+Number(epochProgress||0))/count)),epochTimes=rows.map(r=>Number(get(r,'timing.epoch_seconds',NaN))),typical=median(epochTimes),remaining=Math.max(0,Number(active.through_epoch)-epoch+1-Number(epochProgress||0));return {config:c,phases,activeIndex,active,start,epoch,phaseProgress,typical,transitionEta:Number.isFinite(typical)?remaining*typical:NaN}}
function trainingAssessment(d,p,a,issues,curr){let rows=a.metrics||[];if(!rows.length)return {level:'warn',label:'Too early to assess',detail:'No completed validation epoch is available yet.',html:'<span class=muted>Assessment starts after epoch 1 validation.</span>'};let first=rows[0],last=latest(rows),train0=Number(first.train_loss),train1=Number(last.train_loss),val0=Number(get(first,'validation.loss',NaN)),val1=Number(get(last,'validation.loss',NaN)),tok0=Number(get(first,'validation.teacher_forced_token_accuracy',NaN)),tok1=Number(get(last,'validation.teacher_forced_token_accuracy',NaN)),seq0=Number(get(first,'validation.teacher_forced_sequence_accuracy',NaN)),seq1=Number(get(last,'validation.teacher_forced_sequence_accuracy',NaN)),trainChange=100*(train1/train0-1),valChange=100*(val1/val0-1),tokenPoints=100*(tok1-tok0),sequencePoints=100*(seq1-seq0),learning=train1<train0&&seq1>seq0,pressure=valChange>15||tokenPoints < -1,level=issues.length?'bad':(learning&&pressure?'warn':learning?'ok':'warn'),label=issues.length?'Infrastructure attention required':(learning&&pressure?'Acceptable to continue — watch validation':learning?'Healthy learning trend':'Learning signal is not yet convincing'),patience=Number(last.patience??0),patienceLimit=Number(d.training_contract?.math_training?.early_stop_patience??0),times=rows.map(r=>({epoch:r.epoch,seconds:Number(get(r,'timing.epoch_seconds',NaN))})),typical=median(times.map(x=>x.seconds)),outliers=times.filter(x=>Number.isFinite(typical)&&x.seconds>typical*1.75),phaseName=String(curr.active.name||'current phase').replaceAll('_',' '),detail=learning?'Training loss is falling and exact-sequence accuracy is materially above epoch 1.':'The completed epochs do not yet show both falling training loss and improving exact-sequence accuracy.',next=curr.activeIndex<=1&&curr.epoch<=15&&pressure?'The decisive check is 3–5 completed epochs into difficulty 2: by epochs 14–15, validation loss should stabilize or fall and token accuracy should hold or recover while sequence accuracy rises.':'Continue to judge the run on validation accuracy, validation loss, and accepted-checkpoint selection together.',valInterpretation=valChange<=0?'Improving':valChange>15?'Under pressure':'Slightly elevated',tokenInterpretation=tokenPoints>=0?'Improving on full mixed validation':'Full mixed validation under watch';let metricRows=[['Train loss',n(train0)+' → '+n(train1),signed(trainChange),'Lower is better'],['Validation loss',n(val0)+' → '+n(val1),signed(valChange),valInterpretation],['Token accuracy',pct(tok0)+' → '+pct(tok1),signed(tokenPoints,2,' pp'),tokenInterpretation],['Exact-sequence accuracy',pct(seq0)+' → '+pct(seq1),signed(sequencePoints,2,' pp'),'Checkpoint signal'],['Early-stop patience',patience+' / '+n(patienceLimit),'',patience<=1?'Normal single-epoch noise':'Watch consecutive non-improvements']];let html=`<div class=callout><span class="pill ${level}">${esc(label)}</span><p>${esc(detail)} The model is still in <b>${esc(phaseName)}</b>, while validation covers all difficulty levels.</p><p><b>Next decision point:</b> ${esc(next)}</p></div>`+tbl(metricRows,['Signal','Epoch 1 → latest','Change','Interpretation'])+`<p class=muted>Typical epoch: ${dur(typical)}. ${outliers.length?esc(outliers.map(x=>'Epoch '+x.epoch).join(', '))+' ran unusually slowly; subsequent timing recovered.':'No material timing outlier detected.'}</p>`;return {level,label,detail,html}}
function breakdownTables(validation){let b=validation?.breakdowns||{},html='';for(const key of ['by_source','by_family','by_difficulty']){let rows=Object.entries(b[key]||{}).map(([name,v])=>[name,n(v.examples),n(v.language_loss),pct(v.teacher_forced_token_accuracy),pct(v.teacher_forced_sequence_accuracy)]);if(rows.length)html+=`<h3>${esc(key.replace('by_','By '))}</h3>`+tbl(rows,['Cohort','N','Loss','Token','Sequence'])}return html||'<span class=muted>Aggregate metrics only for this active run. Cohort breakdowns start with the updated trainer.</span>'}
function draw(d){if(d.error){$('stamp').textContent='SSH error: '+d.error;return}let p=d.pipeline||{},a=d.current_stage_artifact||{},status=a.status||{},live=status.metrics||{},rows=a.metrics||[],last=latest(rows),v=last.validation||{},g=v.generation||{},q=d.data_preparation||{},done=q.completed??q.total_records??0,total=q.total??q.total_records??0,dp=total?done/total:0,issues=health(d,p,a),batch=live.epoch_batch_completed,totalBatch=live.epoch_batches_total,ep=live.epoch_progress??(totalBatch?batch/totalBatch:0);
$('stamp').textContent='Updated '+new Date(d.updated_unix*1000).toLocaleString()+' · refreshes every 30 seconds · '+(d.disk||[]).join('');let curr=curriculumState(d,status,live,rows,ep),assessment=trainingAssessment(d,p,a,issues,curr);
$('summary').innerHTML=`<div class=card><b>Pipeline</b><div class="metric ${p.state==='error'?'bad':'ok'}">${esc(p.state||'missing')}</div><code>${esc(p.current_stage||'not running')}</code><div class=muted>stage ${n(p.current_stage_index)} / ${n(p.stage_count)}</div></div><div class=card><b>System health</b><div class="metric health ${issues.length?'bad':'ok'}">${issues.length?'Attention':'Healthy'}</div><div class=muted>${esc(issues.join(' · ')||'Processes, metrics and stderr look normal')}</div></div><div class=card><b>Learning assessment</b><div class="metric health ${assessment.level}">${esc(assessment.level==='warn'?'Watch':assessment.level==='bad'?'Attention':'Healthy')}</div><div class=muted>${esc(assessment.label)}</div></div><div class=card><b>Current epoch</b><div class=metric>${n(status.epoch??live.epoch)}</div><div>${n(batch)} / ${n(totalBatch)} batches</div><div class=bar><i style="width:${Math.max(0,Math.min(100,100*Number(ep||0)))}%"></i></div></div><div class=card><b>Data</b><div class=metric>${n(done)} / ${n(total)}</div><div class=bar><i style="width:${100*dp}%"></i></div><div class=muted>${(100*dp).toFixed(1)}% prepared</div></div><div class=card><b>Revision</b><div class=metric><code>${esc((p.repository_revision||'').slice(0,12)||'—')}</code></div><div class=muted>${esc(p.project||'')}</div></div>`;
$('current').innerHTML=tbl([[status.state||'',status.epoch??live.epoch??'',status.global_step??'',n(live.train_loss_so_far),n(live.learning_rate),n(live.examples_per_second),n(live.steps_per_second),dur(status.elapsed_seconds),dur(live.eta_seconds_to_max_epochs_excluding_validation)]],['State','Epoch','Global step','Rolling train loss','LR','Examples/s','Steps/s','Elapsed','ETA'])+`<details><summary>All current status metrics</summary><pre>${esc(JSON.stringify(status,null,2))}</pre></details>`;
let difficulty=d.data_manifest?.splits?.train?.difficulty_counts||{},examples=Number(curr.config.examples_per_epoch||0),descriptions={foundations:'Variables on both sides, one-operation MathQA, and low-complexity examples across the configured DeepMind modules.',rational_and_nested:'Adds signed fractions, nested parentheses, shorter multi-step language problems, and medium-complexity examples.',systems_and_language:'Adds two-variable systems, longer and distractor word problems, the hardest complexity tier, and the full training set.'},phaseRows=[],cumulative=0,previous=0;for(const [index,phase] of curr.phases.entries()){cumulative+=Number(difficulty[String(phase.max_difficulty)]||0);let start=previous+1,end=Number(phase.through_epoch),state=index<curr.activeIndex?'Completed':index===curr.activeIndex?'Active':'Scheduled',sampling=cumulative<examples?(examples/Math.max(1,cumulative)).toFixed(2)+'× average reuse':'One shuffled pass';phaseRows.push([String(phase.name||'').replaceAll('_',' '),start+'–'+end,'≤ '+phase.max_difficulty,n(cumulative),sampling,state,descriptions[phase.name]||'']);previous=end}let mt=d.training_contract?.math_training||{},validationN=d.training_contract?.validation_examples??d.data_manifest?.splits?.validation?.count??0;
$('curriculum').innerHTML=`<div class=callout><span class="pill ok">Active: ${esc(String(curr.active.name||'unknown').replaceAll('_',' '))}</span><p>Phase progress: <b>${(100*curr.phaseProgress).toFixed(1)}%</b> · next phase after epoch ${n(curr.active.through_epoch)} · estimated transition in <b>${dur(curr.transitionEta)}</b>.</p><div class=bar><i style="width:${100*curr.phaseProgress}%"></i></div></div>`+tbl(phaseRows,['Phase','Epochs','Difficulty','Eligible unique rows','400k examples/epoch','State','What is added'])+`<p><b>Validation is deliberately harder than the active curriculum:</b> all ${n(validationN)} mixed holdout examples are evaluated after every epoch (difficulty 1: ${n(d.data_manifest?.splits?.validation?.difficulty_counts?.['1'])}, difficulty 2: ${n(d.data_manifest?.splits?.validation?.difficulty_counts?.['2'])}, difficulty 3: ${n(d.data_manifest?.splits?.validation?.difficulty_counts?.['3'])}). It is not filtered to the active phase.</p><p class=muted>Training contract: minimum ${n(mt.minimum_epochs)} epochs, maximum ${n(mt.max_epochs)}, early-stop patience ${n(mt.early_stop_patience)} after the minimum, batch size ${n(mt.batch_size)}. Configuration: ${esc(d.training_contract?.config_path||'unavailable')}.</p>`;
$('assessment').innerHTML=assessment.html;
let accSeries=[{label:'Token accuracy',color:'#60a5fa',values:rows.map(r=>({x:r.epoch,y:Number(get(r,'validation.teacher_forced_token_accuracy',NaN))}))},{label:'Sequence accuracy',color:'#5bd190',values:rows.map(r=>({x:r.epoch,y:Number(get(r,'validation.teacher_forced_sequence_accuracy',NaN))}))},{label:'Generation accuracy',color:'#c084fc',values:rows.map(r=>({x:r.epoch,y:Number(get(r,'validation.generation.accuracy',NaN))}))},{label:'Valid answer rate',color:'#ffc857',values:rows.map(r=>({x:r.epoch,y:Number(get(r,'validation.generation.valid_rate',NaN))}))}],lossSeries=[{label:'Train loss',color:'#60a5fa',values:rows.map(r=>({x:r.epoch,y:Number(r.train_loss)}))},{label:'Validation loss',color:'#ff7d74',values:rows.map(r=>({x:r.epoch,y:Number(get(r,'validation.loss',NaN))}))}];
let trendRows=rows.map(r=>[r.epoch,n(r.global_step),n(r.train_loss),n(get(r,'validation.loss')),pct(get(r,'validation.teacher_forced_token_accuracy')),pct(get(r,'validation.teacher_forced_sequence_accuracy')),pct(get(r,'validation.generation.valid_rate')),pct(get(r,'validation.generation.accuracy')),n(r.selection_metric),n(r.best_metric),r.patience??'',n(r.learning_rate),get(r,'curriculum.phase',''),get(r,'curriculum.max_difficulty',''),dur(get(r,'timing.epoch_seconds'))]);
$('trend').innerHTML='<h3>Accuracy</h3>'+lineChart(accSeries,{zeroOne:true})+'<h3>Loss</h3>'+lineChart(lossSeries)+tbl(trendRows,['Epoch','Step','Train loss','Val loss','Token','Sequence','Gen valid','Gen correct','Selection','Best','Patience','LR','Curriculum','Difficulty','Epoch time']);
$('breakdowns').innerHTML=breakdownTables(v);
let failureRows=(g.failure_examples||[]).map(x=>[x.source||'',x.family||'',x.difficulty??'',x.problem||'',x.expected_answer||'',x.parsed_answer??'no valid answer',x.generation||'']);
$('generation').innerHTML=(Object.keys(g).length?tbl([[n(g.examples),pct(g.valid_rate),pct(g.accuracy),pct(g.canonical_string_accuracy),dur(g.elapsed_seconds)]],['N','Valid format','Equivalent answer','Canonical','Time'])+tbl(failureRows,['Source','Family','Difficulty','Problem','Expected','Parsed','Raw generation']):'<span class=muted>Per-epoch native generation starts with the updated trainer. The current run will still perform full generation checkpoint selection after training.</span>');
$('stages').innerHTML=tbl(Object.entries(p.stages||{}).map(([name,x])=>[name,x.state||'',x.returncode??'',x.started_unix?new Date(x.started_unix*1000).toLocaleString():'',x.completed_unix?new Date(x.completed_unix*1000).toLocaleString():'']),['Stage','State','Code','Started','Completed']);$('artifacts').textContent=JSON.stringify(d.stage_artifacts||{},null,2);
$('gpu').innerHTML=tbl((d.gpu?.gpus||[]).map(x=>[x.index,x.name,x.utilization_percent+'%',x.memory_used_mib+' / '+x.memory_total_mib+' MiB',x.temperature_c+'°C']),['#','GPU','Util','Memory','Temp']);
$('processes').innerHTML=tbl((d.processes||[]).map(x=>[x.pid,x.ppid,x.elapsed,x.state,x.cpu_percent+'%',x.command]),['PID','PPID','Elapsed','State','CPU','Command']);
$('checkpoints').innerHTML=tbl((d.checkpoints||[]).map(x=>[x.path.split('/').slice(-2).join('/'),n(x.size_bytes),new Date(x.modified_unix*1000).toLocaleString()]),['Checkpoint','Bytes','Modified']);
let wr=(d.wandb?.runs||[]).map(x=>[esc(x.run_name||''),esc(x.group||''),/^https?:/.test(x.url||'')?`<a href="${esc(x.url)}" target=_blank>open run</a>`:esc(x.url||'pending')]);$('wandb').innerHTML=tbl(wr,['Run','Group','URL'],true)+(!wr.length?'<p class=muted>Run created when training begins.</p>':'');
$('stdout').textContent=d.logs?.stdout?.text||'';$('stderr').textContent=d.logs?.stderr?.text||''}
async function load(){try{let response=await fetch('/api/status',{cache:'no-store'});draw(await response.json())}catch(e){$('stamp').textContent='Dashboard request failed: '+e}}load();setInterval(load,30000)
</script></body></html>"""


def make_handler(probe: RemoteProbe) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        server_version = "CFTNRemoteDashboard/1.0"

        def send_body(self, body: bytes, content_type: str, status: HTTPStatus = HTTPStatus.OK) -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header(
                "Content-Security-Policy",
                "default-src 'self'; style-src 'unsafe-inline'; script-src 'unsafe-inline'",
            )
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:  # noqa: N802
            path = urlparse(self.path).path
            if path == "/":
                self.send_body(PAGE.encode("utf-8"), "text/html; charset=utf-8")
            elif path == "/healthz":
                self.send_body(b'{"ok":true}\n', "application/json; charset=utf-8")
            elif path == "/api/status":
                self.send_body(json.dumps(probe.fetch()).encode("utf-8"), "application/json; charset=utf-8")
            else:
                self.send_body(b"not found\n", "text/plain; charset=utf-8", HTTPStatus.NOT_FOUND)

        def do_POST(self) -> None:  # noqa: N802
            self.send_body(b"read-only dashboard\n", "text/plain; charset=utf-8", HTTPStatus.METHOD_NOT_ALLOWED)

        def log_message(self, _format: str, *_args: object) -> None:
            return

    return Handler


def main() -> None:
    parser = argparse.ArgumentParser(description="Serve local CFTN V2 progress over SSH")
    parser.add_argument("--ssh-host", required=True)
    parser.add_argument("--ssh-port", type=int, default=22)
    parser.add_argument("--identity-file", required=True)
    parser.add_argument("--host", default="0.0.0.0", help="local/LAN bind address")
    parser.add_argument("--port", type=int, default=8787)
    parser.add_argument("--ssh-timeout", type=int, default=15)
    args = parser.parse_args()
    probe = RemoteProbe(args.ssh_host, args.ssh_port, args.identity_file, args.ssh_timeout)
    server = ThreadingHTTPServer((args.host, args.port), make_handler(probe))
    print(f"CFTN dashboard listening on http://{args.host}:{args.port}", flush=True)
    try:
        server.serve_forever(poll_interval=0.5)
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
