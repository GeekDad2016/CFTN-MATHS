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

artifact_root = Path("/workspace/cftn-text/artifacts/v2_broad_math_400k_r4")
data_root = Path("/workspace/cftn-text/data/v2_broad_math_400k_r4")
markers = ("run_v2.py", "prepare_v2_data", "train_math_tower", "train_v2_dispatcher", "train_v1_3_integration", "evaluate_v2", "evaluate_v1_3")
sensitive = ("api_key", "token", "secret", "authorization", "password")

def read_json(path):
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else {}
    except (OSError, ValueError, json.JSONDecodeError):
        return {}

def redact(value):
    if isinstance(value, dict):
        return {str(key): ("[redacted]" if any(marker in str(key).lower() for marker in sensitive) else redact(item)) for key, item in value.items()}
    if isinstance(value, list):
        return [redact(item) for item in value]
    return value

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
stage = str(pipeline.get("current_stage") or "")
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
stdout = tail(artifact_root / "pipeline_logs" / f"{stage}.stdout.log") if stage else {}
stderr = tail(artifact_root / "pipeline_logs" / f"{stage}.stderr.log") if stage else {}
disk = command(["df", "-h", "/workspace"]).splitlines()[-1:] 
print(json.dumps({"format": "cftn_text_remote_dashboard_v1", "updated_unix": time.time(), "artifact_root": str(artifact_root), "data_root": str(data_root), "pipeline": pipeline, "data_preparation": redact(read_json(data_root / "prepare_status.json")), "gpu": {"gpus": gpus}, "processes": processes, "checkpoints": checkpoints, "wandb": {"preflight": redact(read_json(artifact_root / "startup_preflight.json")).get("wandb", {}), "runs": wandb_runs}, "logs": {"stdout": stdout, "stderr": stderr}, "disk": disk}, separators=(",", ":")))
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
                "format": "cftn_text_remote_dashboard_v1",
                "error": f"remote SSH probe failed: {detail[-1000:]}",
                "updated_unix": time.time(),
            }
        except (OSError, subprocess.SubprocessError, ValueError, json.JSONDecodeError) as exc:
            return {"format": "cftn_text_remote_dashboard_v1", "error": str(exc), "updated_unix": time.time()}


PAGE = """<!doctype html><html><head><meta charset=\"utf-8\"><meta name=\"viewport\" content=\"width=device-width,initial-scale=1\"><title>CFTN V2 progress</title><style>
body{margin:0;padding:20px;background:#10151b;color:#e7edf3;font:15px system-ui,sans-serif}.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(245px,1fr));gap:12px;margin:14px 0}.card{background:#18212b;border:1px solid #2e3c4a;border-radius:9px;padding:13px}.muted{color:#9caaba}.ok{color:#5bd190}.bad{color:#ff7d74}.bar{height:10px;border-radius:9px;background:#293846;overflow:hidden}.bar i{display:block;height:100%;background:#5bd190}pre{margin:0;max-height:320px;overflow:auto;white-space:pre-wrap;overflow-wrap:anywhere;background:#0d1218;border-radius:7px;padding:10px}table{border-collapse:collapse;width:100%}th,td{padding:6px;border-bottom:1px solid #2d3c4a;text-align:left;vertical-align:top}code{font-size:12px}</style></head><body><h1>CFTN V2 progress</h1><div class=\"muted\" id=\"stamp\">Connecting through SSH…</div><div id=\"summary\" class=\"grid\"></div><div class=\"card\"><h2>Pipeline stages</h2><div id=\"stages\"></div></div><div class=\"grid\"><div class=\"card\"><h2>GPU</h2><div id=\"gpu\"></div></div><div class=\"card\"><h2>Processes</h2><div id=\"processes\"></div></div></div><div class=\"grid\"><div class=\"card\"><h2>Checkpoints</h2><div id=\"checkpoints\"></div></div><div class=\"card\"><h2>W&amp;B</h2><div id=\"wandb\"></div></div></div><div class=\"grid\"><div class=\"card\"><h2>Current stdout</h2><pre id=\"stdout\"></pre></div><div class=\"card\"><h2>Current stderr</h2><pre id=\"stderr\"></pre></div></div><script>
const $=x=>document.getElementById(x),esc=x=>String(x??'').replace(/[&<>\"]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','\"':'&quot;'}[c])),num=x=>Number(x||0).toLocaleString();function tbl(rows,heads){return rows?.length?'<table><tr>'+heads.map(x=>'<th>'+esc(x)+'</th>').join('')+'</tr>'+rows.map(r=>'<tr>'+r.map(x=>'<td>'+esc(x)+'</td>').join('')+'</tr>').join('')+'</table>':'<span class=muted>None yet.</span>'}function draw(d){if(d.error){$('stamp').textContent='SSH error: '+d.error;return}let p=d.pipeline||{},q=d.data_preparation||{},done=q.completed??q.total_records??0,total=q.total??q.total_records??0,percent=total?100*done/total:0;$('stamp').textContent='Updated '+new Date(d.updated_unix*1000).toLocaleString()+' · refreshes every 10 seconds · '+(d.disk||[]).join('');$('summary').innerHTML=`<div class=card><b>Pipeline</b><h2 class=${p.state==='error'?'bad':'ok'}>${esc(p.state||'missing')}</h2><code>${esc(p.current_stage||'not running')}</code></div><div class=card><b>Data preparation</b><h2>${num(done)} / ${num(total)}</h2><div class=bar><i style=\"width:${percent}%\"></i></div><div class=muted>${percent.toFixed(1)}% · ${esc(q.split||q.phase||'')}</div></div><div class=card><b>Revision</b><h2><code>${esc((p.repository_revision||'').slice(0,12)||'—')}</code></h2><span>${esc(p.project||'')}</span></div>`;$('stages').innerHTML=tbl(Object.entries(p.stages||{}).map(([n,v])=>[n,v.state||'',v.returncode??'',v.completed_unix?new Date(v.completed_unix*1000).toLocaleTimeString():'' ]),['Stage','State','Code','Completed']);$('gpu').innerHTML=tbl((d.gpu?.gpus||[]).map(g=>[g.index,g.name,g.utilization_percent+'%',g.memory_used_mib+' / '+g.memory_total_mib+' MiB',g.temperature_c+'°C']),['#','GPU','Util','Memory','Temp']);$('processes').innerHTML=tbl((d.processes||[]).map(x=>[x.pid,x.elapsed,x.state,x.cpu_percent+'%',x.command]),['PID','Elapsed','State','CPU','Command']);$('checkpoints').innerHTML=tbl((d.checkpoints||[]).map(x=>[x.path.split('/').slice(-2).join('/'),num(x.size_bytes),new Date(x.modified_unix*1000).toLocaleString()]),['Checkpoint','Bytes','Modified']);$('wandb').innerHTML=tbl((d.wandb?.runs||[]).map(x=>[x.run_name||'',x.group||'',x.url||'pending']),['Run','Group','URL'])+(!(d.wandb?.runs||[]).length?'<p class=muted>Run created when the first training stage starts.</p>':'');$('stdout').textContent=d.logs?.stdout?.text||'';$('stderr').textContent=d.logs?.stderr?.text||''}async function load(){try{draw(await (await fetch('/api/status',{cache:'no-store'})).json())}catch(e){$('stamp').textContent='Dashboard request failed: '+e}}load();setInterval(load,10000)</script></body></html>"""


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
