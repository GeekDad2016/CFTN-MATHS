from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from cftn_text.config import load_config
from cftn_text.runpod_control import RunPodSupervisor, create_control_server


def _environment_flag(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().casefold() in {"1", "true", "yes", "on"}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run one resumable V2 pipeline with an authenticated RunPod API"
    )
    parser.add_argument("--config", default="config/v2_broad_math.yaml")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--host", default=os.environ.get("CFTN_CONTROL_HOST", "0.0.0.0"))
    parser.add_argument(
        "--port", type=int, default=int(os.environ.get("CFTN_CONTROL_PORT", "8000"))
    )
    parser.add_argument("--no-autostart", action="store_true")
    parser.add_argument("--no-wandb", action="store_true")
    args = parser.parse_args()

    repository_root = Path(__file__).resolve().parents[1]
    config_path = (repository_root / args.config).resolve()
    config = load_config(config_path)
    token = os.environ.get("CFTN_CONTROL_API_TOKEN", "")
    supervisor = RunPodSupervisor(
        repository_root=repository_root,
        artifact_root=config["project"]["artifact_root"],
        data_root=config["project"]["data_root"],
        config_path=config_path,
        api_token=token,
        allow_updates=_environment_flag("CFTN_CONTROL_ALLOW_UPDATES", False),
        device=args.device,
        wandb=not args.no_wandb,
    )
    server = create_control_server(supervisor, args.host, args.port)
    address, port = server.server_address[:2]
    pod_id = os.environ.get("RUNPOD_POD_ID")
    proxy_url = f"https://{pod_id}-{port}.proxy.runpod.net" if pod_id else None
    connection = {
        "format": "cftn_text_runpod_connection_v1",
        "bind_address": str(address),
        "port": int(port),
        "proxy_url": proxy_url,
        "authentication": "Authorization: Bearer $CFTN_CONTROL_API_TOKEN",
        "updates_enabled": supervisor.allow_updates,
    }
    (supervisor.control_root / "api_connection.json").write_text(
        json.dumps(connection, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(connection, indent=2), flush=True)
    if not args.no_autostart:
        print(json.dumps(supervisor.autostart(), indent=2), flush=True)
    try:
        server.serve_forever(poll_interval=1.0)
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
