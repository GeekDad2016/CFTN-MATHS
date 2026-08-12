from __future__ import annotations

import argparse
import json
import os
import urllib.error
import urllib.parse
import urllib.request
from typing import Any


def _request(
    base_url: str,
    token: str,
    path: str,
    *,
    method: str = "GET",
    payload: dict[str, Any] | None = None,
) -> Any:
    body = None
    headers = {
        "Accept": "application/json",
        "Authorization": f"Bearer {token}",
    }
    if payload is not None:
        body = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"
    request = urllib.request.Request(
        f"{base_url.rstrip('/')}{path}",
        data=body,
        headers=headers,
        method=method,
    )
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            return json.load(response)
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise SystemExit(f"RunPod API returned HTTP {exc.code}: {detail}") from exc


def main() -> None:
    parser = argparse.ArgumentParser(description="Authenticated CFTN RunPod client")
    parser.add_argument(
        "--url",
        default=os.environ.get("CFTN_REMOTE_API_URL"),
        help="RunPod proxy base URL; defaults to CFTN_REMOTE_API_URL",
    )
    parser.add_argument(
        "--token",
        default=os.environ.get("CFTN_REMOTE_API_TOKEN"),
        help="Bearer token; defaults to CFTN_REMOTE_API_TOKEN",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("status")
    subparsers.add_parser("checkpoints")
    logs = subparsers.add_parser("logs")
    logs.add_argument("--stage", default="supervisor")
    logs.add_argument("--stream", choices=("stdout", "stderr"), default="stderr")
    logs.add_argument("--lines", type=int, default=100)
    subparsers.add_parser("pause-after-stage")
    subparsers.add_parser("resume")
    update = subparsers.add_parser("update-resume")
    update.add_argument("--revision", required=True)
    update.add_argument("--expected-current-revision")
    update.add_argument("--no-resume", action="store_true")
    args = parser.parse_args()
    if not args.url:
        parser.error("--url or CFTN_REMOTE_API_URL is required")
    if not args.token:
        parser.error("--token or CFTN_REMOTE_API_TOKEN is required")

    if args.command == "status":
        result = _request(args.url, args.token, "/v1/status")
    elif args.command == "checkpoints":
        result = _request(args.url, args.token, "/v1/checkpoints")
    elif args.command == "logs":
        query = urllib.parse.urlencode(
            {"stage": args.stage, "stream": args.stream, "lines": args.lines}
        )
        result = _request(args.url, args.token, f"/v1/logs?{query}")
    elif args.command == "pause-after-stage":
        result = _request(
            args.url, args.token, "/v1/actions/pause-after-stage", method="POST", payload={}
        )
    elif args.command == "resume":
        result = _request(
            args.url, args.token, "/v1/actions/resume", method="POST", payload={}
        )
    else:
        result = _request(
            args.url,
            args.token,
            "/v1/actions/update-resume",
            method="POST",
            payload={
                "revision": args.revision,
                "expected_current_revision": args.expected_current_revision,
                "resume": not args.no_resume,
            },
        )
    print(json.dumps(result, indent=2, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
