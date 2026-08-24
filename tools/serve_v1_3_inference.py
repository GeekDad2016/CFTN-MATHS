from __future__ import annotations

import argparse
import json
import os
import socket
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Mapping
from urllib.parse import urlparse

from cftn_text.v1_3_inference import LoadedV13Runtime


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
STATIC_ROOT = REPOSITORY_ROOT / "cftn_text" / "web_static"
SEALED_V1_3_CONFIG = Path(r"G:\ctfn-text\config\v1_3_multi_specialist.yaml")
MAX_REQUEST_BYTES = 64 * 1024
MAX_PROMPT_BYTES = 4096


class InferenceApplication:
    def __init__(self, runtime: Any) -> None:
        self.runtime = runtime

    def health(self) -> dict[str, Any]:
        return dict(self.runtime.health())

    def infer_payload(self, payload: Any) -> dict[str, Any]:
        if not isinstance(payload, Mapping):
            raise ValueError("request body must be a JSON object")
        prompt = payload.get("prompt")
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("prompt must be a non-empty string")
        prompt_bytes = len(prompt.encode("utf-8"))
        if prompt_bytes > MAX_PROMPT_BYTES:
            raise ValueError(
                f"prompt has {prompt_bytes} bytes, exceeding {MAX_PROMPT_BYTES}"
            )
        towers = payload.get("towers")
        if towers is not None and not isinstance(towers, Mapping):
            raise ValueError("towers must be a JSON object")
        generation = payload.get("generation", {})
        if not isinstance(generation, Mapping):
            raise ValueError("generation must be a JSON object")
        return self.runtime.infer(
            prompt,
            enabled_towers=towers,
            max_gpt_new_tokens=_bounded_integer(
                generation.get("gpt_max_new_tokens", 32), 1, 128,
                "generation.gpt_max_new_tokens",
            ),
            max_specialist_new_tokens=_bounded_integer(
                generation.get("specialist_max_new_tokens", 96), 1, 256,
                "generation.specialist_max_new_tokens",
            ),
        )


def _bounded_integer(value: Any, minimum: int, maximum: int, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{label} must be an integer")
    if value < minimum or value > maximum:
        raise ValueError(f"{label} must be within {minimum}..{maximum}")
    return int(value)


def _local_addresses(port: int) -> list[str]:
    addresses = {"127.0.0.1"}
    try:
        for item in socket.getaddrinfo(socket.gethostname(), None, socket.AF_INET):
            addresses.add(str(item[4][0]))
    except OSError:
        pass
    return [f"http://{address}:{port}/" for address in sorted(addresses)]


def make_handler(application: InferenceApplication):
    class InferenceHandler(BaseHTTPRequestHandler):
        server_version = "CFTNInference/1.0"

        def log_message(self, format: str, *args: Any) -> None:
            print(f"{self.address_string()} - {format % args}")

        def _security_headers(self) -> None:
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("X-Frame-Options", "DENY")
            self.send_header("Referrer-Policy", "no-referrer")
            self.send_header("Cache-Control", "no-store")
            self.send_header(
                "Content-Security-Policy",
                "default-src 'self'; script-src 'self'; style-src 'self'; "
                "img-src 'self' data:; connect-src 'self'; frame-ancestors 'none'",
            )

        def _write_bytes(
            self, status: HTTPStatus, body: bytes, content_type: str
        ) -> None:
            self.send_response(int(status))
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self._security_headers()
            self.end_headers()
            self.wfile.write(body)

        def _write_json(self, status: HTTPStatus, payload: Mapping[str, Any]) -> None:
            body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self._write_bytes(status, body, "application/json; charset=utf-8")

        def do_GET(self) -> None:
            path = urlparse(self.path).path
            if path == "/api/health":
                self._write_json(HTTPStatus.OK, application.health())
                return
            static_files = {
                "/": ("index.html", "text/html; charset=utf-8"),
                "/index.html": ("index.html", "text/html; charset=utf-8"),
                "/app.css": ("app.css", "text/css; charset=utf-8"),
                "/app.js": ("app.js", "text/javascript; charset=utf-8"),
            }
            selected = static_files.get(path)
            if selected is None:
                self._write_json(
                    HTTPStatus.NOT_FOUND,
                    {"ok": False, "error": "not found"},
                )
                return
            filename, content_type = selected
            self._write_bytes(
                HTTPStatus.OK,
                (STATIC_ROOT / filename).read_bytes(),
                content_type,
            )

        def do_POST(self) -> None:
            if urlparse(self.path).path != "/api/infer":
                self._write_json(
                    HTTPStatus.NOT_FOUND,
                    {"ok": False, "error": "not found"},
                )
                return
            content_type = self.headers.get("Content-Type", "")
            if not content_type.lower().startswith("application/json"):
                self._write_json(
                    HTTPStatus.UNSUPPORTED_MEDIA_TYPE,
                    {"ok": False, "error": "Content-Type must be application/json"},
                )
                return
            try:
                content_length = int(self.headers.get("Content-Length", "0"))
            except ValueError:
                content_length = -1
            if content_length < 1 or content_length > MAX_REQUEST_BYTES:
                self._write_json(
                    HTTPStatus.REQUEST_ENTITY_TOO_LARGE,
                    {"ok": False, "error": "request body size is invalid"},
                )
                return
            try:
                payload = json.loads(self.rfile.read(content_length).decode("utf-8"))
                result = application.infer_payload(payload)
            except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
                self._write_json(
                    HTTPStatus.BAD_REQUEST,
                    {"ok": False, "error": str(exc)},
                )
                return
            except Exception as exc:
                self._write_json(
                    HTTPStatus.INTERNAL_SERVER_ERROR,
                    {"ok": False, "error": f"inference failed: {exc}"},
                )
                return
            self._write_json(HTTPStatus.OK, result)

    return InferenceHandler


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Serve the accepted V1.3 typed CFTN inference path on the LAN."
    )
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--config",
        default=str(
            SEALED_V1_3_CONFIG
            if SEALED_V1_3_CONFIG.is_file()
            else REPOSITORY_ROOT / "config" / "v1_3_multi_specialist.yaml"
        ),
    )
    parser.add_argument(
        "--v1-1-artifact-root",
        default=r"G:\ctfn-text\artifacts\v1_1_algorithmic_linear_equations",
    )
    parser.add_argument(
        "--v1-2-artifact-root",
        default=r"G:\ctfn-text\artifacts\v1_2_conditional_bridge",
    )
    parser.add_argument(
        "--v1-3-data-root",
        default=r"G:\ctfn-text\data\manifests\v1_3_multi_specialist",
    )
    parser.add_argument(
        "--v1-3-artifact-root",
        default=r"G:\ctfn-text\artifacts\v1_3_multi_specialist",
    )
    parser.add_argument(
        "--collaboration-checkpoint",
        default=str(
            Path(r"G:\ctfn-text\artifacts\v1_3_multi_specialist")
            / "oracle_hard_answer_bus_recovery"
            / "oracle_hard_answer_bus_recovery.best.pth"
        ),
    )
    parser.add_argument(
        "--dispatcher-checkpoint",
        default=str(
            Path(r"C:\CFTN\learned_dispatcher_v1_3_final_v2")
            / "learned_dispatcher.best.pth"
        ),
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.port < 1 or args.port > 65535:
        raise SystemExit("--port must be within 1..65535")
    os.environ["CFTN_V1_1_ARTIFACT_ROOT"] = str(args.v1_1_artifact_root)
    os.environ["CFTN_V1_2_ARTIFACT_ROOT"] = str(args.v1_2_artifact_root)
    os.environ["CFTN_V1_3_DATA_ROOT"] = str(args.v1_3_data_root)
    os.environ["CFTN_V1_3_ARTIFACT_ROOT"] = str(args.v1_3_artifact_root)
    print("Loading V1.3 model, specialists, collaboration state, and dispatcher...")
    runtime = LoadedV13Runtime(
        config_path=args.config,
        collaboration_checkpoint=args.collaboration_checkpoint,
        dispatcher_checkpoint=args.dispatcher_checkpoint,
        device=args.device,
    )
    server = ThreadingHTTPServer(
        (str(args.host), int(args.port)),
        make_handler(InferenceApplication(runtime)),
    )
    print("CFTN inference console is ready:")
    for address in _local_addresses(int(args.port)):
        print(f"  {address}")
    print("No authentication is enabled. Keep this service on a trusted private LAN.")
    try:
        server.serve_forever(poll_interval=0.25)
    except KeyboardInterrupt:
        print("Stopping inference console.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
