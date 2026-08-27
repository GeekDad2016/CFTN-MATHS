"""Start or exactly resume the competency-gated V2 math curriculum on RunPod.

The default ``auto`` command is intentionally the only operator entry point:
it resumes an existing compatible artifact, otherwise it starts a new artifact
from the newest preserved checkpoint of the previous full-supervision run.
"""
from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path
import subprocess
import sys

import torch

from cftn_text.checkpoint import atomic_json_dump, latest_checkpoint, load_checkpoint
from cftn_text.config import load_config
from cftn_text.data_generator import file_sha256
from cftn_text.full_math_data import audit_full_data
from cftn_text.training import train_math_tower
from tools.pilot_verified_math import assert_idle


DEFAULT_ARTIFACT_ROOT = Path("/workspace/cftn-text/artifacts/v2_broad_math_400k_r4")
DEFAULT_DATA = Path("/workspace/cftn-text/data/full_supervision_v1_20260826")
DEFAULT_PROJECT = Path("/workspace/CFTN-MATHS")
DEFAULT_OUTPUT = DEFAULT_ARTIFACT_ROOT / "math_competency_curriculum_v3"
DEFAULT_PREVIOUS = DEFAULT_ARTIFACT_ROOT / "math_full_supervision_v1"
DEFAULT_PROTECTED = DEFAULT_ARTIFACT_ROOT / "math/math.best.pth"
DEFAULT_WORK = Path("/tmp/cftn-math-competency-v3")
DEFAULT_SETTINGS = "config/v2_full_supervision_v3.json"


def checked_settings(path: str | Path) -> dict:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    training = value["math_training"]
    format_name = value.get("format")
    transition_policy = value.get("curriculum", {}).get("transition_policy")
    if format_name == "cftn_full_math_training_v2":
        expected_policy = "competency_gated_v1"
        expected_max_epochs = 100
        maximum_learning_rate = 1e-4
        expected_examples = 400_000
    elif format_name == "cftn_full_math_training_v3":
        expected_policy = "competency_gated_v2"
        expected_max_epochs = 42
        maximum_learning_rate = 2e-5
        expected_examples = 100_000
    else:
        raise ValueError("invalid competency math safety contract format")
    if (
        not value.get("require_acceptance_for_best")
        or not value.get("promote_final_phase_only")
        or value.get("production_acceptance")
        or value.get("remaining_pipeline_enabled")
        or transition_policy != expected_policy
        or training.get("max_epochs") != expected_max_epochs
        or training.get("objective") != "computation_roles_v1"
        or training.get("role_weights") != [0.25, 0.5, 0.25]
        or training.get("input_view") != "shared_problem_v1"
        or training.get("target_mode") != "full_trace_v1"
        or not 0 < float(training.get("learning_rate", 0)) <= maximum_learning_rate
        or int(value.get("curriculum", {}).get("examples_per_epoch", 0))
        != expected_examples
        or float(value["retention_baseline"]["maximum_drop"]) > 0.03
    ):
        raise ValueError("invalid competency math safety contract")
    phases = value.get("phases", [])
    if (
        not phases
        or sum(int(phase["maximum_epochs"]) for phase in phases)
        != expected_max_epochs
    ):
        raise ValueError("competency phase maxima must match max_epochs")
    if format_name == "cftn_full_math_training_v3":
        entrance = value.get("zero_update_entrance", {})
        preservation = value.get("preservation_distillation", {})
        if (
            not entrance.get("enabled")
            or int(entrance.get("maximum_skipped_phases", 0)) != len(phases) - 1
            or not preservation.get("enabled")
            or set(preservation.get("sources", []))
            != {"deepmind_mathematics", "gsm8k"}
            or preservation.get("baseline_correct_only") is not True
            or not 0 < float(preservation.get("weight", 0)) <= 0.1
        ):
            raise ValueError("invalid V3 entrance or preservation contract")
    verified_sources = {"verified_school_full", "cftn_generated"}
    for index, phase in enumerate(phases):
        if (
            int(phase["minimum_epochs"]) < 2
            or int(phase["maximum_epochs"]) < int(phase["minimum_epochs"])
            or int(phase["advance_after_consecutive_passes"]) < 2
        ):
            raise ValueError("competency phases require bounded repeated evidence")
        groups = phase.get("quota_groups", [])
        if sum(int(group["examples"]) for group in groups) != expected_examples:
            raise ValueError("phase quota groups must equal examples_per_epoch")
        names = [str(group.get("name", "")) for group in groups]
        if len(set(names)) != len(names) or any(not name for name in names):
            raise ValueError("phase quota groups require unique names")
        if any("mathqa" in group.get("filters", {}).get("sources", []) for group in groups):
            raise ValueError("quarantined MathQA rows cannot enter training")
        verified_examples = sum(
            int(group["examples"])
            for group in groups
            if set(group.get("filters", {}).get("sources", [])) <= verified_sources
        )
        required_fraction = 0.70 if index < 3 else 0.50
        if verified_examples / expected_examples < required_fraction:
            raise ValueError("phase has insufficient verified procedural supervision")
    for phase in phases[:2]:
        if (
            float(phase["minimum_generation_accuracy"]) < 0.99
            or float(phase["minimum_valid_rate"]) < 1.0
            or min(phase["minimum_generation_accuracy_by_family"].values()) < 0.99
            or min(phase["minimum_trace_exact_by_family"].values()) < 0.95
        ):
            raise ValueError("foundational gates may not be weakened")
    if not phases[-1].get("stop_on_pass"):
        raise ValueError("final curriculum phase must stop only after acceptance")
    return value


def _clean_revision() -> str:
    if subprocess.check_output(["git", "status", "--porcelain"], text=True).strip():
        raise ValueError("clean tested revision required")
    return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()


def _verify_source(path: Path, config: dict) -> tuple[dict, str]:
    digest = file_sha256(path)
    checkpoint = load_checkpoint(path, expected_stage="math", map_location="cpu")
    effective = checkpoint.get("extra", {}).get("effective_math_tower")
    if not isinstance(effective, dict) or effective != config["math_tower"]:
        raise ValueError("initial checkpoint architecture differs from competency tower")
    return checkpoint, digest


def run(args: argparse.Namespace) -> None:
    ignored: set[int] = set()
    if args.launcher_pid:
        launcher = Path(f"/proc/{args.launcher_pid}/cmdline")
        if launcher.exists():
            argv = launcher.read_bytes().decode().split("\0")
            if "tools.run_v2_math_curriculum" not in argv or "auto" not in argv:
                raise ValueError("unexpected launcher process")
            ignored.add(args.launcher_pid)
    assert_idle(ignore_pids=ignored)
    settings = checked_settings(args.settings)
    revision = _clean_revision()
    manifest = audit_full_data(args.data)
    config = load_config(args.config)
    config["data"]["full_supervision_root"] = str(Path(args.data).resolve())
    config["data"]["full_supervision_sha256"] = manifest["manifest_sha256"]
    config["math_tower"]["layers"] = 24
    output, work = Path(args.output), Path(args.work)
    settings_sha256 = file_sha256(args.settings)

    if file_sha256(args.protected) != settings["protected_checkpoint_sha256"]:
        raise ValueError("protected checkpoint hash mismatch")
    initial_checkpoint: Path | None = None
    if args.resume:
        if not output.is_dir() or not work.is_dir():
            raise FileNotFoundError("resume requires existing output and work directories")
        contract = json.loads((output / "recovery_contract.json").read_text())
        if contract.get("format") != settings["format"]:
            raise ValueError("resume artifact uses a different curriculum format")
        if contract.get("settings_sha256") != settings_sha256:
            raise ValueError("resume settings changed")
        if contract.get("repository_revision") != revision:
            raise ValueError("resume requires the exact tested repository revision")
        if contract.get("data_manifest_sha256") != manifest["manifest_sha256"]:
            raise ValueError("resume data manifest changed")
        source_path = Path(contract["source_checkpoint"])
        if file_sha256(source_path) != contract["source_checkpoint_sha256"]:
            raise ValueError("resume source checkpoint changed")
    else:
        if output.exists() or work.exists():
            raise FileExistsError("fresh competency output/work paths required")
        if not args.source:
            raise ValueError("fresh competency training requires a source checkpoint")
        source_path = Path(args.source).resolve()
        _, source_sha256 = _verify_source(source_path, config)
        output.mkdir(parents=True, exist_ok=False)
        work.mkdir(parents=True, exist_ok=False)
        contract = copy.deepcopy(settings)
        contract.update(
            repository_revision=revision,
            settings_path=str(Path(args.settings).resolve()),
            settings_sha256=settings_sha256,
            source_checkpoint=str(source_path),
            source_checkpoint_sha256=source_sha256,
            source_checkpoint_mode="weights_only_fresh_optimizer_scheduler",
            data_root=str(Path(args.data).resolve()),
            data_manifest_sha256=manifest["manifest_sha256"],
        )
        atomic_json_dump(contract, output / "recovery_contract.json")
        atomic_json_dump(config, output / "effective_config.json")
        atomic_json_dump(manifest, output / "data_manifest.json")
        initial_checkpoint = source_path

    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = True
    wandb = {
        "enabled": args.wandb_mode != "disabled",
        "mode": args.wandb_mode,
        "project": "cftn-text-v2",
        "group": (
            "competency-curriculum-v3"
            if settings["format"] == "cftn_full_math_training_v3"
            else "competency-curriculum-v2"
        ),
        "run_name": output.name,
    }
    result = train_math_tower(
        config,
        resume=args.resume,
        initial_checkpoint=initial_checkpoint,
        artifact_directory=output,
        working_directory=work,
        recovery_contract=contract,
        require_calibration=False,
        disable_early_stopping=True,
        wandb_options=wandb,
    )
    if file_sha256(args.protected) != settings["protected_checkpoint_sha256"]:
        raise RuntimeError("protected checkpoint changed")
    if file_sha256(source_path) != contract["source_checkpoint_sha256"]:
        raise RuntimeError("source checkpoint changed")
    atomic_json_dump(
        {
            "state": result["state"],
            "production_acceptance": False,
            "remaining_pipeline_enabled": False,
            "next_gate": "unchanged full native evaluation before downstream release",
            "protected_sources_preserved": True,
        },
        output / "release_status.json",
    )
    print(json.dumps(result), flush=True)


def _next_attempt(output: Path) -> tuple[Path, Path, Path]:
    for attempt in range(0, 1000):
        stem = output if attempt == 0 else Path(str(output) + f".attempt_{attempt:03d}")
        stdout = Path(str(stem) + ".stdout.log")
        stderr = Path(str(stem) + ".stderr.log")
        launch = Path(str(stem) + ".launcher.json")
        if not stdout.exists() and not stderr.exists() and not launch.exists():
            return stdout, stderr, launch
    raise RuntimeError("no free launch-attempt suffix")


def auto(args: argparse.Namespace) -> None:
    assert_idle()
    checked_settings(args.settings)
    output = Path(args.output)
    resume = output.is_dir()
    source: Path | None = None
    if resume:
        summary = output / "summary.json"
        if summary.is_file():
            result = json.loads(summary.read_text())
            if result.get("state") in {"completed", "failed_acceptance"}:
                print(json.dumps({"action": "terminal", "result": result}), flush=True)
                return
        if not (output / "recovery_contract.json").is_file() or latest_checkpoint(output) is None:
            raise ValueError("existing output is not safely resumable")
    else:
        source = Path(args.source).resolve() if args.source else latest_checkpoint(args.previous_artifact)
        if source is None or not source.is_file():
            raise FileNotFoundError("no previous full-supervision checkpoint is available")
    stdout_path, stderr_path, launcher_path = _next_attempt(output)
    command = [sys.executable, "-u", "-m", "tools.run_v2_math_curriculum", "run"]
    for name in ("data", "protected", "output", "work", "config", "settings", "wandb_mode"):
        command += ["--" + name.replace("_", "-"), str(getattr(args, name))]
    if source is not None:
        command += ["--source", str(source)]
    if resume:
        command.append("--resume")
    command += ["--launcher-pid", str(os.getpid())]
    with stdout_path.open("x") as stdout, stderr_path.open("x") as stderr:
        process = subprocess.Popen(
            command,
            stdin=subprocess.DEVNULL,
            stdout=stdout,
            stderr=stderr,
            start_new_session=True,
        )
    launch = {
        "pid": process.pid,
        "pgid": process.pid,
        "mode": "resume" if resume else "start",
        "command": command,
        "source_checkpoint": str(source) if source is not None else None,
        "stdout": str(stdout_path),
        "stderr": str(stderr_path),
        "worktree": str(Path.cwd()),
    }
    atomic_json_dump(launch, launcher_path)
    print(json.dumps(launch), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", nargs="?", choices=("auto", "run"), default="auto")
    parser.add_argument("--project", default=str(DEFAULT_PROJECT))
    parser.add_argument("--data", default=str(DEFAULT_DATA))
    parser.add_argument("--previous-artifact", default=str(DEFAULT_PREVIOUS))
    parser.add_argument("--source")
    parser.add_argument("--protected", default=str(DEFAULT_PROTECTED))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--work", default=str(DEFAULT_WORK))
    parser.add_argument("--config", default="config/v2_broad_math.yaml")
    parser.add_argument("--settings", default=DEFAULT_SETTINGS)
    parser.add_argument("--wandb-mode", choices=("online", "offline", "disabled"), default="online")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--launcher-pid", type=int)
    args = parser.parse_args()
    if os.name != "posix":
        raise RuntimeError("V2 math training and validation run on RunPod only")
    os.chdir(args.project)
    if args.command == "run":
        import fcntl

        lock_path = Path(args.output).parent / ".math_competency_training.lock"
        with lock_path.open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            run(args)
    else:
        auto(args)


if __name__ == "__main__":
    main()
