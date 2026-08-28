"""Full repaired V2 math curriculum, with immutable data and one detached run."""
from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path
import subprocess
import sys
import torch

from cftn_text.checkpoint import atomic_json_dump, atomic_torch_save, build_checkpoint
from cftn_text.config import config_sha256, load_config
from cftn_text.data_generator import file_sha256
from cftn_text.full_math_data import audit_full_data, prepare_full_data
from cftn_text.training import train_math_tower
from tools.pilot_verified_math import assert_idle


def checked_settings(path):
    value = json.loads(Path(path).read_text())
    training = value["math_training"]
    if (value["format"] != "cftn_full_math_training_v1" or not value["require_acceptance_for_best"]
            or not value["promote_final_phase_only"] or value["production_acceptance"]
            or value["remaining_pipeline_enabled"] or training["max_epochs"] != 100
            or training["role_weights"] != [.25, .5, .25]
            or training["objective"] != "computation_roles_v1"
            or not 0 < training["learning_rate"] <= 1e-4
            or training["target_mode"] != "full_trace_v1"
            or training["input_view"] != "shared_problem_v1"
            or value["retention_baseline"]["maximum_drop"] > .03):
        raise ValueError("invalid full-training safety contract")
    if [p["through_epoch"] for p in value["phases"]] != [20, 40, 60, 100]:
        raise ValueError("unexpected curriculum deadlines")
    for phase in value["phases"]:
        if sum(phase["source_quotas"].values()) != value["curriculum"]["examples_per_epoch"]:
            raise ValueError("source quotas must equal epoch sample count")
    for phase in value["phases"][:3]:
        if (phase["minimum_generation_accuracy"] < .99 or phase["minimum_valid_rate"] < 1
                or min(phase["minimum_generation_accuracy_by_family"].values()) < .99
                or min(phase["minimum_trace_exact_by_family"].values()) < .95):
            raise ValueError("foundational gates may not be weakened")
    return value


def run(args):
    ignored = set()
    if args.launcher_pid:
        proc = Path(f"/proc/{args.launcher_pid}/cmdline")
        if proc.exists():
            argv = proc.read_bytes().decode().split("\0")
            if "tools.train_v2_full_supervision" not in argv or "launch" not in argv:
                raise ValueError("unexpected launcher process")
            ignored.add(args.launcher_pid)
    assert_idle(ignore_pids=ignored)
    settings = checked_settings(args.settings)
    if subprocess.check_output(["git", "status", "--porcelain"], text=True).strip():
        raise ValueError("clean tested revision required")
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    for path, digest in ((args.source, settings["source_checkpoint_sha256"]),
                         (args.protected, settings["protected_checkpoint_sha256"])):
        if file_sha256(path) != digest:
            raise ValueError("immutable source/protected hash mismatch")
    # train_math_tower performs the mandatory full audit before GPU allocation;
    # reading its identity here avoids doing that multi-minute audit twice.
    manifest = json.loads((Path(args.data) / "manifest.json").read_text())
    output, work = Path(args.output), Path(args.work)
    if not args.resume:
        output.mkdir(parents=True, exist_ok=False)
        work.mkdir(parents=True, exist_ok=False)
    config = load_config(args.config)
    config["data"]["full_supervision_root"] = str(Path(args.data).resolve())
    config["data"]["full_supervision_sha256"] = manifest["manifest_sha256"]
    config["math_tower"]["layers"] = 24
    # A conversion creates a NEW weights-only initialization checkpoint. It
    # does not mark the trial checkpoint accepted or alter either source file.
    converted = output / "initialization.not_accepted.pth"
    if not args.resume:
        source = torch.load(args.source, map_location="cpu", weights_only=False)
        if source["format"] != "cftn_v2_school_trial_not_promotable_v1" or source["epoch"] != 3:
            raise ValueError("unexpected school warm-start format/epoch")
        if source["effective_math_tower"] != config["math_tower"]:
            raise ValueError("warm-start architecture mismatch")
        payload = build_checkpoint(stage="math", epoch=3, global_step=source["global_step"],
                                   model_state=source["model_state"], optimizer_state={}, scheduler_state={},
                                   scaler_state=None, config_sha256=config_sha256(config),
                                   manifest_sha256=source["contract"]["manifest_sha256"], best_metric=0.0, patience=0,
                                   extra={"production_acceptance": False, "initialization_only": True,
                                          "original_source_sha256": settings["source_checkpoint_sha256"]})
        atomic_torch_save(payload, converted)
        del source, payload
    contract = copy.deepcopy(settings)
    for phase in contract["phases"]:
        phase["balance_families_within_source"] = True
    contract.update(repository_revision=revision, original_source_checkpoint=str(Path(args.source).resolve()),
                    original_source_sha256=settings["source_checkpoint_sha256"], source_checkpoint=str(converted.resolve()),
                    source_checkpoint_sha256=file_sha256(converted), data_root=str(Path(args.data).resolve()),
                    data_manifest_sha256=manifest["manifest_sha256"])
    if args.resume:
        previous = json.loads((output / "recovery_contract.json").read_text())
        if contract != previous:
            raise ValueError("resume contract changed")
    else:
        atomic_json_dump(contract, output / "recovery_contract.json")
        atomic_json_dump(config, output / "effective_config.json")
        atomic_json_dump(manifest, output / "data_manifest.json")
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = True
    wandb = {"enabled": args.wandb_mode != "disabled", "mode": args.wandb_mode,
             "project": "cftn-text-v2", "group": "full-supervision-r1", "run_name": output.name}
    result = train_math_tower(config, resume=args.resume, initial_checkpoint=None if args.resume else converted,
                              artifact_directory=output, working_directory=work, recovery_contract=contract,
                              require_calibration=False, disable_early_stopping=True, wandb_options=wandb)
    for path, digest in ((args.source, settings["source_checkpoint_sha256"]),
                         (args.protected, settings["protected_checkpoint_sha256"])):
        if file_sha256(path) != digest:
            raise RuntimeError("immutable source/protected checkpoint changed")
    atomic_json_dump({"state": result["state"], "production_acceptance": False,
                      "remaining_pipeline_enabled": False,
                      "next_gate": "unchanged full native evaluation before any downstream release",
                      "protected_sources_preserved": True}, output / "release_status.json")
    print(json.dumps(result), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "audit", "launch", "run"))
    parser.add_argument("--data", required=True)
    parser.add_argument("--parent")
    parser.add_argument("--expected-parent-manifest-sha256")
    parser.add_argument("--source")
    parser.add_argument("--protected")
    parser.add_argument("--output")
    parser.add_argument("--work")
    parser.add_argument("--config", default="config/v2_broad_math.yaml")
    parser.add_argument("--settings", default="config/v2_full_supervision.json")
    parser.add_argument("--wandb-mode", choices=("online", "offline", "disabled"), default="offline")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--launcher-pid", type=int)
    args = parser.parse_args()
    if os.name != "posix":
        raise RuntimeError("full dataset building, tests, and training run on RunPod only")
    if args.command in ("prepare", "audit"):
        kwargs = {}
        if args.expected_parent_manifest_sha256:
            kwargs["expected_parent_manifest_sha256"] = args.expected_parent_manifest_sha256
        result = (
            prepare_full_data(args.parent, args.data, **kwargs)
            if args.command == "prepare"
            else audit_full_data(args.data, **kwargs)
        )
        print(json.dumps(result), flush=True)
    elif args.command == "run":
        # Serializes all full-supervision launches without preventing monitoring.
        import fcntl
        with (Path(args.output).parent / ".full_supervision_training.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            run(args)
    else:
        assert_idle()
        checked_settings(args.settings)
        paths = [Path(args.output), Path(str(args.output) + ".stdout.log"), Path(str(args.output) + ".stderr.log"), Path(str(args.output) + ".launcher.json")]
        if any(p.exists() for p in paths) or Path(args.work).exists():
            raise FileExistsError("fresh output/work/log paths required")
        command = [sys.executable, "-u", "-m", "tools.train_v2_full_supervision", "run"]
        for name in ("data", "source", "protected", "output", "work", "config", "settings", "wandb_mode"):
            command += ["--" + name.replace("_", "-"), str(getattr(args, name))]
        command += ["--launcher-pid", str(os.getpid())]
        with paths[1].open("x") as stdout, paths[2].open("x") as stderr:
            proc = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=stdout, stderr=stderr, start_new_session=True)
        launch = {"pid": proc.pid, "pgid": proc.pid, "command": command, "worktree": str(Path.cwd())}
        atomic_json_dump(launch, paths[3])
        print(json.dumps(launch), flush=True)


if __name__ == "__main__":
    main()
