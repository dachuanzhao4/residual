#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import os
import re
import shlex
import subprocess
import sys
from pathlib import Path
from statistics import mean, pstdev
from typing import Sequence


VAL_LINE_RE = re.compile(
    r"Epoch\s+(?P<epoch>\d+)/(?P<epochs>\d+)\s+Val Loss:\s+(?P<loss>[0-9.]+),\s+Val Acc@1:\s+(?P<acc1>[0-9.]+),\s+Val Acc@5:\s+(?P<acc5>[0-9.]+)"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="PoC sweep (50e): ViT-S/CIFAR10 over linear/orthogonal/ours with different (alpha,beta) inits.")
    parser.add_argument("--nproc-per-node", type=int, default=8)
    parser.add_argument("--torchrun", type=str, default="torchrun")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--results-root", type=Path, default=Path("results-poc-vit-s-cifar10"))
    parser.add_argument("--summary-csv", type=Path, default=Path("results-poc-vit-s-cifar10/summary.csv"))
    parser.add_argument("--max-train-steps", type=int, default=None, help="Override max_train_steps (for debugging).")
    parser.add_argument("--epochs", type=int, default=None, help="Override epochs (for debugging).")
    parser.add_argument("--save-every-epochs", type=int, default=None, help="Override save_every_epochs to control checkpoints.")
    parser.add_argument("--save-every-steps", type=int, default=None, help="Override save_every_steps to control checkpoints.")
    return parser.parse_args()


def cmd_to_str(cmd: Sequence[str]) -> str:
    return " ".join(shlex.quote(c) for c in cmd)


def capture_dirs(path: Path) -> set[Path]:
    if not path.exists():
        return set()
    return {p for p in path.iterdir() if p.is_dir()}


def find_new_dir(before: set[Path], after: set[Path]) -> Path:
    created = sorted(after - before)
    if not created:
        raise RuntimeError("No new results directory found after run.")
    if len(created) > 1:
        created.sort(key=lambda p: p.stat().st_mtime)
    return created[-1]


def parse_val_metrics(log_path: Path) -> dict:
    best_acc1 = -1.0
    best_epoch = None
    last = None
    with log_path.open("r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            m = VAL_LINE_RE.search(line)
            if not m:
                continue
            epoch = int(m.group("epoch"))
            acc1 = float(m.group("acc1"))
            acc5 = float(m.group("acc5"))
            loss = float(m.group("loss"))
            last = {"epoch": epoch, "acc1": acc1, "acc5": acc5, "loss": loss}
            if acc1 > best_acc1:
                best_acc1 = acc1
                best_epoch = epoch

    if last is None:
        raise RuntimeError(f"No validation metrics found in {log_path}")
    return {
        "best_epoch": int(best_epoch) if best_epoch is not None else last["epoch"],
        "best_acc1": float(best_acc1),
        "last_epoch": int(last["epoch"]),
        "last_acc1": float(last["acc1"]),
        "last_acc5": float(last["acc5"]),
        "last_loss": float(last["loss"]),
    }


def main() -> None:
    args = parse_args()
    methods = {
        "linear": Path("configs/poc/vit_s_cifar10_linear_50e.yaml"),
        "orthogonal": Path("configs/poc/vit_s_cifar10_orthogonal_50e.yaml"),
        "ours_a0b0_drift": Path("configs/poc/vit_s_cifar10_ours_a0b0_drift_50e.yaml"),
        "ours_a1e-4_b0_drift": Path("configs/poc/vit_s_cifar10_ours_a1e-4_b0_drift_50e.yaml"),
    }

    args.results_root.mkdir(parents=True, exist_ok=True)
    args.summary_csv.parent.mkdir(parents=True, exist_ok=True)

    records: list[dict] = []

    for method, cfg in methods.items():
        if not cfg.exists():
            raise SystemExit(f"Missing config: {cfg}")
        out_dir = args.results_root / method
        out_dir.mkdir(parents=True, exist_ok=True)

        for seed in args.seeds:
            before = capture_dirs(out_dir)
            cmd = [
                args.torchrun,
                "--standalone",
                f"--nproc_per_node={args.nproc_per_node}",
                "train_classifier.py",
                "--config_file",
                str(cfg),
                "--results_dir",
                str(out_dir),
                "--seed",
                str(seed),
                "--project",
                "neural-sde",
                "--log_backend",
                "csv",
            ]
            if args.max_train_steps is not None:
                cmd.extend(["--max_train_steps", str(args.max_train_steps)])
            if args.epochs is not None:
                cmd.extend(["--epochs", str(args.epochs)])
            if args.save_every_epochs is not None:
                cmd.extend(["--save_every_epochs", str(args.save_every_epochs)])
            if args.save_every_steps is not None:
                cmd.extend(["--save_every_steps", str(args.save_every_steps)])

            env = os.environ.copy()

            print(f"\n=== {method} seed={seed} ===")
            print(cmd_to_str(cmd))
            proc = subprocess.run(cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            sys.stdout.write(proc.stdout)
            sys.stdout.flush()
            if proc.returncode != 0:
                raise RuntimeError(f"Run failed: method={method}, seed={seed}, exit={proc.returncode}")

            after = capture_dirs(out_dir)
            run_dir = find_new_dir(before, after)
            log_path = run_dir / "log.txt"
            metrics = parse_val_metrics(log_path)

            record = {"method": method, "seed": seed, "run_dir": str(run_dir), **metrics}
            records.append(record)

            with args.summary_csv.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(records[0].keys()))
                writer.writeheader()
                writer.writerows(records)

    print("\n=== Summary (best_acc1 mean±std over seeds) ===")
    for method in methods.keys():
        vals = [r["best_acc1"] for r in records if r["method"] == method]
        if not vals:
            continue
        mu = mean(vals)
        sd = pstdev(vals) if len(vals) > 1 else 0.0
        print(f"{method:>18}: {mu:.4f} ± {sd:.4f} (n={len(vals)})")

    print(f"\nWrote summary CSV: {args.summary_csv}")


if __name__ == "__main__":
    main()

