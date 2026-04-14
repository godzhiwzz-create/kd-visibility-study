"""DADG training script skeleton.

This script is a skeleton that Codex / the server-side runner will wire
into the existing YOLOv8 KD framework under `/root/kd_visibility/core/kd/`
(server-only; not present in the public repo). Integration points are
marked with `# INTEGRATE:` comments.

Local usage (syntax check only):
    python3 -m py_compile gate/training/train_dadg.py

Server usage:
    python3 gate/training/train_dadg.py --config gate/configs/dadg.yaml [--override key=val]

Critical invariants (DO NOT violate without ablation intent):
    * `cfg.dadg.stop_gradient_from_kd = True` in all main runs.
    * Gate optimizer sees ONLY `L_gate` gradients.
    * Student optimizer sees KD + detection gradients, weighted by
      `stop_gradient(gate_output)`.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path

import torch
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from gate.losses import (  # noqa: E402
    attention_divergence,
    dadg_gate_loss,
    feature_divergence,
    localization_divergence,
    stack_divergences,
)
from gate.models import build_dadg  # noqa: E402


# --------------------------------------------------------------------------
# Config loading
# --------------------------------------------------------------------------

def load_config(path: str, overrides: list[str] | None = None) -> dict:
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)
    for kv in overrides or []:
        key, _, val = kv.partition("=")
        _set_nested(cfg, key.split("."), _parse_value(val))
    return cfg


def _set_nested(d: dict, keys: list[str], value) -> None:
    cur = d
    for k in keys[:-1]:
        cur = cur.setdefault(k, {})
    cur[keys[-1]] = value


def _parse_value(v: str):
    try:
        return json.loads(v)
    except json.JSONDecodeError:
        return v


# --------------------------------------------------------------------------
# Framework integration adapters (server-side plug-in points)
# --------------------------------------------------------------------------

def build_teacher_student(cfg: dict):
    """Return (teacher_model, student_model, dataloader).

    INTEGRATE on server:
        from core.kd.models import load_yolov8_teacher, build_yolov8_student
        from core.kd.data import build_dataloader
        teacher = load_yolov8_teacher(cfg["teacher"]["weights"]).eval()
        for p in teacher.parameters(): p.requires_grad_(False)
        student = build_yolov8_student(cfg["student"])
        loader = build_dataloader(cfg["data"])
        return teacher, student, loader
    """
    raise NotImplementedError(
        "Wire up to server-side core.kd module. See INTEGRATE docstring."
    )


def compute_kd_features(teacher, student, batch) -> dict:
    """Forward teacher + student, return aligned intermediate tensors.

    Expected keys in the returned dict:
        student_feat, teacher_feat     — (B, C, H, W) backbone / neck features
        student_attn, teacher_attn     — (B, H, W) attention maps
        student_boxes, teacher_boxes   — (B, N, 4) matched regression outputs
        detection_loss                 — scalar, standard supervised loss
        feature_kd_loss                — scalar, feature distillation loss
        attention_kd_loss              — scalar
        localization_kd_loss           — scalar
        input_image                    — (B, 3, H, W) for gate forward

    INTEGRATE: use server's existing KD hooks.
    """
    raise NotImplementedError


# --------------------------------------------------------------------------
# Training loop
# --------------------------------------------------------------------------

def train(cfg: dict) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = Path(cfg["experiment"]["output_dir"]) / cfg["experiment"]["name"]
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "config.yaml").write_text(yaml.safe_dump(cfg, allow_unicode=True))

    torch.manual_seed(cfg["experiment"]["seed"])

    teacher, student, loader = build_teacher_student(cfg)
    teacher.to(device)
    student.to(device)
    gate = build_dadg(**{k: cfg["gate"][k] for k in ("in_channels", "widths", "mlp_hidden", "dropout")})
    gate.to(device)

    # Two separate optimizers — gate gradients never touch student/teacher.
    gate_optim = torch.optim.AdamW(
        gate.parameters(),
        lr=cfg["gate"]["lr"],
        weight_decay=cfg["gate"]["weight_decay"],
    )
    # INTEGRATE: student optimizer comes from YOLOv8 trainer; shown here as skeleton.
    student_optim = torch.optim.AdamW(student.parameters(), lr=1e-3)

    tau = float(cfg["dadg"]["tau"])
    floor = float(cfg["dadg"]["entropy_floor"])
    stop_grad = bool(cfg["dadg"]["stop_gradient_from_kd"])
    kd_w = cfg["kd"]

    traj_path = output_dir / "gate_trajectory.csv"
    traj_f = open(traj_path, "w", newline="")
    traj_writer = csv.writer(traj_f)
    traj_writer.writerow(
        ["epoch", "step", "w_feat", "w_attn", "w_loc", "gate_loss", "kd_loss"]
    )

    step = 0
    for epoch in range(int(cfg["student"]["epochs"])):
        for batch in loader:
            # 1) Teacher/student forward — gather per-branch signals and KD losses.
            feats = compute_kd_features(teacher, student, batch)
            img = feats["input_image"].to(device)

            # 2) Divergences, detached from teacher/student graphs — used as
            #    stable supervision targets for the gate.
            with torch.no_grad():
                d_feat = feature_divergence(feats["student_feat"].detach(), feats["teacher_feat"].detach())
                d_attn = attention_divergence(feats["student_attn"].detach(), feats["teacher_attn"].detach())
                d_loc = localization_divergence(
                    feats["student_boxes"].detach(),
                    feats["teacher_boxes"].detach(),
                    reduction=cfg["dadg"]["loc_reduction"],
                )
                divs = stack_divergences(d_feat, d_attn, d_loc).to(device)

            # 3) Gate forward + gate loss.
            gate_w = gate(img)
            g_loss, g_log = dadg_gate_loss(
                gate_w, divs, tau=tau, entropy_floor=floor,
                entropy_weight=cfg["dadg"]["entropy_weight"],
            )
            gate_optim.zero_grad(set_to_none=True)
            g_loss.backward()
            gate_optim.step()

            # 4) KD loss — use the updated gate weights but detached so KD
            #    gradients DO NOT flow into gate. This is the collapse firewall.
            with torch.no_grad():
                gate_w_for_kd = gate(img).detach() if stop_grad else gate(img)
            if not stop_grad:
                # ABLATION ONLY — allow gradient to simulate collapse
                gate_w_for_kd = gate(img)

            w_f, w_a, w_l = gate_w_for_kd[:, 0].mean(), gate_w_for_kd[:, 1].mean(), gate_w_for_kd[:, 2].mean()

            kd_loss = (
                kd_w["feature_weight"] * w_f * feats["feature_kd_loss"]
                + kd_w["attention_weight"] * w_a * feats["attention_kd_loss"]
                + kd_w["localization_weight"] * w_l * feats["localization_kd_loss"]
                + kd_w["detection_weight"] * feats["detection_loss"]
            )
            student_optim.zero_grad(set_to_none=True)
            kd_loss.backward()
            student_optim.step()

            # 5) Log.
            if step % int(cfg["logging"]["log_every"]) == 0:
                traj_writer.writerow([
                    epoch, step,
                    float(gate_w[:, 0].mean()), float(gate_w[:, 1].mean()), float(gate_w[:, 2].mean()),
                    float(g_log["gate/total_loss"]), float(kd_loss.detach()),
                ])
                traj_f.flush()
            step += 1

        # INTEGRATE: YOLOv8 per-epoch val hook → save best.pt to output_dir/weights/
    traj_f.close()
    (output_dir / "done.flag").touch()


# --------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=str)
    parser.add_argument("--override", nargs="*", default=[], help="dot.key=value")
    args = parser.parse_args()
    cfg = load_config(args.config, args.override)
    train(cfg)


if __name__ == "__main__":
    main()
