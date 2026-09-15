#!/usr/bin/env python3

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Generate randomly-initialized DNN checkpoints for the cuMAC ``trtEngine``
latency benchmark.

Given a ``modelConfig.yaml`` describing one or more MLPs (e.g. ``mlp_4x512``:
input 512, four hidden layers ``[512, 512, 512, 512]``, output 32), this script
builds each network, randomizes its parameters, and writes one ``<name>.pt``
checkpoint per model. Each run first wipes prior artifacts from the output
directory, then (by default) also pre-exports the cached ``<name>.onnx`` and a
ready-to-run ``<name>.infConfig.yaml`` next to the checkpoint -- so the model
benchmarks immediately even where the runtime python lacks the ``onnx`` package
(e.g. inside the Aerial build container) -- and a single ``model_summary.pdf``
describing every generated model (layer structure, activation, parameter counts).

The checkpoints are *bit-for-bit compatible* with the bundled ``pt_to_onnx.py``
exporter that ``trtEngine`` invokes on a ``.pt`` cache miss: each network is the
deployable actor path ``actor(encoder(obs)) -> logits``, where (``act`` is the
configurable ``activation``: SiLU by default, or ReLU)

    encoder = mlp(input_dim, encoder_layers, act) + LayerNorm(hidden) + act
    actor   = mlp(hidden, [*actor_head_layers, output_dim], act)

and the saved file is ``{"policy": state_dict, "model_config": {...}}`` -- the
same structure produced by the DRL trainer. By default every hidden layer goes
into the encoder trunk and the actor is a single output projection, so the
network is exactly ``input -> hidden_layers -> output``.

Typical usage::

    python3 gen_models.py                       # uses ./modelConfig.yaml
    python3 gen_models.py -c modelConfig.yaml -o generatedModels
"""
from __future__ import annotations

import argparse
import glob
import importlib.util
import math
import os
import subprocess
import sys
from typing import Any, Sequence

try:
    import torch
    import torch.nn as nn
except Exception as exc:  # pragma: no cover - clearer error than a raw traceback
    sys.stderr.write(
        "ERROR: gen_models.py requires PyTorch. Install it with "
        "'python3 -m pip install torch'.\n"
        f"Import error: {exc}\n"
    )
    raise SystemExit(2)

try:
    import yaml
except Exception as exc:  # pragma: no cover
    sys.stderr.write(
        "ERROR: gen_models.py requires PyYAML. Install it with "
        "'python3 -m pip install pyyaml'.\n"
        f"Import error: {exc}\n"
    )
    raise SystemExit(2)


_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
# pt_to_onnx.py lives one directory up (examples/ml/pt_to_onnx.py).
_PT_TO_ONNX = os.path.normpath(os.path.join(_THIS_DIR, os.pardir, "pt_to_onnx.py"))

# Artifact patterns this script (and the trtEngine cache it feeds) produce. Used
# to wipe a previous run's output so each generation starts from a clean slate.
_GENERATED_GLOBS = ("*.pt", "*.onnx", "*.engine", "*.trt", "*.plan", "*.infConfig.yaml", "*.pdf")


def _load_pt_to_onnx_module():
    """Import the sibling ``pt_to_onnx.py`` as a module.

    Reusing the exporter's own network definition (``ActorLogitsModule``) and
    helpers (``activation_class``) guarantees the generated ``state_dict``
    keys/shapes and activation choices always match what the exporter rebuilds,
    even if the architecture there changes.
    """
    if not os.path.exists(_PT_TO_ONNX):
        raise FileNotFoundError(
            f"could not find pt_to_onnx.py next to this script (looked at {_PT_TO_ONNX})"
        )
    spec = importlib.util.spec_from_file_location("cumac_pt_to_onnx", _PT_TO_ONNX)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module


def clean_output_dir(output_dir: str) -> int:
    """Delete previously-generated artifacts (.pt/.onnx/.engine/.infConfig.yaml,
    plus the trtEngine engine cache) from ``output_dir``.

    Only the known generated file types are removed, so an accidentally shared
    output directory keeps any unrelated files. Returns the count removed.
    """
    if not os.path.isdir(output_dir):
        return 0
    removed = 0
    for pattern in _GENERATED_GLOBS:
        for path in sorted(glob.glob(os.path.join(output_dir, pattern))):
            try:
                os.remove(path)
                removed += 1
            except OSError as err:
                sys.stderr.write(f"WARNING: could not remove {path}: {err}\n")
    return removed


def onnx_sidecar_path(output_dir: str, name: str, obs_normalize: bool) -> str:
    """Path of the cached ONNX the trtEngine benchmark looks for next to a .pt.

    Mirrors trtEngine's ``ensureOnnxFromPt`` naming: ``<stem>.onnx``, with a
    ``.obsnorm`` infix when the observation normalizer is folded into the graph.
    """
    stem = f"{name}.obsnorm" if obs_normalize else name
    return os.path.join(output_dir, f"{stem}.onnx")


def export_onnx(pt_path: str, onnx_path: str, *, include_obs_normalizer: bool,
                input_name: str = "obs", output_name: str = "logits") -> None:
    """Export ``pt_path`` to ``onnx_path`` via the bundled pt_to_onnx.py exporter.

    Runs the exact same command the trtEngine benchmark would, but in this
    script's interpreter -- so the cached ONNX is produced wherever ``onnx`` is
    installed (e.g. on the host), letting the benchmark run even where its own
    python lacks the ``onnx`` package (e.g. inside the build container).
    """
    cmd = [
        sys.executable, _PT_TO_ONNX,
        "--checkpoint", pt_path,
        "--output", onnx_path,
        "--input-name", input_name,
        "--output-name", output_name,
    ]
    if include_obs_normalizer:
        cmd.append("--include-obs-normalizer")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0 or not os.path.exists(onnx_path):
        raise RuntimeError((result.stderr or result.stdout or "").strip()
                           or f"pt_to_onnx.py exited with code {result.returncode}")


def _as_int_list(values: Sequence[Any]) -> list[int]:
    return [int(v) for v in values]


def resolve_layers(spec: dict[str, Any]) -> tuple[list[int], list[int], list[int]]:
    """Resolve (hidden_layers, encoder_layers, actor_head_layers) for one model.

    ``hidden_layers`` is the full list of hidden widths. It can be given
    directly, or via the ``hidden`` + ``depth`` shorthand. The actor-critic
    split defaults to "all hidden layers in the encoder, actor is just the
    output projection"; it can be overridden with explicit ``encoder_layers``
    and/or ``actor_head_layers``.
    """
    hidden_layers = spec.get("hidden_layers")
    if hidden_layers is None:
        hidden = spec.get("hidden")
        depth = spec.get("depth")
        if hidden is None or depth is None:
            raise ValueError(
                "each model needs 'hidden_layers' (a list) or both 'hidden' and 'depth'"
            )
        hidden_layers = [int(hidden)] * int(depth)
    hidden_layers = _as_int_list(hidden_layers)
    if not hidden_layers:
        raise ValueError("'hidden_layers' must contain at least one layer")

    enc = spec.get("encoder_layers")
    act = spec.get("actor_head_layers")
    if enc is None and act is None:
        encoder_layers = list(hidden_layers)
        actor_head_layers: list[int] = []
    else:
        encoder_layers = _as_int_list(enc) if enc is not None else list(hidden_layers)
        actor_head_layers = _as_int_list(act) if act is not None else []
    if not encoder_layers:
        raise ValueError("'encoder_layers' must contain at least one layer")
    return hidden_layers, encoder_layers, actor_head_layers


def randomize_parameters(model: nn.Module, seed: int) -> None:
    """Deterministically (re)initialize every parameter from a random source.

    Linear layers get the standard Kaiming-uniform weight init; LayerNorm gains
    a randomized (rather than identity) affine transform so the saved model is
    genuinely random end-to-end. The seed makes the result reproducible.
    """
    torch.manual_seed(int(seed))
    for module in model.modules():
        if isinstance(module, nn.Linear):
            nn.init.kaiming_uniform_(module.weight, a=math.sqrt(5))
            if module.bias is not None:
                fan_in = module.weight.size(1)
                bound = 1.0 / math.sqrt(fan_in) if fan_in > 0 else 0.0
                nn.init.uniform_(module.bias, -bound, bound)
        elif isinstance(module, nn.LayerNorm):
            if module.elementwise_affine:
                nn.init.normal_(module.weight, mean=1.0, std=0.05)
                nn.init.normal_(module.bias, mean=0.0, std=0.02)


def build_checkpoint(
    actor_module_cls,
    *,
    input_dim: int,
    output_dim: int,
    hidden_layers: list[int],
    encoder_layers: list[int],
    actor_head_layers: list[int],
    obs_normalize: bool,
    seed: int,
    activation: str,
) -> tuple[dict[str, Any], nn.Module]:
    """Build one randomized actor network.

    Returns the trainer-style checkpoint dict and the built ``nn.Module`` (the
    latter so callers can introspect the exact layer structure for reporting).
    """
    model = actor_module_cls(
        obs_dim=input_dim,
        action_dim=output_dim,
        encoder_layers=encoder_layers,
        actor_head_layers=actor_head_layers,
        include_obs_normalizer=False,
        activation=activation,
    )
    randomize_parameters(model, seed)
    model.eval()

    state = dict(model.state_dict())  # encoder.* and actor.* tensors

    # Always store the running-observation normalizer buffers so the checkpoint
    # works whether or not the benchmark folds the normalizer into the graph
    # (this mirrors the real trainer checkpoints, which always carry them).
    if obs_normalize:
        torch.manual_seed(int(seed) + 9973)
        state["obs_normalizer.rms.mean"] = torch.randn(input_dim)
        state["obs_normalizer.rms.var"] = torch.rand(input_dim) + 0.5
        state["obs_normalizer.rms.count"] = torch.tensor(1.0e4)
    else:
        state["obs_normalizer.rms.mean"] = torch.zeros(input_dim)
        state["obs_normalizer.rms.var"] = torch.ones(input_dim)
        state["obs_normalizer.rms.count"] = torch.tensor(0.0)

    model_config = {
        "obs_dim": int(input_dim),
        "action_dim": int(output_dim),
        "hidden": int(hidden_layers[0]) if hidden_layers else 0,
        "depth": len(encoder_layers),
        "hidden_layers": list(hidden_layers),
        "encoder_layers": list(encoder_layers),
        "actor_head_layers": list(actor_head_layers),
        "obs_normalize": bool(obs_normalize),
        "activation": str(activation),
        "generated_by": "gen_models.py",
    }
    return {"policy": state, "model_config": model_config}, model


def describe_model(model: nn.Module) -> list[tuple[str, str, int]]:
    """Ordered ``(label, detail, num_params)`` rows for the deployable policy.

    Walks the actual ``encoder``/``actor`` submodules (and the optional folded
    observation normalizer) so the reported structure always matches the graph
    that gets exported to ONNX -- including the activation modules and the
    LayerNorm at the end of the encoder trunk.
    """
    rows: list[tuple[str, str, int]] = []

    def _detail(m: nn.Module) -> str:
        if isinstance(m, nn.Linear):
            tail = "" if m.bias is not None else "  (no bias)"
            return f"{m.in_features} -> {m.out_features}{tail}"
        if isinstance(m, nn.LayerNorm):
            return f"normalized_shape={tuple(m.normalized_shape)}"
        return ""  # parameterless activation

    if getattr(model, "obs_normalizer", None) is not None:
        rows.append(("obs_normalizer", "folds raw obs (non-trainable)", 0))
    for seq_name in ("encoder", "actor"):
        seq = getattr(model, seq_name, None)
        if seq is None:
            continue
        for idx, layer in enumerate(seq):
            n_params = sum(t.numel() for t in layer.parameters())
            rows.append((f"{seq_name}.{idx} {type(layer).__name__}", _detail(layer), n_params))
    return rows


def write_summary_pdf(path: str, summaries: list[dict[str, Any]]) -> str:
    """Write a single PDF summarizing every generated model.

    Prefers ``reportlab`` (rich tables); falls back to ``matplotlib`` so the
    summary still renders wherever either package is installed. Returns the name
    of the backend used; raises RuntimeError if neither package is available.
    """
    try:
        _write_pdf_reportlab(path, summaries)
        return "reportlab"
    except ImportError:
        pass
    try:
        _write_pdf_matplotlib(path, summaries)
        return "matplotlib"
    except ImportError as exc:
        raise RuntimeError(
            "needs the 'reportlab' or 'matplotlib' package "
            f"(e.g. 'python3 -m pip install reportlab'): {exc}"
        ) from exc


def _summary_facts(s: dict[str, Any]) -> list[tuple[str, str]]:
    """The per-model key/value facts shown above the layer table (both backends)."""
    return [
        ("Structure", s["structure"]),
        ("Activation", str(s["activation"])),
        ("Encoder layers", str(s["encoder_layers"])),
        ("Actor head layers", str(s["actor_head_layers"]) if s["actor_head_layers"] else "[] (output projection only)"),
        ("LayerNorm", "yes (after encoder trunk)"),
        ("Obs normalizer", "on (folded into graph)" if s["obs_normalize"] else "off"),
        ("Engine precision", str(s["precision"])),
        ("RNG seed", str(s["seed"])),
        ("Trainable params (policy)", f"{s['num_params']:,}"),
    ]


def _write_pdf_reportlab(path: str, summaries: list[dict[str, Any]]) -> None:
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import LETTER
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.lib.units import inch
    from reportlab.platypus import (
        PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle,
    )

    styles = getSampleStyleSheet()
    doc = SimpleDocTemplate(
        path, pagesize=LETTER, title="cuMAC trtEngine model summary",
        leftMargin=0.7 * inch, rightMargin=0.7 * inch,
        topMargin=0.7 * inch, bottomMargin=0.7 * inch,
    )
    story: list[Any] = [
        Paragraph("cuMAC trtEngine - generated model summary", styles["Title"]),
        Paragraph(
            f"{len(summaries)} model(s); each network is actor(encoder(obs)) -> logits.",
            styles["Normal"],
        ),
        Spacer(1, 0.2 * inch),
    ]

    for i, s in enumerate(summaries):
        if i:
            story.append(PageBreak())
        story.append(Paragraph(str(s["name"]), styles["Heading1"]))

        facts = Table(
            [[k, v] for k, v in _summary_facts(s)],
            colWidths=[2.0 * inch, 4.6 * inch], hAlign="LEFT",
        )
        facts.setStyle(TableStyle([
            ("GRID", (0, 0), (-1, -1), 0.4, colors.lightgrey),
            ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, -1), 9),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("ROWBACKGROUNDS", (0, 0), (-1, -1), [colors.whitesmoke, colors.white]),
        ]))
        story.append(facts)
        story.append(Spacer(1, 0.18 * inch))

        story.append(Paragraph("Layer structure", styles["Heading2"]))
        rows: list[list[str]] = [["#", "Module", "Detail", "Params"]]
        for j, (label, detail, n_params) in enumerate(s["layers"]):
            rows.append([str(j), label, detail, f"{n_params:,}" if n_params else "-"])
        rows.append(["", "Total trainable params", "", f"{s['num_params']:,}"])
        layer_tbl = Table(
            rows, colWidths=[0.4 * inch, 2.5 * inch, 2.7 * inch, 1.0 * inch], hAlign="LEFT",
        )
        layer_tbl.setStyle(TableStyle([
            ("GRID", (0, 0), (-1, -1), 0.4, colors.lightgrey),
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#dfe6f3")),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTNAME", (0, -1), (-1, -1), "Helvetica-Bold"),
            ("BACKGROUND", (0, -1), (-1, -1), colors.whitesmoke),
            ("ALIGN", (0, 0), (0, -1), "RIGHT"),
            ("ALIGN", (3, 0), (3, -1), "RIGHT"),
            ("FONTSIZE", (0, 0), (-1, -1), 9),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ]))
        story.append(layer_tbl)

    doc.build(story)


def _write_pdf_matplotlib(path: str, summaries: list[dict[str, Any]]) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages

    with PdfPages(path) as pdf:
        for s in summaries:
            fig = plt.figure(figsize=(8.5, 11))
            ax = fig.add_subplot(111)
            ax.axis("off")
            lines = [f"Model: {s['name']}", ""]
            for k, v in _summary_facts(s):
                lines.append(f"{k + ':':28s}{v}")
            lines += ["", "Layer structure:",
                      f"  {'#':>2}  {'Module':26.26s} {'Detail':26.26s} {'Params':>12s}",
                      "  " + "-" * 70]
            for j, (label, detail, n_params) in enumerate(s["layers"]):
                shown = f"{n_params:,}" if n_params else "-"
                lines.append(f"  {j:>2}  {label:26.26s} {detail:26.26s} {shown:>12s}")
            lines += ["  " + "-" * 70,
                      f"  {'':>2}  {'Total trainable params':26s} {'':26s} {s['num_params']:>12,}"]
            ax.text(0.0, 1.0, "\n".join(lines), va="top", ha="left",
                    family="monospace", fontsize=9, transform=ax.transAxes)
            pdf.savefig(fig)
            plt.close(fig)


def _bool_str(value: bool) -> str:
    return "true" if value else "false"


def write_inf_config(
    path: str,
    *,
    pt_name: str,
    script_path: str,
    input_dim: int,
    output_dim: int,
    obs_normalize: bool,
    precision: str,
    use_cuda_graph: bool,
    workspace_mib: int,
    batch_size: int,
    max_batch_size: int,
    warmup_iters: int,
    timing_iters: int,
    copy_input_each_iter: bool,
) -> None:
    """Emit an infConfig.yaml (minimal-YAML dialect the C++ benchmark parses).

    ``script_path`` (the pt_to_onnx.py exporter) is written relative to this
    config file so the config stays valid regardless of where the source tree is
    mounted (e.g. on the host vs. inside the Aerial build container).
    """
    text = f"""# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Auto-generated by gen_models.py for model '{pt_name}'. Run with:
#   ./trtEngine -c {os.path.basename(path)}

model:
  path: {pt_name}
  inputName: obs
  outputName: logits
  obsDim: {int(input_dim)}
  actionDim: {int(output_dim)}
  includeObsNormalizer: {_bool_str(obs_normalize)}

precision: {precision}

engine:
  useCudaGraph: {_bool_str(use_cuda_graph)}
  workspaceMiB: {int(workspace_mib)}
  cache: true
  cacheDir: ""
  forceRebuild: false

conversion:
  python: python3
  script: {script_path}

runtime:
  gpuId: 0
  batchSize: {int(batch_size)}
  maxBatchSize: {int(max_batch_size)}

latency:
  warmupIters: {int(warmup_iters)}
  timingIters: {int(timing_iters)}
  copyInputEachIter: {_bool_str(copy_input_each_iter)}

verbose: true
"""
    with open(path, "w") as f:
        f.write(text)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "-c",
        "--config",
        default=os.path.join(_THIS_DIR, "modelConfig.yaml"),
        help="Path to modelConfig.yaml (default: alongside this script).",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        default=None,
        help="Override the output directory from the config.",
    )
    parser.add_argument(
        "--no-inf-config",
        action="store_true",
        help="Do not emit the companion infConfig.<name>.yaml files.",
    )
    parser.add_argument(
        "--no-clean",
        action="store_true",
        help="Keep existing artifacts instead of wiping the output directory first.",
    )
    parser.add_argument(
        "--no-onnx",
        action="store_true",
        help="Do not pre-export the cached .onnx next to each .pt (the benchmark "
        "would then convert the .pt at runtime, which needs 'onnx' in its python).",
    )
    parser.add_argument(
        "--no-pdf",
        action="store_true",
        help="Do not write the model_summary.pdf describing the generated models.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    config_path = os.path.abspath(args.config)
    if not os.path.exists(config_path):
        sys.stderr.write(f"ERROR: model config not found: {config_path}\n")
        return 2
    config_dir = os.path.dirname(config_path)

    with open(config_path) as f:
        cfg = yaml.safe_load(f) or {}

    models = cfg.get("models")
    if not models:
        sys.stderr.write(f"ERROR: '{config_path}' has no 'models' list.\n")
        return 2

    defaults = dict(cfg.get("defaults", {}))
    emit_inf_config = bool(cfg.get("emit_inf_config", True)) and not args.no_inf_config
    emit_onnx = bool(cfg.get("emit_onnx", True)) and not args.no_onnx
    emit_pdf = bool(cfg.get("emit_pdf", True)) and not args.no_pdf

    out_dir_rel = args.output_dir or cfg.get("output_dir", "generatedModels")
    output_dir = out_dir_rel if os.path.isabs(out_dir_rel) else os.path.join(config_dir, out_dir_rel)
    os.makedirs(output_dir, exist_ok=True)

    pt_module = _load_pt_to_onnx_module()
    actor_module_cls = pt_module.ActorLogitsModule

    print(f"[gen_models] config:      {config_path}")
    print(f"[gen_models] output dir:  {output_dir}")
    print(f"[gen_models] exporter:    {_PT_TO_ONNX}")
    if not args.no_clean:
        removed = clean_output_dir(output_dir)
        print(f"[gen_models] cleaned:     removed {removed} existing artifact(s)")
    print(f"[gen_models] {len(models)} model(s) to generate")
    print("-" * 72)

    seen_names: set[str] = set()
    summaries: list[dict[str, Any]] = []
    for index, raw in enumerate(models):
        spec = dict(defaults)
        spec.update(raw or {})

        name = spec.get("name")
        if not name:
            sys.stderr.write(f"ERROR: model #{index} is missing 'name'.\n")
            return 2
        if name in seen_names:
            sys.stderr.write(f"ERROR: duplicate model name '{name}'.\n")
            return 2
        seen_names.add(name)

        input_dim = int(spec.get("input_dim", spec.get("obs_dim", 0)))
        output_dim = int(spec.get("output_dim", spec.get("action_dim", 0)))
        if input_dim <= 0 or output_dim <= 0:
            sys.stderr.write(
                f"ERROR: model '{name}' needs positive 'input_dim' and 'output_dim'.\n"
            )
            return 2

        try:
            hidden_layers, encoder_layers, actor_head_layers = resolve_layers(spec)
        except ValueError as err:
            sys.stderr.write(f"ERROR: model '{name}': {err}\n")
            return 2

        obs_normalize = bool(spec.get("obs_normalize", False))
        seed = int(spec.get("seed", 0)) + index

        activation = str(spec.get("activation", "silu")).strip().lower()
        try:
            pt_module.activation_class(activation)  # validate early for a clear error
        except ValueError as err:
            sys.stderr.write(f"ERROR: model '{name}': {err}\n")
            return 2

        checkpoint, model = build_checkpoint(
            actor_module_cls,
            input_dim=input_dim,
            output_dim=output_dim,
            hidden_layers=hidden_layers,
            encoder_layers=encoder_layers,
            actor_head_layers=actor_head_layers,
            obs_normalize=obs_normalize,
            seed=seed,
            activation=activation,
        )

        pt_name = f"{name}.pt"
        pt_path = os.path.join(output_dir, pt_name)
        torch.save(checkpoint, pt_path)

        num_params = sum(t.numel() for k, t in checkpoint["policy"].items()
                         if not k.startswith("obs_normalizer."))
        trunk = " -> ".join(str(w) for w in hidden_layers)
        summaries.append({
            "name": name,
            "structure": f"{input_dim} -> [{trunk}] -> {output_dim}",
            "activation": activation,
            "encoder_layers": encoder_layers,
            "actor_head_layers": actor_head_layers,
            "obs_normalize": obs_normalize,
            "precision": str(spec.get("precision", "fp16")),
            "seed": seed,
            "num_params": num_params,
            "layers": describe_model(model),
        })
        print(f"[{index}] {name}")
        print(f"      structure:  {input_dim} -> [{trunk}] -> {output_dim}  ({activation.upper()}, LayerNorm trunk)")
        print(f"      split:      encoder_layers={encoder_layers} actor_head_layers={actor_head_layers}")
        print(f"      obs_norm:   {obs_normalize}   seed={seed}   params={num_params:,}")
        print(f"      saved:      {pt_path}")

        if emit_onnx:
            onnx_path = onnx_sidecar_path(output_dir, name, obs_normalize)
            try:
                export_onnx(pt_path, onnx_path, include_obs_normalizer=obs_normalize)
                print(f"      onnx:       {onnx_path}")
            except Exception as err:  # noqa: BLE001 - keep the .pt; ONNX is a cache
                sys.stderr.write(
                    f"WARNING: ONNX pre-export for '{name}' failed: {err}\n"
                    f"         The .pt was still written; the benchmark will try to convert it\n"
                    f"         at runtime (needs the 'onnx' package in that python).\n"
                )

        if emit_inf_config:
            inf_path = os.path.join(output_dir, f"{name}.infConfig.yaml")
            # Path to the exporter, relative to the emitted config's directory,
            # so the config resolves identically on the host and in the container.
            script_path = os.path.relpath(_PT_TO_ONNX, output_dir)
            write_inf_config(
                inf_path,
                pt_name=pt_name,
                script_path=script_path,
                input_dim=input_dim,
                output_dim=output_dim,
                obs_normalize=obs_normalize,
                precision=str(spec.get("precision", "fp16")),
                use_cuda_graph=bool(spec.get("use_cuda_graph", True)),
                workspace_mib=int(spec.get("workspace_mib", 1024)),
                batch_size=int(spec.get("batch_size", 32)),
                max_batch_size=int(spec.get("max_batch_size", spec.get("batch_size", 32))),
                warmup_iters=int(spec.get("warmup_iters", 200)),
                timing_iters=int(spec.get("timing_iters", 2000)),
                copy_input_each_iter=bool(spec.get("copy_input_each_iter", False)),
            )
            print(f"      infConfig:  {inf_path}")
        print()

    if emit_pdf and summaries:
        pdf_path = os.path.join(output_dir, "model_summary.pdf")
        try:
            backend = write_summary_pdf(pdf_path, summaries)
            print(f"[gen_models] model summary PDF ({backend}): {pdf_path}")
        except Exception as err:  # noqa: BLE001 - PDF is a convenience artifact
            sys.stderr.write(f"WARNING: model summary PDF not written: {err}\n")

    print("-" * 72)
    print(f"[gen_models] done: {len(models)} checkpoint(s) written to {output_dir}")
    if emit_inf_config:
        first = models[0].get("name")
        print("[gen_models] benchmark one with, e.g.:")
        print(f"    ./trtEngine -c {os.path.join(output_dir, first + '.infConfig.yaml')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
