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

"""Convert a trained PPO actor-critic ``.pt`` checkpoint into a deployable
actor-only ONNX graph that the cuMAC ``trtEngine`` can build a TensorRT engine
from.

The cuMAC ``trtEngine`` consumes ONNX (or a serialized TensorRT engine), not a
raw PyTorch checkpoint. The checkpoint produced by the DRL MU-MIMO link
adaptation trainer stores the full actor-critic ``state_dict`` plus a
``model_config`` block describing the architecture. Only the *actor* path
(``actor(encoder(obs))``) is needed for real-time MCS-offset inference, so this
script rebuilds that sub-network directly from the checkpoint -- without
importing the training package -- loads the trained weights, and exports it
with a dynamic batch axis.

This mirrors the deployable module used by the trainer's own latency benchmark
(``scripts/benchmark_latency_models.py::ActorLogitsModule``): a single input
``obs`` of shape ``[batch, obs_dim]`` and a single output ``logits`` of shape
``[batch, action_dim]``. Optionally the leading running observation normalizer
can be folded into the graph (``--include-obs-normalizer``) so the engine
accepts raw, un-normalized observations.

Typical usage (invoked automatically by trtEngine on a ``.pt`` cache miss):

    python3 pt_to_onnx.py \
        --checkpoint trainedModels/.../checkpoint_..._update_008500.pt \
        --output     trainedModels/.../checkpoint_..._update_008500.onnx
"""
from __future__ import annotations

import argparse
import sys
from typing import Any, Sequence

try:
    import torch
    import torch.nn as nn
except Exception as exc:  # pragma: no cover - clearer error than a raw traceback
    sys.stderr.write(
        "ERROR: pt_to_onnx.py requires PyTorch. Install it with "
        "'python3 -m pip install torch'.\n"
        f"Import error: {exc}\n"
    )
    raise SystemExit(2)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", required=True, help="Path to the trained .pt checkpoint.")
    parser.add_argument("--output", required=True, help="Destination .onnx path.")
    parser.add_argument("--input-name", default="obs", help="ONNX graph input tensor name.")
    parser.add_argument("--output-name", default="logits", help="ONNX graph output tensor name.")
    parser.add_argument("--opset", type=int, default=17, help="ONNX opset version.")
    parser.add_argument(
        "--include-obs-normalizer",
        action="store_true",
        help="Fold the running observation normalizer into the graph so the "
        "engine consumes raw (un-normalized) observations.",
    )
    parser.add_argument(
        "--sample-batch",
        type=int,
        default=1,
        help="Batch size of the tracing sample (axis 0 is exported dynamic).",
    )
    return parser.parse_args()


# Selectable hidden-layer activations. SiLU (x*sigmoid(x)) is the DRL trainer
# default; ReLU is offered because TensorRT fuses ReLU directly into the
# preceding GEMM epilogue (one kernel per layer), whereas SiLU stays a separate
# kernel -- so ReLU lowers the trtEngine kernel count / inference latency. Both
# are parameterless, so the checkpoint state_dict layout is identical either way.
_ACTIVATIONS = {
    "silu": nn.SiLU,
    "swish": nn.SiLU,  # alias
    "relu": nn.ReLU,
}


def activation_class(name: str):
    """Map an activation name ("silu"/"swish"/"relu") to its nn.Module class."""
    key = str(name).strip().lower()
    try:
        return _ACTIVATIONS[key]
    except KeyError:
        supported = ", ".join(sorted(set(_ACTIVATIONS)))
        raise ValueError(f"unsupported activation '{name}' (supported: {supported})")


def _mlp(in_dim: int, layer_dims: Sequence[int], activation_cls) -> nn.Sequential:
    """Re-implementation of the trainer's ``ppo.mlp`` builder.

    Linear layers with ``activation_cls`` inserted *between* layers (not after
    the final one), so the module indexing matches the checkpoint keys exactly
    (Linear at even indices, activation at odd indices).
    """
    if not layer_dims:
        raise ValueError("layer_dims must contain at least one layer")
    layers: list[nn.Module] = []
    dim = int(in_dim)
    for idx, out_dim in enumerate(layer_dims):
        layers.append(nn.Linear(dim, int(out_dim)))
        if idx < len(layer_dims) - 1:
            layers.append(activation_cls())
        dim = int(out_dim)
    return nn.Sequential(*layers)


class _ObsNormalizer(nn.Module):
    """Minimal mirror of the trainer's ``ObsNormalizer`` forward pass."""

    def __init__(self, obs_dim: int, clip: float = 10.0) -> None:
        super().__init__()
        self.clip = float(clip)
        # Sub-module name ``rms`` + buffer names mirror the checkpoint keys
        # ``obs_normalizer.rms.{mean,var,count}``.
        self.rms = nn.Module()
        self.rms.register_buffer("mean", torch.zeros(int(obs_dim)))
        self.rms.register_buffer("var", torch.ones(int(obs_dim)))
        self.rms.register_buffer("count", torch.tensor(0.0))

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        normalized = (obs - self.rms.mean) / torch.sqrt(self.rms.var + 1e-8)
        return normalized.clamp(-self.clip, self.clip)


class ActorLogitsModule(nn.Module):
    """Deployable actor-only network: ``actor(encoder(obs))`` -> logits.

    Optionally prefixed by the observation normalizer. This is the exact graph
    executed at inference time for MCS-offset selection; the critic head and
    the reward normalizer are training-only and intentionally dropped.
    """

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        encoder_layers: Sequence[int],
        actor_head_layers: Sequence[int],
        include_obs_normalizer: bool,
        activation: str = "silu",
    ) -> None:
        super().__init__()
        act_cls = activation_class(activation)
        hidden = int(encoder_layers[-1])
        self.obs_normalizer = _ObsNormalizer(obs_dim) if include_obs_normalizer else None
        # encoder = mlp(obs_dim, encoder_layers, act) + LayerNorm(hidden) + act
        self.encoder = _mlp(obs_dim, encoder_layers, act_cls)
        self.encoder.append(nn.LayerNorm(hidden))
        self.encoder.append(act_cls())
        # actor = mlp(hidden, [*actor_head_layers, action_dim], act)
        self.actor = _mlp(hidden, [*actor_head_layers, action_dim], act_cls)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        if self.obs_normalizer is not None:
            obs = self.obs_normalizer(obs)
        return self.actor(self.encoder(obs))


def _sorted_layer_widths(state: dict[str, Any], prefix: str) -> list[int]:
    """Output widths of the 2D (Linear) weights under ``prefix`` in index order."""

    def sort_key(key: str) -> list[Any]:
        return [int(part) if part.isdigit() else part for part in key.split(".")]

    return [
        int(state[key].shape[0])
        for key in sorted((k for k in state if k.startswith(prefix)), key=sort_key)
        if key.endswith(".weight") and state[key].dim() == 2
    ]


def infer_architecture(checkpoint: dict[str, Any]) -> dict[str, Any]:
    """Resolve obs_dim / encoder_layers / actor_head_layers / action_dim.

    Prefers the checkpoint's ``model_config`` block and falls back to shape
    inference from the ``state_dict`` so older checkpoints still convert.
    """
    state = checkpoint["policy"]
    model_cfg = dict(checkpoint.get("model_config", {}))

    encoder_widths = _sorted_layer_widths(state, "encoder.")
    actor_widths = _sorted_layer_widths(state, "actor.")
    if not encoder_widths or not actor_widths:
        raise RuntimeError(
            "Could not locate encoder/actor Linear layers in the checkpoint "
            "'policy' state_dict; this does not look like an actor-critic checkpoint."
        )

    encoder_layers = [int(w) for w in model_cfg.get("encoder_layers", encoder_widths)]
    # actor widths from state = [*actor_head_layers, action_dim]; the last is the
    # action dimension, the rest are the head hidden layers.
    action_dim = int(actor_widths[-1])
    actor_head_layers = [int(w) for w in model_cfg.get("actor_head_layers", actor_widths[:-1])]

    first_encoder_weight = state["encoder.0.weight"]
    obs_dim = int(model_cfg.get("obs_dim", first_encoder_weight.shape[1]))

    # Activation is parameterless, so it is not recoverable from the state_dict;
    # take it from model_config and default to SiLU (the trainer default) for
    # older checkpoints that predate this field.
    activation = str(model_cfg.get("activation", "silu"))

    return {
        "obs_dim": obs_dim,
        "action_dim": action_dim,
        "encoder_layers": encoder_layers,
        "actor_head_layers": actor_head_layers,
        "obs_normalize": bool(model_cfg.get("obs_normalize", True)),
        "activation": activation,
    }


def build_deployable(checkpoint: dict[str, Any], include_obs_normalizer: bool) -> tuple[ActorLogitsModule, dict[str, Any]]:
    arch = infer_architecture(checkpoint)
    if include_obs_normalizer and not arch["obs_normalize"]:
        sys.stderr.write(
            "WARNING: --include-obs-normalizer requested but the checkpoint was "
            "trained with obs_normalize=False; exporting an identity normalizer.\n"
        )
    model = ActorLogitsModule(
        obs_dim=arch["obs_dim"],
        action_dim=arch["action_dim"],
        encoder_layers=arch["encoder_layers"],
        actor_head_layers=arch["actor_head_layers"],
        include_obs_normalizer=include_obs_normalizer,
        activation=arch["activation"],
    )

    state = checkpoint["policy"]
    wanted_prefixes = ("encoder.", "actor.")
    if include_obs_normalizer:
        wanted_prefixes = ("obs_normalizer.", *wanted_prefixes)
    filtered = {k: v for k, v in state.items() if k.startswith(wanted_prefixes)}

    missing, unexpected = model.load_state_dict(filtered, strict=False)
    # ``missing`` is expected to be empty; ``unexpected`` is empty because we
    # pre-filtered. Surface anything surprising so a silent architecture
    # mismatch does not produce a wrong-but-runnable engine.
    real_missing = [k for k in missing if not k.endswith(".count")]
    if real_missing or unexpected:
        raise RuntimeError(
            "Checkpoint/architecture mismatch while loading the deployable actor:\n"
            f"  missing keys:    {real_missing}\n"
            f"  unexpected keys: {list(unexpected)}"
        )
    model.eval()
    return model, arch


def main() -> int:
    args = parse_args()
    print(f"[pt_to_onnx] loading checkpoint: {args.checkpoint}", flush=True)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict) or "policy" not in checkpoint:
        sys.stderr.write(
            "ERROR: checkpoint does not contain a 'policy' state_dict; "
            "is this a DRL actor-critic checkpoint?\n"
        )
        return 2

    model, arch = build_deployable(checkpoint, args.include_obs_normalizer)
    obs_dim = arch["obs_dim"]
    action_dim = arch["action_dim"]
    print(
        f"[pt_to_onnx] architecture: obs_dim={obs_dim} action_dim={action_dim} "
        f"encoder_layers={arch['encoder_layers']} actor_head_layers={arch['actor_head_layers']} "
        f"activation={arch['activation']} "
        f"obs_normalizer={'on' if args.include_obs_normalizer else 'off'}",
        flush=True,
    )

    sample = torch.randn(max(1, int(args.sample_batch)), obs_dim, dtype=torch.float32)
    with torch.no_grad():
        reference = model(sample)
    if reference.shape[-1] != action_dim:
        sys.stderr.write(
            f"ERROR: exported output dim {reference.shape[-1]} != action_dim {action_dim}.\n"
        )
        return 2

    torch.onnx.export(
        model,
        sample,
        args.output,
        input_names=[args.input_name],
        output_names=[args.output_name],
        dynamic_axes={args.input_name: {0: "batch"}, args.output_name: {0: "batch"}},
        opset_version=int(args.opset),
        do_constant_folding=True,
        dynamo=False,
    )
    print(
        f"[pt_to_onnx] wrote {args.output} "
        f"(input '{args.input_name}'[batch,{obs_dim}] -> output '{args.output_name}'[batch,{action_dim}])",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
