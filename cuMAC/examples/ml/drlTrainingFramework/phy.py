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

"""PHY abstraction the link adaptation framework needs, and a reference model.

Link adaptation only ever asks the PHY four questions:

``mcs_spectral_eff``
    What spectral efficiency does MCS ``m`` carry?
``cqi_to_sinr_db``
    What SINR does a reported CQI correspond to?
``eesm_effective_sinr_all_mcs``
    Given a UE's per-resource SINRs, what single effective SINR does its
    transport block see?
``bler_from_sinr_mcs``
    At that effective SINR, what is the block error rate of MCS ``m``?

:class:`PhyAbstraction` is the whole contract. Everything else in the package
is written against it, so a host simulator can hand over its own calibrated
link-to-system mapping and nothing above has to change.

:class:`NrPhyAbstraction` is a **reference** implementation, provided so the
package is runnable and testable on its own. Its MCS spectral efficiencies and
CQI efficiencies are the 3GPP tables and are exact; its required-SINR,
EESM-beta and waterfall-steepness values come from the analytic model documented
on the class and are *not* a substitute for measured link-level curves. Replace
it with the host PHY abstraction before reading anything into absolute BLER or
throughput numbers.
"""

from __future__ import annotations

__all__ = [
    "CQI_MODULATION_ORDER",
    "CQI_SPECTRAL_EFF",
    "DB_FLOOR",
    "DEFAULT_PHY",
    "MCS_MODULATION_ORDER",
    "MCS_SPECTRAL_EFF",
    "NUM_MCS",
    "NrPhyAbstraction",
    "PhyAbstraction",
    "power_to_db",
]

import math
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import torch

# NR MCS table 2 (TS 38.214 table 5.1.3.1-2), indices 0..27. Indices 28..31 are
# reserved for retransmissions and are never an action, so the action space is
# exactly these 28 entries.
NUM_MCS = 28

# Modulation order Qm per MCS index, from the same table.
MCS_MODULATION_ORDER: tuple[int, ...] = (
    2, 2, 2, 2, 2,
    4, 4, 4, 4, 4, 4,
    6, 6, 6, 6, 6, 6, 6, 6, 6,
    8, 8, 8, 8, 8, 8, 8, 8,
)

# Spectral efficiency in bit/s/Hz per layer, Qm * R, from the same table.
MCS_SPECTRAL_EFF: torch.Tensor = torch.tensor(
    (
        0.2344, 0.3770, 0.6016, 0.8770, 1.1758,
        1.4766, 1.6953, 1.9141, 2.1602, 2.4063, 2.5703,
        2.7305, 3.0293, 3.3223, 3.6094, 3.9023, 4.2129, 4.5234, 4.8164, 5.1152,
        5.3320, 5.5547, 5.8906, 6.2266, 6.5703, 6.9141, 7.1602, 7.4063,
    ),
    dtype=torch.float32,
)

# Modulation order per CQI index, CQI table 2 (TS 38.214 table 5.2.2.1-3).
# Index 0 is "out of range" and carries no efficiency.
CQI_MODULATION_ORDER: tuple[int, ...] = (
    2, 2, 2, 2, 4, 4, 4, 6, 6, 6, 6, 6, 8, 8, 8, 8,
)

# Spectral efficiency per CQI index from the same table, with index 0 set to the
# lowest reportable efficiency so the conversion stays monotone and finite.
CQI_SPECTRAL_EFF: torch.Tensor = torch.tensor(
    (
        0.0762,
        0.1523, 0.3770, 0.8770,
        1.4766, 1.9141, 2.4063,
        2.7305, 3.3223, 3.9023, 4.5234, 5.1152,
        5.5547, 6.2266, 6.9141, 7.4063,
    ),
    dtype=torch.float32,
)

# Implementation margin over the Shannon SINR for a given spectral efficiency,
# keyed by modulation order. Higher-order constellations pay more for shaping
# loss and receiver imperfection, so the margin grows with Qm.
_MODULATION_MARGIN_DB: dict[int, float] = {2: 1.0, 4: 1.6, 6: 2.6, 8: 4.2}

# EESM beta per modulation order, in linear SINR. Beta controls how harshly the
# aggregation is dominated by a UE's weakest resource; it grows with Qm because
# a denser constellation is less tolerant of a deep fade on part of its
# allocation.
_MODULATION_EESM_BETA: dict[int, float] = {2: 1.4, 4: 5.0, 6: 14.0, 8: 40.0}

# Standard-normal quantile at 0.1, i.e. Q^-1(0.1). Used to place the waterfall
# so that the required SINR of an MCS is its SINR at 10% BLER.
_Q_INVERSE_AT_10_PERCENT = 1.2815515655446004

# BLER floor and ceiling. Exact zeros make a log-domain goodput comparison
# degenerate and exact ones make every MCS look equally hopeless.
_BLER_MIN = 1.0e-6
_BLER_MAX = 1.0 - 1.0e-6

# dB value reported for a power ratio of exactly zero, and for the effective
# SINR of a UE holding no resources.
DB_FLOOR = -100.0


def power_to_db(ratio: torch.Tensor, *, floor_db: float = DB_FLOOR) -> torch.Tensor:
    """Convert a nonnegative power ratio to dB, flooring exact zeros.

    A single-layer resource has exactly zero residual interference, so the
    naive conversion would produce ``-inf`` and poison the observation.
    """

    floor_linear = 10.0 ** (float(floor_db) / 10.0)
    return 10.0 * torch.log10(ratio.clamp_min(floor_linear))


def _required_sinr_db(
    spectral_eff: torch.Tensor,
    modulation_order: tuple[int, ...],
) -> torch.Tensor:
    """SINR needed to carry ``spectral_eff``, from Shannon plus a Qm margin.

    ``SINR_req = 2^SE - 1`` inverts the AWGN capacity, and the per-Qm margin in
    :data:`_MODULATION_MARGIN_DB` covers the gap a real coded modulation leaves
    against it.
    """

    margin = torch.tensor(
        [_MODULATION_MARGIN_DB[order] for order in modulation_order],
        dtype=torch.float32,
    )
    shannon_db = 10.0 * torch.log10(torch.pow(2.0, spectral_eff) - 1.0)
    return shannon_db + margin


@runtime_checkable
class PhyAbstraction(Protocol):
    """Link-to-system mapping the link adaptation framework depends on.

    Implement this over the host simulator's own PHY abstraction to replace the
    reference model. Every method is batched and shape-preserving; none of them
    may allocate on a different device than their input.
    """

    @property
    def num_mcs(self) -> int:
        """Number of selectable MCS indices, i.e. the action space size."""

    def mcs_spectral_eff(
        self,
        *,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> torch.Tensor:
        """Per-layer spectral efficiency ``[num_mcs]`` in bit/s/Hz."""

    def cqi_to_sinr_db(self, cqi: torch.Tensor) -> torch.Tensor:
        """SINR in dB a reported wideband CQI corresponds to, shape preserved."""

    def eesm_effective_sinr_all_mcs(
        self,
        sinr_db: torch.Tensor,
        resource_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Aggregate per-resource SINR into one effective SINR per MCS.

        Args:
            sinr_db: ``[..., resource]`` per-resource SINR in dB.
            resource_mask: ``[..., resource]`` boolean, true where the resource
                belongs to this UE's transport block.

        Returns:
            ``[..., num_mcs]`` effective SINR in dB. The EESM beta is MCS
            dependent, so the effective SINR is too. Rows with no resource
            return :data:`DB_FLOOR`.
        """

    def bler_from_sinr_mcs(
        self,
        sinr_db: torch.Tensor,
        mcs: torch.Tensor,
    ) -> torch.Tensor:
        """Block error rate at ``sinr_db`` for ``mcs``, broadcast together."""


@dataclass(frozen=True)
class NrPhyAbstraction:
    """Reference PHY abstraction over the 3GPP MCS and CQI tables.

    The tables themselves are exact. The mapping from an MCS to a BLER curve is
    a model, in three parts:

    * **Required SINR.** ``2^SE - 1`` inverts the AWGN capacity for that MCS's
      spectral efficiency, plus a modulation-order margin for the gap a real
      coded modulation leaves against capacity. This lands within about a dB of
      published NR AWGN curves across the table.
    * **Waterfall.** BLER is ``Q((SINR - centre) / sigma)``, with ``centre``
      placed so that BLER is exactly 10% at the required SINR. ``sigma``
      controls how steep the transition is; the default 0.8 dB puts the 10% to
      0.1% span at roughly 1.5 dB, which is typical of an NR LDPC block.
    * **EESM beta.** Keyed by modulation order, growing with Qm.

    None of the three is a measurement. Substitute a calibrated implementation
    of :class:`PhyAbstraction` before drawing conclusions from absolute BLER or
    throughput; the framework reads nothing but this interface.
    """

    waterfall_sigma_db: float = 0.8
    #: Multiplies the per-Qm EESM beta, for calibrating against measured curves.
    eesm_beta_scale: float = 1.0

    def __post_init__(self) -> None:
        if not math.isfinite(self.waterfall_sigma_db) or self.waterfall_sigma_db <= 0.0:
            raise ValueError("waterfall_sigma_db must be positive and finite")
        if not math.isfinite(self.eesm_beta_scale) or self.eesm_beta_scale <= 0.0:
            raise ValueError("eesm_beta_scale must be positive and finite")

    @property
    def num_mcs(self) -> int:
        return NUM_MCS

    def mcs_spectral_eff(
        self,
        *,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> torch.Tensor:
        return MCS_SPECTRAL_EFF.to(device=device, dtype=dtype)

    def required_sinr_db(
        self,
        *,
        device: torch.device | None = None,
    ) -> torch.Tensor:
        """SINR at 10% BLER per MCS index, shape ``[num_mcs]``."""

        return _required_sinr_db(MCS_SPECTRAL_EFF, MCS_MODULATION_ORDER).to(device)

    def eesm_beta(self, *, device: torch.device | None = None) -> torch.Tensor:
        """Linear-domain EESM beta per MCS index, shape ``[num_mcs]``."""

        beta = torch.tensor(
            [_MODULATION_EESM_BETA[order] for order in MCS_MODULATION_ORDER],
            dtype=torch.float32,
        )
        return (beta * float(self.eesm_beta_scale)).to(device)

    def cqi_to_sinr_db(self, cqi: torch.Tensor) -> torch.Tensor:
        """Map a reported wideband CQI index to the SINR it implies.

        The same capacity-plus-margin model as the MCS table, applied to the
        CQI table's spectral efficiencies, so a CQI and the MCS it would select
        are quoted on one consistent SINR scale.
        """

        table = _required_sinr_db(CQI_SPECTRAL_EFF, CQI_MODULATION_ORDER).to(cqi.device)
        index = cqi.long().clamp(0, table.numel() - 1)
        return table[index]

    def eesm_effective_sinr_all_mcs(
        self,
        sinr_db: torch.Tensor,
        resource_mask: torch.Tensor,
    ) -> torch.Tensor:
        """EESM aggregation over a UE's resources, evaluated for every MCS.

        ``SINR_eff = -beta * ln( mean_i exp(-SINR_i / beta) )`` in the linear
        domain, computed through :func:`torch.logsumexp` so a high-SINR resource
        cannot underflow the mean to zero.
        """

        if sinr_db.shape != resource_mask.shape:
            raise ValueError(
                "sinr_db and resource_mask must have the same shape, got "
                f"{tuple(sinr_db.shape)} and {tuple(resource_mask.shape)}"
            )
        beta = self.eesm_beta(device=sinr_db.device).to(sinr_db.dtype)
        sinr_linear = torch.pow(
            torch.tensor(10.0, device=sinr_db.device, dtype=sinr_db.dtype),
            sinr_db / 10.0,
        )
        # [..., resource, mcs]: one exponent per resource and MCS.
        exponent = -sinr_linear.unsqueeze(-1) / beta
        exponent = exponent.masked_fill(~resource_mask.unsqueeze(-1), -float("inf"))
        count = resource_mask.sum(dim=-1, keepdim=True).to(sinr_db.dtype)
        log_mean = torch.logsumexp(exponent, dim=-2) - torch.log(
            count.clamp_min(1.0)
        )
        effective_db = 10.0 * torch.log10(
            (-beta * log_mean).clamp_min(10.0 ** (DB_FLOOR / 10.0))
        )
        # A UE holding no resource has no transport block; report the floor
        # rather than a value derived from an empty aggregation.
        return torch.where(
            count > 0.0,
            effective_db,
            torch.full_like(effective_db, DB_FLOOR),
        )

    def bler_from_sinr_mcs(
        self,
        sinr_db: torch.Tensor,
        mcs: torch.Tensor,
    ) -> torch.Tensor:
        """Gaussian-waterfall BLER, 10% at the MCS's required SINR."""

        table = self.required_sinr_db(device=sinr_db.device).to(sinr_db.dtype)
        required = table[mcs.long().clamp(0, table.numel() - 1)]
        sigma = float(self.waterfall_sigma_db)
        centre = required - _Q_INVERSE_AT_10_PERCENT * sigma
        # Q(x) = 0.5 * erfc(x / sqrt(2)).
        bler = 0.5 * torch.erfc((sinr_db - centre) / (sigma * math.sqrt(2.0)))
        return bler.clamp(_BLER_MIN, _BLER_MAX)


#: Reference PHY abstraction used when a simulator adapter supplies none.
DEFAULT_PHY = NrPhyAbstraction()
