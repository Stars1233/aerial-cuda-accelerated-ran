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

"""Test ISAC (Integrated Sensing and Communications) functionality."""

import gc
import os
import weakref
from typing import Optional, Tuple

import cupy as cp
import numpy as np
import pytest

from gpu3gppchan import (
    AntPanelConfig,
    ExternalConfig,
    Scenario,
    SimConfig,
    SpstParam,
    StParam,
    StatisticalChannel,
    SystemLevelConfig,
)
from gpu3gppchan import _gpu3gppchan as _native

# Import helper functions from existing test
from test_chmod_sls import (
    read_config,
    generate_cell_configs,
    generate_ut_configs,
    allocate_cir_arrays,
    allocate_cfr_arrays
)


def create_isac_channel_model(
    config_file: Optional[str] = None,
    n_st: Optional[int] = None,
    isac_type: Optional[int] = None,
    use_custom_cell_and_ut_configs: bool = False,
) -> Tuple[StatisticalChannel, SimConfig, SystemLevelConfig, ExternalConfig]:
    """Create an ISAC-enabled channel model configuration.

    Args:
        config_file: Path to YAML configuration file (optional)
        n_st: Number of sensing targets (None = use YAML value)
        isac_type: ISAC type (None = use YAML value)
        use_custom_cell_and_ut_configs: If True, override cell and UT configs
            via helper generators.

    Returns:
        Tuple of (chan_model, sim_config, system_config, external_config)
    """
    # Read config without creating model (avoid double creation)
    if config_file is None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        config_file = os.path.join(
            script_dir,
            "../../config/statistic_channel_config_isac.yaml")
        config_file = os.path.normpath(config_file)
    system_config, link_config, sim_config, external_config = read_config(config_file)
    # Override ISAC parameters only if specified
    if isac_type is not None:
        system_config.isac_type = isac_type
    if n_st is not None:
        system_config.n_st = n_st

    if use_custom_cell_and_ut_configs:  # optional: override cell and UT configurations
        # Generate cell and UT configurations (reuse external_config from read_config)
        external_config.cell_config = generate_cell_configs(system_config, external_config)
        external_config.ut_config = generate_ut_configs(system_config, external_config)
        # For monostatic sensing, configure BS as TX/RX
        effective_isac_type = (
            system_config.isac_type
            if system_config.isac_type is not None
            else isac_type
        )
        if effective_isac_type == 1:
            if len(external_config.cell_config) > 0:
                external_config.cell_config[0].monostatic_ind = 1
                external_config.cell_config[0].second_ant_panel_idx = 0
                print("Configured Cell 0 as monostatic sensing TX/RX")
    # Create channel model once with full ISAC configuration
    chan_model = StatisticalChannel(
        sim_config=sim_config,
        system_level_config=system_config,
        link_level_config=link_config,
        external_config=external_config
    )
    return chan_model, sim_config, system_config, external_config


def test_uint16_cuda_array_rejects_uint32() -> None:
    """Keep the native CUDA-array dtype contract exact."""

    class FakeCudaArray:
        def __init__(self, typestr: str, address: int = 1) -> None:
            self.__cuda_array_interface__ = {
                "shape": (1,),
                "strides": None,
                "typestr": typestr,
                "data": (address, False),
                "version": 3,
            }

    _native.CudaArrayUint16(FakeCudaArray("<u2"))
    with pytest.raises(RuntimeError, match="uint16"):
        _native.CudaArrayUint16(FakeCudaArray("<u4"))


def test_cuda_array_rejects_null_pointer_for_nonempty_shape() -> None:
    """Reject nonempty CUDA-array metadata that exposes a null pointer."""

    class FakeCudaArray:
        __cuda_array_interface__ = {
            "shape": (1,),
            "strides": None,
            "typestr": "<f4",
            "data": (0, False),
            "version": 3,
        }

    with pytest.raises(
        ValueError,
        match="^non-empty CUDA array must expose a non-null data pointer$",
    ):
        _native.CudaArrayFloat(FakeCudaArray())

    class EmptyCudaArray:
        __cuda_array_interface__ = {
            "shape": (0,),
            "strides": None,
            "typestr": "<f4",
            "data": (0, False),
            "version": 3,
        }

    assert _native.CudaArrayFloat(EmptyCudaArray()).size == 0


def test_cuda_array_retains_producer_lifetime() -> None:
    """Keep the CUDA-array producer alive for the lifetime of the native view."""

    class FakeCudaArray:
        __cuda_array_interface__ = {
            "shape": (1,),
            "strides": None,
            "typestr": "<f4",
            "data": (1, False),
            "version": 3,
        }

    producer = FakeCudaArray()
    producer_ref = weakref.ref(producer)
    view = _native.CudaArrayFloat(producer)

    del producer
    gc.collect()
    assert producer_ref() is not None

    del view
    gc.collect()
    assert producer_ref() is None


def test_fixed_array_constructors_require_exact_lengths() -> None:
    """Constructor arrays reject partial initialization; capacity fields do not."""
    with pytest.raises(ValueError, match="exactly match"):
        AntPanelConfig(
            n_ant=1,
            ant_size=[1, 1, 1, 1],
            ant_spacing=[0.0, 0.0, 0.5, 0.5],
            ant_polar_angles=[0.0, 0.0],
            ant_model=0,
        )

    with pytest.raises(ValueError, match="exactly match"):
        AntPanelConfig(
            n_ant=1,
            ant_size=[1, 1, 1, 1, 1],
            ant_spacing=[0.0, 0.5],
            ant_polar_angles=[0.0, 0.0],
            ant_model=0,
        )

    config = SystemLevelConfig(
        scenario=Scenario.UMa,
        n_site=1,
        n_sector_per_site=1,
        n_ut=1,
        isd=500.0,
        ut_drop_cells=[0],
    )
    assert config.n_site == 1


def test_fixed_array_setter_requires_exact_length() -> None:
    """Reject partial fixed-array updates without changing the old value."""
    target = StParam()
    target.velocity = [1.0, 2.0, 3.0]

    with pytest.raises(ValueError, match="exactly match"):
        target.velocity = [4.0, 5.0]

    assert target.velocity == [1.0, 2.0, 3.0]


def test_spst_setters_validate_before_commit() -> None:
    """An invalid SPST update must leave the target configuration usable."""
    target = StParam()
    target.n_spst = 1
    target.spst_configs = [SpstParam()]

    with pytest.raises(ValueError, match="n_spst"):
        target.n_spst = 2
    assert target.n_spst == 1

    with pytest.raises(ValueError, match="n_spst"):
        target.spst_configs = []
    assert len(target.spst_configs) == 1


@pytest.mark.parametrize("isac_type,n_st", [
    (1, 5),  # Monostatic sensing with 5 targets
])
def test_isac_channel_model(isac_type, n_st):
    """Test ISAC channel model with different configurations.

    Args:
        isac_type: ISAC type (0: comm only, 1: monostatic, 2: bistatic)
        n_st: Number of sensing targets
    """
    print(f"\n{'=' * 60}")
    print(f"Testing ISAC: isac_type={isac_type}, n_st={n_st}")
    print(f"{'=' * 60}")
    # Create ISAC-enabled channel model (uses YAML values, no overrides)
    config_file = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "../../config/statistic_channel_config_isac.yaml")
    chan_model, sim_config, system_config, external_config = create_isac_channel_model(
        config_file=config_file,
        n_st=n_st,
        isac_type=isac_type
    )
    # Setup simulation parameters
    n_active_cells = system_config.n_site * system_config.n_sector_per_site
    n_snapshots = sim_config.n_snapshot_per_slot
    # For ISAC monostatic mode, we need larger buffers:
    # - BS has 64 antennas (from config panel_0)
    # - Monostatic sensing uses BS->ST->BS (64x64)
    # - Combined taps can be larger than default 24
    if system_config.isac_type == 1:  # Monostatic sensing
        n_bs_ant = 64  # Match the BS antenna config
        n_ut_ant = 64  # For monostatic, RX is also BS
        # +3 represents ISAC_NUM_REFERENCE_POINTS (background reference points).
        # Current common case is st_rcs_model == 1 where n_spst_total == n_st.
        isac_num_reference_points = 3
        if getattr(system_config, "st_rcs_model", 1) == 1:
            n_spst_total = n_st
        else:
            st_cfg = external_config.st_config
            n_spst_total = sum(max(1, int(getattr(st, "n_spst", 1))) for st in st_cfg)
            if n_spst_total == 0:
                n_spst_total = n_st
        max_taps = (n_spst_total + isac_num_reference_points) * 24
    else:
        n_bs_ant = 4
        n_ut_ant = 4
        max_taps = 24
    # Setup active links - use all UTs from config
    active_cells = list(range(n_active_cells))
    active_uts = []
    for _ in range(len(active_cells)):
        cell_uts = list(range(system_config.n_ut))
        active_uts.append(cell_uts)
    print(f"  Config: n_site={system_config.n_site}, n_ut={system_config.n_ut}, "
          f"n_st={system_config.n_st}, isac_type={system_config.isac_type}")
    # Allocate buffers with ISAC-appropriate sizes
    cir_coe_per_cell, cir_norm_delay_per_cell, cir_n_taps_per_cell = allocate_cir_arrays(
        sim_config, active_cells, active_uts, n_snapshots, n_ut_ant, n_bs_ant, max_taps
    )
    cfr_prbg_per_cell, cfr_sc_per_cell = allocate_cfr_arrays(
        sim_config, active_cells, active_uts, n_snapshots, n_ut_ant, n_bs_ant
    )
    # Run channel model for one TTI
    # Note: Do NOT pass ut_new_loc/ut_new_velocity as zeros - this would overwrite
    # UT positions to origin (0,0,0), causing d_2d=0 errors in pathloss calculation.
    print("Running channel simulation...")
    chan_model.run(
        ref_time=0.0,
        continuous_fading=1,
        active_cell=active_cells,
        active_ut=active_uts,
        cir_coe=cir_coe_per_cell,
        cir_norm_delay=cir_norm_delay_per_cell,
        cir_n_taps=cir_n_taps_per_cell,
        cfr_prbg=cfr_prbg_per_cell if sim_config.run_mode >= 1 else [],
        cfr_sc=cfr_sc_per_cell if sim_config.run_mode >= 2 else []
    )
    # GPU synchronization
    if sim_config.cpu_only_mode == 0:
        chan_model._stream.synchronize()
    # Verify results
    print("Verifying ISAC channel generation...")
    for cell_id, cir_coe_arr in enumerate(cir_coe_per_cell):
        arr_shape = cir_coe_arr.shape
        total_elements = cir_coe_arr.size
        print(f"  Cell {cell_id}: shape={arr_shape}, total={total_elements}")
    print(f"✓ ISAC test passed for isac_type={isac_type}, n_st={n_st}")
    # Assertions
    assert chan_model is not None
    assert system_config.isac_type == isac_type
    assert system_config.n_st == n_st
    array_module = np if sim_config.cpu_only_mode == 1 else cp
    for cir_coe_arr in cir_coe_per_cell:
        assert bool(array_module.all(array_module.isfinite(cir_coe_arr)))
        assert int(array_module.count_nonzero(cir_coe_arr)) > 0


if __name__ == "__main__":
    # Run pytest tests
    try:
        import subprocess
        import sys
        print("Running ISAC pytest tests...")
        subprocess.run([sys.executable, "-m", "pytest", __file__, "-v", "-s"])
    except Exception as e:
        print(f"Could not run pytest: {e}")
