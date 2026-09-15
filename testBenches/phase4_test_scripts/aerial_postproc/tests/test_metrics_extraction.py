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

import importlib.util
from pathlib import Path
from types import ModuleType


def load_metrics_extraction_module() -> ModuleType:
    module_path = Path(__file__).resolve().parents[1] / "scripts" / "dashboard" / "metrics_extraction.py"
    spec = importlib.util.spec_from_file_location("metrics_extraction", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load module spec from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write_metadata(scenario_dir: Path, test_case: str) -> None:
    scenario_dir.mkdir()
    (scenario_dir / "metadata.txt").write_text(
        "\n".join(
            [
                "jenkins_pipeline=Nightly_Pipelines/Aerial_cuphy_control_plane_GH_nightly_pipeline",
                "jenkins_id=gh_485-20260519-212006",
                f"test_case={test_case}",
                "test_duration=300",
                "mr=",
                "jenkins_test_result=True",
            ]
        )
        + "\n"
    )


def test_phase_4_metadata_parses_gh_nightly_test_case(tmp_path: Path) -> None:
    metrics_extraction = load_metrics_extraction_module()
    scenario_dir = tmp_path / "F08_20C_59c_BFP9_STT455000_EH_GC_1P"
    write_metadata(scenario_dir, "F08_20C_59c_BFP9_STT455000_EH_GC_1P")

    info, _ = metrics_extraction.read_jenkins_test_information(str(scenario_dir), "phase_4")

    assert info["cell_count"] == 20
    assert info["pattern"] == "59c"
    assert info["bfp"] == 9
    assert info["test_name"] == "F08_20C_59c_BFP9_STT455000_EH_GC_1P"
    assert info["dual_port"] is False
    assert info["eh"] is True
    assert info["gc"] is True


def test_phase_4_metadata_ignores_csv_suffix_in_test_case(tmp_path: Path) -> None:
    metrics_extraction = load_metrics_extraction_module()
    scenario_dir = tmp_path / "F08_20C_59c_BFP9_STT455000_EH_GC_1P"
    write_metadata(
        scenario_dir,
        "F08_20C_59c_BFP9_STT455000_EH_GC_1P,[99.5:99.5:99.5:99.5:99.5],300,true",
    )

    info, _ = metrics_extraction.read_jenkins_test_information(str(scenario_dir), "phase_4")

    assert info["cell_count"] == 20
    assert info["pattern"] == "59c"
    assert info["bfp"] == 9
    assert info["test_name"] == "F08_20C_59c_BFP9_STT455000_EH_GC_1P"
    assert info["dual_port"] is False
    assert info["eh"] is True
    assert info["gc"] is True
