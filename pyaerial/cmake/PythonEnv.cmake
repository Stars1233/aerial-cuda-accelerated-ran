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

include_guard(GLOBAL)

function(pyaerial_get_arch out_var)
    if(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|arm64")
        set(${out_var} "arm64" PARENT_SCOPE)
    elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|amd64|AMD64")
        set(${out_var} "amd64" PARENT_SCOPE)
    else()
        message(FATAL_ERROR "Unsupported architecture for pyAerial Python targets: ${CMAKE_SYSTEM_PROCESSOR}")
    endif()
endfunction()

function(pyaerial_set_uses_terminal target_name)
    if(CMAKE_VERSION VERSION_GREATER_EQUAL "3.27")
        set_target_properties(${target_name} PROPERTIES USES_TERMINAL TRUE)
    endif()
endfunction()

function(pyaerial_add_python_env_targets
         extension_stamp
         pycuphy_pkg_dir
         clean_staged_libs_script
         gpu3gppchan_wheel_stamp
         gpu3gppchan_wheel_dir
         gpu3gppchan_python_venv)
    set(PYAERIAL_EXTENSION_STAMP "${extension_stamp}")
    set(PYAERIAL_PYCUPHY_PKG_DIR "${pycuphy_pkg_dir}")
    set(PYAERIAL_CLEAN_STAGED_LIBS_SCRIPT "${clean_staged_libs_script}")
    set(PYAERIAL_GPU3GPPCHAN_WHEEL_STAMP "${gpu3gppchan_wheel_stamp}")
    set(PYAERIAL_GPU3GPPCHAN_WHEEL_DIR "${gpu3gppchan_wheel_dir}")
    set(PYAERIAL_GPU3GPPCHAN_PYTHON_VENV "${gpu3gppchan_python_venv}")
    # Custom targets (cmake --build --target …):
    #   pyaerial_setup        — stage _pycuphy, then venv + editable install
    #   pyaerial_lint         — static analysis
    #   pyaerial_test         — pytest suite
    #   pyaerial_all          — lint + test
    #   pyaerial_clean_venv   — remove pyaerial/.venv
    #
    # CTest tests (ctest --preset pyaerial-x86|pyaerial-arm, label pyaerial):
    #   pyaerial_build_setup  — fixture: runs pyaerial_setup
    #   pyaerial_lint / pyaerial_test
    #
    # See pyaerial/README.md for full documentation.

    pyaerial_get_arch(PYAERIAL_ARCH)

    set(PYAERIAL_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}")
    set(PYAERIAL_VENV "${PYAERIAL_SOURCE_DIR}/.venv")
    set(PYAERIAL_VENV_STAMP "${CMAKE_CURRENT_BINARY_DIR}/pyaerial_venv.stamp")

    add_custom_target(pyaerial_clean_venv
        COMMAND "${CMAKE_COMMAND}"
            -DPYAERIAL_PYCUPHY_PKG_DIR="${PYAERIAL_PYCUPHY_PKG_DIR}"
            -P "${PYAERIAL_CLEAN_STAGED_LIBS_SCRIPT}"
        COMMAND "${CMAKE_COMMAND}" -E rm -rf "${PYAERIAL_VENV}"
        COMMAND "${CMAKE_COMMAND}" -E rm -f "${PYAERIAL_VENV_STAMP}"
        COMMAND "${CMAKE_COMMAND}" -E rm -f "${PYAERIAL_EXTENSION_STAMP}"
        COMMENT "Removing pyAerial Python virtual environment and staged native artifacts"
    )

    add_custom_command(
        OUTPUT "${PYAERIAL_VENV_STAMP}"
        COMMAND "${CMAKE_COMMAND}" -E env
            PYAERIAL_ARCH=${PYAERIAL_ARCH}
            GPU3GPPCHAN_WHEEL_DIR=${PYAERIAL_GPU3GPPCHAN_WHEEL_DIR}
            bash "${PYAERIAL_SOURCE_DIR}/scripts/setup_venv.sh"
        COMMAND "${CMAKE_COMMAND}" -E touch "${PYAERIAL_VENV_STAMP}"
        WORKING_DIRECTORY "${PYAERIAL_SOURCE_DIR}"
        DEPENDS
            "${PYAERIAL_EXTENSION_STAMP}"
            "${PYAERIAL_GPU3GPPCHAN_WHEEL_STAMP}"
            "${PYAERIAL_SOURCE_DIR}/pyproject.toml"
            "${PYAERIAL_SOURCE_DIR}/setup.py"
            "${PYAERIAL_SOURCE_DIR}/scripts/setup_venv.sh"
        COMMENT "Setting up pyAerial Python virtual environment"
    )

    add_custom_target(pyaerial_setup
        DEPENDS "${PYAERIAL_VENV_STAMP}"
        COMMENT "pyAerial venv ready (stamp: ${PYAERIAL_VENV_STAMP})"
    )

    add_custom_target(pyaerial_lint
        COMMAND bash "${PYAERIAL_SOURCE_DIR}/scripts/run_static_tests.sh"
        WORKING_DIRECTORY "${PYAERIAL_SOURCE_DIR}"
        COMMENT "Running pyAerial static analysis"
        DEPENDS pyaerial_setup
    )
    pyaerial_set_uses_terminal(pyaerial_lint)

    add_custom_target(pyaerial_test
        COMMAND "${CMAKE_COMMAND}" -E env
            CUDA_MODULE_LOADING=LAZY
            PYTHONUNBUFFERED=1
            bash "${PYAERIAL_SOURCE_DIR}/scripts/run_unit_tests.sh" trt
        COMMAND "${CMAKE_COMMAND}" -E env
            CUDA_MODULE_LOADING=LAZY
            PYTHONUNBUFFERED=1
            bash "${PYAERIAL_SOURCE_DIR}/scripts/run_unit_tests.sh" pytest
        COMMAND "${CMAKE_COMMAND}" -E env
            CUDA_MODULE_LOADING=LAZY
            PYTHONUNBUFFERED=1
            PYTHON_BIN=${PYAERIAL_GPU3GPPCHAN_PYTHON_VENV}/bin/python
            bash "${PYAERIAL_SOURCE_DIR}/../testBenches/gpu3GPPChan/test/run_python_tests.sh"
        WORKING_DIRECTORY "${PYAERIAL_SOURCE_DIR}"
        COMMENT "Running pyAerial unit tests"
        DEPENDS pyaerial_setup gpu3gppchan_python_setup
    )
    pyaerial_set_uses_terminal(pyaerial_test)

    add_custom_target(pyaerial_all
        DEPENDS pyaerial_lint pyaerial_test
        COMMENT "Running all pyAerial Python checks"
    )

    if(ENABLE_TESTS)
        add_test(NAME pyaerial_build_setup
            COMMAND ${CMAKE_COMMAND} --build ${CMAKE_BINARY_DIR} --target pyaerial_setup
        )
        set_tests_properties(pyaerial_build_setup
            PROPERTIES FIXTURES_SETUP pyaerial_env LABELS "pyaerial"
        )

        add_test(NAME pyaerial_lint
            COMMAND ${CMAKE_COMMAND} --build ${CMAKE_BINARY_DIR} --target pyaerial_lint
        )
        set_tests_properties(pyaerial_lint
            PROPERTIES FIXTURES_REQUIRED pyaerial_env LABELS "pyaerial"
        )

        add_test(NAME pyaerial_test
            COMMAND ${CMAKE_COMMAND} --build ${CMAKE_BINARY_DIR} --target pyaerial_test
        )
        set_tests_properties(pyaerial_test
            PROPERTIES
                FIXTURES_REQUIRED pyaerial_env
                LABELS "pyaerial;requires-gpu;requires-tvs"
        )
    endif()
endfunction()
