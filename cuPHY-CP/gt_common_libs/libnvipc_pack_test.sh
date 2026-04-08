#!/bin/bash -e

# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

LOCAL_DIR=$(dirname $(readlink -f "$0"))
cd $LOCAL_DIR


TARNAME=$(ls -rt nvipc_src.*.tar.gz 2>/dev/null | tail -n1)

if [ -z "$TARNAME" ] || [ ! -f "$TARNAME" ]; then
    echo "run pack_nvipc.sh first"
    exit 1
fi

cat > Dockerfile <<EOF
FROM ubuntu:22.04

RUN apt update && apt install -y build-essential cmake pkg-config libpcap-dev libcunit1-dev libnuma-dev

ADD ./$TARNAME /src/

RUN mv /src/nvipc_src.* /src/nvipc_src

# Test with default options
RUN  cd /src/nvipc_src && \
  cmake -Bbuild -DENABLE_TESTS=ON && cmake --build build && cmake --install build

# Test without fmtlog support for nvIPC
RUN  cd /src/nvipc_src && \
  cmake -Bbuild.no_fmtlog -DENABLE_TESTS=ON -DNVIPC_FMTLOG_ENABLE=OFF && cmake --build build.no_fmtlog && cmake --install build.no_fmtlog

# Test with CUDA disabled
RUN  cd /src/nvipc_src && \
  cmake -Bbuild.no_cuda -DENABLE_TESTS=ON -DNVIPC_CUDA_ENABLE=OFF && cmake --build build.no_cuda && cmake --install build.no_cuda

RUN  cd /src/nvipc_src && \
  cmake -Bbuild.oai -DENABLE_TESTS=ON -DNVIPC_DPDK_ENABLE=OFF -DNVIPC_DOCA_ENABLE=OFF -DNVIPC_CUDA_ENABLE=OFF -DENABLE_SLT_RSP=ON && \
  cmake --build build.oai && cmake --install build.oai


WORKDIR /src/nvipc_src
RUN echo "#! /bin/bash -e\nfor dir in \\\$(ls -d ./build*); do \\\$dir/nvIPC/tests/cunit/nvipc_cunit 3 2; done" > run_tests.sh && chmod +x ./run_tests.sh
CMD /src/nvipc_src/run_tests.sh
EOF
docker build . -t nvipc_tester
docker run --rm nvipc_tester
exit $?
