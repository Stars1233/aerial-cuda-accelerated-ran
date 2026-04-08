#!/bin/bash -ue

# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

# For create a nvipc source code release package to partner

LOCAL_DIR=$(dirname $(readlink -f "$0"))
cd $LOCAL_DIR

if [ ! -v cuBB_SDK ]; then
    cuBB_SDK=$(realpath $LOCAL_DIR/../..)
fi

DATE=$(date -u "+%Y.%m.%d")
TEMP_SRC_DIR=${TEMP_SRC_DIR:-/tmp/nvipc_src.$DATE}
rm -rf $TEMP_SRC_DIR/* || true
rm -rf $TEMP_SRC_DIR || true
# create externals
mkdir -p $TEMP_SRC_DIR/external
(cd $TEMP_SRC_DIR/external &&
    #cp -a ../../cuPHY/external/libyaml $TEMP_SRC_DIR/external
    git clone -b 0.2.5 https://github.com/yaml/libyaml.git && (cd libyaml && git checkout 2c891fc7a770e8ba2fec34fc6b545c672beb37e6) &&
    #cp -a ../../cuPHY/external/fmtlog $TEMP_SRC_DIR/external/fmtlog
    git clone https://github.com/MengRao/fmtlog.git && (cd fmtlog && git checkout acd521b1a64480354136a745c511358da1ec7dc5 && git apply -v $cuBB_SDK/cuPHY-CP/container/patches/fmtlog.patch && git restore CMakeLists.txt) &&
        (cd fmtlog && git clone https://github.com/fmtlib/fmt.git && (cd fmt && git checkout e69e5f977d458f2650bb346dadf2ad30c5320281)) &&
    #cp -a ../../cuPHY/external/yaml-cpp $TEMP_SRC_DIR/external
    git clone -b 0.8.0 https://github.com/jbeder/yaml-cpp.git && (cd yaml-cpp && git checkout f7320141120f720aecc4c32be25586e7da9eb978)
)

# copy required source files
cp -a nvIPC $TEMP_SRC_DIR
cp -a ../../cuPHY/nvlog $TEMP_SRC_DIR
cp CMakeLists.txt $TEMP_SRC_DIR
cp README.md $TEMP_SRC_DIR
cp test_ipc.sh $TEMP_SRC_DIR
cp nvipc_unit_test.sh $TEMP_SRC_DIR

# Patch the nvlog CMakeLists.txt
sed -i '/find_package(fmt REQUIRED)/d' $TEMP_SRC_DIR/nvlog/CMakeLists.txt
sed -i '/target_link_libraries(nvlog PUBLIC aerial_sdk_version/a target_include_directories(nvlog PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/../external/fmtlog)' $TEMP_SRC_DIR/nvlog/CMakeLists.txt
sed -i '/target_include_directories(nvlog_static INTERFACE/a target_include_directories(nvlog_static PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/../external/fmtlog)' $TEMP_SRC_DIR/nvlog/CMakeLists.txt

TARNAME=nvipc_src.$DATE.tar.gz
tar zcvf $TARNAME -C $(dirname $TEMP_SRC_DIR) $(basename $TEMP_SRC_DIR)

echo "---------------------------------------------"
echo "Pack nvipc source code finished:"
echo "$LOCAL_DIR/$TARNAME"
