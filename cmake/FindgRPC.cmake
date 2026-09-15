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

# Minimal FindgRPC module used when gRPC is installed from the Ubuntu apt
# packages (libgrpc++-dev / libgrpc-dev), which - unlike an upstream source
# build - do NOT ship gRPCConfig.cmake. Consumers first try
# find_package(gRPC CONFIG QUIET) and fall back to this module-mode
# find_package(gRPC REQUIRED); this shim recreates the gRPC::grpc++ and
# gRPC::grpc++_reflection imported targets from the pkg-config metadata that
# the apt packages do provide (grpc++.pc).

find_package(PkgConfig REQUIRED)

pkg_check_modules(PC_GRPCPP QUIET IMPORTED_TARGET grpc++)

# grpc++ uses protobuf symbols directly. The upstream gRPCConfig.cmake carries
# protobuf as a transitive interface dependency of gRPC::grpc++, but jammy's
# grpc++.pc does not list it - so pull protobuf in here to keep -lprotobuf on
# the link line for consumers (e.g. the grpc test executables).
pkg_check_modules(PC_PROTOBUF QUIET IMPORTED_TARGET protobuf)

# grpc++_reflection has no .pc file; locate the shared object directly.
find_library(GRPCPP_REFLECTION_LIBRARY
    NAMES grpc++_reflection
    HINTS ${PC_GRPCPP_LIBDIR} ${PC_GRPCPP_LIBRARY_DIRS})

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(gRPC
    REQUIRED_VARS PC_GRPCPP_FOUND PC_PROTOBUF_FOUND GRPCPP_REFLECTION_LIBRARY
    VERSION_VAR PC_GRPCPP_VERSION)

if(gRPC_FOUND AND NOT TARGET gRPC::grpc++)
    add_library(gRPC::grpc++ INTERFACE IMPORTED)
    set_target_properties(gRPC::grpc++ PROPERTIES
        INTERFACE_LINK_LIBRARIES "PkgConfig::PC_GRPCPP;PkgConfig::PC_PROTOBUF")
endif()

if(gRPC_FOUND AND NOT TARGET gRPC::grpc++_reflection)
    add_library(gRPC::grpc++_reflection INTERFACE IMPORTED)
    set_target_properties(gRPC::grpc++_reflection PROPERTIES
        INTERFACE_LINK_LIBRARIES "${GRPCPP_REFLECTION_LIBRARY};PkgConfig::PC_GRPCPP;PkgConfig::PC_PROTOBUF")
endif()
