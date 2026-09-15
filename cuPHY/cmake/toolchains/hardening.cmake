# Shared hardening flag logic included by each platform toolchain after it
# declares AERIAL_FORTIFY_LEVEL and AERIAL_STACK_PROTECTOR in CACHE.
# Sets CMAKE_{C,CXX,CUDA}_FLAGS_INIT, which must be done during toolchain
# processing (these variables have no effect if set from CMakeLists.txt).

set(_valid_stack_protectors none strong)
if(NOT AERIAL_STACK_PROTECTOR IN_LIST _valid_stack_protectors)
    message(FATAL_ERROR "AERIAL_STACK_PROTECTOR must be 'none' or 'strong', got '${AERIAL_STACK_PROTECTOR}'")
endif()
if(NOT AERIAL_FORTIFY_LEVEL MATCHES "^[0-9]+$")
    message(FATAL_ERROR "AERIAL_FORTIFY_LEVEL must be a non-negative integer, got '${AERIAL_FORTIFY_LEVEL}'")
endif()

set(_harden -U_FORTIFY_SOURCE)
if(NOT AERIAL_FORTIFY_LEVEL STREQUAL "0")
    list(APPEND _harden -D_FORTIFY_SOURCE=${AERIAL_FORTIFY_LEVEL})
endif()
if(AERIAL_STACK_PROTECTOR STREQUAL "strong")
    list(APPEND _harden -fstack-protector-strong -fstack-clash-protection)
else()
    list(APPEND _harden -fno-stack-protector -fno-stack-clash-protection)
endif()

list(JOIN _harden " " _harden_str)
set(CMAKE_C_FLAGS_INIT   "${_harden_str}")
set(CMAKE_CXX_FLAGS_INIT "${_harden_str}")
string(REPLACE " " "," _harden_nvcc "${_harden_str}")
set(CMAKE_CUDA_FLAGS_INIT "-Xcompiler=${_harden_nvcc}")
