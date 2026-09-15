/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef EXCEPTION_H
#define EXCEPTION_H

#include <iostream>
#include <exception>
#include "nvlog.hpp"

#ifndef TAG_EXCP
#define TAG_EXCP (NVLOG_TAG_BASE_CUPHY_DRIVER + 31)                ///< Logging tag for exception handling ("DRV.EXCP")
#endif

/**
 * @brief PHY driver exception handler class
 *
 * Custom exception class that extends std::runtime_error with an error code.
 * Used throughout cuPHYDriver to provide detailed error information including
 * both a numeric error code and descriptive message.
 */
class pd_exc_h : public std::runtime_error {
    int err_code_;                                                  ///< Error code associated with this exception

public:
    /**
     * @brief Construct a PHY driver exception
     *
     * @param err_code - Error code (typically errno value)
     * @param what     - Descriptive error message
     */
    pd_exc_h(int err_code, const std::string& what) :
        std::runtime_error(what),
        err_code_(err_code) {}
    int err_code() const { return err_code_; }                      ///< Get the error code associated with this exception
};

/**
 * @brief Standard exception catching macro with error code return
 *
 * Catches all exception types and returns appropriate error codes:
 * - pd_exc_h exceptions: return the embedded error code
 * - std::exception: return EINVAL
 * - All other exceptions: return EINVAL
 * Logs all exceptions with file, function, and line information.
 * Use in API functions that return int error codes.
 */
#define PHYDRIVER_CATCH_EXCEPTIONS()                         \
    catch(pd_exc_h const& e)                                 \
    {                                                        \
        NVLOGE_FMT(TAG_EXCP, AERIAL_CUPHYDRV_API_EVENT, "{} {} line {} exception: {}", __FILE__, __func__, __LINE__, e.what()); \
        return e.err_code();                                 \
    }                                                        \
    catch(std::exception const& e)                           \
    {                                                        \
        NVLOGE_FMT(TAG_EXCP, AERIAL_CUPHYDRV_API_EVENT, "{} {} line {} exception: {}", __FILE__, __func__, __LINE__, e.what()); \
        return EINVAL;                                       \
    }                                                        \
    catch(...)                                               \
    {                                                        \
        NVLOGE_FMT(TAG_EXCP, AERIAL_CUPHYDRV_API_EVENT, "{} {} line {} uncaught exception", __FILE__, __func__, __LINE__); \
        return EINVAL;                                       \
    }

/**
 * @brief Exception catching macro for void API functions
 *
 * Same logging as PHYDRIVER_CATCH_EXCEPTIONS but returns from the function with no value.
 */
#define PHYDRIVER_CATCH_EXCEPTIONS_VOID()                    \
    catch(pd_exc_h const& e)                                 \
    {                                                        \
        NVLOGE_FMT(TAG_EXCP, AERIAL_CUPHYDRV_API_EVENT, "{} {} line {} exception: {}", __FILE__, __func__, __LINE__, e.what()); \
        return;                                              \
    }                                                        \
    catch(std::exception const& e)                           \
    {                                                        \
        NVLOGE_FMT(TAG_EXCP, AERIAL_CUPHYDRV_API_EVENT, "{} {} line {} exception: {}", __FILE__, __func__, __LINE__, e.what()); \
        return;                                              \
    }                                                        \
    catch(...)                                               \
    {                                                        \
        NVLOGE_FMT(TAG_EXCP, AERIAL_CUPHYDRV_API_EVENT, "{} {} line {} uncaught exception", __FILE__, __func__, __LINE__); \
        return;                                              \
    }

/**
 * @brief Exception catching macro with custom return value
 *
 * Similar to PHYDRIVER_CATCH_EXCEPTIONS but returns a user-specified value
 * instead of error codes. Catches all exception types and returns rval for any exception.
 * Logs all exceptions with file, function, and line information.
 * Use in functions that return non-int types (e.g., pointers, booleans).
 *
 * @param rval - Value to return when any exception is caught
 */
#define PHYDRIVER_CATCH_EXCEPTIONS_RETVAL(rval)              \
    catch(pd_exc_h const& e)                                 \
    {                                                        \
        NVLOGE_FMT(TAG_EXCP, AERIAL_CUPHYDRV_API_EVENT, "{} {} line {} exception: {}", __FILE__, __func__, __LINE__, e.what()); \
        return rval;                                         \
    }                                                        \
    catch(std::exception const& e)                           \
    {                                                        \
        NVLOGE_FMT(TAG_EXCP, AERIAL_CUPHYDRV_API_EVENT, "{} {} line {} exception: {}", __FILE__, __func__, __LINE__, e.what()); \
        return rval;                                         \
    }                                                        \
    catch(...)                                               \
    {                                                        \
        NVLOGE_FMT(TAG_EXCP, AERIAL_CUPHYDRV_API_EVENT, "{} {} line {} uncaught exception", __FILE__, __func__, __LINE__); \
        return rval;                                         \
    }

/**
 * @brief Exception catching macro with re-throw
 *
 * Catches all exception types, logs them, and re-throws the exception to propagate
 * it up the call stack. Allows intermediate layers to log exceptions while still
 * letting higher-level handlers decide how to handle them.
 * Use in internal functions where exceptions should be logged but not handled.
 */
#define PHYDRIVER_CATCH_THROW_EXCEPTIONS()                   \
    catch(pd_exc_h const& e)                                 \
    {                                                        \
        NVLOGE_FMT(TAG_EXCP, AERIAL_CUPHYDRV_API_EVENT, "{} {} line {} exception: {}", __FILE__, __func__, __LINE__, e.what()); \
        throw e;                                             \
    }                                                        \
    catch(std::exception const& e)                           \
    {                                                        \
        NVLOGE_FMT(TAG_EXCP, AERIAL_CUPHYDRV_API_EVENT, "{} {} line {} exception: {}", __FILE__, __func__, __LINE__, e.what()); \
        throw e;                                             \
    }                                                        \
    catch(...)                                               \
    {                                                        \
        NVLOGE_FMT(TAG_EXCP, AERIAL_CUPHYDRV_API_EVENT, "{} {} line {} uncaught exception", __FILE__, __func__, __LINE__); \
    }

/**
 * @brief Exception catching macro with fatal exit
 *
 * Catches all exception types, logs them as FATAL errors, and immediately exits
 * the L1 process with EXIT_FAILURE. Use only in critical initialization or
 * unrecoverable error conditions where continuing execution would be unsafe.
 * Uses NVLOGF_FMT for fatal-level logging before termination.
 */
#define PHYDRIVER_CATCH_EXCEPTIONS_FATAL_EXIT()              \
    catch(pd_exc_h const& e)                                 \
    {                                                        \
        NVLOGF_FMT(TAG_EXCP, AERIAL_CUPHYDRV_API_EVENT, "{} {} line {} exception: {}", __FILE__, __func__, __LINE__, e.what()); \
        EXIT_L1(EXIT_FAILURE);                                             \
    }                                                        \
    catch(std::exception const& e)                           \
    {                                                        \
        NVLOGF_FMT(TAG_EXCP, AERIAL_CUPHYDRV_API_EVENT, "{} {} line {} exception: {}", __FILE__, __func__, __LINE__, e.what()); \
        EXIT_L1(EXIT_FAILURE);                                             \
    }                                                        \
    catch(...)                                               \
    {                                                        \
        NVLOGF_FMT(TAG_EXCP, AERIAL_CUPHYDRV_API_EVENT, "{} {} line {} uncaught exception", __FILE__, __func__, __LINE__); \
        EXIT_L1(EXIT_FAILURE);                                             \
    }

/**
 * @brief Macro to throw PHY driver exception with error code and message
 *
 * Convenience macro for throwing pd_exc_h exceptions. Constructs and throws
 * an exception with the specified error code and descriptive message.
 *
 * @param err_code - Error code (typically errno value)
 * @param what     - Descriptive error message string
 */
#define PHYDRIVER_THROW_EXCEPTIONS(err_code, what) \
    throw pd_exc_h(err_code, what);

#endif
