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

"""Simple clang-tidy cache wrapper compatible with CMAKE_CXX_CLANG_TIDY.

Usage: clang_tidy_cache.py clang-tidy <file> [args...] [--stamp-file=<path>] [--no-cache]

Caches clang-tidy results based on:
- Source file content hash
- clang-tidy version
- clang-tidy arguments
- Compile database entry (compiler flags, defines, includes)
- Nearest applicable .clang-tidy path and content

Optional --stamp-file=<path> argument for CUDA clang-tidy targets:
- Touches stamp file only on success (exit code 0)
- If file is not in compile database, skips analysis (touches stamp, exits 0)
- Enables proper incremental builds where failures are re-checked

Optional --no-cache flag disables cache read/write while retaining:
- DB membership checks (files not in compile DB are skipped)
- Stamp file semantics (touch on success, don't touch on failure)
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import pickle
import shlex
import subprocess
import sys
from pathlib import Path

# Set up logger
logger = logging.getLogger(__name__)


def get_cache_dir() -> Path | None:
    """Get or create cache directory, returning None if caching is unavailable."""
    try:
        cache_dir = Path.home() / ".cache" / "clang-tidy-cache"
        cache_dir.mkdir(parents=True, exist_ok=True)
        return cache_dir
    except (OSError, RuntimeError) as e:
        logger.debug(f"Failed to create cache directory: {e}")
        return None


def find_nearest_clang_tidy_config(source_file: str) -> Path | None:
    """Find the nearest .clang-tidy config applicable to a source file."""
    try:
        source_path = Path(source_file).resolve()
    except (OSError, RuntimeError) as e:
        logger.debug(f"Failed to resolve source path for clang-tidy config: {e}")
        return None
    for directory in (source_path.parent, *source_path.parent.parents):
        config_path = directory / ".clang-tidy"
        try:
            if config_path.is_file():
                return config_path
        except OSError as e:
            logger.debug(f"Failed to inspect clang-tidy config {config_path}: {e}")
    return None


COMPILER_LAUNCHERS = {"ccache", "sccache", "distcc", "icecc"}


def get_header_dependencies_from_compiler(
    source_file: str,
    compile_command: str,
) -> list[str]:
    """Get header dependencies by running compiler preprocessor."""
    # Extract compiler and flags from compile command, preserving quoted paths.
    try:
        cmd_parts = shlex.split(compile_command)
    except ValueError as e:
        logger.debug(f"Failed to parse compile command: {e}")
        return []

    # Remove common compiler launchers before invoking the compiler directly.
    while cmd_parts and Path(cmd_parts[0]).name in COMPILER_LAUNCHERS:
        cmd_parts.pop(0)
    if not cmd_parts:
        return []

    compiler = cmd_parts[0]

    # Build preprocessor command to get dependencies
    # Use -M for all dependencies or -MM to skip system headers
    preproc_cmd = [compiler, "-MM", source_file]

    # Extract relevant flags from compile command
    # We need -I (include paths), -D (defines), -std (standard), -isystem, etc.
    i = 1
    while i < len(cmd_parts):
        part = cmd_parts[i]
        # Include flags that affect preprocessing
        if part.startswith(("-I", "-D", "-std=", "--std=")) or part in {"-isystem", "-include"}:
            if part in {"-I", "-D", "-isystem", "-include"} and i + 1 < len(cmd_parts):
                preproc_cmd.extend([part, cmd_parts[i + 1]])
                i += 2
            else:
                preproc_cmd.append(part)
                i += 1
        else:
            i += 1

    # Run preprocessor to get dependencies
    try:
        result = subprocess.run(  # noqa: S603
            preproc_cmd,
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )

        if result.returncode != 0:
            logger.debug(f"Preprocessor failed: {result.stderr[:200]}")
            return []

        # Parse Makefile-style output: target: dep1 dep2 \
        #                                     dep3 dep4
        output = result.stdout
        if ":" not in output:
            return []

        deps_str = output.split(":", 1)[1]
        deps_str = deps_str.replace("\\\n", " ").replace("\\", "")
        deps = deps_str.split()

        # Filter to only header files
        return [d.strip() for d in deps if d.endswith((".h", ".hpp", ".hxx", ".cuh", ".hh"))]

    except (subprocess.TimeoutExpired, OSError) as e:
        logger.debug(f"Failed to get dependencies from compiler: {e}")

    return []


def hash_header_dependencies(source_file: str, compile_command: str | None) -> str | None:
    """Hash all header files that source file depends on."""
    if not compile_command:
        logger.debug(f"No compile command found for {source_file}")
        return None

    # Get header dependencies using compiler
    headers = get_header_dependencies_from_compiler(source_file, compile_command)
    if not headers:
        logger.debug(f"No header dependencies found for {source_file}")
        return None

    logger.debug(f"Found {len(headers)} header dependencies")

    # Hash all project headers (system headers already filtered by -MM)
    hasher = hashlib.sha256()
    project_headers = 0

    for header in headers:
        header_path = Path(header)

        # Skip system headers (typically in /usr)
        header_str = str(header_path)
        if header_str.startswith("/usr/"):
            logger.debug(f"Skipping system header: {header}")
            continue

        # Hash header content if it exists and is readable
        try:
            if header_path.exists() and header_path.is_file():
                with header_path.open("rb") as f:
                    content = f.read()
                    hasher.update(content)
                    project_headers += 1
                    logger.debug(f"Hashed project header: {header} ({len(content)} bytes)")
            else:
                logger.debug(f"Header not found or not a file: {header}")
        except OSError as e:
            logger.debug(f"Failed to read header {header}: {e}")

    if project_headers == 0:
        logger.debug("No project headers found to hash")
        return None

    logger.debug(f"Hashed {project_headers} project headers total")
    return hasher.hexdigest()


def compute_cache_key(
    clang_tidy_path: str,
    source_file: str,
    args: list[str],
) -> str | None:
    """Compute cache key from file content, clang-tidy version, arguments, and compile database."""
    hasher = hashlib.sha256()

    # Hash source file content
    try:
        with Path(source_file).open("rb") as f:
            hasher.update(f.read())
    except FileNotFoundError:
        # If file doesn't exist, let clang-tidy handle the error
        return None

    # Hash clang-tidy version
    try:
        version = subprocess.run(  # noqa: S603
            [clang_tidy_path, "--version"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        hasher.update(version.stdout.encode())
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError) as e:
        logger.debug(f"Failed to get clang-tidy version: {e}")

    # Hash arguments (these include checks, compile database, etc.)
    hasher.update(shlex.join(args).encode())

    # Hash the nearest clang-tidy configuration path and content so changes to
    # checks/options invalidate successful cached diagnostics.
    config_path = find_nearest_clang_tidy_config(source_file)
    if config_path is None:
        hasher.update(b"<no-clang-tidy-config>")
    else:
        try:
            hasher.update(str(config_path).encode())
            hasher.update(config_path.read_bytes())
        except OSError as e:
            logger.debug(f"Failed to read clang-tidy config {config_path}: {e}")
            return None

    # Hash compile database entry for this source file
    # This ensures cache invalidates when include paths, defines, or compiler flags change
    compile_db_entry = get_compile_db_entry(source_file, args)
    if compile_db_entry:
        hasher.update(compile_db_entry.encode())
        logger.debug(f"Hashed compile DB entry ({len(compile_db_entry)} chars)")

    # Hash header dependencies
    # This ensures cache invalidates when included headers change
    header_hashes = hash_header_dependencies(source_file, compile_db_entry)
    if header_hashes:
        hasher.update(header_hashes.encode())
        logger.debug("Hashed header dependencies")

    return hasher.hexdigest()


def get_compile_db_entry(source_file: str, args: list[str]) -> str | None:
    """Extract the compile database entry for the source file."""
    # Find -p argument (compile database directory)
    compile_db_dir = None
    for i, arg in enumerate(args):
        if arg == "-p" and i + 1 < len(args):
            compile_db_dir = args[i + 1]
            break

    if not compile_db_dir:
        logger.debug(f"No -p argument found in: {args}")
        return None

    # Load compile_commands.json
    compile_db_path = Path(compile_db_dir) / "compile_commands.json"
    if not compile_db_path.exists():
        return None

    try:
        with compile_db_path.open("r") as f:
            compile_commands = json.load(f)

        # Find entry for this source file. Relative entry paths are relative
        # to the entry's compilation directory, not this process's cwd.
        source_file_abs = str(Path(source_file).resolve())
        for entry in compile_commands:
            entry_dir = Path(entry.get("directory", compile_db_path.parent))
            if not entry_dir.is_absolute():
                entry_dir = compile_db_path.parent / entry_dir
            entry_file = Path(entry.get("file", ""))
            if not entry_file.is_absolute():
                entry_file = entry_dir / entry_file
            entry_file = str(entry_file.resolve())
            if entry_file == source_file_abs:
                # Hash the compile command (includes all flags, includes, defines)
                cmd = entry.get("command")
                if cmd:
                    return cmd
                entry_args = entry.get("arguments")
                if entry_args:
                    return shlex.join(entry_args) if isinstance(entry_args, list) else str(entry_args)
                return None

    except (json.JSONDecodeError, OSError, KeyError) as e:
        logger.debug(f"Failed to read compile database: {e}")

    return None


def get_cached_result(cache_key: str | None) -> dict[str, str | int] | None:
    """Retrieve cached result if it exists."""
    if not cache_key:
        return None

    cache_dir = get_cache_dir()
    if cache_dir is None:
        return None
    cache_file = cache_dir / f"{cache_key}.pkl"
    if not cache_file.exists():
        return None

    try:
        with cache_file.open("rb") as f:
            result = pickle.load(f)  # noqa: S301
            if not isinstance(result, dict) or result.get("returncode") != 0:
                logger.debug("Ignoring non-success cache entry")
                return None
            return result
    except (
        pickle.UnpicklingError,
        OSError,
        EOFError,
        AttributeError,
        ImportError,
        TypeError,
        ValueError,
    ) as e:
        logger.debug(f"Failed to load cached result: {e}")
        return None


def parse_size(size_str: str | None) -> int | None:
    """Parse size string like '2G' or '500M' into bytes."""
    if not size_str:
        return None
    size_str = size_str.strip().upper()
    try:
        if size_str.endswith("G"):
            return int(float(size_str[:-1]) * 1024**3)
        if size_str.endswith("M"):
            return int(float(size_str[:-1]) * 1024**2)
        return int(size_str)
    except (TypeError, ValueError, OverflowError) as e:
        logger.debug(f"Invalid CLANG_TIDY_CACHE_MAX_SIZE={size_str!r}: {e}")
        return None


def enforce_cache_limit(cache_dir: Path) -> None:
    """Delete oldest cache entries if over size limit (default 2GB)."""
    max_size = parse_size(os.environ.get("CLANG_TIDY_CACHE_MAX_SIZE", "2G"))
    if not max_size:
        return

    # Get all cache files with size and access time
    entries = []
    total_size = 0
    try:
        for entry in cache_dir.glob("*.pkl"):
            try:
                stat = entry.stat()
                entries.append((stat.st_atime, stat.st_size, entry))
                total_size += stat.st_size
            except OSError as e:
                logger.debug(f"Failed to stat cache file {entry}: {e}")
    except OSError as e:
        logger.debug(f"Failed to enumerate cache directory {cache_dir}: {e}")
        return

    if total_size <= max_size:
        return

    # Delete oldest until at 90% of limit
    target = int(max_size * 0.9)
    entries.sort()  # Sort by access time

    for _atime, size, filepath in entries:
        if total_size <= target:
            break
        try:
            filepath.unlink()
            total_size -= size
        except OSError as e:
            logger.debug(f"Failed to delete cache file {filepath}: {e}")


def save_cached_result(cache_key: str | None, result: dict[str, str | int]) -> None:
    """Save result to cache."""
    if not cache_key:
        return

    # Only successful diagnostics are cacheable; failures must always be rerun.
    if result.get("returncode") != 0:
        return

    cache_dir = get_cache_dir()
    if cache_dir is None:
        return
    try:
        enforce_cache_limit(cache_dir)
    except OSError as e:
        logger.debug(f"Failed to enforce cache limit: {e}")
        return

    cache_file = cache_dir / f"{cache_key}.pkl"
    try:
        with cache_file.open("wb") as f:
            pickle.dump(result, f)
    except (OSError, pickle.PickleError, TypeError) as e:
        logger.debug(f"Failed to write cache file: {e}")


def run_clang_tidy(clang_tidy_path: str, source_file: str, args: list[str]) -> dict[str, str | int]:
    """Run clang-tidy and capture output."""
    cmd = [clang_tidy_path, source_file, *args]

    result = subprocess.run(  # noqa: S603
        cmd,
        capture_output=True,
        text=True,
        check=False,
    )

    return {
        "stdout": result.stdout,
        "stderr": result.stderr,
        "returncode": result.returncode,
    }


def main() -> None:
    """Run clang-tidy with caching and optional stamp file management."""
    # Set up logging based on environment variable
    debug = os.environ.get("CLANG_TIDY_CACHE_DEBUG", "").lower() in ("1", "true", "yes")
    logging.basicConfig(
        level=logging.DEBUG if debug else logging.WARNING,
        format="[CACHE] %(message)s",
        stream=sys.stderr,
    )

    min_args = 3
    if len(sys.argv) < min_args:
        logger.error(
            "Usage: clang_tidy_cache.py clang-tidy <file> [args...] "
            "[--stamp-file=<path>] [--no-cache]"
        )
        sys.exit(1)

    # Extract custom arguments (not passed to clang-tidy)
    stamp_file: Path | None = None
    no_cache = False
    filtered_argv = []
    for arg in sys.argv:
        if arg.startswith("--stamp-file="):
            stamp_file = Path(arg.split("=", 1)[1])
            logger.debug(f"Stamp file: {stamp_file}")
        elif arg == "--no-cache":
            no_cache = True
            logger.debug("Cache disabled via --no-cache flag")
        else:
            filtered_argv.append(arg)

    clang_tidy_path = filtered_argv[1]

    # Find the source file - it's the argument that looks like a source file
    source_file = None
    source_idx = None
    for i, arg in enumerate(filtered_argv[2:], start=2):
        if arg.endswith((".cpp", ".cu", ".c", ".cc", ".cxx", ".h", ".hpp")):
            source_file = arg
            source_idx = i
            break

    if not source_file:
        logger.error("No source file found in arguments")
        sys.exit(1)

    # Get all args except the source file
    args = filtered_argv[2:source_idx] + filtered_argv[source_idx + 1 :]

    logger.debug(f"Running: {' '.join(filtered_argv)}")
    logger.debug(f"argc={len(filtered_argv)}, argv={filtered_argv}")

    # When stamp file is provided, check if source is in compile database
    # If not, skip analysis (touch stamp and exit 0)
    if stamp_file:
        compile_db_entry = get_compile_db_entry(source_file, args)
        if not compile_db_entry:
            rel_path = Path(source_file).name
            logger.warning(f"Skipping {rel_path} - not in compile database")
            stamp_file.touch()
            sys.exit(0)

    # When cache is disabled, just run clang-tidy with stamp semantics
    if no_cache:
        logger.debug(f"Running clang-tidy (cache disabled) for {source_file}")
        result = run_clang_tidy(clang_tidy_path, source_file, args)

        if result["returncode"] == 0:
            if stamp_file:
                stamp_file.touch()
                logger.debug("Touched stamp file (success, no cache)")
        else:
            exit_code = result["returncode"]
            logger.debug(f"clang-tidy failed for {source_file} (exit code {exit_code})")
            if stamp_file:
                logger.debug("NOT touching stamp file (failure - will re-run next build)")

        if result["stdout"]:
            sys.stdout.write(result["stdout"])
        if result["stderr"]:
            sys.stderr.write(result["stderr"])
        sys.exit(int(result["returncode"]))

    # Compute cache key
    cache_key = compute_cache_key(clang_tidy_path, source_file, args)

    if cache_key:
        logger.debug(f"Key: {cache_key[:16]}...")

    # Check cache
    cached = get_cached_result(cache_key)
    if cached:
        # Cache hit - touch stamp if success
        logger.debug(f"Cache hit for {source_file}")
        if stamp_file and cached["returncode"] == 0:
            stamp_file.touch()
            logger.debug("Touched stamp file (cache hit)")
        if cached["stdout"]:
            sys.stdout.write(cached["stdout"])
        if cached["stderr"]:
            sys.stderr.write(cached["stderr"])
        sys.exit(int(cached["returncode"]))

    # Cache miss - run clang-tidy
    logger.debug(f"Cache miss for {source_file}, running clang-tidy")
    result = run_clang_tidy(clang_tidy_path, source_file, args)

    # Only save to cache and touch stamp if clang-tidy succeeded (exit code 0)
    # Don't cache failures so fixes can be verified without clearing cache
    if result["returncode"] == 0:
        save_cached_result(cache_key, result)
        logger.debug(f"Cached successful result for {source_file}")
        if stamp_file:
            stamp_file.touch()
            logger.debug("Touched stamp file (success)")
    else:
        exit_code = result["returncode"]
        logger.debug(f"Not caching failed result for {source_file} (exit code {exit_code})")
        if stamp_file:
            logger.debug("NOT touching stamp file (failure - will re-run next build)")

    # Output results
    if result["stdout"]:
        sys.stdout.write(result["stdout"])
    if result["stderr"]:
        sys.stderr.write(result["stderr"])

    sys.exit(int(result["returncode"]))


if __name__ == "__main__":
    main()
