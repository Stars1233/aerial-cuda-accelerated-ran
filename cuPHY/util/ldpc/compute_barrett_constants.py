#!/usr/bin/env python3

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

#
# Compute the Barrett-reduction constants used by the CLMAD-based CRC path in
# ldpc2_crc_dispatch.cuh.
#
# For a CRC polynomial p of degree T (e.g. CRC-24B has T=24, p = 0x01800063):
#   gstar    = p mod x^T               = G_CRC_xxx with the leading x^T bit cleared
#   qplusX   = floor(x^(2T)   / p)     (degree T,  used for S=T  postadjust)
#   qplusCRC = floor(x^(32+T) / p)     (degree 32, used for S=32 preadjust)
#
# opt_reduction<S, T, QPLUS, GSTAR>(c) computes c * x^T mod p when
# QPLUS = floor(x^(S+T) / p).
#
# This file is informational; the constants are baked into
# ldpc2_crc_dispatch.cuh and the build does not invoke this script.

def clmul(a: int, b: int) -> int:
    """Carry-less polynomial multiplication of two polynomials in GF(2)[x].

    Each integer is interpreted as a polynomial where bit ``i`` is the
    coefficient of ``x^i``. Multiplication is performed in GF(2)[x], so
    coefficient addition is XOR (no carries).

    Args:
        a: First polynomial, encoded as a non-negative integer
            (bit ``i`` = coefficient of ``x^i``).
        b: Second polynomial, same encoding. Must be non-negative; a
            negative value would not terminate the loop because Python's
            negative ints have infinitely many leading 1 bits.

    Returns:
        The polynomial product ``a * b`` in GF(2)[x], encoded the same way.
        Bit width is up to ``a.bit_length() + b.bit_length() - 1``.

    Raises:
        ValueError: If ``b`` is negative.

    Examples:
        >>> clmul(0b11, 0b11)   # (x+1)*(x+1) = x^2 + 1 in GF(2)
        5
        >>> clmul(0x3, 0x5)     # (x+1)*(x^2+1) = x^3 + x^2 + x + 1
        15
        >>> clmul(0, 0xDEADBEEF)
        0
    """
    if a < 0 or b < 0:
        raise ValueError(f"multiplicands must be non-negative; got a={a}, b={b}")
    r = 0
    while b:
        if b & 1:
            r ^= a
        a <<= 1
        b >>= 1
    return r

def poly_div_mod(a: int, p: int) -> tuple[int, int]:
    """Polynomial long division in GF(2)[x].

    Computes ``a = q * p + r`` such that ``deg(r) < deg(p)``, using XOR for
    subtraction (i.e. arithmetic over GF(2)).

    Args:
        a: Dividend polynomial, encoded with bit ``i`` = coefficient of ``x^i``.
            Must be non-negative.
        p: Divisor polynomial, same encoding. Must be strictly positive;
            ``p <= 0`` leaves ``deg_p`` invalid and the loop would either
            spin forever (``p == 0`` never reduces ``a``) or operate on a
            nonsense polynomial (negative ``p`` has infinite leading 1 bits).

    Returns:
        A ``(quotient, remainder)`` pair, both encoded the same way. The
        remainder has degree strictly less than ``deg(p)``.

    Raises:
        ValueError: If ``p <= 0`` or ``a < 0``.

    Examples:
        >>> q, r = poly_div_mod(0b1011, 0b11)   # (x^3+x+1) / (x+1)
        >>> q, r                                # quotient x^2+x, remainder 1
        (6, 1)
        >>> # floor(x^32 / CRC-24B) == 0x1FF (matches Barrett qplus derivation)
        >>> q, _ = poly_div_mod(1 << 32, 0x01800063)
        >>> hex(q)
        '0x1ff'
    """
    if p <= 0:
        raise ValueError("divisor polynomial p must be strictly positive")
    if a < 0:
        raise ValueError("dividend polynomial a must be non-negative")
    deg_p = p.bit_length() - 1
    q = 0
    while a.bit_length() - 1 >= deg_p:
        shift = a.bit_length() - 1 - deg_p
        q |= 1 << shift
        a ^= p << shift
    return q, a

def barrett(p: int, T: int) -> tuple[int, int, int]:
    """Compute the Barrett-reduction constants for a degree-``T`` CRC polynomial.

    Returns the three constants needed by ``opt_reduction`` for the CLMAD-based
    CRC path: ``gstar`` (the polynomial without its leading term), and two
    pre-computed quotients ``qplusX`` and ``qplusCRC`` used for the postadjust
    (``S=T``) and preadjust (``S=32``) reductions respectively.

    Args:
        p: Full CRC generator polynomial of degree ``T``, encoded with bit ``i``
            = coefficient of ``x^i``. Must satisfy ``p.bit_length() - 1 == T``.
        T: Degree of ``p`` (16 for CRC-16, 24 for CRC-24A/B). Must be positive.

    Returns:
        A 3-tuple ``(gstar, qplusX, qplusCRC)``:

        * ``gstar``    = ``p mod x^T``           (low ``T`` bits of ``p``)
        * ``qplusX``   = ``floor(x^(2*T) / p)``   (degree ``T``)
        * ``qplusCRC`` = ``floor(x^(32+T) / p)`` (degree 32)

    Raises:
        ValueError: If ``T`` is non-positive, ``p <= 0``, or ``p`` does not
            have the declared degree (``p.bit_length() - 1 != T``). The
            degree check matters: silently passing the wrong ``(p, T)`` pair
            would produce plausible-looking but useless Barrett constants.

    Examples:
        >>> g, qX, qC = barrett(0x01800063, 24)  # CRC-24B
        >>> hex(g), hex(qX), hex(qC)
        ('0x800063', '0x1ffff83', '0x1ffff83ff')
        >>> g, qX, qC = barrett(0x011021, 16)    # CRC-16
        >>> hex(g), hex(qX), hex(qC)
        ('0x1021', '0x11130', '0x111303471')
        >>> barrett(0x011021, 24)                # wrong T for this poly
        Traceback (most recent call last):
            ...
        ValueError: polynomial p has degree 16 but T=24; the leading bit of p must be exactly x^T (got p=0x11021)
    """
    if T <= 0:
        raise ValueError("degree T must be strictly positive")
    if p <= 0:
        raise ValueError("polynomial p must be strictly positive")
    actual_deg = p.bit_length() - 1
    if actual_deg != T:
        raise ValueError(
            f"polynomial p has degree {actual_deg} but T={T}; "
            f"the leading bit of p must be exactly x^T (got p=0x{p:X})"
        )
    gstar      = p & ((1 << T) - 1)
    qplusX,   _ = poly_div_mod(1 << (2 * T),     p)
    qplusCRC, _ = poly_div_mod(1 << (32 + T),    p)
    return gstar, qplusX, qplusCRC

def opt_reduction(c: int, S: int, T: int, QPLUS: int, GSTAR: int) -> int:
    """Reference implementation of the device-side ``opt_reduction``.

    Computes ``c * x^T mod p`` (low ``T`` bits) when ``QPLUS = floor(x^(S+T)/p)``
    and ``GSTAR = p mod x^T``. Mirrors the CUDA template
    ``opt_reduction<S, T, QPLUS, GSTAR>`` in ``ldpc2_crc_dispatch.cuh``.

    Args:
        c: Input polynomial of degree ``< S``, encoded as a non-negative
            integer.
        S: Input bit-width assumed when picking the shift amount. The funnel
            shift extracts bits ``[S, S+32)`` of ``c * QPLUS``.
        T: Output degree (low ``T`` bits of the returned value are valid).
        QPLUS: Barrett quotient ``floor(x^(S+T) / p)`` for the CRC polynomial.
        GSTAR: ``p mod x^T`` (the polynomial without its leading ``x^T`` term).

    Returns:
        The value ``c * x^T mod p`` as an integer in the range ``[0, 2^T)``.

    Raises:
        ValueError: If any of ``c``, ``QPLUS``, or ``GSTAR`` is negative
            (propagated from :func:`clmul`).

    Examples:
        >>> # CRC-24B: opt_reduction(1) should yield 1 * x^24 mod p = gstar
        >>> opt_reduction(1, 32, 24, 0x1FFFF83FF, 0x800063)
        8388707
        >>> hex(_)
        '0x800063'
    """
    step1 = clmul(c, QPLUS)
    step2 = (step1 >> S) & ((1 << 32) - 1)
    step3 = clmul(step2, GSTAR)
    return step3 & ((1 << T) - 1)

def crc_clmad_partial(word: int, lut: int, T: int, GSTAR: int,
                      QPLUSX: int, QPLUSCRC: int) -> int:
    """Reference for one per-word CLMAD partial CRC.

    Computes ``word * x^T * lut mod p`` using the same two-step
    (preadjust + postadjust) structure as the device-side
    ``compute_crc_clmad``. Used to validate the Barrett constants by
    comparison against ground-truth polynomial arithmetic.

    Args:
        word: 32-bit polynomial word (``selected_word`` in the device code).
        lut: Per-word LUT value (``G_CRC_xxx_P_LUT[k]``) of degree ``< T``.
        T: CRC degree (16 or 24).
        GSTAR: ``p mod x^T``.
        QPLUSX: ``floor(x^(2*T) / p)``, used for the ``S=T`` postadjust.
        QPLUSCRC: ``floor(x^(32+T) / p)``, used for the ``S=32`` preadjust.

    Returns:
        The partial CRC contribution ``(word * x^T * lut) mod p`` in
        ``[0, 2^T)``.

    Raises:
        ValueError: If any negative value is passed (propagated from
            :func:`clmul`).

    Examples:
        >>> # Round-trip against ground truth for one random sample (CRC-24B)
        >>> g, qX, qC = barrett(0x01800063, 24)
        >>> got = crc_clmad_partial(0xDEADBEEF, 0x123456, 24, g, qX, qC)
        >>> want = poly_mod(clmul(clmul(0xDEADBEEF, 1 << 24), 0x123456),
        ...                 0x01800063)
        >>> got == want
        True
    """
    preadjust = opt_reduction(word, 32, T, QPLUSCRC, GSTAR)
    AxB = clmul(preadjust, lut)
    lsb = AxB & ((1 << T) - 1)
    msb = AxB >> T
    return opt_reduction(msb, T, T, QPLUSX, GSTAR) ^ lsb

def poly_mod(a: int, p: int) -> int:
    """Polynomial remainder ``a mod p`` in GF(2)[x].

    Thin wrapper around :func:`poly_div_mod` that discards the quotient.

    Args:
        a: Dividend polynomial, encoded as a non-negative integer.
        p: Divisor polynomial, encoded as an integer. Must be strictly
            positive.

    Returns:
        ``a mod p`` as an integer with degree strictly less than ``deg(p)``.

    Raises:
        ValueError: If ``p <= 0`` or ``a < 0`` (propagated from
            :func:`poly_div_mod`).

    Examples:
        >>> hex(poly_mod(1 << 32, 0x01800063))   # x^32 mod CRC-24B
        '0x804221'
        >>> poly_mod(0, 0x011021)
        0
    """
    _, r = poly_div_mod(a, p)
    return r

if __name__ == "__main__":
    import random
    import sys

    POLYS = [
        ("CRC-16",  0x011021,   16),
        ("CRC-24A", 0x01864CFB, 24),
        ("CRC-24B", 0x01800063, 24),
    ]

    print(f"{'name':<8} {'gstar':>10}  {'qplusX':>10}  {'qplusCRC':>20}")
    for name, p, T in POLYS:
        gstar, qX, qC = barrett(p, T)
        print(f"{name:<8} 0x{gstar:08X}  0x{qX:08X}  0x{qC:016X}")

    # Self-check: per-word formula matches poly-arithmetic ground truth.
    # Track an aggregate failure flag so this script can be used as a
    # validator in CI -- any mismatch must make the process exit non-zero.
    print()
    random.seed(0)
    any_failure = False
    for name, p, T in POLYS:
        gstar, qX, qC = barrett(p, T)
        ok = True
        for _ in range(1000):
            word = random.randrange(1 << 32)
            lut  = random.randrange(1 << T)
            expected = poly_mod(clmul(clmul(word, 1 << T), lut), p)
            got = crc_clmad_partial(word, lut, T, gstar, qX, qC)
            if expected != got:
                ok = False
                print(f"  MISMATCH {name}: word=0x{word:08X} lut=0x{lut:06X} "
                      f"expected=0x{expected:06X} got=0x{got:06X}")
                break
        print(f"{name}: self-check {'PASS' if ok else 'FAIL'} (1000 random samples)")
        if not ok:
            any_failure = True

    sys.exit(1 if any_failure else 0)
