#!/usr/bin/env python3
"""Compare two little-endian float32 dumps produced by the MMID test."""

import argparse
import math
import struct
import sys


def load(path):
    try:
        with open(path, "rb") as handle:
            data = handle.read()
    except OSError as exc:
        raise ValueError("cannot read {}: {}".format(path, exc))
    if not data:
        raise ValueError("{} is empty".format(path))
    if len(data) % 4:
        raise ValueError("{} has {} bytes, not a multiple of 4".format(path, len(data)))
    values = struct.unpack("<{}f".format(len(data) // 4), data)
    bad = next((index for index, value in enumerate(values) if not math.isfinite(value)), None)
    if bad is not None:
        raise ValueError("{} contains non-finite float at index {}".format(path, bad))
    return data, values


def compare(reference_path, candidate_path, atol, rtol):
    reference_bytes, reference = load(reference_path)
    candidate_bytes, candidate = load(candidate_path)
    if len(reference) != len(candidate):
        raise ValueError("element count mismatch: {} vs {}".format(len(reference), len(candidate)))

    n = len(reference)
    max_abs = 0.0
    max_rel = 0.0
    sum_sq = 0.0
    sum_ref = 0.0
    worst = 0
    gross = 0
    exact = 0
    for index, (ref, value) in enumerate(zip(reference, candidate)):
        diff = abs(ref - value)
        if diff > max_abs:
            max_abs = diff
            worst = index
        max_rel = max(max_rel, diff / (abs(ref) + 1e-6))
        sum_sq += diff * diff
        sum_ref += ref * ref
        if reference_bytes[index * 4:(index + 1) * 4] == candidate_bytes[index * 4:(index + 1) * 4]:
            exact += 1
        if diff > atol + rtol * abs(ref):
            gross += 1

    rms_abs = math.sqrt(sum_sq / n)
    rms_ref = math.sqrt(sum_ref / n)
    rms_rel = rms_abs / rms_ref if rms_ref else (0.0 if rms_abs == 0.0 else math.inf)
    print(
        "elements={} bit-identical={} ({:.2f}%) max_abs_diff={:.3e} at {} "
        "(a={:.6f} b={:.6f}) max_rel_diff={:.3e} rms_rel_diff={:.3e}".format(
            n, exact, 100.0 * exact / n, max_abs, worst,
            reference[worst], candidate[worst], max_rel, rms_rel))
    print("elements differing by more than atol + rtol*abs(reference): {} (atol={} rtol={})".format(
        gross, atol, rtol))
    return 0 if gross == 0 else 1


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reference")
    parser.add_argument("candidate")
    parser.add_argument("--atol", type=float, default=1e-3)
    parser.add_argument("--rtol", type=float, default=1e-3)
    args = parser.parse_args(argv)
    if args.atol < 0 or args.rtol < 0 or not math.isfinite(args.atol) or not math.isfinite(args.rtol):
        parser.error("--atol and --rtol must be finite non-negative numbers")
    try:
        return compare(args.reference, args.candidate, args.atol, args.rtol)
    except (OSError, struct.error, ValueError) as exc:
        print("ERROR: {}".format(exc), file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
