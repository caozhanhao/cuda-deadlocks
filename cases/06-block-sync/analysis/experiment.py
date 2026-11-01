#!/usr/bin/env python3
"""Generate the four B-register controls; add --run to execute them."""

import argparse
import hashlib
from pathlib import Path
import subprocess


HERE = Path(__file__).resolve().parent
SHA256 = "51e45257238de2ffc740d5b2cf0896918475acd8487e35cc5598b0968ac8336b"
TEXT_OFFSET = 0x780  # Kernel start within reference.cubin.
PAIRS = ((0x3c0, 0x470), (0x5e0, 0x680))  # BSSY/BSYNC offsets within the kernel.
CASES = {"B0-B0": (0, 0), "B0-B3": (0, 3), "B3-B0": (3, 0), "B3-B3": (3, 3)}


def patch():
    original = (HERE / "reference.cubin").read_bytes()
    if hashlib.sha256(original).hexdigest() != SHA256:
        raise SystemExit("reference.cubin: SHA256 mismatch")

    output = HERE / "build"
    output.mkdir(exist_ok=True)
    cubins = {}
    for case, registers in CASES.items():
        binary = bytearray(original)
        for offsets, register in zip(PAIRS, registers):
            for offset in offsets:
                # B-register operand: bits 19:16, in the third byte.
                index = TEXT_OFFSET + offset + 2
                binary[index] = (binary[index] & 0xf0) | register
        cubins[case] = output / (case + ".cubin")
        cubins[case].write_bytes(binary)
    return cubins


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="store_true", help="Run each case three times")
    args = parser.parse_args()
    cubins = patch()
    if not args.run:
        print("Wrote four cubins to " + str(HERE / "build"))
        return 0

    for case, cubin in cubins.items():
        for repeat in range(3):
            result = subprocess.run([str(HERE / "loader"), str(cubin)],
                                    stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                    text=True, timeout=8)
            print(f"{case} [{repeat + 1}/3]: {result.stdout.strip()}", flush=True)
            if result.returncode not in (0, 2):
                return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
