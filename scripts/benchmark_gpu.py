from __future__ import annotations

import argparse
import json

from teukolsky.accelerated.validation import benchmark_mode


CASES = {
    "generic": dict(s=-2, ell=2, m=2, a=0.5, p=10.0, e=0.2, x=0.7, n=0, k=0),
    "eccentric": dict(s=-2, ell=2, m=2, a=0.5, p=10.0, e=0.2, x=1.0, n=0, k=0),
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Run CPU vs GPU benchmark cases for Teukolsky point-particle modes.")
    parser.add_argument("--case", choices=("generic", "eccentric", "all"), default="all")
    args = parser.parse_args()

    selected = CASES.items() if args.case == "all" else [(args.case, CASES[args.case])]
    results = {name: benchmark_mode(**case) for name, case in selected}
    print(json.dumps(results, indent=2, default=str))


if __name__ == "__main__":
    main()
