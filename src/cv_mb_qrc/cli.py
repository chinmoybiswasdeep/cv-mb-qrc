"""Command-line interfaces for evidence inspection."""

import argparse
import json
import sys

from .reservoirs.results import verify_result_directory


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="cvmbqrc")
    subparsers = parser.add_subparsers(dest="command", required=True)
    verify = subparsers.add_parser("verify-run", help="verify a content-indexed run")
    verify.add_argument("path")
    args = parser.parse_args(argv)
    try:
        report = verify_result_directory(args.path)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"invalid run: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(report, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
