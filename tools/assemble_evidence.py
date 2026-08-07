from __future__ import annotations

import argparse
import json

from cftn_text.evidence import assemble_evidence_report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Assemble standalone, shared-view, causal, and architecture evidence"
    )
    parser.add_argument("--math-report", required=True)
    parser.add_argument("--shared-report", required=True)
    parser.add_argument("--synergy-report", required=True)
    parser.add_argument("--architecture-comparison")
    parser.add_argument("--output-root")
    args = parser.parse_args()
    report = assemble_evidence_report(
        args.math_report,
        args.shared_report,
        args.synergy_report,
        architecture_comparison_path=args.architecture_comparison,
        output_root=args.output_root,
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
