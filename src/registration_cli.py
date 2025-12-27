import argparse
import json
from pathlib import Path
from typing import Optional

from .registration_evaluation import evaluate_registration


def _print_report(res: dict):
    print("Registration Evaluation Report")
    print("------------------------------")
    metrics = res.get('metrics', {})
    for k, v in metrics.items():
        print(f"{k:25s}: {v:.4f}")
    print("------------------------------")
    print(f"Final score               : {res.get('final_score', 0.0):.4f}")
    extra = res.get('extra', {})
    if extra:
        print("\nAdditional info:")
        print(json.dumps(extra, indent=2))


def main(argv: Optional[list] = None):
    parser = argparse.ArgumentParser(description="Evaluate MRI registration quality")
    parser.add_argument('fixed', help='Fixed image (NIfTI)')
    parser.add_argument('warped', help='Warped moving image (NIfTI)')
    parser.add_argument('--deformation', help='Deformation field (NIfTI vector image)', default=None)
    parser.add_argument('--landmarks', help='NPZ file containing "fixed" and "moving" arrays', default=None)
    parser.add_argument('--weights', help='JSON string or file specifying per-metric weights', default=None)
    args = parser.parse_args(argv)

    weights = None
    if args.weights:
        try:
            # try parse as JSON string first
            weights = json.loads(args.weights)
        except Exception:
            # try load file
            p = Path(args.weights)
            if p.exists():
                weights = json.loads(p.read_text())

    res = evaluate_registration(args.fixed, args.warped, deformation_field_path=args.deformation, landmarks_path=args.landmarks, weights=weights)
    _print_report(res)


if __name__ == '__main__':
    main()
