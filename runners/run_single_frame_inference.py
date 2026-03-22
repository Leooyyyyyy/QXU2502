import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.runtime_inference import Phase12InferenceEngine
from core.runtime_inference import RuntimeConfig


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run single-frame inference with the frozen Phase 1.2 model.")
    parser.add_argument("image_path", help="Path to the input image.")
    parser.add_argument("--checkpoint", default="artifacts/experiments/phase1.2_baseline_refined/best_checkpoint.pth")
    parser.add_argument("--config", default="artifacts/experiments/phase1.2_baseline_refined/config.json")
    parser.add_argument("--feature-mode", default="landmarks_v1")
    parser.add_argument("--confidence-threshold", type=float, default=0.55)
    parser.add_argument("--posture-margin-threshold", type=float, default=0.08)
    parser.add_argument("--correctness-threshold", type=float, default=0.50)
    parser.add_argument("--print-freeze-summary", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    runtime_config = RuntimeConfig(
        checkpoint_path=args.checkpoint,
        config_path=args.config,
        feature_mode=args.feature_mode,
        confidence_threshold=args.confidence_threshold,
        posture_margin_threshold=args.posture_margin_threshold,
        correctness_threshold=args.correctness_threshold,
        static_image_mode=True,
    )

    engine = Phase12InferenceEngine(runtime_config)
    try:
        if args.print_freeze_summary:
            print(json.dumps(engine.describe_frozen_model(), indent=2))

        result = engine.predict_image_file(args.image_path, use_stabilization=False)
        print(json.dumps(result, indent=2))
    finally:
        engine.close()


if __name__ == "__main__":
    main()
