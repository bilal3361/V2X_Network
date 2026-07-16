from __future__ import annotations

import argparse
from pathlib import Path

from v2x_task5_common import (
    FEATURE_SCALER_PATH,
    MODEL_METADATA_PATH,
    MODEL_PATH,
    TARGET_SCALER_PATH,
    TRAINING_DATASET_PATH,
    export_scalers_and_metadata,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export task 5 scalers and metadata for real-time inference."
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=TRAINING_DATASET_PATH,
        help="Training dataset CSV used to recreate the model scalers.",
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=MODEL_PATH,
        help="Saved Keras trajectory model.",
    )
    parser.add_argument(
        "--feature-scaler",
        type=Path,
        default=FEATURE_SCALER_PATH,
        help="Output path for the feature scaler.",
    )
    parser.add_argument(
        "--target-scaler",
        type=Path,
        default=TARGET_SCALER_PATH,
        help="Output path for the target scaler.",
    )
    parser.add_argument(
        "--metadata",
        type=Path,
        default=MODEL_METADATA_PATH,
        help="Output path for task 5 model metadata.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metadata = export_scalers_and_metadata(
        dataset_path=args.dataset,
        model_path=args.model,
        feature_scaler_path=args.feature_scaler,
        target_scaler_path=args.target_scaler,
        metadata_path=args.metadata,
    )

    print("Task 5 inference artifacts exported.")
    print(f"Feature scaler: {args.feature_scaler}")
    print(f"Target scaler : {args.target_scaler}")
    print(f"Metadata      : {args.metadata}")
    print(f"Model         : {args.model}")
    print(f"Training rows : {metadata['training_rows']}")
    print(
        "Vehicles      : "
        f"train={metadata['unique_train_vehicles']}, "
        f"validation={metadata['unique_validation_vehicles']}, "
        f"test={metadata['unique_test_vehicles']}"
    )


if __name__ == "__main__":
    main()
