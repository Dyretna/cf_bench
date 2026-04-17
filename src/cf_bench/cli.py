import argparse
import logging

import yaml

from .dice_batch_runner import run_pipeline


def main():
    parser = argparse.ArgumentParser(
        description="Run counterfactual explanation pipeline"
    )
    parser.add_argument(
        "--config", required=True, help="Path to YAML configuration file"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug output (shows detailed dtype info, query details, etc.)",
    )
    args = parser.parse_args()

    # Configure logging
    if args.debug:
        logging.basicConfig(
            level=logging.DEBUG,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        )
        logging.info("Debug mode enabled")
    else:
        logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    run_pipeline(cfg)


if __name__ == "__main__":
    main()
