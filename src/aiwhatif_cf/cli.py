import argparse

import yaml

from .runner import run_pipeline


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    run_pipeline(cfg)


if __name__ == "__main__":
    main()
