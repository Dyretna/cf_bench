"""
Add Gaussian noise to the BMI column in cfcheck.csv.

BMI is the only continuous feature in the pipeline. All other features are
ordinal (integer-valued Likert scales) and cannot meaningfully receive
continuous noise.

This script tests pipeline robustness: do we still find valid counterfactuals
when the input data has small measurement noise on BMI?

Usage:
    python scripts/add_noise.py
    python scripts/add_noise.py --sigma 2.0 --seed 99

Output:
    data/cfcheck_noisy.csv   (same format as cfcheck.csv)
"""

import argparse

import numpy as np
import pandas as pd

BMI_MIN = 15.0
BMI_MAX = 40.0


def add_bmi_noise(df: pd.DataFrame, sigma: float, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    df = df.copy()
    noise = rng.normal(loc=0.0, scale=sigma, size=len(df))
    df["bmi"] = (df["bmi"] + noise).clip(BMI_MIN, BMI_MAX)
    return df


def main():
    parser = argparse.ArgumentParser(description="Add BMI noise to cfcheck.csv")
    parser.add_argument("--sigma", type=float, default=1.0,
                        help="Standard deviation of Gaussian noise (default: 1.0)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility (default: 42)")
    parser.add_argument("--input", default="data/cfcheck.csv",
                        help="Input CSV path")
    parser.add_argument("--output", default="data/cfcheck_noisy.csv",
                        help="Output CSV path")
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    print(f"Loaded {len(df)} rows from {args.input}")
    print(f"BMI before noise:  mean={df['bmi'].mean():.3f}  std={df['bmi'].std():.3f}")
    print(f"  values: {df['bmi'].tolist()}")

    df_noisy = add_bmi_noise(df, sigma=args.sigma, seed=args.seed)
    print(f"\nBMI after noise (sigma={args.sigma}):")
    print(f"  mean={df_noisy['bmi'].mean():.3f}  std={df_noisy['bmi'].std():.3f}")
    print(f"  values: {df_noisy['bmi'].round(4).tolist()}")

    bmi_diff = (df_noisy["bmi"] - df["bmi"]).round(4)
    print(f"\nBMI changes (noise applied): {bmi_diff.tolist()}")

    df_noisy.to_csv(args.output, index=False)
    print(f"\nSaved noisy dataset to {args.output}")


if __name__ == "__main__":
    main()
