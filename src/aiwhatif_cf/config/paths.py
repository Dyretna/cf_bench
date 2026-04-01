import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()
DATA_DIR = Path(os.getenv("DATA_DIR"))
MODELS_DIR = Path(os.getenv("MODELS_DIR"))
CF_OUTPUTS = Path(os.getenv("CF_OUTPUTS"))

TRAIN_DATA_PATH = DATA_DIR / "05_single_target" / "ess_ready_v2_hltprhb_train.csv"
TEST_DATA_PATH = DATA_DIR / "05_single_target" / "ess_ready_v2_hltprhb_test.csv"
MODEL_PATH_HB = MODELS_DIR / "rf_hltprhb_2026-03-31.pkl"
MODEL_PATH_HC = MODELS_DIR / "rf_hltprhc_2026-03-31.pkl"


def check_paths():
    return (
        f"DATA_DIR: {DATA_DIR} | is_dir: {DATA_DIR.is_dir()}\n"
        f"MODELS_DIR: {MODELS_DIR} | is_dir: {MODELS_DIR.is_dir()}\n"
        f"CF_OUTPUTS: {CF_OUTPUTS} | is_dir: {CF_OUTPUTS.is_dir()}\n"
        f"TRAIN_DATA_PATH: {TRAIN_DATA_PATH} | is_file: {TRAIN_DATA_PATH.is_file()}\n"
        f"TEST_DATA_PATH: {TEST_DATA_PATH} | is_file: {TEST_DATA_PATH.is_file()}\n"
        f"MODEL_PATH_HB: {MODEL_PATH_HB} | is_file: {MODEL_PATH_HB.is_file()}\n"
        f"MODEL_PATH_HC: {MODEL_PATH_HC} | is_file: {MODEL_PATH_HC.is_file()}"
    )
