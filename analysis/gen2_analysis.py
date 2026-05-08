"""
Generation 2 — Automatic experiment comparison.

Auto-discovers all annotated.csv files in cf_outputs/, computes
a standard set of metrics for each run, and exports a comparison table.

Usage:
    python analysis/gen2_analysis.py
    python analysis/gen2_analysis.py --dir cf_outputs/predictors_vs_threshold
    python analysis/gen2_analysis.py --save

Metrics computed per experiment:
    validity_%        — valid CFs / total CFs × 100
    solved_%          — patients with ≥1 valid CF / total patients × 100
    avg_nchanged      — mean features changed per valid CF (valid CFs only)
    avg_nchanged_all  — mean features changed per CF (all CFs, valid + invalid)
    avg_gower_valid   — mean Gower distance of valid CFs (lower = more similar to original)
    avg_risk_before_% — mean baseline risk (× 100)
    avg_risk_after_%  — mean risk after CF (valid CFs only, × 100)
    min_risk_after_%  — lowest risk achieved across all valid CFs (× 100)
    risk_reduction_%  — mean (risk_before - risk_after) / risk_before × 100 (valid CFs)
    avg_gen_time_sec  — mean CF generation time per patient (seconds)
    top_features      — features most often changed in valid CFs
"""
import pandas as pd 
from pathlib import Path 
import argparse                 # lire les arguments de la ligne de commande 
import glob                     # trouver tous les fichiers qui matchent un pattern 

FEATURES = ["etfruit", "eatveg" , "cgtsmok" , "alcfreq" , "slprl", "paccnois", "bmi" , "dosprt"]

FILES = glob.glob("cf_outputs/**/annotated.csv", recursive=True)



def compute_validity_rate( cf_valid , cf_all) :
    return (len(cf_valid) / len(cf_all)) * 100

def compute_solved_rate(cf_all , original_rows):
    nb_patients = len(original_rows)
    solved = cf_all.groupby("query_index")["valid"].apply(lambda x : (x == True).any())
    return round(solved.sum()/ nb_patients * 100, 1)
    
def compute_avg_nchanged(cf_valid):
    if "Nchanged" not in cf_valid.columns:
        return None
    values = pd.to_numeric(cf_valid["Nchanged"], errors="coerce")
    return round(values.mean(), 2)

def compute_avg_gower_valid(cf_valid):
    if "gower_distance" not in cf_valid.columns:
        return None
    values = pd.to_numeric(cf_valid["gower_distance"], errors="coerce")
    return round(values.mean(), 4)

def compute_avg_risk_before_rate(cf_all):
    values = pd.to_numeric(cf_all["risk_before"], errors="coerce")
    return round(values.mean() * 100, 1)

def compute_avg_risk_after_rate(cf_valid): 
    values = pd.to_numeric(cf_valid["predicted_risk_after"], errors="coerce")
    return round(values.mean() * 100 ,1)

def compute_risk_reduction_rate(cf_all, cf_valid):
    risk_before = compute_avg_risk_before_rate(cf_all)
    risk_after  = compute_avg_risk_after_rate(cf_valid)
    return round((risk_before - risk_after) / risk_before * 100, 1)

def compute_avg_nchanged_all(cf_all):
    if "Nchanged" not in cf_all.columns:
        return None
    values = pd.to_numeric(cf_all["Nchanged"], errors="coerce")
    return round(values.mean(), 2)

def compute_min_risk_after_rate(cf_valid):
    values = pd.to_numeric(cf_valid["predicted_risk_after"], errors="coerce")
    return round(values.min() * 100, 1)

def compute_avg_gen_time(original_rows):
    if "cf_gen_time_sec" not in original_rows.columns:
        return None
    values = pd.to_numeric(original_rows["cf_gen_time_sec"], errors="coerce")
    return round(values.mean(), 2)

def compute_top_features(cf_valid):
    features_range = []
    for feature in FEATURES : 
        nb_cf_f = cf_valid[feature].notna() & (cf_valid[feature].astype(str)!= "")
        sum_feature = nb_cf_f.sum()
        result_feature = (sum_feature / len(cf_valid))*100
        features_range.append((feature ,result_feature)) 
    features_range.sort(key =lambda x : x[1], reverse=True)
    top4 = features_range[:4]
    return "  |  ".join(f"{feat}: {pct:.0f}%" for feat, pct in top4)
          


def analyse_one(csv_path):

    df = pd.read_csv(csv_path)
    cf_all = df [df["cf_id"] != "original"]
    cf_valid = cf_all [cf_all["valid"] == True]
    original_rows = df [df["cf_id"] == "original"]

    # all metrics calculate below 
    validity_rate = compute_validity_rate( cf_valid , cf_all)

    solved_rate = compute_solved_rate(cf_all , original_rows)

    avg_nchanged = compute_avg_nchanged(cf_valid)

    avg_nchanged_all = compute_avg_nchanged_all(cf_all)

    avg_gower_valid = compute_avg_gower_valid(cf_valid)

    avg_risk_before_rate = compute_avg_risk_before_rate(cf_all)

    avg_risk_after_rate = compute_avg_risk_after_rate(cf_valid)

    min_risk_after_rate = compute_min_risk_after_rate(cf_valid)

    risk_reduction_rate = compute_risk_reduction_rate(cf_all, cf_valid)

    avg_gen_time = compute_avg_gen_time(original_rows)

    top_features = compute_top_features(cf_valid)


    n_patients  = len(original_rows)
    total_cfs   = len(cf_all)
    valid_cfs   = len(cf_valid)

    # dictionary containing everything 
    metrics = {
        "experiment":Path(csv_path).parent.name,
        "n_patients":n_patients,
        "total_cfs":total_cfs ,
        "valid_cfs":valid_cfs,
        "validity_%":validity_rate ,
        "solved_%":solved_rate, 
        "avg_nchanged":avg_nchanged,
        "avg_nchanged_all":avg_nchanged_all,
        "avg_gower_valid":avg_gower_valid,
        "avg_risk_before_%":avg_risk_before_rate,
        "avg_risk_after_%":avg_risk_after_rate,
        "min_risk_after_%":min_risk_after_rate,
        "risk_reduction_%":risk_reduction_rate,
        "avg_gen_time_sec":avg_gen_time,
        "top_features":top_features,
        "csv_path":str(csv_path)
    }

    return metrics


def main():

    rows = []
    for file in FILES : 
        rows.append(analyse_one(file))
    complete_analysis = pd.DataFrame(rows) 
    complete_analysis.to_csv("analysis/gen2_summary.csv", index=False)
    print(complete_analysis.to_string())   

if __name__ == "__main__":
    main()    