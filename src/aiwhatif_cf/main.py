import warnings

import joblib
import pandas as pd

from .config import MODEL_PATH_HB, TEST_DATA_PATH, RandomExplainerProfile, SystemConfig
from .dice_pipeline import DiceCFPipeline

# DiCE random explainer triggers Pandas FutureWarnings due to
# internal .at[] assignments. (floats are wrong Dtype)
# Suppress them to keep output clean.
warnings.filterwarnings("ignore", category=FutureWarning)


def main():
    targets = ["hltprhb"]
    for target in targets:
        # load config, explainer profile, predictor and test data
        system_config = SystemConfig(target=target)
        rand_explainer_profile = RandomExplainerProfile(
            features_to_vary=system_config.features_to_vary
        )
        predictor_model = joblib.load(MODEL_PATH_HB)
        df = pd.read_csv(TEST_DATA_PATH)

        # create an instance and run for generation
        pipeline = DiceCFPipeline(
            system_config=system_config,
            explainer_profile=rand_explainer_profile,
            predictor=predictor_model,
        )

        # query instances for pipeline
        # later transform from json (dict) to df
        query_instances_df = df.loc[df[target] == 1, system_config.feature_cols]
        top5 = query_instances_df.head(5)

        (all_annotated, all_recommendations, all_formated_recommendations) = (
            pipeline.process_batch(df, top5)
        )

        for i, (annotated, recs, formatted) in enumerate(
            zip(all_annotated, all_recommendations, all_formated_recommendations)
        ):
            print("\n=== Query Instance ===")
            print(query_instances_df.iloc[[i]])

            print("\n=== Annotated CFs ===")
            print(annotated)

            print("\n=== Recommendations ===")
            print(recs)

            print("\n=== Formatted Recommendations ===")
            print(formatted)
            print("-" * 80)


if __name__ == "__main__":
    main()
