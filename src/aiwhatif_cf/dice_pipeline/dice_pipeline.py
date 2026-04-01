import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from ..config import (
    GeneticExplainerProfile,
    GradientExplainerProfile,
    RandomExplainerProfile,
    SystemConfig,
)
from .build_explainer import build_explainer
from .recommendations import DiceRecommender
from .risk import RiskEvaluator

explainer_profile = type[
    RandomExplainerProfile | GeneticExplainerProfile | GradientExplainerProfile
]


class DiceCFPipeline:
    def __init__(
        self,
        system_config: SystemConfig,
        explainer_profile: explainer_profile,
        predictor: RandomForestClassifier,
    ):
        self.system_config = system_config
        self.explainer_profile = explainer_profile
        self.predictor = predictor
        self.explainer = None  # assigned when during run
        self.counterfactuals = None

        self.risk_evaluator = RiskEvaluator(
            model=self.predictor,
            feature_cols=self.system_config.feature_cols,
            target_factor=self.system_config.target_factor,
        )

        self.recommender = DiceRecommender(
            feature_cols=self.system_config.feature_cols,
            target=self.system_config.target,
        )

    def process_single(self, df, query_instance):
        cf_result = self.run(df, query_instance)
        annotated = self.annotate(query_instance, cf_result.final_cfs_df)
        recs = self.get_recommendations(query_instance, annotated)
        formatted = self.format_recommendations(query_instance, recs)
        return annotated, recs, formatted

    def process_batch(self, df, query_instances):
        all_annotated = []
        all_recs = []
        all_formatted = []

        cf_result = self.run(df, query_instances)

        for i, cf in enumerate(cf_result.cf_examples_list):
            qi = query_instances.iloc[[i]]
            cf_df = cf.final_cfs_df

            annotated = self.annotate(qi, cf_df)
            recs = self.get_recommendations(qi, annotated)
            formatted = self.format_recommendations(qi, recs)

            all_annotated.append(annotated)
            all_recs.append(recs)
            all_formatted.append(formatted)

        return all_annotated, all_recs, all_formatted

    def run(self, df: pd.DataFrame, query_instances: pd.DataFrame):
        # build explainer model that generates CF
        self.explainer = build_explainer(
            config=self.system_config,
            predictor_model=self.predictor,
            df=df,
            explainer_method=self.explainer_profile.method,
        )

        self.counterfactuals = self.explainer.generate_counterfactuals(
            query_instances=query_instances, **self.explainer_profile.to_cf_params()
        )
        return self.counterfactuals

    def annotate(
        self, query_instances: pd.DataFrame, counterfactuals_df: pd.DataFrame
    ) -> pd.DataFrame:
        return self.risk_evaluator.annotate(query_instances, counterfactuals_df)

    def get_recommendations(
        self, query_instances: pd.DataFrame, annoted_counterfactuals: pd.DataFrame
    ) -> list[dict,]:
        return self.recommender.get_recommendations(
            query_instances,
            annoted_counterfactuals,
        )

    def format_recommendations(
        self,
        query_instances: pd.DataFrame,
        recommendations: list[dict],
        true_outcome: int | str = 1,
    ) -> str:
        return self.recommender.format_recommendations(
            query_instances, recommendations, true_outcome=true_outcome
        )
