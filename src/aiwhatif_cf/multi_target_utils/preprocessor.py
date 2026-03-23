import pandas as pd


class DataPreprocessor:
    def __init__(self, outcome_cols):
        self.outcome_cols = outcome_cols

    def split_X_and_targets(self, df: pd.DataFrame):
        X = df.drop(columns=self.outcome_cols)
        y_dict = {t: df[t] for t in self.outcome_cols}
        return X, y_dict
