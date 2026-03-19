"""
Module: causal_cf_demo.py

This module implements a Wachter-style counterfactual explanation and how
to generate counterfactual explanations using DiCE
(Diverse Counterfactual Explanations) in Python. The example is intentionally
constructed to highlight the conceptual and methodological gap between standard
counterfactual explanations and *certified causal recourse* as defined in
Karimi et al. (2021), "Algorithmic Recourse: From Counterfactual Explanations
to Interventions".

DiCE operates purely in the space of statistical associations learned by a
predictive model. It searches for alternative input configurations that flip
the model's prediction, but it does not incorporate any knowledge about the
causal structure of the domain. As a result, the counterfactuals produced by
DiCE are *feasible only within the model*, not necessarily feasible in the
real world.

In contrast, certified causal recourse requires:
    - an explicit Structural Causal Model (SCM),
    - well-defined structural equations,
    - the ability to compute interventions using Pearl's do-operator,
    - guarantees that recommended changes are causally valid and physically,
      socially, or logically possible.

This module therefore serves two purposes:
    (1) to illustrate how DiCE generates counterfactuals in a correlation-based
        setting, and
    (2) to clarify why such counterfactuals cannot be considered "certified"
        without an underlying causal model.

The Adult Income dataset used here contains no causal graph, no structural
equations, and no encoded constraints on which features can or cannot be
changed. Thus, DiCE may propose counterfactuals that violate real-world
causal dependencies (e.g., changing education without affecting age, or
changing income without affecting education).

This example is pedagogical: it shows how standard CF methods work, and why
causal recourse requires a fundamentally different framework.
"""

import dice_ml
from dice_ml import Dice
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pandas as pd

# Load the Adult Income dataset provided by DiCE.
# This dataset includes demographic and socioeconomic variables, but crucially,
# it does NOT include a causal model describing how these variables influence
# one another. DiCE therefore treats all features as manipulable unless the
# user manually specifies constraints.
from dice_ml.utils import helpers
data_obj = helpers.load_adult_income_dataset()
df = data_obj.dataframe

target = data_obj.outcome_name

X = df.drop(columns=[target])
y = df[target]

# Standard ML train/test split.
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=0
)

# Train a predictive model.
# This model captures correlations in the dataset, not causal mechanisms.
clf = RandomForestClassifier(random_state=0)
clf.fit(X_train, y_train)

# Create a DiCE Data object.
# This object describes the dataset but does not encode causal constraints.
d = dice_ml.Data(
    dataframe=df,
    continuous_features=data_obj.continuous_features,
    outcome_name=target
)

# Wrap the trained model for DiCE.
m = dice_ml.Model(
    model=clf,
    backend="sklearn"
)

# Initialize a DiCE explainer.
# The "random" method searches the input space heuristically.
# No causal reasoning is involved.
exp = Dice(d, m, method="random")

# Select an instance for which we want counterfactuals.
query_instance = X_test.iloc[0:1]

# Generate counterfactual explanations.
# These counterfactuals are valid only in the model's correlation space.
# They are NOT certified causal interventions.
dice_exp = exp.generate_counterfactuals(
    query_instance,
    total_CFs=5,
    desired_class="opposite"
)

# Display the generated counterfactuals.
dice_exp.visualize_as_dataframe()
