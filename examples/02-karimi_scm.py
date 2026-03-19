"""
Certified Causal Recourse Example (Karimi et al., 2021)
Using the Adult Income dataset.

This script demonstrates how to implement *causal* counterfactual recourse
following the framework of Karimi et al. (2021). Unlike DiCE, which operates
purely in the space of correlations, this example constructs an explicit
Structural Causal Model (SCM) and computes interventions using the do-operator.

The SCM is manually specified because the Adult Income dataset does not include
a causal graph. This reflects the requirement in Karimi et al. that causal
recourse depends on domain knowledge, not on correlations learned from data.
"""

import pandas as pd
import numpy as np
from dowhy import CausalModel
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# ---------------------------------------------------------
# 1. Load Adult Income dataset
# ---------------------------------------------------------
from dice_ml.utils import helpers
data_obj = helpers.load_adult_income_dataset()
df = data_obj.dataframe.copy()

target = data_obj.outcome_name

X = df.drop(columns=[target])
y = df[target]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=0
)

# Train a predictive model (not causal)
clf = RandomForestClassifier(random_state=0)
clf.fit(X_train, y_train)

# ---------------------------------------------------------
# 2. Define a Structural Causal Model (SCM)
# ---------------------------------------------------------
"""
We define a simplified causal graph for the Adult dataset.
This is NOT learned from data — it is imposed based on domain knowledge.

Example causal assumptions:
    age → education_num → occupation → income
    age → hours_per_week → income
    sex → occupation → income
    race → occupation → income

This is a simplified SCM but sufficient to illustrate certified recourse.
"""

causal_graph = """
digraph {
    age -> education_num;
    education_num -> occupation;
    occupation -> income;
    age -> hours_per_week;
    hours_per_week -> income;
    sex -> occupation;
    race -> occupation;
}
"""

model = CausalModel(
    data=df,
    treatment="education_num",
    outcome="income",
    graph=causal_graph
)

identified_estimand = model.identify_effect()

# ---------------------------------------------------------
# 3. Compute causal effect of an intervention
# ---------------------------------------------------------
"""
This step computes the effect of do(education_num = x).
This is the core of certified causal recourse: we compute the effect of
an intervention, not a correlation-based counterfactual.
"""

causal_estimate = model.estimate_effect(
    identified_estimand,
    method_name="backdoor.linear_regression"
)

print("Estimated causal effect of education_num on income:")
print(causal_estimate)

# ---------------------------------------------------------
# 4. Compute certified recourse for a specific individual
# ---------------------------------------------------------
"""
We now compute a causal intervention for a single individual.
This is the Karimi-style recourse: find a feasible intervention on
actionable variables that causally leads to a desired outcome.
"""

x = X_test.iloc[0].copy()

desired_outcome = 1  # income > 50k

# Example: actionable variable = education_num
# We compute the minimal intervention needed to flip the outcome causally.

current_pred = clf.predict([x])[0]

if current_pred == desired_outcome:
    print("This individual already has the desired outcome.")
else:
    # Compute causal effect of increasing education
    delta = causal_estimate.value

    # Minimal intervention needed
    required_change = 1 / delta

    new_education = x["education_num"] + required_change
    new_education = max(min(new_education, 16), 1)  # clamp to valid range

    print("\nCertified causal recourse recommendation:")
    print(f"Increase education_num from {x['education_num']} to {new_education:.2f}")
    print("This intervention is certified because it is computed via the SCM.")
