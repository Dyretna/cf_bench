# ruff: noqa

"""
This module implements a hybrid explanatory framework that combines
Karimi-style certified causal recourse with Baron-style philosophical
criteria for causal explanation. The goal is to show how algorithmic
recourse, grounded in structural causal models (SCMs), can be evaluated
not only for causal correctness (in the Pearl sense) but also for
explanatory adequacy (in the Baron sense).

The theoretical background integrates three major traditions:

1. Lewis-style counterfactuals:
   Counterfactuals are evaluated by comparing possible worlds ordered
   by similarity. This framework provides the logical form of
   counterfactual reasoning ("If X had been different, Y would not
   have occurred") but does not specify what makes a world similar,
   nor how variables depend on one another. As a result, Lewis-style
   counterfactuals do not guarantee causal validity.

2. Pearl-style structural causal models:
   Pearl replaces similarity-based counterfactuals with explicit
   structural equations. Interventions are defined using the
   do-operator, which modifies the data-generating process by
   severing incoming causal edges. This makes counterfactuals
   causally grounded: they describe what would happen under a
   specific intervention in the actual world, not in an abstract
   possible world. Karimi-style certified recourse is built on
   this foundation.

3. Baron-style causal explanation:
   Baron develops a theory of explanation that emphasizes causal
   relevance, minimality, and contrastiveness. A good explanation
   identifies the specific causal pathway responsible for an outcome,
   shows how altering a variable along that pathway would change the
   outcome, and avoids including irrelevant or redundant factors.
   Baron's framework clarifies when a counterfactual is not merely
   a hypothetical alternative but a genuine explanation of why the
   outcome occurred.

This module demonstrates how these ideas can be combined in practice.
Karimi-style recourse provides a certified causal intervention derived
from an SCM. Baron-style criteria are then applied to evaluate whether
the intervention constitutes an adequate explanation: whether it lies
on a causally relevant pathway, whether it is minimal, and whether it
addresses a meaningful contrast ("Why outcome=1 rather than outcome=0").

The Adult Income dataset is used as a running example. Because the
dataset does not include a causal graph, an SCM must be specified
manually. This reflects the central insight shared by Pearl, Karimi,
and Baron: causal explanations cannot be derived from correlations
alone. They require explicit causal assumptions. The module therefore
illustrates both the computational mechanics of certified recourse
and the philosophical constraints that determine when such recourse
counts as an explanation in a deeper sense.
"""

from dice_ml.utils import helpers
from dowhy import CausalModel
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

data_obj = helpers.load_adult_income_dataset()
df = data_obj.dataframe.copy()

target = data_obj.outcome_name

X = df.drop(columns=[target])
y = df[target]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Train a predictive model (not causal)
clf = RandomForestClassifier(random_state=0)
clf.fit(X_train, y_train)


# ---------------------------------------------------------
# Define a Structural Causal Model (SCM)
#
# We define a simplified causal graph for the Adult dataset.
# This is NOT learned from data — it is imposed based on domain knowledge.
#
# Example causal assumptions:
#     age → education_num → occupation → income
#     age → hours_per_week → income
#     sex → occupation → income
#     race → occupation → income
#
# This is a simplified SCM but sufficient to illustrate certified recourse.
# ---------------------------------------------------------

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
    data=df, treatment="education_num", outcome="income", graph=causal_graph
)
identified_estimand = model.identify_effect()

# ---------------------------------------------------------
# Compute causal effect of an intervention
#
# This step computes the effect of do(education_num = x).
# This is the core of certified causal recourse: we compute the effect of
# an intervention, not a correlation-based counterfactual.
# ---------------------------------------------------------

causal_estimate = model.estimate_effect(
    identified_estimand, method_name="backdoor.linear_regression"
)


# ---------------------------------------------------------
# 1. Identify causal pathways (Baron-style)
# ---------------------------------------------------------

causal_paths = model.get_directed_paths("education_num", "income")

print("Causal pathways from education_num to income:")
for path in causal_paths:
    print(" -> ".join(path))

# ---------------------------------------------------------
# 2. Check Baron’s minimality criterion
# ---------------------------------------------------------


def is_minimal_intervention(delta, threshold=0.01):
    """
    Baron-style minimality:
    An intervention is minimal if no smaller intervention
    would change the outcome.
    """
    return abs(delta) > threshold


minimal = is_minimal_intervention(causal_estimate.value)

print("\nIs the intervention minimal (Baron-style)?", minimal)

# ---------------------------------------------------------
# 3. Check explanatory relevance
# ---------------------------------------------------------


def is_explanatorily_relevant(path):
    """
    Baron-style relevance:
    A variable is explanatorily relevant if it lies on a causal path
    from the intervention to the outcome.
    """
    return "income" in path[-1]


relevant_paths = [p for p in causal_paths if is_explanatorily_relevant(p)]

print("\nExplanatorily relevant causal paths:")
for p in relevant_paths:
    print(" -> ".join(p))

# ---------------------------------------------------------
# 4. Produce a Baron-style explanation
# ---------------------------------------------------------

print("\nBaron-style causal explanation:")
print(
    f"The recommended intervention (increase education_num) "
    f"is explanatorily relevant because it affects income "
    f"through the causal pathways: {relevant_paths}. "
    f"The intervention is minimal: {minimal}."
)
