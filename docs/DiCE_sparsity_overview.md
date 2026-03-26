# Counterfactuals, Sparsity, and DiCE – A Technical and Psychological Overview

This note summarizes how counterfactual explanations work, why sparsity is essential, how DiCE handles sparsity, and what post‑hoc strategies can be used to make recommendations more understandable and psychologically manageable.

---

## 1. What Are Counterfactuals?

Counterfactuals are hypothetical data points where certain input variables have been changed so that a model’s prediction shifts in a desired direction.

Example (informal):

- Original:
  - BMI = 31
  - Smoking = 5
  - Alcohol = 6
  - Sleep = 4
  - Risk = 0.48

- Counterfactual:
  - BMI = 24
  - Smoking = 3
  - Alcohol = 6
  - Sleep = 4
  - Risk = 0.16

Interpretation: “If you were more like this profile, your predicted risk would look like this.”

Counterfactuals are used for:
- explanations (“why did I get this risk?”)
- recommendations (“what can I change to reduce my risk?”)
- simulations and policy analysis

---

## 2. What Does Sparsity Mean?

**Sparsity** describes how many variables change in a counterfactual.

- High sparsity → few changes
- Low sparsity → many changes

Why it matters:
- fewer changes → easier to understand
- fewer changes → more realistic to implement
- fewer changes → lower cognitive load

Examples:
- Dense CF: changes 6–7 features → cognitively heavy
- Sparse CF: changes 1–3 features → psychologically manageable

Sparsity is therefore both a **technical** and **psychological** design concern.

---

## 3. DiCE and Sparsity – What the Library Provides

DiCE (Diverse Counterfactual Explanations) is a library for generating counterfactuals for a wide range of models.

It provides:
- CF generation for single or multiple instances
- support for many model types (sklearn, PyTorch, etc.)
- parameters for:
  - number of CFs
  - target class or target value
  - which features may change
  - constraints on feature ranges
  - sparsity (global)

---

## 4. `posthoc_sparsity_param` – Global Sparsity in DiCE

DiCE includes a parameter:

**`posthoc_sparsity_param` ∈ [0, 1]**

This controls how aggressively DiCE tries to make CFs more sparse **after** they are generated (post‑hoc).

Typical behavior:
- 0.0 → no extra sparsity
- 0.1 → mild sparsity (common default)
- 0.5 → fairly aggressive
- 1.0 → very aggressive (risk of invalid CFs)

The idea:
1. DiCE generates a CF that reaches the target.
2. DiCE attempts to revert changed features back to their original values.
3. If the CF still reaches the target → the change is removed.
4. This continues until the sparsity target is reached.

This is **global sparsity**: all features are treated equally.

---

## 5. Limitations: No Built‑In Per‑Feature Sparsity

DiCE does not provide a parameter that says:

- “change BMI a lot”
- “change smoking only a little”
- “do not change alcohol at all”

There is no direct per‑feature sparsity control in the API.

However, there are indirect ways to achieve this.

---

## 6. Indirect Per‑Feature Control via Constraints

### 6.1 `features_to_vary`

You can control which features are allowed to change:

- `"all"` → all features may change
- list → only these features may change

This is coarse but effective: some variables can be locked entirely.

### 6.2 `permitted_range`

You can restrict how much each feature may change by specifying ranges.

Conceptual example:

- BMI: allowed to vary within [20, 35]
- Smoking: not allowed to change (same as original)
- Alcohol: allowed within a narrow range
- Sleep: allowed within a reasonable range

This yields:
- more realistic CFs
- implicit per‑feature sparsity (some features barely change)

---

## 7. Custom Post‑Hoc Processing – More Control, More Responsibility

To truly adapt counterfactuals to:
- domain knowledge
- psychological realism
- policy constraints

…you often need more than DiCE’s built‑in mechanisms.
This is where **custom post‑hoc processing** becomes valuable.

General idea:
1. Generate CFs with DiCE (with reasonable constraints).
2. Apply a custom function that:
   - prioritizes certain features (e.g., behaviors over biological factors)
   - limits others (e.g., features that are hard to change)
   - removes “unnecessary” changes
   - ensures the CF still achieves the desired outcome

This can be rule‑based, weighted, or more advanced.

---

## 8. The Psychological Dimension: Why This Is Hard

Recommendations are not just about:

> “Which combination of features reduces risk the most?”

They are also about:

> “Which changes can a person realistically understand and act on?”

Challenges:
- too many changes → the person does nothing
- too large changes → feel unrealistic or overwhelming
- overly technical recommendations → hard to connect to real behavior

Therefore, sparsity and post‑hoc processing are not just technical details; they are central to designing effective, human‑centered recommendations.

In practice, this often requires iterative tuning based on:
- domain expertise
- user studies
- behavioral insights

---

## 9. DiCE API – High‑Level Overview

Typical conceptual call:

- Create an `explainer` with:
  - data
  - model
  - feature metadata
- Call a method to generate counterfactuals for one or more instances.

Key parameters:

- `query_instances` – the individuals to generate CFs for
- `total_CFs` – number of CFs per individual
- `desired_class` or `desired_range` – target outcome (e.g., lower risk)
- `features_to_vary` – which features may change
- `permitted_range` – per‑feature constraints
- `stopping_threshold` – how close the CF must get to the target
- `posthoc_sparsity_param` – global sparsity control
- `posthoc_sparsity_algorithm` – method for sparsity (e.g., “linear”)

---

## 10. Summary

- **Sparsity** is crucial for making counterfactuals understandable and actionable.
- **DiCE** provides:
  - global sparsity via `posthoc_sparsity_param`
  - constraints via `features_to_vary` and `permitted_range`
- **Per‑feature sparsity** requires either smart use of constraints or custom post‑hoc logic.
- **Human‑centered recommendations** require more than technical optimality:
  - they must be psychologically manageable
  - they must feel realistic
  - they must be few and clear

This means sparsity and post‑hoc processing are not optional—they are core design choices in any counterfactual recommendation system. In practice, these components often need to be tuned over time based on data, domain knowledge, and user feedback.
