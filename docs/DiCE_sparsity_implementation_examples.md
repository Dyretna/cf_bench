# Counterfactuals, Sparsity, DiCE, and Practical Implementation Examples

This note provides a cohesive overview of counterfactual explanations, sparsity, DiCE’s API, and practical implementation patterns. It includes concrete code examples demonstrating how to generate counterfactuals, apply constraints, and implement custom post‑hoc sparsity logic.

---

## 1. What Are Counterfactuals?

Counterfactuals are hypothetical versions of an individual where certain input features have been modified so that a predictive model produces a different (usually more desirable) outcome.

Example:
- Original: BMI = 31, Smoking = 5, Risk = 0.48
- Counterfactual: BMI = 24, Smoking = 3, Risk = 0.16

Interpretation: “If you were more like this profile, your predicted risk would be lower.”

Counterfactuals support:
- explanations
- recommendations
- simulation and policy analysis

---

## 2. Sparsity: Why It Matters

**Sparsity** refers to how many features change in a counterfactual.

- High sparsity → few changes → easier to understand and act on
- Low sparsity → many changes → cognitively overwhelming

Human‑centered recommendations require sparsity because:
- people struggle with many simultaneous changes
- small, focused changes are more realistic
- behavior change research shows that fewer recommendations increase adherence

---

## 3. DiCE and Sparsity – What the Library Provides

DiCE offers:
- CF generation for single or multiple instances
- support for many model types
- global sparsity control
- feature constraints
- post‑hoc sparsity optimization

Key parameters:
- `total_CFs`
- `desired_class` or `desired_range`
- `features_to_vary`
- `permitted_range`
- `stopping_threshold`
- `posthoc_sparsity_param`

---

## 4. Example: Basic Counterfactual Generation with DiCE

Below is a typical pattern for generating CFs for a single individual.

    # Create a DiCE explainer (example)
    from dice_ml import Dice
    explainer = Dice(data_interface, model_interface, method="random")

    # Select an individual
    query = df.loc[[4]][feature_cols]

    # Generate CFs
    cf = explainer.generate_counterfactuals(
        query,
        total_CFs=3,
        desired_class="opposite",
        posthoc_sparsity_param=0.1
    )

    cf_df = cf.cf_examples_list[0].final_cfs_df

---

## 5. Global Sparsity: `posthoc_sparsity_param`

DiCE’s built‑in sparsity mechanism works post‑hoc:

- 0.0 → no sparsity
- 0.1 → mild sparsity (default)
- 0.5 → aggressive
- 1.0 → very aggressive

DiCE attempts to revert changed features to their original values while keeping the CF valid.

---

## 6. Indirect Per‑Feature Sparsity via Constraints

### 6.1 Restricting Which Features May Change

    cf = explainer.generate_counterfactuals(
        query,
        total_CFs=3,
        features_to_vary=["bmi", "smoking", "sleep"]
    )

This locks all other features.

### 6.2 Restricting How Much Features May Change

    permitted_range = {
        "bmi": [20, 35],
        "smoking": [3, 5],
        "alcohol": [6, 6],   # locked
        "sleep": [2, 5]
    }

    cf = explainer.generate_counterfactuals(
        query,
        total_CFs=3,
        permitted_range=permitted_range
    )

This indirectly enforces per‑feature sparsity.

---

## 7. Custom Post‑Hoc Sparsity (Most Flexible)

DiCE’s global sparsity is often not enough for real‑world recommendations.
Below is an example of a custom post‑hoc sparsity function that:

- prioritizes certain features
- removes low‑priority changes
- ensures the CF still meets the target

This is where psychological realism enters the pipeline.

    def apply_feature_sparsity(cf_row, original_row, rules, model, threshold):
        """
        rules: dict mapping feature -> priority level
            e.g. { "bmi": "high", "smoking": "medium", "alcohol": "low" }

        priority order:
            high   = keep changes
            medium = keep if needed
            low    = try to revert
            none   = always revert
        """
        priority = {"high": 1, "medium": 2, "low": 3, "none": 999}
        cf = cf_row.copy()

        # Sort features by priority
        ordered = sorted(rules.items(), key=lambda x: priority[x[1]])

        for feature, level in ordered:
            if level == "high":
                continue

            # Try reverting the feature
            original_value = original_row[feature].iloc[0]
            old_value = cf[feature]
            cf[feature] = original_value

            # Check if CF still meets the target
            pred = model.predict_proba(cf.values.reshape(1, -1))[0][1]
            if pred > threshold:
                # Reverting breaks the CF → restore the change
                cf[feature] = old_value

        return cf

Usage example:

    rules = {
        "bmi": "high",
        "smoking": "medium",
        "alcohol": "none",
        "sleep": "low"
    }

    cf_sparse = apply_feature_sparsity(cf_df.iloc[0], query, rules, model, threshold=0.2)

This gives full control over sparsity and aligns recommendations with human behavior.

---

## 8. Psychological Considerations

Effective recommendations must be:
- understandable
- realistic
- limited in number
- aligned with human behavior change research

Counterfactuals that modify many features at once are rarely actionable.
Sparse, focused recommendations are far more effective.

This is why sparsity tuning and post‑hoc processing are essential components of a real‑world CF system.

---

## 9. Summary

- Sparsity is essential for human‑centered counterfactual explanations.
- DiCE provides global sparsity and constraints but not per‑feature sparsity.
- Custom post‑hoc sparsity logic enables fine‑grained, psychologically realistic recommendations.
- Real‑world CF pipelines often combine:
  - DiCE generation
  - constraints
  - custom sparsity
  - domain knowledge
  - behavioral insights

This layered approach produces counterfactuals that are both technically valid and practically meaningful.
