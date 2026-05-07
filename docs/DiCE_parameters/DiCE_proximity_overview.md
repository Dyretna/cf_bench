# Proximity in Counterfactual Explanations and DiCE – A High-Level Overview

This note explains **proximity** in the context of counterfactual explanations, with a focus on how it is understood conceptually and how it fits into DiCE’s way of thinking about “good” counterfactuals. It is intentionally high-level and non-technical in terms of implementation details — those can be added later.

---

## 1. What Is Proximity?

In counterfactual explanations, **proximity** measures *how much* a counterfactual instance differs from the original instance in terms of feature values.

Intuitively:

- If a counterfactual is **close** to the original → high proximity (good, small changes).
- If a counterfactual is **far** from the original → low proximity (bad, large changes).

More precisely:

- Proximity is about the **magnitude of change** in the features.
- It answers the question:
  > “How big are the steps we are asking this person (or system) to take?”

Proximity is one of the core ingredients in defining whether a counterfactual is *reasonable*, *realistic*, and *actionable*.

---

## 2. Proximity vs. Sparsity

It is important to distinguish **proximity** from **sparsity**:

- **Sparsity** = *how many* features change.
  - “Do we change 1 feature or 7 features?”

- **Proximity** = *how much* each feature changes.
  - “Do we change BMI by 1 unit or 10 units?”

You can have:
- high sparsity but low proximity
  - e.g., only BMI changes, but from 31 → 20 (a huge jump)
- low sparsity but high proximity
  - e.g., 4 features change, but each only slightly

A good counterfactual usually aims for:
- **high sparsity** (few features change)
- **high proximity** (small, realistic changes)

---

## 3. Why Proximity Matters Conceptually

Proximity is crucial for several reasons:

### 3.1 Psychological Realism

If a counterfactual suggests very large changes, it may be:

- perceived as unrealistic
- emotionally overwhelming
- easy to dismiss (“this is not me”)

Small, incremental changes are:

- easier to accept
- easier to imagine
- more likely to be acted upon

Proximity is therefore directly tied to **behavioral plausibility**.

---

### 3.2 Domain Realism

In many domains (health, finance, education, etc.), large jumps in certain features are:

- physically impossible (e.g., age)
- biologically constrained (e.g., BMI in a short time)
- socially or economically unrealistic (e.g., income)

Proximity helps ensure that counterfactuals stay within **realistic bounds**.

---

### 3.3 Model Trust and Interpretability

If a model suggests that a person must change a feature by an extreme amount to get a better outcome, it can:

- reduce trust in the model
- raise questions about fairness
- highlight potential issues in the data or model

Proximity is therefore also a lens for **auditing** and **interpreting** models.

---

## 4. How Proximity Is Typically Measured (Conceptually)

Without going into implementation details, proximity is usually measured by some notion of **distance** between the original feature vector and the counterfactual feature vector.

Conceptually:

- Each feature’s change is measured (e.g., difference in value).
- These changes are aggregated into a single number.

Common ideas:
- Sum of absolute differences (how much each feature moved).
- Euclidean distance (geometric distance in feature space).
- Weighted distances (some features are more “costly” to change than others).

The key point:

> Proximity is a **continuous measure** of “how far” the counterfactual is from the original.

---

## 5. Proximity in the DiCE Framework (Conceptual View)

In DiCE, proximity is one of the core components of what makes a counterfactual “good”.

DiCE’s goals include:
- changing the model’s prediction in the desired direction
- keeping the counterfactual **close** to the original
- changing as few features as possible

Proximity in DiCE is part of the **optimization objective**:

- Counterfactuals that are closer to the original are preferred.
- Counterfactuals that require large jumps in feature values are penalized.

Even if we don’t look at the exact formulas, the idea is:

> DiCE tries to find counterfactuals that achieve the desired prediction with **minimal movement** in feature space.

---

## 6. Proximity and Human-Centered Recommendations

When counterfactuals are used to generate recommendations (e.g., “what can this person do to reduce their risk?”), proximity becomes a design choice:

- **High proximity** → small, incremental recommendations
  - “Reduce BMI slightly”, “sleep a bit more”, “reduce smoking one step”

- **Low proximity** → large, drastic recommendations
  - “Lose 20 BMI units”, “completely stop drinking immediately”

From a human-centered perspective:

- High proximity recommendations are more likely to be:
  - accepted
  - attempted
  - sustained

- Low proximity recommendations may:
  - feel impossible
  - trigger resistance
  - be ignored

Thus, proximity is not just a mathematical concept — it is a **behavioral design parameter**.

---

## 7. Proximity, Fairness, and Ethics

Proximity also intersects with fairness and ethics:

- If certain groups systematically require larger changes to achieve the same outcome, this may indicate:
  - bias in the model
  - structural inequalities in the data
  - unfair decision boundaries

Analyzing proximity across groups can reveal:
- who is “closer” to favorable outcomes
- who is “farther away” and why

This makes proximity a useful tool for:
- fairness analysis
- policy discussions
- model auditing

---

## 8. How Proximity Fits into the Bigger Picture

In a full counterfactual pipeline (like the one you are building), proximity is one of several key dimensions:

- **Prediction**: Does the CF achieve the desired outcome?
- **Proximity**: How big are the changes?
- **Sparsity**: How many features change?
- **Feasibility**: Are the changes realistic in the real world?
- **Psychological acceptability**: Will a person even consider these changes?

Proximity is the bridge between:
- the **mathematical** notion of distance in feature space, and
- the **human** notion of “how big a change is this in my life?”

---

## 9. Summary

- **Proximity** measures *how much* a counterfactual differs from the original instance.
- It is distinct from **sparsity**, which measures *how many* features change.
- High proximity (small changes) is usually desirable for:
  - psychological realism
  - domain realism
  - interpretability
  - fairness analysis
- In DiCE, proximity is a core part of the objective that defines “good” counterfactuals.
- In human-centered recommendation systems, proximity is not just a technical detail — it is a design choice that directly affects whether people can and will act on the suggestions.

Implementation details (distance metrics, norms, weighting, etc.) can be layered on top of this conceptual understanding in the next step.
