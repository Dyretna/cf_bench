# Implementing Proximity in Counterfactual Explanations with DiCE

This note focuses on **how proximity is implemented and controlled in practice** when using DiCE for counterfactual explanations. It builds on the conceptual understanding of proximity and now moves into more concrete, implementation-oriented thinking — still at a reasonably high level, but with examples and patterns you can actually use.

---

## 1. Recap: What Proximity Means in Practice

In implementation terms, **proximity** is about:

- how far the counterfactual feature vector is from the original feature vector
- measured via some **distance function** in feature space

In DiCE, proximity is not usually exposed as a single parameter called `proximity`, but it is embedded in:

- the **loss function** used during optimization
- the **distance metric** used to compare original vs. CF
- the **constraints** you impose (e.g., permitted ranges, feature scaling)

So when we talk about “implementing proximity” in DiCE, we are really talking about:

> How do we influence and control the distance between original and counterfactuals?

---

## 2. Where Proximity Lives in DiCE

Even if DiCE doesn’t expose a `proximity_param`, proximity is present in several places:

1. **Internal loss function**
   - DiCE penalizes large deviations from the original instance.
   - This is typically done via an L1 or L2 distance term.

2. **Feature scaling / normalization**
   - Distance is computed in the transformed feature space.
   - Scaling affects how “big” a change is perceived.

3. **Constraints (`permitted_range`)**
   - You can limit how far a feature is allowed to move.
   - This directly bounds proximity.

4. **Post-hoc processing**
   - You can manually adjust CFs after generation to reduce large jumps.

You don’t always see the distance formula directly, but you can **shape** it via these levers.

---

## 3. Practical Levers for Controlling Proximity in DiCE

### 3.1 Feature Scaling and Normalization

Distance is computed in the feature space that DiCE sees.
If features are on very different scales, proximity becomes distorted.

**Example:**

- BMI ranges from 18–40
- Income ranges from 10,000–1,000,000

If you don’t scale, a small relative change in income can dominate the distance, even if BMI changes a lot.

**Practical implication:**

- Use consistent scaling (e.g., standardization or min-max scaling) before building the DiCE data interface.
- This makes proximity more meaningful and balanced across features.

---

### 3.2 Using `permitted_range` to Bound Proximity

`permitted_range` is one of the most direct ways to control proximity per feature.

Conceptually:

- You tell DiCE:
  > “You may move this feature, but only within this interval.”

Example idea:

- BMI: `[24, 32]`
- Smoking: `[3, 5]`
- Alcohol: `[6, 7]`
- Sleep: `[3, 6]`

This ensures that:

- CFs cannot propose extreme changes.
- Proximity is bounded by design.

Even though this is not “proximity” in the loss function, it **shapes the feasible region** in which proximity is computed.

---

### 3.3 Tuning the Optimization Behavior (High-Level)

DiCE’s optimization (especially in gradient-based methods) typically includes:

- a term for prediction loss (how close we are to the desired outcome)
- a term for distance (how far we moved from the original)

While you may not always directly set the weight of the distance term, you can:

- choose methods that emphasize proximity more strongly
- adjust stopping criteria (e.g., `stopping_threshold`)
- combine DiCE with your own post-hoc logic

The key idea:

> You can’t always see the proximity term, but you can influence how hard DiCE tries to stay close.

---

## 4. Example: Using `permitted_range` to Control Proximity

Below is a conceptual example of how you might use `permitted_range` to keep proximity under control in a health risk setting.

Imagine you have features:

- `bmi`
- `smoking`
- `alcohol`
- `sleep`

You want to avoid extreme changes, so you define:

- BMI: allow moderate changes
- Smoking: allow small changes
- Alcohol: almost fixed
- Sleep: allow small improvements

Conceptually:

    permitted_range = {
        "bmi": [24, 32],       # moderate movement allowed
        "smoking": [3, 5],     # small movement
        "alcohol": [6, 6],     # fixed (no movement)
        "sleep": [3, 6],       # small to moderate movement
    }

Then you call DiCE:

    cf = explainer.generate_counterfactuals(
        query_instance,
        total_CFs=3,
        desired_class="opposite",
        permitted_range=permitted_range,
        # proximity is indirectly controlled by these ranges
    )

Effect:

- Proximity is implicitly bounded.
- CFs cannot propose “lose 20 BMI units” or “go from 2 to 10 hours of sleep”.
- You get counterfactuals that are closer to the original in a domain-aware way.

---

## 5. Example: Proximity and Feature Scaling

Suppose you have:

- `bmi` (18–40)
- `age` (18–90)
- `income` (10,000–1,000,000)

If you don’t scale:

- A change in income from 50,000 → 60,000 (10,000 difference) will dominate the distance.
- A change in BMI from 30 → 25 (5 units) will look “small” numerically, even if it’s a big real-world change.

In practice:

- You define a `data_interface` in DiCE with scaled features.
- You ensure that all features are on comparable scales (e.g., 0–1 or standardized).

This makes the proximity term more meaningful:

- A unit of change in BMI is comparable to a unit of change in other features.
- Distance reflects real-world “effort” more closely.

---

## 6. Post-Hoc Proximity Adjustments (Conceptual)

Even after DiCE generates CFs, you might find:

- some CFs have large jumps in certain features
- some CFs are technically valid but psychologically unrealistic

You can then implement **post-hoc proximity adjustments**, for example:

1. For each CF:
   - compute the distance from the original
   - identify features with large changes

2. Apply rules like:
   - cap maximum allowed change per feature
   - discard CFs with total distance above a threshold
   - prefer CFs with smaller distances when ranking

This is not built into DiCE directly, but it is a powerful pattern:

> Use DiCE to generate candidates → use your own logic to enforce proximity constraints that reflect your domain and psychological design.

---

## 7. Proximity and Ranking of Counterfactuals

Even if multiple CFs satisfy the desired prediction, they can differ in proximity.

You can:

- compute a distance score for each CF (e.g., L1 or L2 distance in feature space)
- rank CFs by:
  - lowest distance
  - or a combination of distance and risk reduction

This gives you:

- a way to present the “closest” CFs first
- a way to filter out extreme CFs
- a way to align with human-centered design (smallest changes first)

---

## 8. Proximity as a Design Parameter, Not Just a Metric

The most important implementation insight is this:

> Proximity is not just something you measure — it is something you **design for**.

You design for proximity by:

- scaling features appropriately
- constraining allowed ranges (`permitted_range`)
- choosing DiCE methods that respect distance
- adding post-hoc rules to filter or adjust CFs
- ranking CFs by distance and realism

In other words:

- Proximity is both a **technical construct** (distance in feature space) and a **product decision** (how big changes you are willing to suggest to a user).

---

## 9. Summary

- Proximity in DiCE is implemented through:
  - internal distance terms in the optimization
  - feature scaling
  - constraints like `permitted_range`
  - optional post-hoc filtering and ranking

- You don’t always see a `proximity` parameter, but you can **shape proximity** via:
  - how you prepare data
  - how you configure DiCE
  - how you post-process CFs

- Good proximity behavior is essential for:
  - realistic counterfactuals
