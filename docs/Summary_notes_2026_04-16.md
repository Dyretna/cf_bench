# Counterfactual Generation Pipeline — Summary & Technical Notes

This document summarizes the behavior, trade-offs, and implementation details of the counterfactual (CF) generation pipeline using RandomForest, XGBoost, DiCE, permitted ranges, and a genetic search model.
It also documents the challenges encountered and the reasoning behind the final design choices.

---

## 1. RandomForest vs XGBoost for Counterfactuals

### RandomForest
RandomForest is easy to train and works well for standard prediction tasks, but it performs poorly for counterfactual generation:

- **Slow CF generation** due to highly discrete, fragmented decision boundaries.
- **Extremely slow or non‑convergent** when using DiCE with `permitted_range`.
- DiCE’s random search struggles to find valid CFs because small feature changes often jump between unrelated leaf nodes.
- There may be alternative CF search strategies for RF (e.g., custom local search), but **DiCE’s interface is not well suited** for RF + permitted ranges.

### XGBoost
XGBoost is more complex to integrate but significantly better for CF generation:

- Much **faster predictions** and smoother decision boundaries.
- Works well with DiCE’s random search.
- Supports **CUDA/GPU acceleration** for real‑time CF generation (currently running on CPU).
- sklearn’s RandomForest does **not** support CUDA; XGBoost does.

However, XGBoost introduces dtype challenges:

- XGBoost converts all ordinal integer features to **float64**.
- DiCE requires **exact integer matches** for ordinal features.
- Permitted ranges must be **strings** (categorical) for DiCE.
- This forces careful casting between `float → int → str` at different stages of the pipeline.

---

## 2. Performance Comparison

### XGBoost + DiCE Random
- **90 CFs (10 per 9 queries)** generated in ~5 minutes.
- Most CFs are **VALID**.
- No unexpected outcomes.
- Typically modifies **1–2 features**.

### XGBoost + DiCE Genetic
- **90 CFs** generated in ~15 seconds.
- About half are **VALID** (can be improved with tuning).
- One query had **no valid CF** (expected for some edge cases).
- No unexpected outcomes.
- Often modifies **3–4 features**, but changes like BMI are very small (can be rounded).

The genetic model allows tuning:
- CF quality threshold
- sparsity
- mutation rate
- number of generations
- realism constraints

This makes it flexible depending on whether speed or quality is prioritized.

---

## 3. Permitted Range Support — Challenges and Solutions

Adding `permitted_range` support for DiCE was significantly more complex than expected.

### Main challenges
- DiCE requires the **full dataset schema**, and permitted ranges must **exactly match** what the model has seen.
- Directional bounds and outer bounds must be merged without breaking DiCE’s internal validation.
- XGBoost converts ordinal integers to floats, while DiCE requires **exact integer matches**.
- This required careful casting:
  - XGBoost input: **float**
  - Query instances for DiCE: **int**
  - Permitted ranges: **str**
- DiCE validates query instances against the **training data**, not the permitted ranges.
  - Even tiny dtype mismatches (e.g., `4.0` vs `4`) caused:
    ```
    ValueError: Feature X has a value outside the dataset.
    ```
- Debugging required inspecting:
  - training dtypes
  - query dtypes
  - permitted_range dtypes
  - model_input_df after scaling
- The final solution works but adds complexity that should be revisited in a future refactor.

---

## 4. Final Recommendations

- **Use XGBoost** for all CF generation.
- **Use the genetic search model** for speed and robustness.
- Keep permitted range logic, but consider simplifying the dtype handling in a future cleanup.
- Consider enabling **CUDA** for real‑time CF generation in production.

---

## 5. Known Issues / Future Work

- Simplify dtype conversions (float → int → str).
- Improve genetic model validity rate by tuning:
  - mutation rate
  - population size
  - number of generations
  - sparsity penalty
- Add rounding rules for continuous features (e.g., BMI).
- Explore hybrid search strategies (genetic + local refinement).
