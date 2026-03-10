# INF2008 Group Project - Stage 2 (CRISPDM)

## Project Objective
The primary goal of Stage 2 is to move beyond baseline feasibility and focus on pipeline engineering, ablation studies, and decision making. Instead of brute-forcing high accuracy, the objective is to demonstrate robust data preparation, logical model selection, controlled experimentation, and a mechanical understanding of model failures. Ultimately, the project translates probabilistic or continuous outputs into actionable business policies.

---

## Scope and Strict Constraints
This project operates under strict engineering boundaries required by the assignment rubric.

### Allowed (Stage 2 Expansion)
* **Advanced Data Preparation**: Implementations of feature scaling, advanced encoding (e.g., target encoding), complex imputation, and data imbalance mitigation (e.g., SMOTE) are permitted.
* **Model Ensembling**: Classical ensemble methods like Random Forest, Gradient Boosting, and Model Stacking are allowed.
* **Hyperparameter Tuning**: Grid or random searches are permitted with a limit of roughly 3 parameters per model and 50 total iterations, provided they are justified.

### Strictly Prohibited (Stage 2 Restrictions)
* **Deep Learning / LLMs**: Language models or deep learning cannot be used as the primary predictive component, though they can be used for feature engineering.
* **Massive Automated Search**: Unjustified computational brute-forcing (e.g., 100+ grid points) is strictly prohibited.
* **Test Set Overuse**: The hold-out test set may only be evaluated *once* for final deployment metrics. All model selection and tuning must exclusively utilize cross-validation on the training set.

---

## Task Breakdown

* **Part A: Advanced Data Preparation & Pipeline Engineering**
  * Finalize CRISP-DM Phase 3 by encapsulating all data transformations and the estimator strictly within a formal pipeline object (e.g., `sklearn.pipeline.Pipeline`) to prevent data leakage.
* **Part B: Champion Model Selection**
  * Compare 2 to 3 distinct algorithmic families using k-fold cross-validation or a time-series split. 
  * Declare a "Champion" based on the mean and standard deviation of the primary evaluation metric.
* **Part C: Controlled Ablations & Tuning**
  * Perform a maximum of 4 controlled experiments exclusively on the Champion model. 
  * Maintain an Ablation Log detailing the hypothesis, controlled change, CV metric impact, and conclusion.
* **Part D: Mechanical Failure Analysis**
  * Extract 5 to 10 instances of confident False Positives/False Negatives (classification) or highest absolute errors (regression). 
  * Mechanically explain why the model failed based on feature values, and propose a technical fix.
* **Part E: Decision Making**
  * Logically evaluate the risks of the model's errors based on business context defined in Stage 1. 
  * Argue whether the operating threshold should be shifted (classification) or if a safety margin should be applied (regression).

---

## 📄 Deliverables
* **Deadline**: 3rd April 2026, 2359.
* **Format**: A single slide report in PDF format (max 12 slides).
* **File Naming**: `groupXX_labYY_stage2.pdf`.
