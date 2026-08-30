# Semiconductor Fault Detection using Multivariate Statistical Monitoring & Machine Learning

**A real semiconductor manufacturing dataset (SECOM), analyzed two ways in parallel — classical multivariate
statistical process monitoring and modern machine learning — with a validation strategy comparison that
changes the reported outcome depending on how the train/test split is drawn.**

## 1. Overview

This project adopts a comparative evaluation framework rather than advocating for a single methodology. 
It applies two distinct approaches to the same semiconductor manufacturing sensor data: Statistical Process Monitoring, 
using PCA and Hotelling’s T², grounded in multivariate statistical methods, and Machine Learning, using Isolation Forest, 
Logistic Regression, Random Forest, and LDA. Both approaches are evaluated using a chronologically held-out test set, 
with the analysis focusing on the strengths, limitations, and practical applicability of each method rather than declaring 
a single universal winner.

The key finding is not simply which methodology achieves the highest performance. Rather, the analysis demonstrates that 
the choice of validation strategy can substantially influence reported model performance, potentially to a greater extent 
than the choice of model itself.

## 2. Business Problem

A semiconductor chip passes through hundreds of manufacturing steps, each monitored by sensors. At the end of
the line, it either passes or fails final quality testing and by the time it fails, the materials and
machine time that went into it are already spent. Conventional univariate monitoring (checking each sensor
against its own historical range) misses faults that only appear as an *unusual relationship between
sensors*, not any single sensor drifting out of range.

The project evaluates whether semiconductor manufacturing sensor data can support fault detection through complementary unsupervised, 
supervised, and multivariate statistical approaches. The results show that these methods provide different perspectives on 
potential failures, but their apparent effectiveness depends substantially on how performance is evaluated. In particular, 
the same Random Forest achieved 0.161 Average Precision under a random split compared with 0.067 under chronological validation, 
highlighting that validation strategy can materially affect the credibility of reported model performance.

## 3. Dataset

[SECOM](https://archive.ics.uci.edu/ml/datasets/SECOM) (UCI Machine Learning Repository): 1,567 real
production units, 590 anonymized sensor measurements each, 6.64% failure rate, spanning roughly three months
of real production with a timestamp per unit. Sensor identities are not disclosed — any "which sensor
matters" finding below is a statistical pointer, not a physical manufacturing diagnosis.

## 4. Project Structure

```
Data  →  Data Quality  →  Temporal Validation  →  Multivariate Statistical Process Monitoring
      →  ML Models  →  LDA  →  Evaluation  →  SHAP  →  Tableau Export
```

Data quality decisions (which sensors to drop, how to fill gaps) are made before the split is even chosen;
the chronological split is established before any model,statistical or ML , is fit; every downstream
evaluation uses that same held-out chronological test set, so every number in this report is comparable to
every other number.

## 5. Method Comparison

| Method | Type | Uses Labels? | Purpose | Evaluation Metric |
|---|---|---|---|---|
| PCA | Dimensionality reduction | No | Find the directions of largest sensor variation | Cumulative variance explained |
| Hotelling's T² | Process-monitoring statistic | No | Formally test if a unit deviates from the in-control model | Detection rate / false-alarm rate at control limit |
| Isolation Forest | Anomaly detector | No | Flag units unlike normal production, without needing failure examples | Average Precision |
| Logistic Regression | Supervised classifier | **Yes** | Learn historical failure patterns directly | Average Precision |
| Random Forest | Supervised classifier | **Yes** | Learn historical failure patterns directly | Average Precision |
| LDA | Supervised classifier (multivariate-normal-based) | **Yes** | Classify assuming shared-covariance multivariate normal classes | Average Precision |

**The distinction that matters most in this table isn't the method name — it's the "Uses Labels?" column.**
Isolation Forest and Hotelling's T² characterize *abnormality relative to normal, in-control behavior*; they
never see a failure example. Logistic Regression, Random Forest, and LDA learn *historical failure patterns
directly* — a different task, since they only need to recognize failure modes similar to ones they've
already seen labeled. Both are legitimate, complementary approaches to the same business problem.

## 6. Key Findings

### 6.1 Validation Strategy Changes the Reported Outcome 

The same Random Forest, same data, same tuned hyperparameters, evaluated two ways:

| Split Strategy | Random Forest Average Precision |
|---|---|
| Random 75/25 split | **0.161** |
| Chronological split (train on earliest 75%, test on most recent 25%) | **0.067** |

A random split **mixes observations from different production periods together** across the train/test
boundary it doesn't reproduce a realistic forward-in-time deployment, where a model can only ever be
trained on the past and evaluated on what comes after. (To be precise: the model is not literally trained on
the held-out test rows themselves under a random split — the issue is that random rows from *later*
production periods end up in training, and random rows from *earlier* periods end up in test, which a
genuine forward-deployed model would never experience.) SECOM includes a timestamp specifically because it's
real, time-ordered production data; most work on this dataset ignores that and splits randomly anyway. This
project doesn't, and the gap above a >2x difference in Average Precision from the *same model* — is the
concrete cost of that choice.

**Important Caveat:** Important caveat: This analysis compares one random train–test split with one chronological holdout split. 
While the results demonstrate that the choice of validation strategy can substantially affect reported model performance, 
the chronological Average Precision of 0.067 should not be interpreted as a definitive estimate of future production performance. 
A single chronological holdout may produce an optimistic or pessimistic estimate depending on the characteristics of the 
specific test period. Establishing a more robust estimate would require rolling-origin validation, involving multiple sequential 
chronological train–test splits with performance evaluated and aggregated across periods. Accordingly, this project concludes 
that chronological evaluation provides a materially different and more deployment-relevant assessment than random splitting, 
while recognizing that stronger evidence would require repeated time-based validation.

### 6.2 Among Supervised Models Tested, LDA Scored Highest

| Supervised Model | Average Precision (chronological test set) |
|---|---|
| LDA | **0.102** |
| Logistic Regression | 0.082 |
| Random Forest | 0.067 |

LDA achieved the highest predictive performance among the three supervised models evaluated in this analysis. This result is 
interpreted strictly as indicating that LDA provided the strongest benchmark among the models tested under the specified evaluation 
procedure. However, this performance does not establish that the underlying assumptions of LDA, including multivariate normality 
within classes and equality of covariance matrices across classes, are satisfied by the data. Verification of these assumptions 
would require separate diagnostic assessments, such as testing covariance homogeneity across groups. Therefore, no conclusions are 
drawn regarding the reasons for LDA's superior performance or the suitability of its assumptions; the finding is limited to its 
comparatively stronger performance among the three supervised methods evaluated.


### 6.3 Isolation Forest and Hotelling's T² Answer a Different Question Than Accuracy Can

The Isolation Forest model was trained using zero labeled failure observations, achieving an Average Precision of 0.072 with 
a 95% confidence interval of [0.044, 0.125]. This is particularly relevant in practical deployment settings, where labeled 
examples of all potential future failure modes are unlikely to be available in advance. Accordingly, Isolation Forest may 
provide a means of identifying previously unseen abnormal patterns. However, this does not imply that every future or previously 
unseen failure mode will necessarily be detected. No such guarantee is made.

### 6.4 Accuracy and a Default 0.5 Threshold Are Actively Misleading Here

At the default classification threshold of 0.5, the Random Forest model predicted no failures, resulting in 0% recall and 
0% precision for the failure class, despite the use of class_weight='balanced'. This result is retained explicitly because 
it demonstrates the limitations of accuracy in the presence of substantial class imbalance. With a failure rate of only 6.64%,
a model that consistently predicts the majority class ("pass") can achieve approximately 94% accuracy while providing 
no practical value for failure detection. This finding provides direct motivation for the threshold analysis presented 
subsequently.

## 7. Multivariate Statistical Process Monitoring

Multivariate Statistical Process Monitoring (SPC) is presented here not merely as supplementary coursework, but as an 
independent analytical approach to the same failure-detection problem. The methodology is developed and evaluated in 
accordance with established principles of statistical process monitoring, providing a complementary perspective to the 
supervised and unsupervised machine-learning approaches considered above.

### PCA: Dimensionality Reduction, Checked, Not Assumed

The number of components to retain (90% of variance) isn't asserted, it's checked with a scree plot, a
cumulative-variance plot, and a sensitivity comparison across 80%, 90%, and 95% variance retained:

| Variance Retained | Components | Control Limit | T² Average Precision |
|---|---|---|---|
| 80% | 84 | 116.50 | 0.058 |
| 90% | 124 | 172.61 | 0.062 |
| 95% | 157 | 222.03 | 0.062 |

T²'s detection ability is essentially stable from 90% to 95% (0.062 either way), with a small drop at 80%
(0.058) — the 90% cutoff used throughout this project wasn't a fragile decision; a meaningfully different
cutoff choice wouldn't have changed the reported result.

### Hotelling's T²: The Math, Stated Precisely

For a unit projected onto the retained principal components as scores $t_1, ..., t_k$ with eigenvalues
$\lambda_1, ..., \lambda_k$:

$$T^2 = \sum_{i=1}^{k} \frac{t_i^2}{\lambda_i}$$

This is a distance from the center of "normal" behavior that accounts for the fact that some directions of
variation are naturally wider than others — a unit far from center along a wide direction isn't unusual; a
unit even moderately far along a narrow direction is.

**Phase I (building the model):** the mean, the PCA components, and their eigenvalues are estimated using
only in-control (passing) units from the training period. The control limit comes from the F-distribution:

$$T^2_{limit} = \frac{k(n-1)}{n-k} F_\alpha(k, n-k)$$

At 90% variance (124 components): **control limit = 172.61**.

**Phase II (monitoring):** every test-set unit — which the Phase I model never saw — is projected onto that
same, fixed PCA basis and compared against that same, fixed control limit. Nothing re-estimates during
Phase II. That separation is what makes this genuinely Statistical Process Control, not just "PCA with a
threshold attached."

### T² Detection Performance at the Control Limit

| Metric | Value |
|---|---|
| True positives / False positives / True negatives / False negatives | 3 / 66 / 302 / 21 |
| Detection rate (recall) | **12.5%** |
| Precision | **4.3%** |
| False-alarm rate | **17.9%** |

In SPC terms: at this control limit, T² detected 12.5% of failures while flagging 17.9% of genuinely passing
units as false alarms — reported this way deliberately, since SPC's native language is detection rate and
false-alarm rate at an actual operating limit, not a ranking metric averaged across every possible threshold.

### What Drives the First Few Components

The top sensors by absolute loading are anonymized column identifiers, not physical measurements — this
identifies *which columns* move together most strongly, not what they physically represent:

| Component | Top contributing sensors (by absolute loading) |
|---|---|
| PC1 | sensor_343, sensor_335, sensor_332, sensor_336, sensor_207 |
| PC2 | sensor_27, sensor_25, sensor_26, sensor_299, sensor_164 |
| PC3 | sensor_283, sensor_152, sensor_287, sensor_148, sensor_421 |

## 8. Threshold Analysis

Random Forest's precision and recall computed across a full range of thresholds rather than accepting
scikit-learn's default of 0.5:

| Threshold | Precision | Recall | Flagged | TP | FP | FN |
|---|---|---|---|---|---|---|
| 0.05 | 0.071 | 87.5% | 294 | 21 | 273 | 3 |
| 0.10 | 0.066 | 25.0% | 91 | 6 | 85 | 18 |
| 0.15 | 0.045 | 4.2% | 22 | 1 | 21 | 23 |
| 0.20 and above | 0.000 | 0.0% | ≤3 | 0 | ≤3 | 24 |

The appropriate classification threshold depends on the relative costs of the two types of errors: 
**failing to identify an actual faulty unit** versus **inspecting a unit that is ultimately found to be non-faulty**. 
This represents a substantive operational trade-off for which the dataset does not provide sufficient cost 
information to determine a definitive optimal threshold.

Accordingly, it would be inappropriate to select a single threshold and present it as objectively optimal. 
At a low threshold of **0.05**, the model achieves relatively high recall (**87.5%**); however, 
it simultaneously generates a substantial number of false positives, resulting in more than three-quarters 
of inspected units being flagged unnecessarily. As the threshold increases, precision improves only marginally, 
while recall declines sharply toward zero.

Overall, the results indicate that the supervised signal provided by this Random Forest configuration is relatively weak. 
**No threshold within the evaluated range provides a clearly satisfactory operating point**, and therefore an artificial 
"optimal" threshold is not justified. This limitation is reported explicitly rather than imposing an unsupported 
threshold selection.


## 9. SHAP Explainability

SHAP explains individual Random Forest predictions: which sensors' values pushed a specific unit's prediction
toward "fail." **Correct interpretation, stated once:** this shows statistical association with the model's
output, not physical causation — and since sensor identities are anonymized, findings stay a statistical
pointer rather than a manufacturing root-cause diagnosis.

Top contributing sensors (mean absolute SHAP value): **sensor_59** dominates by a wide margin (0.078, roughly
9x the next highest), followed by sensor_460, sensor_103, sensor_33, sensor_341, sensor_65, sensor_562,
sensor_64, sensor_0, sensor_417 — all clustered much closer together (0.005–0.009).

## 10. Tableau Dashboard

A build plan and CSV exports (`tableau_exports/`) turn the key numbers above into a single, shareable
dashboard panel — the KPI summary, the random-vs-chronological comparison chart, failure rate over time, and
top SHAP features. 
See [`https://public.tableau.com/views/FaultDetection_v2026_1/FaultDetectionExecutiveDashboard?:language=en-US&publish=yes&:sid=&:redirect=auth&:display_count=n&:origin=viz_share_link`]
for the exact build steps and chart-to-data mapping.

## 11. Conclusion

Three things this project actually demonstrates, in order of importance:

1. **Validation strategy had a large effect on the reported outcome** — The impact of the train/test splitting 
strategy was substantially greater than the impact of model selection. 
The **Random Forest evaluated using a random split achieved an Average Precision (AP) of 0.161**, 
whereas the **same Random Forest evaluated using a chronological split achieved an AP of 0.067**. Thus, 
the underlying model and dataset remained unchanged; the substantial difference in performance resulted solely 
from how the training and testing boundary was defined.

However, this comparison is based on **one random split and one chronological split**, rather than repeated
or cross-validated evaluations. Therefore, the results should not be interpreted as a fully validated 
estimate of future production performance. Instead, they demonstrate that
**the choice of evaluation strategy can have a substantial effect on reported model performance**, an 
important consideration addressed further in the Limitations section.


2. **Statistical process monitoring and machine learning are complementary lenses, not competitors.**
PCA combined with Hotelling's T provides a formal framework for characterizing deviations from a 
historical in-control model, using an interpretable and derivable statistical control limit. 
**Isolation Forest** provides a complementary approach to anomaly detection without relying on the 
same distributional assumptions. In contrast, **Logistic Regression, Random Forest, and LDA** learn 
directly from historical examples of failure and therefore approach the problem from a supervised 
classification perspective.

None of these methods is considered universally superior in this analysis, as each addresses a different 
interpretation of the question, **"Does this unit exhibit characteristics associated with a problem?"** 
In a practical production environment, these approaches could potentially be used in combination as 
complementary monitoring mechanisms. However, this project does not directly evaluate such an integrated 
approach and therefore does not make a specific recommendation regarding a production deployment architecture.


3. **The available data supports exploratory quality monitoring, not a validated production early-warning system.**
The analysis is subject to several important limitations. **No cost data are available in the dataset**, preventing 
the calculation of actual business savings or the financial impact of improved failure detection. Furthermore, 
**sensor identities are anonymized**, meaning that explainability results can be interpreted only in statistical 
terms and cannot be directly linked to specific physical components or underlying engineering mechanisms. Finally,
**no lead-time analysis is included**, so the analysis does not quantify how far in advance a potential failure could 
realistically be detected.



## 12. Limitations

- No cost data (false-alarm cost, inspection cost, missed-failure cost) exists in this dataset, so no
  business-savings claim is made anywhere in this report.
- Sensor identities are anonymized — SHAP and PCA-loading findings identify *which column* mattered, never
  the physical process step it represents.
- Models are evaluated as configured in the notebook (Random Forest is hyperparameter-tuned via
  `TimeSeriesSplit`; the others are not independently tuned) — reported scores reflect these specific
  configurations, not an upper bound on what's achievable.
- No detection-lead-time analysis — this measures whether a unit's sensor pattern looks unusual or matches a
  known failure pattern, not how many production steps earlier a failure could have been flagged.
- **Only one chronological split is evaluated.** A single split can be unlucky given the small number of
  test-set failures (24). Rolling-origin validation (multiple sequential splits, evaluated and averaged)
  would turn "chronological evaluation gave a meaningfully different result" into a fully validated estimate
  of future-production performance — this is the single most valuable next addition and is not included here.
- LDA's underlying assumptions (multivariate normality, shared covariance across classes) are not tested
  directly — its top ranking among supervised models is reported as an empirical result, not evidence that
  its assumptions hold for this data.

## 13. Tech Stack

`pandas` `numpy` `scikit-learn` (PCA, Isolation Forest, Logistic Regression, Random Forest with
`TimeSeriesSplit` tuning, LDA, preprocessing) `scipy` (Hotelling's T² control limit) `shap` (explainability)
`matplotlib` `seaborn` · Tableau (dashboard, build plan in `dashboard/`)

## 14. Repo Structure

```
├── notebooks/
│   └── SECOM_Fault_Detection.ipynb
├── dashboard/
│   ├── dashboard_plan.md
│   └── dashboard_preview.png
├── README.md
└── requirements.txt
```

## 15. How to Run

```bash
pip install -r requirements.txt
jupyter notebook notebooks/SECOM_Fault_Detection.ipynb
```
