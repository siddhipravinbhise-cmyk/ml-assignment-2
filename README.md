# Credit Card Default Classification & Interactive Dashboard

**Author:** Siddhi Pravin Bhise  
**BITS Student ID:** 2025ac05921  
**Course:** M.Tech (AIML / DSE) Machine Learning Assignment 2  
**Repository:** [https://github.com/siddhipravinbhise-cmyk/ml-assignment-2](https://github.com/siddhipravinbhise-cmyk/ml-assignment-2)  

---

## a. Problem Statement
Financial institutions face risk when extending credit cards to consumers. Predicting individual default probability minimizes non-performing assets (NPAs) and improves risk control. The target variable is binary: whether a client defaults on payment in the subsequent month (`1 = default`, `0 = non-default`).

## b. Dataset Description
* **Source:** UCI Machine Learning Repository (ID: 350) — *Default of Credit Card Clients Dataset*.
* **Instances:** 30,000 total records (24,000 training / 6,000 test split).
* **Features:** 23 predictive attributes including demographic factors, credit limit, and six-month historical repayment/billing histories.
* **Target:** `default.payment.next.month` (Binary: 0 or 1). Majority class baseline accuracy is **77.88%**.

## c. GitHub Repository Link
[https://github.com/siddhipravinbhise-cmyk/ml-assignment-2](https://github.com/siddhipravinbhise-cmyk/ml-assignment-2)

## d. Models Used & Comparison

### Performance Metrics Table
| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.8108 | 0.7260 | 0.7105 | 0.2442 | 0.3634 | 0.3382 |
| Decision Tree | 0.8222 | 0.7564 | 0.6720 | 0.3828 | 0.4878 | 0.4124 |
| k-NN | 0.8097 | 0.7378 | 0.6290 | 0.3399 | 0.4413 | 0.3620 |
| Naive Bayes | 0.7068 | 0.7377 | 0.4024 | 0.6707 | 0.5030 | 0.3336 |
| Random Forest | 0.8240 | 0.7798 | 0.6922 | 0.3677 | 0.4803 | 0.4141 |
| Gradient Boosting | 0.8227 | 0.7791 | 0.6789 | 0.3760 | 0.4840 | 0.4121 |

---

### Performance Observations
| ML Model Name | Observation about model performance |
|---|---|
| Logistic Regression | High overall precision (71.05%), but suffers from severe false negatives resulting in a low Recall (24.42%). |
| Decision Tree | Captures non-linear feature splits efficiently (`max_depth=6`), scoring a balanced MCC of 0.4124. |
| k-NN | Performs moderately well (Accuracy 80.97%), but distance metrics degrade due to noise in feature scales. |
| Naive Bayes | Exhibits the highest Recall (67.07%) by identifying default risk cases effectively, though with lower overall Precision. |
| Random Forest | Delivers the best overall performance with the highest Accuracy (82.40%), AUC (0.7798), and MCC (0.4141). |
| Gradient Boosting | Nearly matches Random Forest across all metrics with an AUC of 0.7791 and MCC of 0.4121. |
| **Overall Winner for your dataset?** | **Random Forest** is the overall winner. It yields the highest Accuracy (0.8240), AUC (0.7798), and MCC (0.4141), striking the best trade-off between bias and variance on credit default risk. |
