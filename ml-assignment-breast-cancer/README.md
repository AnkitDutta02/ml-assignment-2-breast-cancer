# Breast Cancer Diagnosis — ML Assignment 2

M.Tech (AIML), Machine Learning, BITS Pilani WILP

**Live app:** `https://ml-assignment-2-breast-cancer-sdbt2aeiehzpmodzjc5r5w.streamlit.app/`

**BITS Virtual Lab Execution Screenshot:** 

<img width="1912" height="1017" alt="image" src="https://github.com/user-attachments/assets/3563e72f-f66d-42b9-a283-19b8c6ab0ca4" />

<img width="1917" height="1007" alt="image" src="https://github.com/user-attachments/assets/39fbe087-6fa5-4afa-8f0d-64a50cc7e4ff" />

## a. Problem statement

Given 30 numeric measurements of cell nuclei taken from a digitised fine needle aspirate (FNA) image of a breast mass, predict whether the mass is **malignant (1)** or **benign (0)**. This is a binary classification problem. Five classifiers are trained on the same data and compared on Accuracy, AUC, Precision, Recall, F1 and MCC. The trained models are served through a Streamlit web app that evaluates them on uploaded test data without retraining.

Because a missed cancer (false negative) is far more costly than a false alarm, **Recall on the malignant class** and **MCC** (which is robust to class imbalance) are treated as the most important metrics alongside overall accuracy.

## b. Dataset description

| Item | Detail |
|---|---|
| Name | Breast Cancer Wisconsin (Diagnostic) |
| Source | UCI Machine Learning Repository — https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic (loaded via `sklearn.datasets.load_breast_cancer`) |
| Instances | 569 |
| Features | 30, all continuous (requirement: ≥ 12) |
| Target | `diagnosis` — 1 = malignant (212, 37.3%), 0 = benign (357, 62.7%) |
| Missing values | None |

The features come from ten properties of each cell nucleus — radius, texture, perimeter, area, smoothness, compactness, concavity, concave points, symmetry and fractal dimension. Each property is recorded three ways: the mean (`mean_*`), the standard error (`*_error`) and the "worst" or largest value (`worst_*`), giving 10 × 3 = 30 features.

Note: scikit-learn encodes 0 = malignant. The target was re-encoded so that **1 = malignant**, making malignant the positive class for Precision, Recall and F1.

**Preprocessing.** Stratified 80/20 train/test split (`random_state=42`): 455 training rows, 114 test rows (42 malignant, 72 benign). The test rows are saved as `test_data.csv`. Logistic Regression, KNN and SVM are wrapped in a `Pipeline` with `StandardScaler` fitted on training data only; the Decision Tree and Gaussian Naive Bayes use raw features since they are not sensitive to feature scale.

## c. GitHub repository link

`https://github.com/AnkitDutta02/ml-assignment-2-breast-cancer/tree/main/ml-assignment-breast-cancer`

```
project-folder/
├── app.py                                 # Streamlit app (loads saved models, no retraining)
├── ML_Assignment_2_Classification.ipynb   # EDA, training, evaluation, model saving
├── requirements.txt
├── README.md
├── test_data.csv                          # 114-row held-out test set (features + diagnosis)
├── breast_cancer_full.csv                 # full dataset used for training
├── .streamlit/config.toml                 # app theme
└── model/
    ├── logistic_regression.pkl
    ├── decision_tree.pkl
    ├── knn.pkl
    ├── naive_bayes.pkl
    ├── svm.pkl
    ├── metadata.json                      # feature list, target name, file map
    └── test_metrics.csv                   # metrics from the notebook run
```

Run locally:

```bash
pip install -r requirements.txt
jupyter notebook ML_Assignment_2_Classification.ipynb   # retrains and re-saves models
streamlit run app.py
```

## d. Models used

| Model | Configuration |
|---|---|
| Logistic Regression | StandardScaler + `LogisticRegression(max_iter=1000)` |
| Decision Tree | `DecisionTreeClassifier(max_depth=5)` |
| K-Nearest Neighbor | StandardScaler + `KNeighborsClassifier(n_neighbors=7)` |
| Naive Bayes | `GaussianNB()` |
| Support Vector Machine | StandardScaler + `SVC(kernel="rbf", probability=True)` |

### Comparison table (test set, n = 114)

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.9649 | **0.9960** | 0.9750 | **0.9286** | 0.9512 | 0.9245 |
| Decision Tree | 0.9211 | 0.9448 | 0.9459 | 0.8333 | 0.8861 | 0.8299 |
| KNN | 0.9561 | 0.9825 | 0.9744 | 0.9048 | 0.9383 | 0.9058 |
| Naive Bayes | 0.9386 | 0.9934 | **1.0000** | 0.8333 | 0.9091 | 0.8715 |
| Support Vector Machine | **0.9737** | 0.9947 | **1.0000** | **0.9286** | **0.9630** | **0.9442** |

Confusion matrix counts on the test set (TN / FP / FN / TP): Logistic Regression 71 / 1 / 3 / 39; Decision Tree 70 / 2 / 7 / 35; KNN 71 / 1 / 4 / 38; Naive Bayes 72 / 0 / 7 / 35; SVM 72 / 0 / 3 / 39.

5-fold cross-validated F1 on the training set (mean ± std): Logistic Regression 0.964 ± 0.021, Decision Tree 0.872 ± 0.057, KNN 0.958 ± 0.012, Naive Bayes 0.919 ± 0.036, SVM 0.961 ± 0.008.

### Observations

| ML Model Name | Observation about model performance |
|---|---|
| Logistic Regression | Very strong, near the top on every metric. It has the highest AUC (0.996), meaning its probability ranking of tumours is the best of all five, and it ties SVM on recall (3 missed malignant cases) but makes one false alarm. Its strong result shows the classes are close to linearly separable once features are standardised. It also has the best cross-validated F1 (0.964), so its performance is not a lucky split. |
| Decision Tree | Weakest model on every metric (accuracy 0.921, MCC 0.830, AUC 0.945). It reaches 98.7% training accuracy but only 92.1% on test data, a sign of overfitting even with `max_depth=5`. It misses 7 of 42 malignant cases. Axis-aligned splits approximate the smooth boundary between classes poorly, and its high CV standard deviation (0.057) shows it is also the least stable model. |
| KNN | Good overall (accuracy 0.956, MCC 0.906) once features are scaled, which is essential because `area` values are thousands of times larger than `smoothness`. It misses 4 malignant cases, slightly more than Logistic Regression and SVM, as some malignant tumours sit near benign ones in feature space. Its low CV variance (0.012) shows it is consistent. |
| Naive Bayes | Perfect precision (no false alarms) and a high AUC (0.993), but recall is only 0.833 — it misses 7 malignant cases, tied with the Decision Tree for most. The size features (radius, perimeter, area) are almost perfectly correlated, which violates the Naive Bayes independence assumption; the model effectively counts the same evidence several times and becomes over-confident. Its probability ranking is still good, so lowering the decision threshold could improve recall. |
| Support Vector Machine | Best overall: highest accuracy (0.974), F1 (0.963) and MCC (0.944), with perfect precision and the joint-best recall (0.929). It made only 3 errors, all missed malignant cases, and no false alarms. The RBF kernel captures the slightly non-linear boundary, and it has the lowest CV variance (0.008) of all models. |
| **Overall winner for this dataset** | **Support Vector Machine.** It leads on accuracy, F1 and MCC and makes the fewest errors (3 of 114). Logistic Regression is a very close second, with the best AUC and the best cross-validated F1, and would be a reasonable choice if interpretability (feature coefficients) matters. With only 114 test rows, one tumour changes accuracy by about 0.9%, so the gap between these two models is small. |

## Streamlit app features

- **CSV upload** of test data (sidebar), with a checkbox to use the bundled `test_data.csv` and a button to download it as a template. Uploads are validated for missing columns, missing labels and non-numeric values. Labels may be 0/1 or the original UCI B/M codes.
- **Model selection dropdown** for all five classifiers.
- **Evaluation metrics** — Accuracy, AUC, Precision, Recall, F1 and MCC for the selected model.
- **Confusion matrix and classification report** for the selected model, plus a downloadable predictions file with malignant probabilities.
- **Compare all models** tab showing every model's metrics on the uploaded data in one table and chart.

The app only loads the saved pipelines from `model/` with `joblib`; nothing is retrained on the uploaded data.
