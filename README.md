# AI-Based Cardiovascular Signal Classification System

My graduation thesis project: a machine learning system that sorts heartbeats into normal and arrhythmia classes, using features extracted from two-lead ECG signals in the MIT-BIH Arrhythmia Database.

This is a research project and is not intended for clinical use.

## Dataset

The notebook uses the [ECG Arrhythmia Classification Dataset](https://www.kaggle.com/datasets/sadmansakib7/ecg-arrhythmia-classification-dataset) on Kaggle, which is derived from the [MIT-BIH Arrhythmia Database](https://physionet.org/content/mitdb/1.0.0/) on PhysioNet. It has 100,689 heartbeats, each with 32 features from leads II and V5: RR intervals, P, Q, R, S and T peaks, QRS, PQ, QT and ST intervals, and QRS morphology.

| Class | Meaning | Heartbeats |
|-------|---------|------------|
| N | Normal | 90,083 |
| VEB | Ventricular ectopic | 7,009 |
| SVEB | Supraventricular ectopic | 2,779 |
| F | Fusion | 803 |
| Q | Unknown (removed, too few examples) | 15 |

## Method

1. **Data preparation:** label encoding and a stratified 70/15/15 train, validation and test split.
2. **Class imbalance:** SMOTE, SMOTEENN and a targeted SMOTE strategy, applied only to the training data.
3. **Models:** XGBoost, LightGBM and ElasticNet logistic regression.
4. **Tuning and overfitting checks:** early stopping, threshold tuning, grid search, stratified 5-fold cross-validation, and comparing training and test scores.
5. **Evaluation:** classification reports, confusion matrices and one-vs-rest AUC.

## Results

Results on the test set of 15,102 heartbeats:

| Model | Accuracy | Macro F1 | Macro AUC | Weighted AUC |
|-------|----------|----------|-----------|--------------|
| LightGBM (targeted SMOTE, balanced class weights) | 0.99 | 0.93 | 0.9960 | 0.9984 |
| XGBoost (SMOTEENN, early stopping, threshold tuning) | 0.98 | 0.92 | 0.9951 | 0.9974 |
| ElasticNet logistic regression (SMOTE, grid search) | 0.91 | 0.64 | 0.9593 | 0.9589 |

The two tree-based models clearly outperform the linear model, especially on the rare F and SVEB classes.

## Running it

The notebook was written for Google Colab:

1. Download `MIT-BIH Arrhythmia Database.csv` from the Kaggle dataset. It is not included in this repository.
2. Rename it to `MITArrhythmiaDatabase.csv` and upload it next to `thesis.ipynb`.
3. Run the cells in order.

Outside Colab, install the dependencies first. The saved outputs were produced with scikit-learn 1.6.

```bash
pip install "scikit-learn==1.6.*" pandas numpy matplotlib seaborn imbalanced-learn xgboost lightgbm optuna catboost
```

## Notes

- Heartbeats are split at random, so beats from the same patient record can appear in both the training and test sets. These results may not reflect performance on new patients.
- The "F1 Macro Score by Model" chart near the end of the notebook shows cross-validation scores on oversampled training data for XGBoost (0.9926) and LightGBM (0.9667), but a test score for ElasticNet (0.64). The table above uses test scores for all three models, so it is the one to use for comparing them.
