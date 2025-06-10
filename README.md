# Credit Risk Prediction with DenseNet-BC + Focal Loss

This project demonstrates a deep learning approach for credit risk prediction on imbalanced tabular data using:

- DenseNet-BC (1D) for efficient feature reuse
- Focal Loss to handle class imbalance
- SMOTE to balance the dataset
- Log transformation to reduce the impact of outliers
- Adam optimizer for stable training

# Problem Statement
Given a loan applicant's profile, predict whether they are likely to default (1) or fully repay (0) their loan.

# Techniques Used

| Step                        | Description                                                       |
| --------------------------- | ----------------------------------------------------------------- |
| Missing Values Handling     | Numerical: median imputation<br>Kategorical: most frequent (mode) |
| Outlier Handling            | Log transform on `person_income`, `loan_amnt`, `credit_score`     |
| Data Balancing*             | SMOTE (Synthetic Minority Over-sampling Technique)                |
| Data Split                  | 70% training / 30% testing, with stratified sampling              |
| Loss Function               | Focal Loss (gamma=2, alpha=0.25)                                  |
| Architecture                | DenseNet-BC with Conv1D layers adapted for tabular data           |
| Optimizer*                  | Adam with learning rate = 0.001                                   |
| Regularization              | EarlyStopping based on validation loss                            |


# Model Architecture

Input (features) → Conv1D → Dense Block 1 → Transition Layer → Dense Block 2
→ Global Average Pooling → Dense (ReLU) → Output (Sigmoid)

Dense Block : BatchNorm → ReLU → Conv1D(1) → Conv1D(3) → Concatenate
Transition Layer : BatchNorm → ReLU → Conv1D(1) → AvgPool1D


# Evaluation Metrics

Accuracy : \~90%
F1-Score : 0.90 (both classes)
ROC-AUC  : \~0.89
Confusion Matrix & Loss Plot : Included in notebook


# Files

`notebook.ipynb` – Jupyter Notebook with full pipeline
`loan_data.csv` – Sample input data (if permitted)
`README.md` – You are here


# How to Run

```bash
pip install -r requirements.txt
jupyter notebook
# or open notebook.ipynb in Colab
```


# Use Case

This project is relevant for:

1. Financial institutions assessing risk
2. Machine learning practitioners handling class imbalance
3. Anyone applying deep learning to tabular data


# License

This project is for educational purposes. Secondary dataset taken from kaggle.
