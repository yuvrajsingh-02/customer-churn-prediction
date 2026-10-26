# Customer Churn Prediction & Retention Recommendation System

## 1. Project overview
This project uses a supervised Machine Learning model to estimate whether a subscription customer may churn (leave the service). A Streamlit web app collects a customer profile, displays an estimated churn probability and risk category, and suggests example retention actions.

**Internship domain:** Machine Learning & Applied AI  
**Problem domain:** Customer analytics and retention  
**Model:** Random Forest classifier with preprocessing in a scikit-learn Pipeline

## 2. Features
- Customer profile input through a web interface
- Estimated churn probability and Low / Medium / High risk label
- Rule-based retention suggestions based on profile signals
- Dataset preview and basic churn-rate summary
- Model evaluation metrics saved to `models/metrics.json`

## 3. Project structure
```text
customer_churn_prediction_project/
├── app.py
├── train_model.py
├── requirements.txt
├── README.md
├── PROJECT_REPORT.md
├── model_evaluation.txt
├── data/
│   └── customer_churn.csv
└── models/
    ├── churn_pipeline.joblib
    └── metrics.json
```

## 4. Requirements
- Python 3.10 or newer recommended
- pip
- Internet is needed only to install the Python packages

## 5. Run locally
Open a terminal in this project folder and run:

```bash
python -m venv .venv
```

Windows:
```bash
.venv\\Scripts\\activate
```

macOS/Linux:
```bash
source .venv/bin/activate
```

Install packages:
```bash
pip install -r requirements.txt
```

Train the model:
```bash
python train_model.py
```

Launch the app:
```bash
streamlit run app.py
```

The browser will open the local Streamlit app. If the model is retrained, run the training script again.

## 6. Methodology
1. Load the demonstration dataset.
2. Separate customer attributes from the target column `churn`.
3. Split into training and test sets using stratification.
4. Impute missing values, standardize numeric fields, and one-hot encode categorical fields.
5. Train a Random Forest classifier.
6. Evaluate on a held-out test set using accuracy, precision, recall, F1-score, and ROC-AUC.
7. Use the trained pipeline to estimate risk for a profile entered in the app.

## 7. Important data limitation
The included 1,200-row dataset is **synthetically generated for demonstration and educational use**. It is not real customer data, and evaluation metrics must not be presented as proof of real-world accuracy. For a production system, replace it with a properly licensed, representative dataset; check class imbalance, leakage, calibration, subgroup performance, privacy, and business impact before deployment.

## 8. Future enhancements
- Train and validate using a public, documented telecom churn dataset
- Add model explainability (e.g. SHAP) and calibration
- Compare Logistic Regression, Random Forest, and Gradient Boosting
- Add privacy-aware login, feedback capture, and monitoring
- Evaluate whether retention actions actually reduce churn

## 9. Academic integrity
Understand the code and be ready to explain the dataset, target variable, train/test split, preprocessing, model choice, evaluation metrics, limitations, and the difference between a probability estimate and a certainty.
