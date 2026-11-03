# Financial Fraud Detection & Investigation System

An end-to-end machine learning project designed to identify suspicious financial transactions, evaluate fraud risk, explain model predictions, and support transaction-level investigation using data analytics and explainable AI.

## Project Overview

Financial fraud detection is a critical challenge for financial institutions due to the high volume of transactions and the highly imbalanced nature of fraudulent activity.

This project develops a machine learning-based fraud detection and investigation system using the PaySim synthetic financial transaction dataset. It combines exploratory data analysis, statistical profiling, predictive modelling, SHAP explainability, classification threshold optimisation, and transaction-level investigation.

The objective is to identify potential fraudulent transactions while improving alert precision and supporting informed risk assessment.

## Key Highlights

- Analysed **6.36 million+ financial transactions** to identify fraud patterns and risk indicators.
- Developed Logistic Regression and Random Forest classification models.
- Achieved **0.996 ROC-AUC** and **0.693 Average Precision** with Random Forest using chronological validation.
- Applied SHAP explainability to understand the contribution of transaction features to fraud predictions.
- Optimised classification thresholds, achieving **98.7% precision** and reducing flagged alerts by **81.8%** during evaluation.
- Conducted transaction-level investigation and network-based activity analysis.

## Dataset

**Source:** [PaySim – Synthetic Financial Dataset](https://www.kaggle.com/datasets/ealaxi/paysim1)

The dataset simulates mobile money transactions and contains:

- **6,362,620** transaction records
- **11** original features
- **8,213** fraudulent transactions
- **6,354,407** legitimate transactions

### Key Features

| Feature | Description |
|---|---|
| step | Transaction time step |
| type | Transaction type |
| amount | Transaction amount |
| oldbalanceOrg | Sender's balance before transaction |
| newbalanceOrig | Sender's balance after transaction |
| oldbalanceDest | Recipient's balance before transaction |
| newbalanceDest | Recipient's balance after transaction |
| isFraud | Fraud transaction indicator |

The dataset is highly imbalanced, making fraud detection a challenging classification problem.

## Project Workflow

### 1. Data Cleaning & Exploratory Data Analysis
- Examined dataset structure, data types, missing values, and duplicate records.
- Analysed fraudulent and legitimate transaction distributions.
- Investigated transaction types, amounts, and balance patterns.
- Identified fraud concentration across transaction categories.

### 2. Feature Engineering & Data Preparation
- Selected relevant transaction-level predictors.
- Encoded categorical transaction types.
- Excluded post-transaction information to reduce data leakage.
- Prepared chronological training and testing datasets.

### 3. Predictive Modelling
Developed and evaluated two classification algorithms:

- Logistic Regression
- Random Forest Classifier

Model performance was evaluated using ROC-AUC and Average Precision, considering the highly imbalanced target variable.

### 4. Model Explainability
Applied SHAP (SHapley Additive exPlanations) to interpret model predictions and identify influential transaction characteristics.

Key predictive factors included:
- Sender's previous balance
- Transaction amount
- Recipient's previous balance
- Transaction type
- Transaction time step

### 5. Classification Threshold Optimisation
Optimised the classification threshold to improve fraud alert precision and reduce unnecessary alerts.

The selected threshold was calibrated separately from the final evaluation period.

| Metric | Default Threshold (0.50) | Optimised Threshold (0.80) |
|---|---:|---:|
| Precision | 35.49% | 98.70% |
| Recall | 89.81% | 45.53% |
| F1-Score | 0.5088 | 0.6231 |
| Flagged Alerts | 5,486 | 1,000 |

**Key finding:** Threshold optimisation reduced flagged alerts by 81.8% while increasing precision to 98.7%. This involved a trade-off in recall, which decreased to 45.53%.

### 6. Transaction Investigation & Network Analysis
- Developed transaction-level fraud screening.
- Examined suspicious transactions using predicted fraud probabilities.
- Analysed transaction relationships and counterparty activity through network analysis.
- Generated investigation-oriented insights to support risk assessment.

## Model Performance

| Model | ROC-AUC | Average Precision |
|---|---:|---:|
| Logistic Regression | 0.97509 | 0.32758 |
| Random Forest | 0.99589 | 0.69266 |

Random Forest demonstrated stronger performance across both evaluation metrics.

## Technology Stack

| Category | Technologies |
|---|---|
| Programming | Python |
| Data Analysis | Pandas, NumPy |
| Visualisation | Matplotlib, Seaborn, Plotly |
| Machine Learning | Scikit-learn |
| Model Explainability | SHAP |
| Network Analysis | NetworkX |
| Application Development | Streamlit |
| Database | SQLite |

## Project Structure

```text
AI-Powered-Financial-Fraud-Detection/
│
├── data/
│   └── PS_20174392719_1491204439457_log.csv
│
├── outputs/
│   ├── threshold_optimization.csv
│   ├── threshold_evaluation_comparison.csv
│   ├── threshold_performance.png
│   ├── threshold_evaluation_confusion_matrix.png
│   └── fraud_model_config.json
│
├── main.py
├── app.py
├── requirements_app.txt
└── README.md
```

*Note: Add the remaining output files generated by your analysis to the repository as applicable. The dataset is not included in this repository.*

## Installation & Setup

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/AI-Powered-Financial-Fraud-Detection.git
```

### 2. Navigate to the Project Directory

```bash
cd AI-Powered-Financial-Fraud-Detection
```

### 3. Create a Virtual Environment

```bash
python -m venv venv
```

Activate it on Windows:

```bash
venv\Scripts\activate
```

### 4. Install Dependencies

```bash
pip install -r requirements_app.txt
```

### 5. Add the Dataset

Download the PaySim dataset from Kaggle and place the CSV file inside the `data/` directory.

### 6. Run the Analysis

```bash
python main.py
```

### 7. Launch the Streamlit Application

```bash
streamlit run app.py
```

## Key Business Applications

The analytical approach demonstrated in this project can support:

- Financial transaction risk assessment
- Fraud pattern identification
- Suspicious transaction prioritisation
- Fraud alert optimisation
- Explainable risk-based decision-making
- Transaction monitoring and investigation workflows

## Limitations

- The project uses a synthetic dataset rather than real-world banking transactions.
- Model performance on synthetic data may not generalise to actual financial institutions.
- Threshold optimisation involves a precision-recall trade-off.
- Network analysis is based on a 50,000-transaction sample and is intended for activity exploration, not independent fraud classification.
- The application is a prototype and has not been deployed in a production banking environment.

## Future Enhancements

- Integrate real-time transaction monitoring.
- Explore advanced anomaly detection techniques.
- Develop graph-based fraud detection models.
- Integrate LLM-powered investigation summaries.
- Evaluate model performance on real-world financial datasets.

## Author

**Prerna Kumari**  
M.Sc. Analytics | Data Analytics & Machine Learning
