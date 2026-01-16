# Fraud Detection Strategy Engine

A comprehensive machine learning project for detecting fraudulent transactions using imbalanced data techniques and advanced feature engineering.

## Project Overview

This project implements a fraud detection system that identifies suspicious transactions using machine learning models. The dataset is highly imbalanced (98.9% legitimate, 1.1% fraudulent), requiring sophisticated handling techniques like SMOTE to improve model performance.

## Table of Contents

- [Main Objectives](#main-objectives)
- [Dataset Description](#dataset-description)
- [Project Structure](#project-structure)
- [Key Tasks](#key-tasks)
- [Feature Engineering](#feature-engineering)
- [Models & Results](#models--results)
- [Fraud Characteristics](#fraud-characteristics)
- [Usage Instructions](#usage-instructions)
- [Key Findings](#key-findings)

## Main Objectives

1. **Data Exploration**: Understand the characteristics of fraudulent vs legitimate transactions
2. **Feature Engineering**: Create predictive features from raw transaction data
3. **Model Development**: Build and train machine learning models optimized for fraud detection
4. **Parameter Tuning**: Use GridSearchCV to find optimal hyperparameters
5. **Actionable Insights**: Generate risk scores for business decision-making

## Dataset Description

### Data Source
- **Fraud Dataset**: `imbalancedFraudDF.csv` (138,376 records)
- **IP Mapping**: `IpAddress_to_Country.csv` (IP to country lookup table)

### Original Features
| Feature | Description |
|---------|-------------|
| user_id | Unique user identifier |
| signup_time | Account creation timestamp |
| purchase_time | Transaction timestamp |
| purchase_value | Transaction amount |
| device_id | Device identifier |
| source | Traffic source (SEO, Ads, etc.) |
| browser | Browser type (Chrome, Safari, Firefox, etc.) |
| sex | User gender (M/F) |
| age | User age |
| ip_address | IP address (numeric format) |
| class | Target variable (0=legitimate, 1=fraud) |

### Class Distribution
- **Legitimate Transactions**: 136,961 (98.9%)
- **Fraudulent Transactions**: 1,415 (1.1%)

## Project Structure

```
fraudDetectionCode_Py3.ipynb
├── Part 0: Summary
├── Part 1: Data Import & Loading
├── Part 2: Data Exploration
├── Task 1: IP-to-Country Mapping
├── Part 3a: Feature Engineering (Time Features)
├── Part 3b: Feature Engineering (Categorical & Aggregation)
├── Part 4: Data Split
├── Part 5: Model Training
│   ├── Logistic Regression
│   └── Random Forest
├── Part 6: Parameter Tuning (GridSearchCV)
├── Task 3: Fraud Characteristics Analysis
└── Task 4: Risk Scoring & Decision Rules
```

## Key Tasks

### Task 1: IP-to-Country Mapping
**Objective**: Enrich transaction data with geographic information

**Implementation**:
- Map IP addresses to countries using range-based lookup
- Match IP address to IP range boundaries in the mapping table
- Handle unmatched IPs with 'NA' value

**Performance**: ~62 seconds for 138K lookups (using optimizable loop approach)

**Question**: How can we optimize this lookup process?

### Part 3a & 3b: Feature Engineering

**Time-Based Features**:
- `interval_after_signup`: Seconds between account creation and purchase
- `signup_days_of_year`: Day of year account was created
- `signup_seconds_of_day`: Time of day account was created
- `purchase_days_of_year`: Day of year purchase occurred
- `purchase_seconds_of_day`: Time of day purchase occurred

**Categorical Encoding**:
- One-hot encoding for `source` and `browser`
- Binary encoding for `sex` (M=1, F=0)

**Aggregation Features** (created after train/test split):
- `n_dev_shared`: Count of transactions using same device
- `n_ip_shared`: Count of transactions from same IP
- `n_country_shared`: Count of transactions from same country

**Scaling**: Min-Max normalization (0-1) for aggregation features

## Models & Results

### Model 1: Logistic Regression (Baseline)
- **Framework**: sklearn LogisticRegression with GridSearchCV
- **Hyperparameters Tuned**: 
  - C (regularization strength): [0.01, 0.1, 1, 10, 100]
  - penalty: ['l1', 'l2']
- **Optimization Metric**: F1-score

### Model 2: Random Forest (Best Performance)
- **Framework**: sklearn RandomForestClassifier with GridSearchCV
- **Hyperparameters Tuned**:
  - max_depth: [None, 5, 15]
  - n_estimators: [10, 150]
  - class_weight: [{0: 1, 1: w} for w in [0.2, 1, 100]]
- **Training Method**: SMOTE resampling applied to training data only

### Model 3: Random Forest + SMOTE
- **Technique**: Synthetic Minority Over-sampling (SMOTE)
- **Rationale**: 
  - Handle severe class imbalance
  - Generate synthetic fraudulent samples for training
  - Applied AFTER train/test split to prevent data leakage
- **Results**: Improved recall and F1-score on minority (fraud) class

### Evaluation Metrics Used
- **Accuracy**: Overall correctness
- **Precision**: Of predicted frauds, how many are truly fraudulent
- **Recall**: Of actual frauds, how many did we catch
- **F1-Score**: Harmonic mean of precision and recall
- **ROC-AUC**: Area under the ROC curve
- **Confusion Matrix**: TP, TN, FP, FN breakdown

### Feature Importance (Top Features)
The best Random Forest model identifies these as most predictive:
1. `interval_after_signup` - Time from signup to purchase
2. Device/IP/Country sharing metrics
3. Time-of-day features
4. Traffic source

## Fraud Characteristics

### Key Findings

**1. Device Sharing Pattern**
- Fraudsters tend to reuse devices across multiple transactions
- Devices with higher sharing counts = higher fraud probability
- Legitimate users typically use unique devices

**2. Action Velocity (Most Distinctive)**
- **Fraudulent Average**: ~1-2 seconds after signup
- **Legitimate Average**: ~49 days after signup
- **Insight**: Fraud is characterized by immediate action (bot farm behavior)
- **Finding**: More than 50% of frauds occur within 1 second of signup

**3. IP Address & Country Patterns**
- Certain countries have different fraud rates
- IP reuse is a suspicious indicator
- Geographic consistency matters

**4. Temporal Patterns**
- Time of day and day of year matter
- Some times have higher fraud rates

## Usage Instructions

### Prerequisites
```bash
pip install pandas numpy scikit-learn imbalanced-learn matplotlib seaborn xgboost lightgbm
```

### Running the Notebook

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/loganlaioffer/fraudDetection.git
   ```

2. **Open Notebook**:
   - Open `fraudDetectionCode_Py3.ipynb` in Jupyter
   - **Important**: Click File → Save a Copy in your own drive first

3. **Run Sequentially**:
   - Execute cells from top to bottom
   - Notebook auto-installs required packages
   - Prepare for IP lookup (~62 seconds)

### Important Notes

- **Data Leakage Prevention**: SMOTE applied only to training data, AFTER train/test split
- **Scaler Fitting**: Scaler fitted on training data, then applied to test data
- **Feature Mapping**: Device/IP/Country mappings created separately for train and test due to low overlap

## Key Findings & Recommendations

### Risk Scoring System

Convert fraud probability to actionable risk scores (0-10):

```python
risk_score = int(10 * fraud_probability)
```

**Decision Rules**:
- **Green (1-3)**: Pass transaction - low fraud probability
- **Grey (4-7)**: Manual investigation - medium risk
- **Red (8-9)**: Decline transaction - high fraud probability

### Business Insights

1. **Bot Farm Detection**: Immediate purchases after signup are nearly always fraudulent
2. **Device Fingerprinting**: Monitor devices shared across many accounts
3. **Velocity Checks**: Implement time-based rules (e.g., reject if purchase < 10 seconds after signup)
4. **Geographic Monitoring**: Track transactions from unusual IP/country combinations

### Model Selection

**Choose based on business needs**:
- **High Recall Model**: Catch more fraud but accept more false positives → recall_score optimization
- **High Precision Model**: Minimize false positives but miss some fraud → precision optimization
- **Balanced Model**: F1-score optimization for balance

## Technical Considerations

### Why These Techniques?

| Challenge | Solution | Reason |
|-----------|----------|--------|
| Imbalanced Data | SMOTE | Over-sample minority class without data leakage |
| Feature Scaling | Min-Max Normalization | Required for Logistic Regression with L1/L2 regularization |
| High Cardinality | One-hot encoding + aggregation | Convert categorical features to numericals |
| Train/Test Leakage | Apply transformations after split | Prevent information leak to test set |

### Questions for Further Exploration

1. How can IP lookup be optimized (binary search, hash table)?
2. What's the optimal SMOTE ratio for this imbalanced dataset?
3. Should we use ensemble methods combining LR and RF?
4. How to handle geographic anomalies (e.g., impossible IP transitions)?
5. Can we incorporate user behavior sequences?

## Performance Summary

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|-------|----------|-----------|--------|----------|---------|
| Logistic Regression | Good | - | - | Optimized | - |
| Random Forest (Default) | Very Good | High | Moderate | Good | Excellent |
| Random Forest + SMOTE | Very Good | Moderate | **High** | Good | Excellent |



## References

- [Imbalanced-Learn Documentation](https://imbalanced-learn.org/)
- [Scikit-Learn GridSearchCV](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html)
- [SMOTE Algorithm Paper](https://arxiv.org/abs/1106.1813)


## License

Follow the terms of the original repository: https://github.com/loganlaioffer/fraudDetection

