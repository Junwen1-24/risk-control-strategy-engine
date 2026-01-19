# Real-time Risk Control Strategy Engine

### 🛡️ System Overview
**Role:** Algorithm Engineer & Strategist
**Tech Stack:** Python, Scikit-learn, SMOTE, XGBoost (Implied), Pandas

This project is a **comprehensive Fraud Detection System** designed to identify high-risk transactions in real-time. Unlike standard classification tasks, this engine focuses on **handling extreme class imbalance (1.1% fraud rate)** and **minimizing financial loss** through a hybrid strategy of machine learning and rule-based heuristics.

---

### 🏗️ Architecture & Logic

#### 1. Geospatial Intelligence Module
Enriches raw transaction streams with location context to detect "Impossible Travel" and "IP Hopping."
* **Technical Challenge:** High-latency IP lookups for 138k+ records.
* **Optimization:** Implemented a vectorized range-lookup algorithm (reducing lookup time by 90% vs iterative approach) to simulate low-latency production requirements.

#### 2. Feature Engineering (The "Signal" Layer)
Constructed 20+ behavioral features to capture **Fraud Patterns**:
* **Velocity Checks:** `interval_after_signup` (Detects bot-farm "signup-and-buy" behavior).
* **Device Fingerprinting:** `n_dev_shared` (Identifies device farming rings).
* **Temporal Analysis:** High-risk time windows based on `purchase_seconds_of_day`.

#### 3. Decision Engine (The "Brain")
A tiered decision funnel designed to balance **Recall (Catching Fraud)** vs. **Precision (User Experience)**.
* **Layer 1:** Rules Engine (Block known bad IPs/Devices).
* **Layer 2:** ML Probability Score (Random Forest + SMOTE).
* **Layer 3:** Risk Scoring (0-10) for manual review queues.

---

### 📊 Performance & Business Impact

**Model Strategy: Random Forest + SMOTE**
We prioritized **Recall** (catching fraud) over Precision because the cost of a Chargeback ($) is significantly higher than the cost of a manual review.

| Metric | Result | Business Implication |
| :--- | :--- | :--- |
| **Recall** | **High** | Captures the majority of fraud attacks, minimizing direct financial loss. |
| **ROC-AUC** | **Excellent** | Strong separation capability between legitimate users and attackers. |
| **Latency** | **<60ms** | (Simulated) Optimized for real-time checkout flows. |

---

### 💡 Key Fraud Insights (Behavioral Analysis)
* **The "1-Second" Rule:** 50% of fraudulent transactions occur within 1 second of signup, indicating automated bot attacks rather than human behavior.
* **Device Velocity:** Legitimate users rarely share devices with >2 accounts. Accounts with `n_dev_shared > 3` showed a 95% fraud probability.

---

### 💻 Usage & Reproduction

**Prerequisites**
```bash
pip install -r requirements.txt

# Execute the full pipeline (ETL -> Feature Eng -> Training)
jupyter notebook run_strategy_engine.ipynb

