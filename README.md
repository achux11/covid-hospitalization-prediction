# 📊 COVID-19 Hospitalization Prediction Using XGBoost

> Predicting US COVID-19 hospitalizations with **97.5% accuracy** (R² = 0.975) using machine learning on 1.7 million real-world records.

![Python](https://img.shields.io/badge/Python-3.11-blue?style=for-the-badge&logo=python)
![XGBoost](https://img.shields.io/badge/XGBoost-Regressor-orange?style=for-the-badge)
![ScikitLearn](https://img.shields.io/badge/Scikit--Learn-Metrics-green?style=for-the-badge&logo=scikit-learn)
![Pandas](https://img.shields.io/badge/Pandas-Data_Processing-lightblue?style=for-the-badge&logo=pandas)
![Colab](https://img.shields.io/badge/Google_Colab-Notebook-yellow?style=for-the-badge&logo=googlecolab)

---

## 🎯 Problem Statement

During the COVID-19 pandemic, hospitals faced unexpected surges that led to critical shortages of beds, ventilators, and staff.

**The goal:** Build a machine learning model that accurately predicts the number of hospitalized patients — giving healthcare systems advance warning to prepare resources before a surge hits.

---

## 🔄 Pipeline


---

## 🛠️ Tech Stack

| Layer | Technology | Purpose |
|---|---|---|
| Language | Python 3.11 | End to end development |
| Data Processing | Pandas, NumPy | Cleaning, transformation, feature engineering |
| ML Model | XGBoost | Gradient boosted regression |
| Evaluation | Scikit-learn | MAE, RMSE, R² Score |
| Visualization | Matplotlib, Seaborn | Charts, trends, feature importance |
| Environment | Google Colab | Cloud notebook execution |

---

## 🧠 Feature Engineering

The key insight — COVID hospitalization is a time series problem. Yesterday's count heavily predicts today's count.

| Feature | Type | Why it matters |
|---|---|---|
| lag_1 | Lag | Hospitalizations 1 day ago — strongest predictor |
| lag_7 | Lag | Hospitalizations 7 days ago — weekly pattern |
| lag_14 | Lag | Hospitalizations 14 days ago — incubation cycle |
| rolling_7 | Rolling avg | 7-day trend smoothing |
| rolling_14 | Rolling avg | 14-day trend smoothing |
| current_intensive_care_patients | Clinical | ICU load — leading indicator |
| current_ventilator_patients | Clinical | Severity indicator |
| day_of_week | Time | Weekly admission patterns |
| month | Time | Seasonal wave patterns |
| year | Time | Pandemic phase context |

---

## 📈 Model Performance

| Metric | Value | What it means |
|---|---|---|
| R² Score | **0.975** | Model explains 97.5% of variance |
| MAE | 3,750 patients | Average prediction error |
| RMSE | 5,031 patients | Penalizes large errors |

> An R² of 0.975 means the model captures nearly all the variation in hospitalization numbers — a strong result for real-world healthcare data.

---

## 🔑 Feature Importance

| Rank | Feature | Importance Score | Insight |
|---|---|---|---|
| 🥇 1 | lag_1 | 37.3% | Yesterday drives today |
| 🥈 2 | rolling_7 | 30.9% | Weekly trend is critical |
| 🥉 3 | ICU patients | 30.3% | Clinical severity predicts hospitalizations |
| 4 | Everything else | less than 1% | Minimal contribution |

**Key finding:** The top 3 features alone account for 98.5% of predictive power. Time-based lag features and ICU load are the strongest signals.

---

## 📊 Dataset

| Property | Detail |
|---|---|
| Source | Google Health COVID-19 Open Data |
| Total rows | 1,768,485 |
| Columns | 11 |
| Date range | 2020 — 2022 |
| Filtered to | US national level data |
| Final training rows | 279 (after cleaning and feature engineering) |
| Train size | 279 rows (80%) |
| Test size | 70 rows (20%) |

---

## 💡 Key Insights

- **Yesterday matters most** — lag_1 at 37.3% importance confirms that COVID spreads with a 1-day delay in hospitalization effect
- **ICU load is a leading indicator** — as ICU patients rise, general hospitalizations follow within days
- **Rolling averages smooth noise** — 7-day rolling average captures the true trend better than daily spikes
- **Seasonal patterns exist** — month and day_of_week features showed weekly admission cycles

---

## 🔮 Future Improvements

- Add LSTM or Prophet model for deeper time series modeling
- Expand to state-level and country-level predictions
- Build a real-time prediction dashboard with Streamlit
- Add early warning alert system for surge detection
- Integrate live data pipeline from public health APIs

---

## 🗂️ Project Structure
---

## 👤 Author

**Achintya Vamshi Nudurupati**
Master of Science in Business Analytics and AI — University of Texas at Dallas

GitHub: [@achux11](https://github.com/achux11)

---

*Built using real-world COVID-19 data from Google Health Open Data. This project demonstrates end-to-end machine learning — from raw data cleaning to model evaluation and visualization.*