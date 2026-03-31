# 🏥 Patient Readmission Risk Analysis

## 📌 Project Overview
This project predicts whether a patient is likely to be readmitted within 30 days using healthcare data and machine learning.

## 🎯 Objective
- Identify high-risk patients
- Help hospitals reduce readmission rates
- Improve healthcare decision-making

## 🛠️ Tech Stack
- Python
- Pandas, NumPy
- Scikit-learn
- Matplotlib, Seaborn
- Streamlit

## 📂 Project Structure
patient-readmission-project/
│
├── data/
│   ├── raw/
│   └── processed/
│
├── notebooks/
│
├── src/
│   ├── data_cleaning.py
│   ├── train_model.py
│   ├── evaluate_model.py
│   └── app.py
│
├── models/
│
├── outputs/
│   ├── charts/
│   └── reports/
│
├── requirements.txt
└── README.md

## 🚀 How to Run
1. Activate environment  
   `.\.venv\Scripts\Activate.ps1`

2. Install requirements  
   `pip install -r requirements.txt`

3. Run dashboard  
   `streamlit run src/app.py`

## 📊 Expected Output
- Readmission prediction model
- Data visualizations
- Interactive dashboard

## 📌 Future Improvements
- Deploy on cloud
- Improve model accuracy
- Add real-time prediction