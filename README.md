Multiple Disease Prediction System

A Machine Learning based web application built using Streamlit to predict three major diseases:

Chronic Kidney Disease (CKD)

Indian Liver Patient Disease

Parkinson’s Disease

This project includes data preprocessing, model training, model comparison, and deployment in a unified Streamlit dashboard.

🚀 Features
✔️ Kidney Disease Prediction

Uses cleaned & preprocessed dataset

Handles categorical encoding using OneHotEncoder

Scaling of numerical features using StandardScaler

ML Models Trained:

Random Forest (Best)

KNN

Logistic Regression

Best model saved as best_ckd_model.pkl

✔️ Indian Liver Patient Prediction

Uses 9 key features

StandardScaler + Gradient Boosting/Random Forest (depending on final selection)

Probability-based output with adjustable threshold slider

Model saved as liver_disease_model.pkl

✔️ Parkinson’s Disease Prediction

Uses 22 numerical features (after dropping name)

StandardScaler + Logistic Regression / Random Forest

ROC curve, accuracy, recall & precision analyzed

Model saved as parkinsons_model.pkl

🧪 Machine Learning Workflow
1️⃣ Data Cleaning

Replace missing values

Convert categorical values

Handle '?' entries (in kidney dataset)

Remove or encode non-numeric columns

Apply scaling on numerical features

2️⃣ Feature Engineering

OneHotEncoding for categorical features

StandardScaler for numeric features

Combine transformed features into final training matrix

3️⃣ Model Training

Each disease model used multiple algorithms:

Disease	Algorithms Tested	Best Model
CKD	Random Forest, KNN, Logistic Regression	RandomForestClassifier
Liver	Logistic Regression, RandomForest, XGBoost	RandomForest or LR (your result)
Parkinson’s	Logistic Regression, Decision Tree, Bagging, Random Forest	RandomForest / Logistic Regression

All best-performing models were saved using pickle.

4️⃣ Model Saving

Models are saved in .pkl format:

best_ckd_model.pkl  
liver_disease_model.pkl  
parkinsons_model.pkl  


Each file contains:

The trained ML model

Preprocessing scaler (if required)

🖥️ Streamlit Application
🎯 Key Features:

Sidebar navigation

Separate UI for each disease

Clean input forms

Automatic scaling & prediction

Probability-based decision (for liver model)

📌 Run the App:
streamlit run app.py

📂 Project Structure
MultipleDiseaseApp/
│── app.py
│── best_ckd_model.pkl
│── liver_disease_model.pkl
│── parkinsons_model.pkl
│── kidney_disease.csv
│── indian_liver_patient.csv
│── parkinsons.csv
│── README.md
│── ... (other notebooks)

📈 Results Summary
✔️ CKD

Accuracy: 100%

Best Model: Random Forest

✔️ Liver Disease

Accuracy: ~82–87%

Final Model: Random Forest / Logistic Regression

Provides probability & threshold-based prediction

✔️ Parkinson’s

Accuracy: 94–98%

ROC-AUC: ~0.94

🔧 Tech Stack
Languages

Python

Libraries

Pandas, NumPy

Scikit-learn

XGBoost

Matplotlib, Seaborn

Streamlit

💡 Future Enhancements

Deploy on Render / Railway / HuggingFace

Add Diabetes / Heart disease modules

Add visualizations inside Streamlit

Database integration for patient history
