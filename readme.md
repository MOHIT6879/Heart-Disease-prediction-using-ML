❤️ Heart Disease Prediction using ML 
This project implements a machine learning pipeline to predict the presence of heart disease based on various medical parameters. It includes exploratory data analysis, preprocessing, training multiple models, evaluating performance metrics, and saving the trained models for reuse.

📊 Dataset
Source: dataset.csv

Target Variable: target (1 = disease present, 0 = no disease)

Features include:

Age
Sex
Chest pain type
Resting blood pressure
Cholesterol
Fasting blood sugar
Resting ECG results
Max heart rate achieved
Exercise-induced angina
oldpeak
ST slope

📁 Project Structure
Copy
Edit
├── dataset.csv
├── heart_disease_prediction.py
├── logistic_regression.pkl
├── random_forest.pkl
├── svm.pkl
├── xgboost.pkl
└── README.md

⚙️ Technologies Used
Language: Python

Libraries:

pandas, numpy, matplotlib, seaborn , matplotlib , sklearn

scikit-learn, xgboost, joblib 

🧪 Steps Performed
1. Data Loading & Exploration
Loaded dataset using pandas

Inspected data types, null values, basic statistics

Plotted:

Histograms for feature distributions

Correlation heatmap to identify multicollinearity

2. Data Preprocessing
Converted categorical variables into dummy/one-hot encoding using pd.get_dummies

Normalized features using StandardScaler

Split data into train/test sets (80/20)

3. Model Training
Trained and saved the following models:

✅ Logistic Regression

✅ Random Forest

✅ Support Vector Machine (with probability=True)

✅ XGBoost Classifier

Each model was saved using joblib for future inference.

4. Model Evaluation
Evaluated all models on the test set using:

Accuracy

Precision, Recall, F1 Score

ROC AUC Score

Confusion Matrix

ROC Curve

Each model's performance is visualized and printed clearly.

📈 Example Output

Logistic Regression Accuracy: 0.8689
✅ Saved model as logistic_regression.pkl

Random Forest Accuracy: 0.9016
✅ Saved model as random_forest.pkl
...
Each model’s performance includes plots like:

Confusion matrix heatmap

ROC curve with AUC score

🧠 Key Learnings
Encoding categorical variables improves model compatibility

Standardization is crucial for models like SVM and Logistic Regression

ROC AUC is a better performance metric than accuracy alone for imbalanced datasets

Model serialization using joblib enables easy reuse

📌 How to Run
Install dependencies:

pip install -r requirements.txt
Run the script:


python heart_disease_prediction.py
Models will be saved as:

logistic_regression.pkl

random_forest.pkl

svm.pkl

xgboost.pkl

