from flask import Flask, render_template, request
import numpy as np
import pickle

app = Flask(__name__)

# Load model
model = pickle.load(open("random_forest_model.pkl", "rb"))

@app.route('/')
def home():
    return render_template('index_Churn.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.form

    # Extract inputs
    CreditScore = int(data['CreditScore'])
    Age = int(data['Age'])
    Tenure = int(data['Tenure'])
    Balance = float(data['Balance'])
    NumOfProducts = int(data['NumOfProducts'])
    HasCrCard = int(data['HasCrCard'])
    IsActiveMember = int(data['IsActiveMember'])
    EstimatedSalary = float(data['EstimatedSalary'])

    Geography_Germany = 1 if data['Geography'] == 'Germany' else 0
    Geography_Spain = 1 if data['Geography'] == 'Spain' else 0
    Gender_Male = 1 if data['Gender'] == 'Male' else 0

    # Prepare features
    features = np.array([[CreditScore, Age, Tenure, Balance, NumOfProducts,
                          HasCrCard, IsActiveMember, EstimatedSalary,
                          Geography_Germany, Geography_Spain, Gender_Male]])

    # Predict
    prediction = model.predict(features)[0]
    result = "Client will EXIT " if prediction == 1 else "Client will STAY "

    return render_template('Index_Churn.html', prediction_text=result)

if __name__ == '__main__':
    app.run(debug=True)


