from flask import Flask, render_template, request
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

app = Flask(__name__)

# ... (Your existing code for data preprocessing and model training) ...

@app.route('/', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        temperature = float(request.form['temperature'])
        rainfall = float(request.form['rainfall'])
        humidity = float(request.form['humidity'])
        credit_history = request.form['credit_history']
        collateral = request.form['collateral']

        input_data = pd.DataFrame({
            'temperature': [temperature],
            'rainfall': [rainfall],
            'humidity': [humidity],
            'credit_history': [credit_history],
            'collateral': [collateral]
        })

        input_data = pd.get_dummies(input_data, columns=['credit_history', 'collateral'])

        missing_columns = set(merged_data.columns) - set(input_data.columns)
        for column in missing_columns:
            input_data[column] = 0

        input_data = input_data[merged_data.columns]

        X_input = input_data[features]

        prediction = model.predict(X_input)
        prediction_text = "Paid" if prediction[0] == 1 else "Unpaid"

        return render_template('index.html', prediction_text=prediction_text)

    return render_template('index.html')

if __name__ == '__main__':
    app.run(port=3000, debug=True)
