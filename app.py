from flask import Flask, render_template, request, jsonify
from joblib import load
import pandas as pd

app = Flask(__name__)

# Load the trained model
model = load('fraud_detection_model.joblib')

# Define the feature names
features = ['Time', 'Amount', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9',
            'V10', 'V11', 'V12', 'V13', 'V14', 'V15', 'V16', 'V17', 'V18', 'V19',
            'V20', 'V21', 'V22', 'V23', 'V24', 'V25', 'V26', 'V27', 'V28']

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get input data from the request
        data = request.form.to_dict()
        
        # Create a DataFrame from the input data
        input_data = pd.DataFrame(data, index=[0], columns=features)
        
        # Make predictions
        prediction = model.predict(input_data)
        
        # Return the prediction as JSON
        return jsonify({'prediction': int(prediction[0])})
    
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
