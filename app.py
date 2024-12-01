import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
import math

app = Flask(__name__)

def sigmoid(x):
    return 1 / (1 + math.exp(-x))

# Custom unpickler to handle outdated module references
class CustomUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        # Handle outdated paths for scaler
        if module == 'sklearn.preprocessing.data':
            module = 'sklearn.preprocessing._data'
        # Handle outdated paths for LinearSVM
        elif module == 'sklearn.svm.classes':
            module = 'sklearn.svm._classes'
        # Handle outdated paths for LogisticRegression
        elif module == 'sklearn.linear_model.logistic':
            module = 'sklearn.linear_model._logistic'
        return super().find_class(module, name)

# Load the scaler
with open("scaler.bin", "rb") as f:
    scaler = CustomUnpickler(f).load()

# Load LinearSVM model
with open("LinearSVM.bin", "rb") as f:
    model1 = CustomUnpickler(f).load()

# Load LogisticRegression model
with open("LogisticRegression.bin", "rb") as f:
    model2 = CustomUnpickler(f).load()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    try:
        # Collect features from the user input
        int_features = [[float(x) for x in request.form.values()]]
        
        # Scale the features using the scaler
        final_features = scaler.transform(int_features)
        
        # Model 1 (LinearSVM): Use the sigmoid of the decision function
        out1 = sigmoid(model1.decision_function(final_features))
        
        # Model 2 (LogisticRegression): Use the probability of the positive class
        out2 = model2.predict_proba(final_features)[:, 1]  # Only the positive class probability
        
        # Ensure both outputs are scalar values
        if isinstance(out1, np.ndarray):
            out1 = out1.item()  # Convert to scalar if it's an ndarray
        
        if isinstance(out2, np.ndarray):
            out2 = out2.item()  # Convert to scalar if it's an ndarray
        
        # Debugging: Print values of out1 and out2 to check if they are scalars
        print(f"out1 (sigmoid of decision function): {out1}")
        print(f"out2 (positive class probability): {out2}")
        
        # Compute the average of the two model outputs
        final_output = np.mean([out1, out2])
        
        # Convert the final output to percentage
        output = str(round(final_output * 100, 2))
        
        # Render the result on the HTML page
        return render_template(
            'index.html',
            prediction_text=f'Chance of Alzheimer disease: {output}%'
        )
    
    except Exception as e:
        # Handle any errors and display an error message
        print(f"Error occurred: {e}")
        return render_template('index.html', prediction_text='Error: Unable to process the input.')

if __name__ == "__main__":
    app.run(debug=True)