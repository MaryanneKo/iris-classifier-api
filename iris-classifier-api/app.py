from flask import Flask, request, jsonify
import joblib
import numpy as np

# Load the trained model
model = joblib.load('iris_model.pkl')

# Create the Flask app
app = Flask(__name__)

# Home route
@app.route('/')
def home():
    return "🌸 Welcome to the Iris Classifier API!"

# Prediction route
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Expecting JSON input with 4 features
        data = request.get_json(force=True)
        features = [data['sepal_length'], data['sepal_width'], data['petal_length'], data['petal_width']]
        features_array = np.array(features).reshape(1, -1)

        prediction = model.predict(features_array)[0]

        # Map the prediction to actual iris species
        species = ['setosa', 'versicolor', 'virginica']
        result = species[prediction]

        return jsonify({
            'prediction': result,
            'input': data
        })

    except Exception as e:
        return jsonify({'error': str(e)})

# Run the app
if __name__ == '__main__':
    app.run(debug=True)
