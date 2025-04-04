"#iris-classifier-api"

A lightweight API to classify Iris flowers into species (setosa, versicolor, virginica) using machine learning.

"#Key Features"

1. ML Model:
Trained on the classic Iris dataset (scikit-learn).
Uses a Random Forest classifier for high accuracy.

2. API Endpoints:
POST /predict: Accepts sepal_length, sepal_width, petal_length, petal_width (in cm) and returns the predicted species.

3. Tech Stack:
Python + FastAPI (backend).
Pickle for model serialization.

4. Easy Setup:
- Install dependencies:
pip install -r requirements.txt
- Run locally:
uvicorn main:app --reload

5. Example Request:
curl -X POST "http://localhost:8000/predict" \
-H "Content-Type: application/json" \
-d '{"sepal_length": 5.1, "sepal_width": 3.5, "petal_length": 1.4, "petal_width": 0.2}'
Response:
{"species": "setosa"}

"# Project Structure"
iris-classifier-api/
├── model/               # Trained model (pickle file)
├── main.py              # FastAPI server + routes
├── requirements.txt     # Dependencies
└── README.md            # You are here!

