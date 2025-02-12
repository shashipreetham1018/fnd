from flask import Flask, request, render_template
import joblib
import os

app = Flask(__name__)

# Load the trained model and vectorizer
model_path = "models/model.pkl"
vectorizer_path = "models/vectorizer.pkl"

if os.path.exists(model_path) and os.path.exists(vectorizer_path):
    model = joblib.load(model_path)
    vectorizer = joblib.load(vectorizer_path)
else:
    raise FileNotFoundError("Model or vectorizer file not found. Train the model first.")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.form['news_text']
    transformed_data = vectorizer.transform([data])
    prediction = model.predict(transformed_data)
    result = "Fake News" if prediction[0] == 1 else "Real News"
    return render_template('prediction.html', news_text=data, prediction=result)

if __name__ == '__main__':
    app.run(debug=True)
