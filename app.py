from flask import Flask, request, jsonify, render_template
import joblib

app = Flask(__name__)

# Load model and vectorizer
model = joblib.load('spam_model.pkl')
vectorizer = joblib.load('vectorizer.pkl')

# الصفحة الرئيسية
@app.route('/')
def index():
    return render_template('index.html')

# Endpoint لتوقع السبام (بيشتغل مع JavaScript fetch)
@app.route('/predict', methods=['POST'])
def predict():
    email_text = request.form.get('email')

    if not email_text:
        return jsonify({'result': 'error', 'message': 'No email provided'}), 400

    # Transform and predict
    email_vector = vectorizer.transform([email_text])
    prediction = model.predict(email_vector)[0]
    result = "spam" if prediction == 1 else "not spam"

    return jsonify({'result': result})

if __name__ == '__main__':
    app.run(debug=True)
