import os
import torch
import joblib
from flask import Flask, request, send_from_directory
from transformers import BertForSequenceClassification, BertTokenizer

app = Flask(__name__)

# Path to the models (using relative path)
MODEL_DIR = os.path.join(os.getcwd(), 'models')

# Load BERT model and tokenizer
bert_model = BertForSequenceClassification.from_pretrained(os.path.join(MODEL_DIR, 'bert_model'))
bert_model.load_state_dict(torch.load(os.path.join(MODEL_DIR, 'bert_model.pth'), map_location=torch.device('cpu'), weights_only=True))
bert_model.to('cpu')

tokenizer = BertTokenizer.from_pretrained(os.path.join(MODEL_DIR, 'bert_model'))

# Load Naive Bayes model and vectorizer
naive_bayes_model = joblib.load(os.path.join(MODEL_DIR, 'naive_bayes_model.pkl'))
vectorizer = joblib.load(os.path.join(MODEL_DIR, 'vectorizer.pkl'))

INT_TO_STATUS = {
    0: 'Anxiety',
    1: 'Normal',
    2: 'Depression',
    3: 'Suicidal',
    4: 'Stress',
    5: 'Bipolar',
    6: 'Personality',
}

def predict_bert(statement):
    if not statement.strip():
        return -1

    inputs = tokenizer(statement, return_tensors="pt", padding=True, truncation=True, max_length=128).to('cpu')

    bert_model.eval()
    with torch.no_grad():
        logits = bert_model(**inputs).logits
    return torch.argmax(logits, dim=1).item()

def predict_naive_bayes(statement):
    if not statement.strip():
        return -1

    X_test = vectorizer.transform([statement])
    prediction = naive_bayes_model.predict(X_test)
    return prediction[0]

@app.route('/', methods=['GET', 'POST'])
def index():
    user_input = ""
    bert_predicted_disease = ""
    nb_predicted_disease = ""
    final_predicted_disease = ""

    if request.method == 'POST':
        user_input = request.form['user_input']
        bert_prediction = predict_bert(user_input)
        naive_bayes_prediction = predict_naive_bayes(user_input)

        bert_predicted_disease = INT_TO_STATUS.get(bert_prediction, "Unknown disease")
        nb_predicted_disease = INT_TO_STATUS.get(naive_bayes_prediction, "Unknown disease")

        # Final prediction is BERT's prediction if different
        final_predicted_disease = bert_predicted_disease if bert_predicted_disease != nb_predicted_disease else nb_predicted_disease

    # Return the HTML file as a response directly
    return send_from_directory(os.path.join(os.getcwd(), 'static'), 'index.html')

if __name__ == '__main__':
    app.run(debug=True)
