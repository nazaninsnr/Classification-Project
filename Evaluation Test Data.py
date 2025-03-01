import pandas as pd
import joblib
import torch
from transformers import BertForSequenceClassification, BertTokenizer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import json
import os

# Load status_to_int 
def load_status_to_int():
    try:
        with open(os.path.join(os.getcwd(), "status_to_int.json"), "r") as file:
            status_to_int = json.load(file)
    except FileNotFoundError:
        file_path = os.path.join(os.getcwd(), 'Dataset.csv')
        df = pd.read_csv(file_path).dropna(subset=['statement', 'status'])
        unique_statuses = df['status'].unique()
        status_to_int = {status: idx for idx, status in enumerate(unique_statuses)}
        with open(os.path.join(os.getcwd(), "status_to_int.json"), "w") as file:
            json.dump(status_to_int, file)
    return status_to_int

status_to_int = load_status_to_int()
int_to_status = {v: k for k, v in status_to_int.items()}

# Load Models
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load Bert
bert_model_dir = os.path.join(os.getcwd(), 'models', 'bert_model')
bert_model = BertForSequenceClassification.from_pretrained(bert_model_dir, num_labels=len(status_to_int))
bert_model.load_state_dict(torch.load(f"{bert_model_dir}.pth"))
bert_model.to(device)

# Load Naive Bayes
naive_bayes_model = joblib.load(os.path.join(os.getcwd(), 'models', 'naive_bayes_model.pkl'))
vectorizer = joblib.load(os.path.join(os.getcwd(), 'models', 'vectorizer.pkl'))

# Load Tokenizer
tokenizer = BertTokenizer.from_pretrained(bert_model_dir)

# Predict
def predict_bert(statement):
    """Predict class using BERT model"""
    inputs = tokenizer(statement, return_tensors="pt", padding=True, truncation=True, max_length=128)
    inputs = {key: value.to(device) for key, value in inputs.items()}
    bert_model.eval()
    with torch.no_grad():
        outputs = bert_model(**inputs)
        logits = outputs.logits
    return torch.argmax(logits, dim=1).item()

def predict_naive_bayes(statement):
    """Predict class using Naive Bayes model"""
    tfidf_input = vectorizer.transform([statement])
    return naive_bayes_model.predict(tfidf_input)[0]

# Evaluation Test Data
def evaluate_model():
    file_path = os.path.join(os.getcwd(), 'Dataset.csv')
    df = pd.read_csv(file_path).dropna(subset=['statement', 'status'])
    test_data = df.sample(frac=0.2, random_state=42)  

    true_labels = [status_to_int[label] for label in test_data['status']]
    statements = test_data['statement'].tolist()

    bert_predictions = [predict_bert(statement) for statement in statements]
    nb_predictions = [predict_naive_bayes(statement) for statement in statements]

    # BERT
    bert_metrics = {
        "accuracy": accuracy_score(true_labels, bert_predictions),
        "precision": precision_score(true_labels, bert_predictions, average='weighted'),
        "recall": recall_score(true_labels, bert_predictions, average='weighted'),
        "f1_score": f1_score(true_labels, bert_predictions, average='weighted'),
    }

    # Naive Bayes
    nb_metrics = {
        "accuracy": accuracy_score(true_labels, nb_predictions),
        "precision": precision_score(true_labels, nb_predictions, average='weighted'),
        "recall": recall_score(true_labels, nb_predictions, average='weighted'),
        "f1_score": f1_score(true_labels, nb_predictions, average='weighted'),
    }

    print("Evaluation Results for BERT:")
    for metric, value in bert_metrics.items():
        print(f"{metric.capitalize()}: {value:.4f}")

    print("\nEvaluation Results for Naive Bayes:")
    for metric, value in nb_metrics.items():
        print(f"{metric.capitalize()}: {value:.4f}")

if __name__ == "__main__":
    evaluate_model()
