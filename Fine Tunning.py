import os
import time
import joblib
import torch
import pandas as pd
from transformers import BertForSequenceClassification, BertTokenizer, AdamW
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

# Define relative paths
base_dir = os.getcwd()
data_path = os.path.join(base_dir, 'data', 'Dataset.csv')
model_dir = os.path.join(base_dir, 'models')

os.makedirs(model_dir, exist_ok=True)

# Load dataset
df = pd.read_csv(data_path)

# Select columns
symptoms_col = 'statement'  
target_col = 'status'  

# Remove null values and convert to string
df = df.dropna(subset=[symptoms_col, target_col])
df[symptoms_col] = df[symptoms_col].astype(str)

# Create dictionary mapping statuses to integers
unique_statuses = df[target_col].unique()
status_to_int = {status: idx for idx, status in enumerate(unique_statuses)}
int_to_status = {v: k for k, v in status_to_int.items()}

# Prepare data
pairs = list(zip(df[symptoms_col].tolist(), df[target_col].tolist()))
train_data, test_data = train_test_split(pairs, test_size=0.2, random_state=42)

# Tokenizer
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

def tokenize_data(data):
    inputs = tokenizer(
        [pair[0] for pair in data],
        padding=True,
        truncation=True,
        max_length=128,
        return_tensors="pt"
    )
    labels = torch.tensor([status_to_int.get(pair[1], -1) for pair in data])  
    return inputs, labels

train_inputs, train_labels = tokenize_data(train_data)
test_inputs, test_labels = tokenize_data(test_data)

# Models
bert_model = BertForSequenceClassification.from_pretrained(
    "bert-base-uncased", num_labels=len(status_to_int)
)

vectorizer = TfidfVectorizer(max_features=5000)
X_train_tfidf = vectorizer.fit_transform([pair[0] for pair in train_data])
X_test_tfidf = vectorizer.transform([pair[0] for pair in test_data])

naive_bayes_model = MultinomialNB()
naive_bayes_model.fit(
    X_train_tfidf, [status_to_int[pair[1]] for pair in train_data]
)

# Training settings
epochs = 3
batch_size = 32
learning_rate = 2e-5
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
bert_model.to(device)

train_data_tensor = TensorDataset(
    train_inputs['input_ids'], train_inputs['attention_mask'], train_labels
)
train_loader = DataLoader(train_data_tensor, batch_size=batch_size, shuffle=True)
optimizer = AdamW(bert_model.parameters(), lr=learning_rate)

# Training loop
for epoch in range(epochs):
    bert_model.train()
    total_loss = 0
    start_time = time.time()
    
    for batch_idx, batch in enumerate(train_loader):
        input_ids, attention_mask, labels = [b.to(device) for b in batch]
        optimizer.zero_grad()
        outputs = bert_model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        total_loss += loss.item()
        loss.backward()
        optimizer.step()

        if batch_idx % 100 == 0:
            print(f"Epoch {epoch + 1}/{epochs}, Batch {batch_idx + 1}/{len(train_loader)}, Loss: {loss.item():.4f}")
    
    avg_loss = total_loss / len(train_loader)
    epoch_time = time.time() - start_time
    print(f"Epoch {epoch + 1}/{epochs}, Avg Loss: {avg_loss:.4f}, Time: {epoch_time:.2f} seconds")

# Save models
def save_models():
    bert_model.save_pretrained(os.path.join(model_dir, 'bert_model'))
    torch.save(bert_model.state_dict(), os.path.join(model_dir, 'bert_model.pth'))

    joblib.dump(naive_bayes_model, os.path.join(model_dir, 'naive_bayes_model.pkl'))
    joblib.dump(vectorizer, os.path.join(model_dir, 'vectorizer.pkl'))

    tokenizer.save_pretrained(os.path.join(model_dir, 'bert_model'))

save_models()
