import sys
import csv
import datetime
import re

import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset, RandomSampler, SequentialSampler
from transformers import BertTokenizer, BertForSequenceClassification, AdamW, get_linear_schedule_with_warmup

csv.field_size_limit(sys.maxsize)


def preprocess_text(text):
    text = re.sub(r"http\S+|www\S+|pic.twitter.com\S+", "", text)
    text = re.sub(r"@[A-Za-z0-9]+", "", text)
    return text


data = []
labels = []

with open('dataset_merged.csv') as file:
    csv_data = file.read()

csv_reader = csv.reader(csv_data.splitlines(), delimiter=',')

for row in csv_reader:
    try:
        label = int(row[4])
        text = row[3]

        processed_text = preprocess_text(text)
        data.append(processed_text)
        labels.append(label)
    except Exception as e:
        continue

print("CSV Read Finished")

X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.5,  train_size=0.5)

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)

def encode_data(tokenizer, data, max_length):
    input_ids = []
    attention_masks = []

    for sent in data:
        encoded_dict = tokenizer.encode_plus(
            sent,
            add_special_tokens=True,
            max_length=max_length,
            pad_to_max_length=True,
            return_attention_mask=True,
            return_tensors='pt',
        )
        input_ids.append(encoded_dict['input_ids'])
        attention_masks.append(encoded_dict['attention_mask'])

    input_ids = torch.cat(input_ids, dim=0)
    attention_masks = torch.cat(attention_masks, dim=0)

    return input_ids, attention_masks


max_length = 128
train_inputs, train_masks = encode_data(tokenizer, X_train, max_length)
test_inputs, test_masks = encode_data(tokenizer, X_test, max_length)

batch_size = 32
train_data = TensorDataset(train_inputs, train_masks, torch.tensor(y_train))
train_sampler = RandomSampler(train_data)
train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)

test_data = TensorDataset(test_inputs, test_masks, torch.tensor(y_test))
test_sampler = SequentialSampler(test_data)
test_dataloader = DataLoader(test_data, sampler=test_sampler, batch_size=batch_size)

optimizer = AdamW(model.parameters(), lr=5e-5, eps=1e-8)
epochs = 10
total_steps = len(train_dataloader) * epochs
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

device = torch.device("mps" if torch.backends.mps.is_built() else "cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

loss_fn = nn.CrossEntropyLoss()

for epoch in range(epochs):
    print(f"Epoch {epoch + 1} {datetime.datetime.now()}")
    model.train()
    print("model.train")
    total_loss, total_accuracy = 0, 0
    for step, batch in enumerate(train_dataloader):
        b_input_ids, b_input_mask, b_labels = tuple(t.to(device) for t in batch)

        model.zero_grad()

        outputs = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask, labels=b_labels)
        loss = outputs.loss

        total_loss += loss.item()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

    avg_train_loss = total_loss / len(train_dataloader)
    print(f"Epoch {epoch + 1} Train Loss: {avg_train_loss}")

    model.eval()
    print("model.eval")

    val_accuracy, val_loss = 0, 0
    for batch in test_dataloader:
        b_input_ids, b_input_mask, b_labels = tuple(t.to(device) for t in batch)

        with torch.no_grad():
            outputs = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask, labels=b_labels)

        logits = outputs.logits
        loss = outputs.loss
        val_loss += loss.item()

        logits = logits.detach().cpu().numpy()
        label_ids = b_labels.to('cpu').numpy()

        batch_accuracy = np.sum(np.argmax(logits, axis=1) == label_ids) / len(label_ids)
        val_accuracy += batch_accuracy

    avg_val_accuracy = val_accuracy / len(test_dataloader)
    avg_val_loss = val_loss / len(test_dataloader)

    print(f"Epoch {epoch + 1} Validation. Accuracy: {avg_val_accuracy}, Loss: {avg_val_loss}")

