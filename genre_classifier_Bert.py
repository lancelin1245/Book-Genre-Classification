import os
import sqlite3
import numpy as np

import torch
from transformers import BertTokenizer, BertForSequenceClassification
from torch.utils.data import DataLoader, TensorDataset, random_split

def load_data_from_db(db_path, table_name, columns=None):
    # Connect to the SQLite database
    conn = sqlite3.connect(db_path)
    
    # Prepare the SQL query
    if columns:
        columns_str = ", ".join(columns)
    else:
        columns_str = "*"
    query = f"SELECT {columns_str} FROM {table_name}"
    
    # Execute the query and fetch the results
    cursor = conn.cursor()
    cursor.execute(query)
    results = cursor.fetchall()
    
    # Close the database connection
    conn.close()
    
    return list(zip(*results)) if columns else results

# Define the paths to the SQLite database files
db_path_train = "genre_train.db" 
db_path_test = "genre_test.db"   

# Load the training data from SQLite
train_columns = ["text", "category", "docid"]
X, Y, docid = load_data_from_db(db_path_train, "genre_train", train_columns)

# Load the test data from SQLite
test_columns = ["text"]
Xt, = load_data_from_db(db_path_test, "genre_test", test_columns)

class KeywordModel(object):
    def __init__(self):
        # Change tokenizer and model to 'bert-base-cased'
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
        self.model = BertForSequenceClassification.from_pretrained(
            "bert-base-cased",
            num_labels=4,  # Number of genres
            output_attentions=False,
            output_hidden_states=False
        )
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        #elif torch.backends.mps.is_available(): # Since I am using an Apple Silicon chip
        #    self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")
        self.model.to(self.device)
        
    def fit(self, X, Y, epochs=3, batch_size=16):
        # Tokenize and encode the training data
        input_ids = []
        attention_masks = []

        for sentence in X:
            encoded_dict = self.tokenizer.encode_plus(
                sentence,
                add_special_tokens=True,
                max_length=256,
                padding='max_length',
                truncation=True,
                return_attention_mask=True,
                return_tensors='pt'
            )
            input_ids.append(encoded_dict['input_ids'])
            attention_masks.append(encoded_dict['attention_mask'])

        input_ids = torch.cat(input_ids, dim=0)
        attention_masks = torch.cat(attention_masks, dim=0)
        labels = torch.tensor(Y)

        # Create DataLoader for training data
        dataset = TensorDataset(input_ids, attention_masks, labels)

        # Split the dataset into training and validation sets in a 90-10 ratio
        train_size = int(0.9 * len(dataset))  
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        optimiser = torch.optim.AdamW(self.model.parameters(), lr=2e-5)
        loss_fn = torch.nn.CrossEntropyLoss()

        # Train the model
        self.model.train()
        for epoch in range(epochs):

            train_correct = 0
            train_total = 0

            for batch in train_dataloader:
                input_ids, attention_mask, label = batch
                input_ids = input_ids.to(self.device)
                attention_mask = attention_mask.to(self.device)
                label = label.to(self.device)

                optimiser.zero_grad()
                outputs = self.model(input_ids, attention_mask=attention_mask)
                loss = loss_fn(outputs.logits, label)
                loss.backward()
                optimiser.step()

                _, predicted = torch.max(outputs.logits.data, 1)
                train_total += label.size(0)
                train_correct += (predicted == label).sum().item()

            train_accuracy = train_correct / train_total

            # Validation Check
            self.model.eval()
            total_val_loss = 0
            val_correct = 0
            val_total = 0
            with torch.no_grad():
                for batch in val_dataloader:
                    input_ids, attention_mask, label = batch
                    input_ids = input_ids.to(self.device)
                    attention_mask = attention_mask.to(self.device)
                    label = label.to(self.device)

                    outputs = self.model(input_ids, attention_mask=attention_mask)
                    loss = loss_fn(outputs.logits, label)
                    total_val_loss += loss.item()

                    _, predicted = torch.max(outputs.logits.data, 1)
                    val_total += label.size(0)
                    val_correct += (predicted == label).sum().item()

            avg_val_loss = total_val_loss / len(val_dataloader)
            val_accuracy = val_correct / val_total
            
            print(f"Epoch {epoch+1}/{epochs}, Training Accuracy: {train_accuracy:.4f}, Validation Loss: {avg_val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}")

    def predict(self, Xin):
        Y_test_pred = []

        # Tokenize and encode the test data
        input_ids = []
        attention_masks = []

        for sentence in Xin:
            encoded_dict = self.tokenizer.encode_plus(
                sentence,
                add_special_tokens=True,
                max_length=256,
                padding='max_length',
                truncation=True,
                return_attention_mask=True,
                return_tensors='pt'
            )
            input_ids.append(encoded_dict['input_ids'])
            attention_masks.append(encoded_dict['attention_mask'])

        input_ids = torch.cat(input_ids, dim=0)
        attention_masks = torch.cat(attention_masks, dim=0)

        # Create DataLoader for test data
        dataset = TensorDataset(input_ids, attention_masks)
        test_dataloader = DataLoader(dataset, batch_size=16)

        # Predict using the model
        self.model.eval()
        with torch.no_grad():
            for batch in test_dataloader:
                input_ids, attention_mask = batch
                input_ids = input_ids.to(self.device)
                attention_mask = attention_mask.to(self.device)
                outputs = self.model(input_ids, attention_mask=attention_mask)
                preds = torch.argmax(outputs.logits, dim=1).cpu().numpy()
                Y_test_pred.extend(preds)

        return Y_test_pred


# fit the model on the training data
model = KeywordModel()
model.fit(X, Y)

# predict on the test data
Y_test_pred = model.predict(Xt)

# write out the csv file
# first column is the id, it is the index to the list of test examples
# second column is the prediction as an integer
fout = open("out.csv", "w")
fout.write("Id,Predicted\n")
for i, line in enumerate(Y_test_pred): # Y_test_pred is in the same order as the test data
    fout.write("%d,%d\n" % (i, line))
fout.close()