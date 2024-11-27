from dataclasses import dataclass
import json
import logging
import os
import time

from datasets import Dataset
import pandas as pd
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import get_scheduler

from model import BertForWordBoundaryDetection

if torch.backends.mps.is_available() and torch.backends.mps.is_built():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger(__name__)


@dataclass
class Hyperparameters:
    test_split_size: float = 0.2
    batch_size: int = 8
    epochs: int = 10
    learning_rate: float = 2e-5
    N: int = int(1e4)
    num_warmup_steps: int = 0


# Load hyperparameters
try:
    with open("hyperparameters.json", "r") as f:
        json_data = json.load(f)
        for k, v in json_data.items():
            try:
                json_data[k] = int(v)
            except ValueError:
                try:
                    json_data[k] = float(v)
                except ValueError:
                    logger.warning(f"Failed parsing hyperparameter {k}={v}. {v} invalid literal for int and float.")

    hyperparameters = Hyperparameters(**json_data)
    logger.info(f"Loaded hyperparameters: {json.dumps(hyperparameters.__dict__, indent=4)}")

except FileNotFoundError:
    logger.info("No hyperparameters found; using defaults.")
    hyperparameters = Hyperparameters()

bert = BertForWordBoundaryDetection()
bert.to(device)

# Load the dataset
df = pd.read_csv("training_data.csv", index_col=None)
df = df.loc[df.index.repeat(df["count"])]\
    .reset_index(drop=True)[["text", "label"]]

# Define optimizer and learning rate scheduler
optimizer = AdamW(bert.parameters(), lr=hyperparameters.learning_rate)
num_training_steps = hyperparameters.N * hyperparameters.epochs
scheduler = get_scheduler(
    "linear",
    optimizer=optimizer,
    num_warmup_steps=hyperparameters.num_warmup_steps,
    num_training_steps=num_training_steps,
)

# Initialize binary classification loss
criterion = nn.BCEWithLogitsLoss()

# Training loop
bert.train()
for epoch in range(hyperparameters.epochs):
    start_time = time.time()
    total_train_loss = 0

    # Randomly sample N strings of each state
    df = pd.concat([
        df[df["label"] == 0].sample(hyperparameters.N),
        df[df["label"] == 1].sample(hyperparameters.N)
    ])

    # Convert DataFrame to Hugging Face Dataset
    dataset = Dataset.from_pandas(df)

    # Tokenize dataset
    tokenized_dataset = dataset.map(lambda x: bert.tokenize_function(x["text"]), batched=True)

    # Split into train and validation sets
    train_test_split = tokenized_dataset.train_test_split(
        test_size=hyperparameters.test_split_size
    )

    train_dataset = torch.utils.data.TensorDataset(
        torch.tensor(train_test_split["train"]["input_ids"]).to(device),
        torch.tensor(train_test_split["train"]["attention_mask"]).to(device),
        torch.tensor(train_test_split["train"]["label"]).to(device),
    )

    eval_dataset = torch.utils.data.TensorDataset(
        torch.tensor(train_test_split["test"]["input_ids"]).to(device),
        torch.tensor(train_test_split["test"]["attention_mask"]).to(device),
        torch.tensor(train_test_split["test"]["label"]).to(device),
    )

    # DataLoader for batching
    train_dataloader = DataLoader(
        train_dataset, batch_size=hyperparameters.batch_size, shuffle=True
    )
    eval_dataloader = DataLoader(
        eval_dataset, batch_size=hyperparameters.batch_size
    )

    num_batches = len(train_dataloader)

    for index, batch in enumerate(train_dataloader):
        optimizer.zero_grad()
        input_ids, attention_mask, labels = [b.to(device) for b in batch]
        labels = labels.float().unsqueeze(1)

        outputs = bert(input_ids=input_ids, attention_mask=attention_mask)
        loss = criterion(outputs, labels)
        total_train_loss += loss.item()

        if index % int(num_batches // 10) == 0:
            logger.info(f"Epoch [{epoch + 1}/{hyperparameters.epochs}], Batch [{index + 1}/{num_batches}], Loss: {loss.item()}")

        # Backward pass
        loss.backward()
        optimizer.step()
        scheduler.step()

    avg_train_loss = total_train_loss / len(train_dataloader)
    logger.info(f"Epoch {epoch + 1}/{hyperparameters.epochs} - Training Loss: {avg_train_loss}")

    # Evaluation
    bert.eval()
    total_eval_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for batch in eval_dataloader:
            input_ids, attention_mask, labels = [b.to(device) for b in batch]
            labels = labels.float().unsqueeze(1)
            outputs = bert(input_ids=input_ids, attention_mask=attention_mask)
            loss = criterion(outputs, labels)
            total_eval_loss += loss.item()

            predictions = (torch.sigmoid(outputs) > 0.5).int()
            correct += (predictions.view(labels.shape) == labels.int()).sum().item()
            total += labels.size(0)

    avg_val_loss = total_eval_loss / len(eval_dataloader)
    accuracy = correct / len(eval_dataset)

    logger.info(f"Epoch {epoch + 1}/{hyperparameters.epochs} - Validation Loss: {avg_val_loss}")
    logger.info(f"Epoch {epoch + 1}/{hyperparameters.epochs} - Validation Accuracy: {accuracy}")
    logger.info(f"Epoch {epoch + 1}/{hyperparameters.epochs} - Time: {time.time() - start_time:.2f} seconds")

# Save the model
output_dir = "output"
os.makedirs(output_dir, exist_ok=True)
model_path = os.path.join(output_dir, "pytorch_model.bin")
torch.save(bert.state_dict(), model_path)
logger.info("Training complete. Model saved to %s", model_path)
