import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import RobertaTokenizer, RobertaForSequenceClassification
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
from sklearn.metrics import classification_report
import json, time, os

# ── Config ──────────────────────────────────────────────────────────────────
TRAIN_PATH  = "master_dataset/roberta_train.csv"
TEST_PATH   = "master_dataset/roberta_test.csv"
MODEL_OUT   = "models/roberta_layer1"
BATCH_SIZE  = 32        # safe for 8GB VRAM
MAX_LEN     = 128       # our serialized text is ~150 tokens, slight truncation ok
EPOCHS      = 3
LR          = 2e-5
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Device: {DEVICE}")
print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")

# ── Dataset ──────────────────────────────────────────────────────────────────
class NetworkFlowDataset(Dataset):
    def __init__(self, df, tokenizer):
        self.texts  = df['text'].tolist()
        self.labels = df['label'].tolist()
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        enc = self.tokenizer(
            self.texts[idx],
            max_length=MAX_LEN,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        return {
            'input_ids':      enc['input_ids'].squeeze(),
            'attention_mask': enc['attention_mask'].squeeze(),
            'label':          torch.tensor(self.labels[idx], dtype=torch.long)
        }

# ── Load data ────────────────────────────────────────────────────────────────
print("Loading data...")
train_df = pd.read_csv(TRAIN_PATH)
test_df  = pd.read_csv(TEST_PATH)
print(f"Train: {len(train_df)} | Test: {len(test_df)}")

tokenizer = RobertaTokenizer.from_pretrained('roberta-base')

train_ds = NetworkFlowDataset(train_df, tokenizer)
test_ds  = NetworkFlowDataset(test_df,  tokenizer)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=4, pin_memory=True)
test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

# ── Model ────────────────────────────────────────────────────────────────────
print("Loading RoBERTa...")
model = RobertaForSequenceClassification.from_pretrained('roberta-base', num_labels=2)
model.to(DEVICE)

# Class weights to handle 2.5:1 imbalance
counts     = train_df['label'].value_counts().sort_index().values
weights    = torch.tensor([1.0 / c for c in counts], dtype=torch.float).to(DEVICE)
weights    = weights / weights.sum() * 2
loss_fn    = torch.nn.CrossEntropyLoss(weight=weights)

optimizer  = AdamW(model.parameters(), lr=LR, weight_decay=0.01)
total_steps= len(train_loader) * EPOCHS
scheduler  = get_linear_schedule_with_warmup(optimizer,
                num_warmup_steps=total_steps // 10,
                num_training_steps=total_steps)

# ── Training loop ─────────────────────────────────────────────────────────────
best_f1 = 0.0

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    start = time.time()

    for step, batch in enumerate(train_loader):
        input_ids  = batch['input_ids'].to(DEVICE)
        attn_mask  = batch['attention_mask'].to(DEVICE)
        labels     = batch['label'].to(DEVICE)

        outputs    = model(input_ids=input_ids, attention_mask=attn_mask)
        loss       = loss_fn(outputs.logits, labels)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        total_loss += loss.item()

        if step % 100 == 0:
            elapsed = time.time() - start
            print(f"Epoch {epoch+1} | Step {step}/{len(train_loader)} | "
                  f"Loss: {loss.item():.4f} | Elapsed: {elapsed:.0f}s")

    avg_loss = total_loss / len(train_loader)
    print(f"\nEpoch {epoch+1} avg loss: {avg_loss:.4f}")

    # ── Eval ──────────────────────────────────────────────────────────────────
    model.eval()
    all_preds, all_labels, all_probs = [], [], []

    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch['input_ids'].to(DEVICE)
            attn_mask = batch['attention_mask'].to(DEVICE)
            labels    = batch['label']

            outputs   = model(input_ids=input_ids, attention_mask=attn_mask)
            probs     = torch.softmax(outputs.logits, dim=1)[:, 1].cpu().numpy()
            preds     = outputs.logits.argmax(dim=1).cpu().numpy()

            all_preds.extend(preds)
            all_labels.extend(labels.numpy())
            all_probs.extend(probs)

    report = classification_report(all_labels, all_preds,
                target_names=["Benign", "Attack"], output_dict=True)
    f1 = report['Attack']['f1-score']
    print(classification_report(all_labels, all_preds,
                target_names=["Benign", "Attack"]))

    if f1 > best_f1:
        best_f1 = f1
        os.makedirs(MODEL_OUT, exist_ok=True)
        model.save_pretrained(MODEL_OUT)
        tokenizer.save_pretrained(MODEL_OUT)
        print(f"✅ Best model saved (Attack F1: {f1:.4f})")

print(f"\nTraining complete. Best Attack F1: {best_f1:.4f}")
print(f"Model saved to: {MODEL_OUT}")
