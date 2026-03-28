import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import RobertaTokenizer, RobertaForSequenceClassification
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import pickle, json, time, os

TRAIN_PATH = "Dataset/datasets/unsw_nb15/UNSW_NB15_training-set.csv"
TEST_PATH  = "Dataset/datasets/unsw_nb15/UNSW_NB15_testing-set.csv"
MODEL_OUT  = "models/roberta_unsw"
SCALER_OUT = "models/unsw_scaler.pkl"
FEATS_OUT  = "models/unsw_features.json"
BATCH_SIZE = 32
MAX_LEN    = 128
EPOCHS     = 3
LR         = 2e-5
DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Device: {DEVICE}")

# ── Load and prep data ────────────────────────────────────────────────────────
print("Loading UNSW-NB15...")
train_df = pd.read_csv(TRAIN_PATH)
test_df  = pd.read_csv(TEST_PATH)

# Drop non-numeric and ID columns
drop_cols = ['id', 'attack_cat', 'label']
label_col = 'label'

# Get numeric features only
feature_cols = [c for c in train_df.columns
                if c not in drop_cols
                and train_df[c].dtype in ['float64', 'int64']]

print(f"Features: {len(feature_cols)}")
print(f"Train: {len(train_df)} | Test: {len(test_df)}")
print(f"Label dist:\n{train_df[label_col].value_counts()}")

# Scale features
scaler = StandardScaler()
X_train = scaler.fit_transform(train_df[feature_cols].fillna(0))
X_test  = scaler.transform(test_df[feature_cols].fillna(0))
y_train = train_df[label_col].values
y_test  = test_df[label_col].values

# Save scaler and feature names
with open(SCALER_OUT, "wb") as f:
    pickle.dump(scaler, f)
with open(FEATS_OUT, "w") as f:
    json.dump(feature_cols, f)
print(f"Scaler and features saved")

# ── Text serialization ────────────────────────────────────────────────────────
def serialize(row, feat_names):
    parts = []
    for name, val in zip(feat_names[:25], row[:25]):
        clean = name.lower().replace(" ", "_")
        parts.append(f"{clean}:{val:.4f}")
    return " ".join(parts)

print("Serializing text...")
train_texts = [serialize(X_train[i], feature_cols) for i in range(len(X_train))]
test_texts  = [serialize(X_test[i],  feature_cols) for i in range(len(X_test))]

# ── Dataset ───────────────────────────────────────────────────────────────────
class UNSWDataset(Dataset):
    def __init__(self, texts, labels, tokenizer):
        self.texts     = texts
        self.labels    = labels
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        enc = self.tokenizer(
            self.texts[idx], max_length=MAX_LEN,
            padding='max_length', truncation=True, return_tensors='pt'
        )
        return {
            'input_ids':      enc['input_ids'].squeeze(),
            'attention_mask': enc['attention_mask'].squeeze(),
            'label':          torch.tensor(self.labels[idx], dtype=torch.long)
        }

tokenizer  = RobertaTokenizer.from_pretrained('roberta-base')
train_ds   = UNSWDataset(train_texts, y_train, tokenizer)
test_ds    = UNSWDataset(test_texts,  y_test,  tokenizer)
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=4, pin_memory=True)
test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

# ── Model ─────────────────────────────────────────────────────────────────────
print("Loading RoBERTa...")
model = RobertaForSequenceClassification.from_pretrained('roberta-base', num_labels=2)
model.to(DEVICE)

counts  = np.bincount(y_train)
weights = torch.tensor([1.0/counts[0], 1.0/counts[1]], dtype=torch.float).to(DEVICE)
weights = weights / weights.sum() * 2
loss_fn = torch.nn.CrossEntropyLoss(weight=weights)

optimizer   = AdamW(model.parameters(), lr=LR, weight_decay=0.01)
total_steps = len(train_loader) * EPOCHS
scheduler   = get_linear_schedule_with_warmup(
    optimizer, num_warmup_steps=total_steps//10,
    num_training_steps=total_steps
)

# ── Train ─────────────────────────────────────────────────────────────────────
best_f1 = 0.0
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    start = time.time()

    for step, batch in enumerate(train_loader):
        input_ids = batch['input_ids'].to(DEVICE)
        attn_mask = batch['attention_mask'].to(DEVICE)
        labels    = batch['label'].to(DEVICE)

        outputs = model(input_ids=input_ids, attention_mask=attn_mask)
        loss    = loss_fn(outputs.logits, labels)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        total_loss += loss.item()

        if step % 50 == 0:
            print(f"Epoch {epoch+1} | Step {step}/{len(train_loader)} | "
                  f"Loss: {loss.item():.4f} | Elapsed: {time.time()-start:.0f}s")

    print(f"\nEpoch {epoch+1} avg loss: {total_loss/len(train_loader):.4f}")

    # Eval
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch['input_ids'].to(DEVICE)
            attn_mask = batch['attention_mask'].to(DEVICE)
            outputs   = model(input_ids=input_ids, attention_mask=attn_mask)
            preds     = outputs.logits.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(batch['label'].numpy())

    report = classification_report(all_labels, all_preds,
                target_names=["Normal", "Attack"], output_dict=True)
    f1 = report['Attack']['f1-score']
    print(classification_report(all_labels, all_preds,
                target_names=["Normal", "Attack"]))

    if f1 > best_f1:
        best_f1 = f1
        os.makedirs(MODEL_OUT, exist_ok=True)
        model.save_pretrained(MODEL_OUT)
        tokenizer.save_pretrained(MODEL_OUT)
        print(f"✅ Best UNSW model saved (Attack F1: {f1:.4f})")

print(f"\nTraining complete. Best Attack F1: {best_f1:.4f}")
print(f"Model saved to: {MODEL_OUT}")
