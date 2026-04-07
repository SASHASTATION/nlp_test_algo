import pandas as pd
import numpy as np
import torch
import re
import gc
import os
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    get_linear_schedule_with_warmup,
    DataCollatorWithPadding # Добавлено для ускорения
)
from sklearn.utils.class_weight import compute_class_weight

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# CONFIG
MODEL_NAME = "ai-forever/ruBERT-base"
MAX_LEN = 256
BATCH_SIZE = 64 # Увеличено, т.к. 32GB VRAM это легко позволяет
EPOCHS = 4
LR = 3e-5 # Чуть подняли из-за увеличенного размера батча
WEIGHT_DECAY = 0.01
WARMUP_RATIO = 0.1
NUM_LABELS = 5
N_FOLDS = 5
TRAIN_ALL_FOLDS = True

# Загрузка данных
train_df = pd.read_csv("train_csv.csv")
test_df = pd.read_csv("test_csv.csv")
sample_sub = pd.read_csv("sample_submission_csv.csv")

def clean_text(text):
    if pd.isna(text):
        return ""
    text = str(text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

train_df["text"] = train_df["text"].apply(clean_text)
test_df["text"] = test_df["text"].apply(clean_text)
train_df["label"] = train_df["rate"] - 1

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
# Динамический паддинг: дополняет тексты нулями только до максимума в текущем батче!
collator = DataCollatorWithPadding(tokenizer=tokenizer) 

class ReviewDatasetFast(Dataset):
    def __init__(self, texts, labels=None, tokenizer=None, max_len=256):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len
        
    def __len__(self):
        return len(self.texts)
        
    def __getitem__(self, idx):
        # Убрали padding="max_length" и return_tensors="pt" - это сделает collator
        item = self.tokenizer(
            self.texts[idx],
            truncation=True,
            max_length=self.max_len,
        )
        if self.labels is not None:
            item["labels"] = self.labels[idx]
        return item

class_weights = compute_class_weight("balanced", classes=np.arange(NUM_LABELS), y=train_df["label"].values)
class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32).to(device)

def train_one_fold(fold, train_idx, val_idx, train_df, test_df):
    print(f"\n{'='*60}\nFOLD {fold+1}\n{'='*60}")
    tr_df = train_df.iloc[train_idx]
    vl_df = train_df.iloc[val_idx]
    
    train_ds = ReviewDatasetFast(tr_df["text"].tolist(), tr_df["label"].tolist(), tokenizer, MAX_LEN)
    val_ds = ReviewDatasetFast(vl_df["text"].tolist(), vl_df["label"].tolist(), tokenizer, MAX_LEN)
    test_ds = ReviewDatasetFast(test_df["text"].tolist(), labels=None, tokenizer=tokenizer, max_len=MAX_LEN)

    # Добавлены num_workers и pin_memory для ускорения передачи данных на GPU
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collator, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE*2, shuffle=False, collate_fn=collator, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE*2, shuffle=False, collate_fn=collator, num_workers=4, pin_memory=True)

    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=NUM_LABELS).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    
    total_steps = len(train_loader) * EPOCHS
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(total_steps * WARMUP_RATIO), num_training_steps=total_steps)
    loss_fn = torch.nn.CrossEntropyLoss(weight=class_weights_tensor)

    # Скалер для Mixed Precision (AMP)
    scaler = torch.cuda.amp.GradScaler()

    best_f1 = 0
    best_preds_test = None
    
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        for batch in train_loader:
            optimizer.zero_grad()
            
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            
            # Включаем аппаратное ускорение FP16
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                loss = loss_fn(outputs.logits, labels)
            
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            total_loss += loss.item()
            
        avg_loss = total_loss / len(train_loader)

        model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)
                
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                    
                preds = outputs.logits.argmax(dim=-1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                
        f1 = f1_score(all_labels, all_preds, average="weighted")
        print(f"  Epoch {epoch+1}/{EPOCHS} | Loss: {avg_loss:.4f} | Val F1: {f1:.4f}")
        
        if f1 > best_f1:
            best_f1 = f1
            test_logits = []
            with torch.no_grad():
                for batch in test_loader:
                    input_ids = batch["input_ids"].to(device)
                    attention_mask = batch["attention_mask"].to(device)
                    with torch.autocast(device_type="cuda", dtype=torch.float16):
                        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                    test_logits.append(outputs.logits.float().cpu().numpy())
            best_preds_test = np.concatenate(test_logits, axis=0)
            
    print(f"  Best Val F1: {best_f1:.4f}")
    del model, optimizer, scheduler, scaler
    gc.collect()
    torch.cuda.empty_cache()
    return best_f1, best_preds_test

skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
oof_f1_scores = []
test_logits_all = []

texts = train_df["text"].values
labels = train_df["label"].values

for fold, (train_idx, val_idx) in enumerate(skf.split(texts, labels)):
    f1, test_logits = train_one_fold(fold, train_idx, val_idx, train_df, test_df)
    oof_f1_scores.append(f1)
    test_logits_all.append(test_logits)
    if not TRAIN_ALL_FOLDS:
        break

print(f"\n{'='*60}")
print(f"Mean CV F1: {np.mean(oof_f1_scores):.4f} (+/- {np.std(oof_f1_scores):.4f})")
print(f"{'='*60}")

ensemble_logits = np.mean(test_logits_all, axis=0)
test_preds = ensemble_logits.argmax(axis=1) + 1
submission = sample_sub.copy()
submission["rate"] = test_preds

output_path = os.path.join(os.getcwd(), "submission.csv")
submission.to_csv(output_path, index=False)
print(f"Submission saved: {output_path}")