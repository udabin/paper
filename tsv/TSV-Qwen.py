# TSV layer 추가 Hallucination Detection
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from functools import partial
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from sklearn.metrics import roc_auc_score
from torch.utils.data import Dataset, DataLoader
import scipy.stats
import ot

# CUDA 설정 및 경로 초기화
sys.path.append("/data/")
torch.cuda.empty_cache()
torch.cuda.reset_max_memory_allocated()

# 모델 설정
MODEL_NAME = "Qwen/Qwen2.5-7B"
device = "cuda" if torch.cuda.is_available() else "cpu"

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16,
    device_map="auto",
    low_cpu_mem_usage=True,
    trust_remote_code=True
).to(device)

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
model.eval()

# TruthfulQA 데이터셋
full_data = load_dataset("truthful_qa", "multiple_choice")["validation"]
train_test_split = full_data.train_test_split(test_size=0.25, seed=0)
train_data = train_test_split["train"]
test_data = train_test_split["test"]
validation_data = test_data[:30]

# 파라미터
BATCH_SIZE = 8 #128
steering_strength = 5.0
target_layer = 4
initial_epochs = 50 #20
augmented_epochs = 50 #20
learning_rate = 0.005
alpha = 0.99 # EMA decay rate
cp = 10 #concentration parameter(k)
num_sample = 10

# 데이터셋 정의
class TruthfulQADataset(Dataset):
    def __init__(self, hf_dataset):
        self.data = hf_dataset
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        item = self.data[idx]
        return {"question": item["question"], "mc1_targets": item["mc1_targets"]}

class CleanedTruthfulQADataset(Dataset):
    def __init__(self, data_list):
        self.data = data_list
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        return {
            "question": self.data[idx]["question"],
            "mc1_targets": {
                "labels": [1],
                "choices": [self.data[idx]["mc1_targets"]]
            }
        }

def collate_fn(batch, tokenizer):
    questions = [item["question"] for item in batch]
    targets = [item["mc1_targets"]["choices"][item["mc1_targets"]["labels"].index(1)] for item in batch]
    input_encodings = tokenizer(questions, padding=True, truncation=True, return_tensors="pt")
    target_encodings = tokenizer(targets, padding=True, truncation=True, return_tensors="pt")
    return {
        "input_ids": input_encodings["input_ids"],
        "attention_mask": input_encodings["attention_mask"],
        "labels": target_encodings["input_ids"]
    }

collate_fn_with_tokenizer = partial(collate_fn, tokenizer=tokenizer)

# TSV Layer 정의
class TSVLayer(nn.Module):
    def __init__(self, hidden_dim, steering_strength):
        super().__init__()
        self.tsv = nn.Parameter(torch.randn(hidden_dim))
        self.prototype = nn.Parameter(torch.zeros(hidden_dim))
        self.steering_strength = steering_strength
        self.loss_fn = nn.NLLLoss()

    def forward(self, hidden_states):
        return hidden_states + self.steering_strength * self.tsv

    def compute_loss(self, logits, target_ids):
        log_probs = torch.log_softmax(logits, dim=-1)
        min_length = min(log_probs.shape[1], target_ids.shape[1])
        return self.loss_fn(
            log_probs[:, :min_length, :].reshape(-1, log_probs.size(-1)),
            target_ids[:, :min_length].reshape(-1)
        )

    def update_prototype(self, hidden_states, alpha=0.1):
        with torch.no_grad():
            if hidden_states.dim() == 3:
                hidden_states = hidden_states.mean(dim=1)
            if hidden_states.shape[1] != self.prototype.shape[0]:
                hidden_states = F.interpolate(hidden_states.unsqueeze(1), size=self.prototype.shape[0], mode='linear', align_corners=False).squeeze(1)
            self.prototype.data = alpha * self.prototype.data + (1 - alpha) * hidden_states.mean(dim=0)
            self.prototype.data = self.prototype / self.prototype.norm()

# 모델 래핑
class ModifiedQwenModel(nn.Module):
    def __init__(self, qwen_model, tsv_layer, target_layer):
        super().__init__()
        self.model = qwen_model
        self.tsv_layer = tsv_layer
        self.target_layer = target_layer

    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.model.model(input_ids, attention_mask=attention_mask, output_hidden_states=True)
        hidden_states = outputs.hidden_states
        modified_hidden_state = self.tsv_layer(hidden_states[self.target_layer])
        logits = self.model.lm_head(modified_hidden_state.to(torch.float16))
        loss = self.tsv_layer.compute_loss(logits, labels) if labels is not None else None
        return logits, loss, modified_hidden_state

# 학습 데이터셋 분할 및 초기화
initial_train_data = train_data.select(range(32))
remaining_train_data = train_data.select(range(32, len(train_data)))
initial_dataset = TruthfulQADataset(initial_train_data)
remaining_dataset = TruthfulQADataset(remaining_train_data)

initial_loader = DataLoader(initial_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn_with_tokenizer)
remaining_loader = DataLoader(remaining_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn_with_tokenizer)

# TSV 레이어 및 수정된 모델 정의
tsv_layer = TSVLayer(model.config.hidden_size, steering_strength).to(device)
modified_model = ModifiedQwenModel(model, tsv_layer, target_layer=target_layer).to(device)

# 학습
def train_tsv(epochs=initial_epochs, learning_rate=learning_rate, alpha=alpha, weight_decay=0.01):
    optimizer = optim.AdamW([tsv_layer.tsv], lr=learning_rate, weight_decay=weight_decay)
    for epoch in range(epochs):
        total_loss = 0
        for batch in initial_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            logits, loss, modified_hidden_state = modified_model(input_ids, attention_mask, labels)
            if loss is not None:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                tsv_layer.update_prototype(modified_hidden_state, alpha=alpha)
                total_loss += loss.item()
            torch.cuda.empty_cache()
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {total_loss / len(initial_loader):.4f}")

def compute_truthfulness_score(hidden_state, kappa=cp):
    hidden_state = hidden_state.cpu().numpy()
    prototype = tsv_layer.prototype.detach().cpu().numpy()
    score = np.exp(kappa * np.dot(prototype, hidden_state.T) - np.max(kappa * np.dot(prototype, hidden_state.T)))
    return score[0] / np.sum(score)

def evaluate_auroc(threshold=0.5):
    scores, labels = [], []
    for data in test_data:
        inputs = tokenizer(data["question"], return_tensors="pt").to(device)
        with torch.no_grad():
            _, _, hidden_state = modified_model(inputs["input_ids"], inputs["attention_mask"])
        score = compute_truthfulness_score(hidden_state.mean(dim=0))
        scores.append(score)
        labels.append(1 if score >= threshold else 0)
    print(f"AUROC Score: {roc_auc_score(labels, scores):.4f}")

def detect_hallucination_on_test(threshold=0.5, num_samples=num_sample):
    for i in range(min(num_samples, len(test_data))):
        q = test_data[i]["question"]
        print(f"\nTest Sample {i+1}: {q}")
        inputs = tokenizer(q, return_tensors="pt").to(device)
        with torch.no_grad():
            _, _, hidden_state = modified_model(inputs["input_ids"], inputs["attention_mask"])
        score = compute_truthfulness_score(hidden_state.mean(dim=0))
        print(f"Truthfulness Score: {score:.4f}")
        print("[Hallucination Detected]" if score < threshold else "[Truthful]")


# run
train_tsv()
evaluate_auroc()
detect_hallucination_on_test()