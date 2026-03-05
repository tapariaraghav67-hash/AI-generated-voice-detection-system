import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import Wav2Vec2Model

from training.dataset import VoiceDataset
from model.classifier import VoiceClassifier

# 2. Config / Hyperparameters
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 8
EPOCHS = 5
LR = 1e-4

#Load datasets
train_ds = VoiceDataset("data/train")
val_ds = VoiceDataset("data/val")

train_loader = DataLoader(
    train_ds, batch_size=BATCH_SIZE, shuffle=True
)
val_loader = DataLoader(
    val_ds, batch_size=BATCH_SIZE
)

#Load models
wav2vec = Wav2Vec2Model.from_pretrained(
    "facebook/wav2vec2-base"
).to(DEVICE)

# Freeze wav2vec
for param in wav2vec.parameters():
    param.requires_grad = False
for name, param in wav2vec.named_parameters():
    if "encoder.layers.10" in name or "encoder.layers.11" in name:
        param.requires_grad = True


classifier = VoiceClassifier().to(DEVICE)

# Loss & Optimizer
criterion = nn.BCELoss()
optimizer = torch.optim.AdamW([
    {
        "params": classifier.parameters(),
        "lr": 1e-4
    },
    {
        "params": wav2vec.encoder.layers[-2:].parameters(),
        "lr": 1e-5
    }
])


# Training Loop
for epoch in range(EPOCHS):
    classifier.train()
    total_loss = 0

    for waveforms, labels in train_loader:
        waveforms = waveforms.to(DEVICE)
        labels = labels.to(DEVICE).unsqueeze(1)

        outputs = wav2vec(waveforms).last_hidden_state


        pooled = outputs.mean(dim=1)
        preds = classifier(pooled)

        loss = criterion(preds, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)
    print(f"Epoch {epoch+1} | Train Loss: {avg_loss:.4f}")

#Validation Loop
classifier.eval()
correct = 0
total = 0

with torch.no_grad():
    for waveforms, labels in val_loader:
        waveforms = waveforms.to(DEVICE)
        labels = labels.to(DEVICE)

        outputs = wav2vec(waveforms).last_hidden_state
        pooled = outputs.mean(dim=1)
        preds = classifier(pooled).squeeze()

        predicted = (preds > 0.5).float()
        correct += (predicted == labels).sum().item()
        total += labels.size(0)

accuracy = correct / total
print(f"Validation Accuracy: {accuracy:.2%}")

#Save Model
torch.save(
    classifier.state_dict(),
    "model/model.pt"
)

print("Training complete. Model saved.")