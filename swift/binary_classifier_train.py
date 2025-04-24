import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
from sklearn.metrics import roc_auc_score

# ====== Config ======
MODEL_NAME = 'deepseek1_3b'
if MODEL_NAME=='deepseek1_3b':
    INPUT_DIM = 2048
elif MODEL_NAME=='qwen2b':
    INPUT_DIM=1536
elif MODEL_NAME=='intern1b':
    INPUT_DIM=896

HIDDEN_DIM = 512
EPOCHS = 10
BATCH_SIZE = 32
LR = 1e-4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_SAVE_PATH = "best_binary_model.pt"
TRAIN_DATA_PATH = f'/home/hice1/hshih35/.cache/kagglehub/datasets/parthplc/facebook-hateful-meme-dataset/versions/1/data/{MODEL_NAME}_binary_data_train.pt'
TEST_DATA_PATH = f'/home/hice1/hshih35/.cache/kagglehub/datasets/parthplc/facebook-hateful-meme-dataset/versions/1/data/{MODEL_NAME}_binary_data_dev.pt'


# ====== Dataset ======
data = torch.load(TRAIN_DATA_PATH)
X = torch.cat(data['embedding']).float()
y = torch.Tensor(data['groundtruth']).float()
print(type(X), type(y))
print(X.shape, y.shape)





# Dataset and DataLoader
dataset = TensorDataset(X, y)
train_size = int(0.8* X.shape[0])
val_size = X.shape[0]-train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

# ====== Model ======
class BinaryClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)  # single output for binary classification
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)  # shape: (batch_size,)

model = BinaryClassifier(INPUT_DIM, HIDDEN_DIM).to(DEVICE)

# ====== Loss & Optimizer ======
criterion = nn.BCEWithLogitsLoss()  # combines sigmoid + binary cross entropy
optimizer = optim.Adam(model.parameters(), lr=LR)

# ====== Training Loop ======
best_val_acc = 0.0

for epoch in range(EPOCHS):
    model.train()
    total_loss, correct = 0.0, 0

    for inputs, labels in train_loader:
        inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * inputs.size(0)
        preds = torch.sigmoid(outputs) > 0.5
        correct += (preds == labels.bool()).sum().item()

    train_acc = correct / len(train_loader.dataset)
    avg_loss = total_loss / len(train_loader.dataset)

    # ====== Validation ======
    model.eval()
    val_correct = 0
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            outputs = model(inputs)
            preds = torch.sigmoid(outputs) > 0.5
            val_correct += (preds == labels.bool()).sum().item()

    val_acc = val_correct / len(val_loader.dataset)

    print(f"Epoch {epoch+1}: Train Loss={avg_loss:.4f}, Train Acc={train_acc:.4f}, Val Acc={val_acc:.4f}")

    # Save model if it’s the best so far
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), MODEL_SAVE_PATH)
        print(f"Saved new best model at epoch {epoch+1} (Val Acc={val_acc:.4f})")

print(f"\n Training complete. Best Validation Accuracy: {best_val_acc:.4f}")


# ====== Testing ======
test_data = torch.load(TEST_DATA_PATH)
test_X = torch.cat(test_data['embedding']).float()
test_y = torch.Tensor(test_data['groundtruth']).float()


# Dataset and DataLoader
test_dataset = TensorDataset(test_X, test_y)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True)



print("\n Evaluating on test set with best model...")

# Load best model
model.load_state_dict(torch.load(MODEL_SAVE_PATH))
model.eval()

all_preds = []
all_probs = []
all_labels = []

with torch.no_grad():
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
        outputs = model(inputs)
        probs = torch.sigmoid(outputs)
        preds = probs > 0.5

        all_probs.extend(probs.cpu().numpy())
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

torch.save(all_probs, f'all_probs_{MODEL_NAME}.pt')
torch.save(all_labels, 'all_labels.pt')


# Accuracy
test_acc = (torch.tensor(all_preds) == torch.tensor(all_labels)).sum().item() / len(all_labels)

# AUROC
try:
    auroc = roc_auc_score(all_labels, all_probs)
except ValueError:
    auroc = None
    print("⚠️ AUROC could not be computed (perhaps only one class in test set).")

print(f"Test Accuracy = {test_acc:.4f}")
if auroc is not None:
    print(f"Test AUROC     = {auroc:.4f}")

