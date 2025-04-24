from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
import torch
import numpy as np



MODEL_NAME = 'deepseek1_3b'
TRAIN_DATA_PATH = f'/home/hice1/hshih35/.cache/kagglehub/datasets/parthplc/facebook-hateful-meme-dataset/versions/1/data/{MODEL_NAME}_binary_data_train.pt'
TEST_DATA_PATH = f'/home/hice1/hshih35/.cache/kagglehub/datasets/parthplc/facebook-hateful-meme-dataset/versions/1/data/{MODEL_NAME}_binary_data_dev.pt'

# ====== Prepare Data ======
data = torch.load(TRAIN_DATA_PATH)
X = torch.cat(data['embedding']).float().cpu().numpy()
y = torch.Tensor(data['groundtruth']).float().cpu().numpy()


X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)


test_data = torch.load(TEST_DATA_PATH)
X_test = torch.cat(test_data['embedding']).float().cpu().numpy()
y_test = torch.Tensor(test_data['groundtruth']).float().cpu().numpy()


# ====== Train Random Forest ======
clf = RandomForestClassifier(n_estimators=100, random_state=42)
print('training random forest ...')
clf.fit(X_train, y_train)
print('Training Complete!')

# ====== Evaluate on validation set ======
val_preds = clf.predict(X_val)
val_probs = clf.predict_proba(X_val)[:, 1]
val_acc = accuracy_score(y_val, val_preds)
val_auroc = roc_auc_score(y_val, val_probs)
print(f"Validation Accuracy: {val_acc:.4f}, AUROC: {val_auroc:.4f}")

# ====== Test set evaluation ======
test_preds = clf.predict(X_test)
test_probs = clf.predict_proba(X_test)[:, 1]
test_acc = accuracy_score(y_test, test_preds)
test_auroc = roc_auc_score(y_test, test_probs)
print(f"Test Accuracy: {test_acc:.4f}, AUROC: {test_auroc:.4f}")
