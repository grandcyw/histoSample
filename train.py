from models import DenseNetClassifier, LongNetClassifier, DNNClassifier,vit
import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score

def evaluate_model(model, test_features, test_labels):
    with torch.no_grad():
        logits = model(test_features)
        preds = torch.argmax(logits, dim=1)
        acc = accuracy_score(test_labels, preds)
        f1 = f1_score(test_labels, preds)
    return {"accuracy": acc, "f1": f1}

# Split data
train_features, test_features, train_labels, test_labels = train_test_split(
    features, labels, test_size=0.2
)

# Train and compare models
models = {
    "DenseNet": DenseNetClassifier,
    # "LongNet": LongNetClassifier,
    # "DNN": DNNClassifier,
}

results = {}
for name, model_class in models.items():
    model = train_model(train_features, train_labels, model_class)
    results[name] = evaluate_model(model, test_features, test_labels)

print(results)