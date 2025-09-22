# pipelines/modeling.py
import logging
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc, precision_recall_curve
import os

log = logging.getLogger(__name__)

def evaluate_model(name, model, X_train, y_train, X_test, y_test, artifact_dir="./artifacts"):
    os.makedirs(artifact_dir, exist_ok=True)
    log.info(f"Starting evaluation for {name}...")
    
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)

    if hasattr(model, "predict_proba"):
        y_prob_test = model.predict_proba(X_test)[:, 1]
    else:
        y_prob_test = model.decision_function(X_test)

    train_acc = accuracy_score(y_train, y_pred_train)
    test_acc = accuracy_score(y_test, y_pred_test)
    
    report = classification_report(y_test, y_pred_test)
    log.info(f"{name} Classification Report:\n{report}")

    # Confusion matrix plot
    cm = confusion_matrix(y_test, y_pred_test)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title(f"{name} - Confusion Matrix")
    plt.savefig(os.path.join(artifact_dir, f"{name}_confusion_matrix.png"))
    plt.close()

    # ROC Curve
    fpr, tpr, _ = roc_curve(y_test, y_prob_test)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(6,5))
    plt.plot(fpr, tpr, label=f"{name} (AUC={roc_auc:.2f})")
    plt.plot([0,1],[0,1],'k--')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend()
    plt.savefig(os.path.join(artifact_dir, f"{name}_roc_curve.png"))
    plt.close()

    return {"train_accuracy": train_acc, "test_accuracy": test_acc, "roc_auc": roc_auc}
