import os
import logging
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_curve,
    auc
)

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def evaluate_model(name, model, X_train, y_train, X_test, y_test, artifact_dir="./artifacts"):
    """
    Evaluate model performance: metrics, confusion matrix, ROC curve, feature importance.
    Save plots in artifact_dir and return metrics dict.
    """
    os.makedirs(artifact_dir, exist_ok=True)
    log.info(f"Evaluating {name}...")

    # Predictions
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)

    # Probabilities for ROC
    if hasattr(model, "predict_proba"):
        y_prob_test = model.predict_proba(X_test)[:, 1]
    else:
        y_prob_test = model.decision_function(X_test)

    # Accuracy
    train_acc = accuracy_score(y_train, y_pred_train)
    test_acc = accuracy_score(y_test, y_pred_test)

    # Classification report
    report = classification_report(y_test, y_pred_test)
    log.info(f"{name} Classification Report:\n{report}")

    # ------------------------
    # Confusion matrix
    # ------------------------
    cm = confusion_matrix(y_test, y_pred_test)
    cm_path = os.path.join(artifact_dir, f"{name}_confusion_matrix.png")
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title(f"{name} Confusion Matrix")
    plt.savefig(cm_path)
    plt.close()
    log.info(f"Saved confusion matrix: {cm_path}")

    # ------------------------
    # ROC Curve
    # ------------------------
    fpr, tpr, _ = roc_curve(y_test, y_prob_test)
    roc_auc = auc(fpr, tpr)
    roc_path = os.path.join(artifact_dir, f"{name}_roc_curve.png")
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, label=f"{name} (AUC={roc_auc:.2f})")
    plt.plot([0, 1], [0, 1], "k--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend()
    plt.savefig(roc_path)
    plt.close()
    log.info(f"Saved ROC curve: {roc_path}")

    # ------------------------
    # Feature importance (CatBoost-specific)
    # ------------------------
    if hasattr(model, "get_feature_importance"):
        importance = model.get_feature_importance()
        feat_names = X_train.columns
        plt.figure(figsize=(8, 6))
        sns.barplot(x=importance, y=feat_names)
        plt.title(f"{name} Feature Importance")
        feat_path = os.path.join(artifact_dir, f"{name}_feature_importance.png")
        plt.savefig(feat_path)
        plt.close()
        log.info(f"Saved feature importance: {feat_path}")

    return {"train_accuracy": train_acc, "test_accuracy": test_acc, "roc_auc": roc_auc,
            "confusion_matrix_plot": cm_path, "roc_curve_plot": roc_path, "feature_importance_plot": feat_path if hasattr(model, "get_feature_importance") else None}
