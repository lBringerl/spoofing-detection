import os

from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    roc_curve,
    precision_recall_curve,
    precision_score,
    recall_score,
    f1_score
)
import matplotlib.pyplot as plt


def calculate_metrics(model, test_pool, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    test_preds = model.predict_proba(test_pool)[:, 1]
    test_pred_labels = model.predict(test_pool)

    val_auc = roc_auc_score(test_pool.get_label(), test_preds)
    val_ap = average_precision_score(test_pool.get_label(), test_preds)
    val_precision = precision_score(test_pool.get_label(), test_pred_labels)
    val_recall = recall_score(test_pool.get_label(), test_pred_labels)
    val_f1 = f1_score(test_pool.get_label(), test_pred_labels)

    metrics_file = os.path.join(output_dir, "metrics.txt")
    with open(metrics_file, "w") as f:
        f.write(f"AUC: {val_auc:.4f}\n")
        f.write(f"AP: {val_ap:.4f}\n")
        f.write(f"Precision: {val_precision:.4f}\n")
        f.write(f"Recall: {val_recall:.4f}\n")
        f.write(f"F1: {val_f1:.4f}\n")
    print(f"AUC: {val_auc:.4f}")
    print(f"AP: {val_ap:.4f}")
    print(f"Precision: {val_precision:.4f}")
    print(f"Recall: {val_recall:.4f}")
    print(f"F1: {val_f1:.4f}")

    fpr, tpr, _ = roc_curve(test_pool.get_label(), test_preds)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve (AUC = {val_auc:.4f})')
    plt.savefig(os.path.join(output_dir, "roc_curve.png"))
    plt.show()
    plt.close()

    precision, recall, _ = precision_recall_curve(test_pool.get_label(), test_preds)
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f'Precision-Recall Curve (AP = {val_ap:.4f})')
    plt.savefig(os.path.join(output_dir, "pr_curve.png"))
    plt.show()
    plt.close()
