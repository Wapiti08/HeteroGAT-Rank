'''
 # @ Create Time: 2025-04-30 16:41:47
 # @ Modified time: 2025-04-30 16:41:50
 # @ Description: functions to eval model and save metrics locally
 '''

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc


def evaluate(logits, labels, threshold=0.5):
    ''' evaluate binary classification predictions using sklearn

    args:
        logits (Tensor): model raw outputs of shape [batch_size]
        threshold: threshold to convert probabilities to class labels
    
    returns:
        dict: contains acc, prec, recall, F1, and AUC
    '''

    with torch.no_grad():
        y_true = labels.detach().cpu().numpy().astype(int)
        # interpret output as prob -- (0,1)
        y_prob = torch.sigmoid(logits).detach().cpu().numpy()
        y_pred = (y_prob >= threshold).astype(int)

        metrics = {
            "accuracy": accuracy_score(y_true, y_pred),
            "precision": precision_score(y_true, y_pred, zero_division=0),
            "recall": recall_score(y_true, y_pred, zero_division=0),
            "f1": f1_score(y_true, y_pred, zero_division=0)
        }

        # compute auc only if both classes are present
        if len(set(y_true)) > 1:
            metrics['auc'] = roc_auc_score(y_true, y_prob)
        else:
            metrics['auc'] = float('nan')
        
        return metrics
    
def plot_loss_curve(loss_list, model_name):
    """
    plot loss convergence curve

    Parameters:
    - loss_list (list of float): the loss value of every epoch
    - model_name (str): model name to be used in the plot title and filename
    """
    plt.figure()
    plt.plot(range(1, len(loss_list) + 1), loss_list, marker='o')
    plt.title(f"{model_name} Training Loss Curve")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"training_loss_{model_name.lower()}.png")
    plt.show()

def plot_roc(y_true, y_prob, save_path = "roc_curve.png"):
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.cpu()
    
    if isinstance(y_prob, torch.Tensor):
        y_prob = y_prob.cpu()
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)

    plt.figure()
    plt.plot(fpr, tpr, label=f"ROC Curve (AUC = {roc_auc:.2f})")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Receiver Operating Characteristic")
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.savefig(save_path)
    plt.close()


def plot_metrics_bar(metrics, save_path = "metrics_bar.png"):
    keys = ['accuracy', 'precision', 'recall', 'f1', 'auc']
    values = [metrics[k] for k in keys if k in metrics]

    plt.figure(figsize=(6, 4))
    plt.bar(keys, values, color='skyblue')
    plt.ylim(0, 1)
    plt.ylabel("Score")
    plt.title("Evaluation Metrics")
    for i, v in enumerate(values):
        plt.text(i, v + 0.02, f"{v:.2f}", ha='center')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()








