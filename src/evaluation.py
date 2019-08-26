from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score


def get_model_accuracy(true_y, pred_y):
    return accuracy_score(true_y, pred_y)


def get_model_confusion_matrix(true_y, pred_y):
    return confusion_matrix(true_y, pred_y).ravel()


def calculate_ratios(tp, fp, fn, tn):
    true_pos_rate = tp / (tp + fn)  # TPR / Recall / Sensitivity
    false_pos_rate = fp / (fp + tn)  # FPR = 1 - Specificity
    return true_pos_rate, false_pos_rate

