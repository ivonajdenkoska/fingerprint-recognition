from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score


def get_model_accuracy(true_y, pred_y):
    return accuracy_score(true_y, pred_y)


def get_model_confusion_matrix(true_y, pred_y):
    return confusion_matrix(true_y, pred_y).ravel()


def get_precision_score(true_y, pred_y):
    return precision_score(true_y, pred_y)


def get_recall_score(true_y, pred_y):
    return recall_score(true_y, pred_y)


def calculate_ratios(tp, fp, fn, tn):
    true_pos_rate = tp / (tp + fn)  # TPR / Recall / Sensitivity
    false_pos_rate = fp / (fp + tn)  # FPR = 1 - Specificity
    return true_pos_rate, false_pos_rate

