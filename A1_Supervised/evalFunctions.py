import numpy as np

def calcAccuracy(LPred, LTrue):
    """Calculates prediction accuracy from data labels.

    Args:
        LPred (array): Predicted data labels.
        LTrue (array): Ground truth data labels.

    Retruns:
        acc (float): Prediction accuracy.
    """
    # --------------------------------------------
    # === Your code here =========================
    acc = sum(LPred == LTrue)/len(LPred)
    # ============================================
    
    return acc


def calcConfusionMatrix(LPred, LTrue):
    """Calculates a confusion matrix from data labels.

    Args:
        LPred (array): Predicted data labels.
        LTrue (array): Ground truth data labels.

    Returns:
        cM (array): Confusion matrix, with predicted labels in the rows
            and actual labels in the columns.
    """

    # --------------------------------------------
    # === Your code here =========================
    num_classes = len(set(LTrue))
    cM = np.zeros((num_classes, num_classes), dtype=int)
    for actual, prediction in zip(LTrue, LPred):
        cM[actual, prediction] += 1
    # ============================================
    return cM


def calcAccuracyCM(cM):
    """Calculates prediction accuracy from a confusion matrix.

    Args:
        cM (array): Confusion matrix, with predicted labels in the rows
            and actual labels in the columns.

    Returns:
        acc (float): Prediction accuracy.
    """

    # --------------------------------------------
    # === Your code here =========================
    correct_predictions = np.diag(cM).sum()
    total_predictions = cM.sum()
    acc = correct_predictions / total_predictions
    # ============================================
    
    return acc
