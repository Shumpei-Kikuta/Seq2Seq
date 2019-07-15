import numpy as np
import torch

def calc_accuracy(pred_Y, Y):
    """pred_Yと、Yが列方向に一致しているか"""
    same_num = np.equal(pred_Y, Y).all(axis=1).sum()
    same_rate = same_num / len(Y)
    return same_rate