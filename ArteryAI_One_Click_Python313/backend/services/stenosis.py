import cv2
import numpy as np

def estimate_narrowing(mask):
    '''
    Preliminary geometric estimate from a binary segmentation mask.
    This is NOT validated clinical stenosis grading.
    It compares the narrowest row-wise width with the median width
    among detected rows; angiographic orientation/calibration matters.
    '''
    binary = (mask > 0).astype("uint8")
    if binary.sum() < 20:
        return None, "Segmentation mask contains too little vessel area."
    widths=[]
    for row in binary:
        xs=np.where(row > 0)[0]
        if len(xs) >= 2:
            widths.append(float(xs.max()-xs.min()+1))
    if len(widths) < 5:
        return None, "Could not identify a sufficiently continuous vessel profile."
    arr=np.array(widths, dtype=float)
    reference=float(np.percentile(arr, 75))
    narrow=float(np.percentile(arr, 10))
    if reference <= 0:
        return None, "Unable to estimate reference vessel width."
    pct=float(np.clip((1.0 - narrow/reference)*100.0, 0, 99))
    return round(pct, 1), "Preliminary geometric estimate; not a clinical diagnosis."
