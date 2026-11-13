
from pathlib import Path
import cv2
import numpy as np

root = Path("backend/app/data/arcade/arcade/arcade")

syntax_path = root / "syntax/val/images/1.png"
stenosis_path = root / "stenosis/val/images/1.png"

syntax = cv2.imread(str(syntax_path), cv2.IMREAD_GRAYSCALE)
stenosis = cv2.imread(str(stenosis_path), cv2.IMREAD_GRAYSCALE)

if syntax is None:
    print("Syntax image 1.png not found.")

if stenosis is None:
    print("Stenosis image 1.png not found.")

if syntax is not None and stenosis is not None:
    print("Syntax shape:", syntax.shape)
    print("Stenosis shape:", stenosis.shape)
    print("Identical images:", np.array_equal(syntax, stenosis))
    print("Mean absolute difference:",
          np.mean(np.abs(
              syntax.astype(np.float32) -
              stenosis.astype(np.float32)
          )))
