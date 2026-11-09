
from pathlib import Path
from backend.app.services.segmentation import predict_stenosis

image_path = Path(
    "backend/app/data/arcade/arcade/arcade/stenosis/val/images/1.png"
)

result = predict_stenosis(str(image_path))

print("Return type:", type(result))
print("Returned value:")

if isinstance(result, dict):
    for key, value in result.items():
        print(
            f"Key: {key}",
            f"| Type: {type(value)}",
            f"| Shape: {getattr(value, 'shape', 'N/A')}"
        )
else:
    print(result)
