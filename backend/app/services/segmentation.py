
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn


# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DEVICE = torch.device("cpu")

SYNTAX_MODEL_PATH = (
    PROJECT_ROOT / "results" / "unet_syntax_improved.pth"
)

STENOSIS_MODEL_PATH = (
    PROJECT_ROOT / "results" / "attention_unet_stenosis_v4.pth"
)

SYNTAX_THRESHOLD = 0.5
STENOSIS_THRESHOLD = 0.20


# --------------------------------------------------
# SHARED BUILDING BLOCK
# --------------------------------------------------

class DoubleConv(nn.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.ReLU(inplace=True),

            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.layers(x)


# --------------------------------------------------
# SYNTAX U-NET
# --------------------------------------------------

class UNet(nn.Module):

    def __init__(self):
        super().__init__()

        self.enc1 = DoubleConv(1, 16)
        self.enc2 = DoubleConv(16, 32)
        self.enc3 = DoubleConv(32, 64)

        self.pool = nn.MaxPool2d(2)

        self.bottleneck = DoubleConv(64, 128)

        self.up3 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec3 = DoubleConv(128, 64)

        self.up2 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.dec2 = DoubleConv(64, 32)

        self.up1 = nn.ConvTranspose2d(32, 16, 2, stride=2)
        self.dec1 = DoubleConv(32, 16)

        self.output = nn.Conv2d(16, 1, 1)

    def forward(self, x):

        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))

        b = self.bottleneck(self.pool(e3))

        d3 = self.up3(b)
        d3 = self.dec3(torch.cat([d3, e3], dim=1))

        d2 = self.up2(d3)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))

        d1 = self.up1(d2)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))

        return self.output(d1)


# --------------------------------------------------
# ATTENTION GATE
# --------------------------------------------------

class AttentionGate(nn.Module):

    def __init__(
        self,
        gate_channels,
        skip_channels,
        intermediate_channels
    ):
        super().__init__()

        self.gate_conv = nn.Sequential(
            nn.Conv2d(
                gate_channels,
                intermediate_channels,
                1
            ),
            nn.BatchNorm2d(intermediate_channels)
        )

        self.skip_conv = nn.Sequential(
            nn.Conv2d(
                skip_channels,
                intermediate_channels,
                1
            ),
            nn.BatchNorm2d(intermediate_channels)
        )

        self.attention = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Conv2d(intermediate_channels, 1, 1),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

    def forward(self, gate, skip):

        gate_features = self.gate_conv(gate)
        skip_features = self.skip_conv(skip)

        attention_map = self.attention(
            gate_features + skip_features
        )

        return skip * attention_map


# --------------------------------------------------
# ATTENTION U-NET
# --------------------------------------------------

class AttentionUNet(nn.Module):

    def __init__(self):
        super().__init__()

        self.enc1 = DoubleConv(1, 16)
        self.enc2 = DoubleConv(16, 32)
        self.enc3 = DoubleConv(32, 64)

        self.pool = nn.MaxPool2d(2)

        self.bottleneck = DoubleConv(64, 128)

        self.up3 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.att3 = AttentionGate(64, 64, 32)
        self.dec3 = DoubleConv(128, 64)

        self.up2 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.att2 = AttentionGate(32, 32, 16)
        self.dec2 = DoubleConv(64, 32)

        self.up1 = nn.ConvTranspose2d(32, 16, 2, stride=2)
        self.att1 = AttentionGate(16, 16, 8)
        self.dec1 = DoubleConv(32, 16)

        self.output = nn.Conv2d(16, 1, 1)

    def forward(self, x):

        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))

        b = self.bottleneck(self.pool(e3))

        d3 = self.up3(b)
        a3 = self.att3(d3, e3)
        d3 = self.dec3(torch.cat([d3, a3], dim=1))

        d2 = self.up2(d3)
        a2 = self.att2(d2, e2)
        d2 = self.dec2(torch.cat([d2, a2], dim=1))

        d1 = self.up1(d2)
        a1 = self.att1(d1, e1)
        d1 = self.dec1(torch.cat([d1, a1], dim=1))

        return self.output(d1)


# --------------------------------------------------
# LOAD TRAINED MODELS
# --------------------------------------------------

def load_trained_model(model_class, checkpoint_path):

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Model checkpoint not found: {checkpoint_path}"
        )

    trained_model = model_class().to(DEVICE)

    state_dict = torch.load(
        checkpoint_path,
        map_location=DEVICE,
        weights_only=True
    )

    trained_model.load_state_dict(
        state_dict,
        strict=True
    )

    trained_model.eval()

    return trained_model


syntax_model = load_trained_model(
    UNet,
    SYNTAX_MODEL_PATH
)

stenosis_model = load_trained_model(
    AttentionUNet,
    STENOSIS_MODEL_PATH
)


# --------------------------------------------------
# IMAGE PREPROCESSING
# --------------------------------------------------

def preprocess_image(image_path):

    image = cv2.imread(
        str(image_path),
        cv2.IMREAD_GRAYSCALE
    )

    if image is None:
        raise ValueError(
            "Unable to read the uploaded image."
        )

    height, width = image.shape

    normalized = image.astype(np.float32) / 255.0

    tensor = torch.from_numpy(
        normalized
    ).unsqueeze(0).unsqueeze(0)

    return image, tensor, height, width


# --------------------------------------------------
# COMMON PREDICTION FUNCTION
# --------------------------------------------------

def run_prediction(image_path, trained_model, threshold):

    image, tensor, height, width = preprocess_image(image_path)

    # Run model prediction.
    with torch.no_grad():

        logits = trained_model(tensor)

        probabilities = torch.sigmoid(logits)

    # Convert probability tensor to NumPy array.
    probability_map = (
        probabilities.squeeze().cpu().numpy()
    )

    # Validate model output.
    if probability_map.ndim != 2:
        raise ValueError(
            "Unexpected model output dimensions. "
            f"Expected a 2D probability map, got {probability_map.shape}."
        )

    if not np.isfinite(probability_map).all():
        raise ValueError(
            "Model produced non-finite probability values."
        )

    # Resize probability map to original image dimensions.
    # Linear interpolation preserves probability values.
    if probability_map.shape != (height, width):

        probability_map = cv2.resize(
            probability_map.astype(np.float32),
            (width, height),
            interpolation=cv2.INTER_LINEAR
        )

    # Keep probabilities within valid range.
    probability_map = np.clip(
        probability_map,
        0.0,
        1.0
    ).astype(np.float32)

    # Generate binary mask from the resized probability map.
    binary_mask = (
        probability_map >= threshold
    ).astype(np.uint8) * 255

    # Final dimension validation.
    if (
        binary_mask.shape != (height, width)
        or probability_map.shape != (height, width)
    ):
        raise ValueError(
            "Prediction dimensions do not match original image dimensions."
        )

    return {
        "mask": binary_mask,
        "probability_map": probability_map,
        "original_image": image,
        "height": height,
        "width": width
    }


# --------------------------------------------------
# PUBLIC PREDICTION FUNCTIONS
# --------------------------------------------------

def predict_segmentation(image_path):

    return run_prediction(
        image_path,
        syntax_model,
        SYNTAX_THRESHOLD
    )


def predict_stenosis(image_path):

    return run_prediction(
        image_path,
        stenosis_model,
        STENOSIS_THRESHOLD
    )
