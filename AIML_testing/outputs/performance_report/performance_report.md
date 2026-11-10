# Segmentation Performance Report

## Evaluation Setup

- Dataset: ARCADE processed stenosis test set
- Common test images: 300
- Evaluation threshold recorded: 0.3
- Models: U-Net and Attention U-Net
- Metrics: Dice Score and Intersection over Union (IoU)

**Verification:** Ensure both source evaluation runs used threshold 0.3.
The threshold is not stored in the current per-image CSV files.

## Model Comparison

| Metric | U-Net | Attention U-Net | Difference (Attention - U-Net) |
|---|---:|---:|---:|
| Mean Dice | 38.30% | 50.90% | +12.61 percentage points |
| Mean IoU | 27.31% | 37.47% | +10.16 percentage points |

## Additional Statistics

| Statistic | U-Net Dice | Attention U-Net Dice | U-Net IoU | Attention U-Net IoU |
|---|---:|---:|---:|---:|
| Median | 0.3947 | 0.5644 | 0.2459 | 0.3931 |
| Standard deviation | 0.2717 | 0.2425 | 0.2182 | 0.2075 |
| Minimum Dice | 0.0000 | 0.0000 | - | - |
| Maximum Dice | 0.9548 | 0.9453 | - | - |

## Results

- Higher mean Dice Score: **Attention U-Net**
- Higher mean IoU Score: **Attention U-Net**

A higher Dice or IoU indicates better overlap with the reference masks
on this test set. Review visual predictions as well. These results alone
do not establish clinical suitability.

## Generated Files

- `model_summary.csv`: summary statistics.
- `per_image_comparison.csv`: metrics for each common test image.
- `performance_comparison.png`: comparison chart.
