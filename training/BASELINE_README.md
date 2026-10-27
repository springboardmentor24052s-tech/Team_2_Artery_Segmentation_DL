ArteryAI — baseline U-Net
What this baseline does
- Trains a binary U-Net on dataset/syntax.
- Rasterizes COCO polygon annotations into masks.
- Unions anatomical category polygons and excludes the category named stenosis.
- Reports validation/test loss, Dice, and IoU; saves the best checkpoint, metrics, training history, and prediction/ground-truth masks.
This is an initial research baseline. It does not yet detect or quantify stenosis and is not a clinical tool.