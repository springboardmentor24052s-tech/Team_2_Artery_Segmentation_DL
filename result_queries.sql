
-- 1. Count segmentation results
SELECT COUNT(*) AS total_segmentation_results
FROM segmentation_results;

-- 2. Count stenosis results
SELECT COUNT(*) AS total_stenosis_results
FROM stenosis_results;

-- 3. Compare segmentation model performance
-- Compare segmentation models using Dice score
SELECT
    model_name,
    AVG(dice_score) AS average_dice_score
FROM segmentation_results
GROUP BY model_name
ORDER BY average_dice_score DESC NULLS LAST;
