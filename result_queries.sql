
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

-- 4. Complete model performance analytics

SELECT
    model_name,
    COUNT(*) AS total_results,
    ROUND(AVG(dice_score), 4) AS average_dice,
    ROUND(AVG(iou_score), 4) AS average_iou,
    ROUND(AVG(precision_score), 4) AS average_precision,
    ROUND(AVG(recall_score), 4) AS average_recall,
    ROUND(AVG(f1_score), 4) AS average_f1
FROM segmentation_results
GROUP BY model_name
ORDER BY average_dice DESC NULLS LAST;


-- 5. Stenosis severity distribution

SELECT
    severity,
    COUNT(*) AS total_cases,
    ROUND(AVG(confidence_score), 4) AS average_confidence
FROM stenosis_results
GROUP BY severity
ORDER BY total_cases DESC;

-- 6. Case-wise medical image and analysis report
-- Avoid duplicate combinations of segmentation and stenosis results


WITH segmentation_summary AS (
    SELECT
        image_id,
        COUNT(*) AS total_segmentation_results,
        STRING_AGG(DISTINCT model_name, ', ') AS model_names,
        ROUND(AVG(dice_score), 4) AS average_dice,
        ROUND(AVG(iou_score), 4) AS average_iou
    FROM segmentation_results
    GROUP BY image_id
),
stenosis_summary AS (
    SELECT
        image_id,
        COUNT(*) AS total_stenosis_results,
        STRING_AGG(DISTINCT severity, ', ') AS detected_severities,
        ROUND(AVG(confidence_score), 4) AS average_confidence,
        ROUND(AVG(estimated_diameter), 4) AS average_estimated_diameter
    FROM stenosis_results
    GROUP BY image_id
)
SELECT
    c.case_id,
    c.case_name,
    c.imaging_modality,
    c.target_vessel,
    m.image_id,
    m.image_name,
    COALESCE(s.total_segmentation_results, 0) AS total_segmentation_results,
    s.model_names,
    s.average_dice,
    s.average_iou,
    COALESCE(st.total_stenosis_results, 0) AS total_stenosis_results,
    st.detected_severities,
    st.average_confidence,
    st.average_estimated_diameter
FROM cases c
JOIN medical_images m
    ON c.case_id = m.case_id
LEFT JOIN segmentation_summary s
    ON m.image_id = s.image_id
LEFT JOIN stenosis_summary st
    ON m.image_id = st.image_id
ORDER BY c.case_id, m.image_id;

-- 7. Count images and analysis results for each case

SELECT
    c.case_id,
    c.case_name,
    COUNT(DISTINCT m.image_id) AS total_images,
    COUNT(DISTINCT s.segmentation_id) AS total_segmentation_results,
    COUNT(DISTINCT st.stenosis_id) AS total_stenosis_results
FROM cases c
LEFT JOIN medical_images m
    ON c.case_id = m.case_id
LEFT JOIN segmentation_results s
    ON m.image_id = s.image_id
LEFT JOIN stenosis_results st
    ON m.image_id = st.image_id
GROUP BY c.case_id, c.case_name
ORDER BY c.case_id;


-- 8. Verify database foreign-key relationships

SELECT
    tc.table_name,
    kcu.column_name,
    ccu.table_name AS referenced_table,
    ccu.column_name AS referenced_column,
    tc.constraint_name
FROM information_schema.table_constraints AS tc
JOIN information_schema.key_column_usage AS kcu
    ON tc.constraint_name = kcu.constraint_name
    AND tc.constraint_schema = kcu.constraint_schema
JOIN information_schema.constraint_column_usage AS ccu
    ON ccu.constraint_name = tc.constraint_name
    AND ccu.constraint_schema = tc.constraint_schema
WHERE tc.constraint_type = 'FOREIGN KEY'
AND tc.constraint_schema = 'public'
ORDER BY tc.table_name;



-- 9. Verify all project database tables

SELECT
    table_name
FROM information_schema.tables
WHERE table_schema = 'public'
AND table_type = 'BASE TABLE'
AND table_name IN (
    'users',
    'cases',
    'medical_images',
    'segmentation_results',
    'stenosis_results'
)
ORDER BY table_name;

-- 10. Verify table columns and data types

SELECT
    table_name,
    column_name,
    data_type,
    is_nullable,
    column_default
FROM information_schema.columns
WHERE table_schema = 'public'
AND table_name IN (
    'users',
    'cases',
    'medical_images',
    'segmentation_results',
    'stenosis_results'
)
ORDER BY table_name, ordinal_position;


-- 11. Verify foreign-key data consistency

SELECT
    'Cases without valid users' AS check_name,
    COUNT(*) AS invalid_records
FROM cases c
LEFT JOIN users u
    ON c.user_id = u.user_id
WHERE u.user_id IS NULL

UNION ALL

SELECT
    'Images without valid cases',
    COUNT(*)
FROM medical_images m
LEFT JOIN cases c
    ON m.case_id = c.case_id
WHERE c.case_id IS NULL

UNION ALL

SELECT
    'Segmentation results without valid images',
    COUNT(*)
FROM segmentation_results s
LEFT JOIN medical_images m
    ON s.image_id = m.image_id
WHERE m.image_id IS NULL

UNION ALL

SELECT
    'Stenosis results without valid images',
    COUNT(*)
FROM stenosis_results s
LEFT JOIN medical_images m
    ON s.image_id = m.image_id
WHERE m.image_id IS NULL;

-- 12. Check existing medical image records
SELECT *
FROM medical_images
ORDER BY image_id;

-- 13. Insert medical image details for segmentation testing
INSERT INTO medical_images (
    case_id,
    image_name,
    image_path,
    image_format,
    image_size
)
VALUES (
    1,
    'segmentation_test.png',
    'uploads/segmentation_test.png',
    'PNG',
    204800
)
RETURNING *;

-- 14. Insert segmentation results
INSERT INTO segmentation_results (
    image_id,
    mask_path,
    model_name,
    dice_score,
    iou_score,
    processed_mask_path,
    centerline_path,
    model_version,
    precision_score,
    recall_score,
    f1_score,
    mean_vessel_diameter
)
VALUES (
    2,
    'results/mask.png',
    'U-Net',
    0.95000,
    0.91000,
    'results/processed_mask.png',
    'results/centerline.png',
    '1.0',
    0.94000,
    0.93000,
    0.93500,
    2.5000
)
RETURNING *;

-- 15. Retrieve segmentation results by image ID
SELECT *
FROM segmentation_results
WHERE image_id = 2
ORDER BY created_at DESC;

-- 16. Retrieve segmentation results by case ID
SELECT
    s.segmentation_id,
    s.image_id,
    m.case_id,
    s.model_name,
    s.dice_score,
    s.iou_score,
    s.precision_score,
    s.recall_score,
    s.f1_score,
    s.created_at
FROM segmentation_results s
JOIN medical_images m
    ON s.image_id = m.image_id
WHERE m.case_id = 1
ORDER BY s.created_at DESC;

-- 17. Retrieve the latest segmentation result
SELECT *
FROM segmentation_results
ORDER BY created_at DESC, segmentation_id DESC
LIMIT 1;

-- 18. Insert stenosis detection results
INSERT INTO stenosis_results (
    image_id,
    location,
    severity,
    confidence_score,
    estimated_diameter
)
VALUES (
    2,
    'LAD - Proximal segment',
    'Moderate',
    0.92000,
    1.8000
)
RETURNING *;

-- 19. Retrieve stenosis results by image ID
SELECT *
FROM stenosis_results
WHERE image_id = 2
ORDER BY created_at DESC;

-- 20. Retrieve stenosis results by case ID
SELECT
    st.stenosis_id,
    st.image_id,
    m.case_id,
    st.location,
    st.severity,
    st.confidence_score,
    st.estimated_diameter,
    st.created_at
FROM stenosis_results st
JOIN medical_images m
    ON st.image_id = m.image_id
WHERE m.case_id = 1
ORDER BY st.created_at DESC;

-- 21. Filter stenosis results by severity
SELECT
    stenosis_id,
    image_id,
    location,
    severity,
    confidence_score,
    estimated_diameter,
    created_at
FROM stenosis_results
WHERE severity = 'Moderate'
ORDER BY created_at DESC;

-- 22. Sort stenosis results by confidence score
SELECT
    stenosis_id,
    image_id,
    location,
    severity,
    confidence_score,
    estimated_diameter,
    created_at
FROM stenosis_results
ORDER BY confidence_score DESC;

-- 23. Retrieve the latest stenosis results
SELECT
    stenosis_id,
    image_id,
    location,
    severity,
    confidence_score,
    estimated_diameter,
    created_at
FROM stenosis_results
ORDER BY created_at DESC, stenosis_id DESC
LIMIT 5;

-- 24. Get total registered users
SELECT COUNT(*) AS total_registered_users
FROM users;

-- 25. Get total medical cases
SELECT COUNT(*) AS total_medical_cases
FROM cases;

-- 26. Get total uploaded medical images
SELECT COUNT(*) AS total_uploaded_images
FROM medical_images;

-- 27. Get total stenosis detections
SELECT COUNT(*) AS total_stenosis_detections
FROM stenosis_results;

-- 28. Retrieve recent analysis activity
SELECT
    m.image_id,
    m.image_name,
    'Segmentation' AS analysis_type,
    s.model_name AS model_or_severity,
    s.created_at AS analysis_date
FROM segmentation_results s
JOIN medical_images m
    ON s.image_id = m.image_id

UNION ALL

SELECT
    m.image_id,
    m.image_name,
    'Stenosis' AS analysis_type,
    st.severity AS model_or_severity,
    st.created_at AS analysis_date
FROM stenosis_results st
JOIN medical_images m
    ON st.image_id = m.image_id

ORDER BY analysis_date DESC
LIMIT 10;


