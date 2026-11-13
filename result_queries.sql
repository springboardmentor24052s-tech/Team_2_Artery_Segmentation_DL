-- 1. Count segmentation results
SELECT COUNT(*) AS total_segmentation_results
FROM segmentation_results AS s
JOIN medical_images AS m ON s.image_id = m.image_id
JOIN cases AS c ON m.case_id = c.case_id
WHERE (:user_id IS NULL OR c.user_id = :user_id);


-- 2. Count stenosis results
SELECT COUNT(*) AS total_stenosis_results
FROM stenosis_results AS st
JOIN medical_images AS m ON st.image_id = m.image_id
JOIN cases AS c ON m.case_id = c.case_id
WHERE (:user_id IS NULL OR c.user_id = :user_id);


-- 3. Compare segmentation model performance
SELECT s.model_name, AVG(s.dice_score) AS average_dice_score
FROM segmentation_results AS s
JOIN medical_images AS m ON s.image_id = m.image_id
JOIN cases AS c ON m.case_id = c.case_id
WHERE (:user_id IS NULL OR c.user_id = :user_id)
GROUP BY s.model_name
ORDER BY average_dice_score DESC NULLS LAST;


-- 4. Complete model performance analytics
SELECT
    s.model_name,
    COUNT(*) AS total_results,
    ROUND(AVG(s.dice_score), 4) AS average_dice,
    ROUND(AVG(s.iou_score), 4) AS average_iou,
    ROUND(AVG(s.precision_score), 4) AS average_precision,
    ROUND(AVG(s.recall_score), 4) AS average_recall,
    ROUND(AVG(s.f1_score), 4) AS average_f1
FROM segmentation_results AS s
JOIN medical_images AS m ON s.image_id = m.image_id
JOIN cases AS c ON m.case_id = c.case_id
WHERE (:user_id IS NULL OR c.user_id = :user_id)
GROUP BY s.model_name
ORDER BY average_dice DESC;


-- 5. Stenosis severity distribution
SELECT
    st.severity,
    COUNT(*) AS total_cases,
    ROUND(AVG(st.confidence_score), 4) AS average_confidence
FROM stenosis_results AS st
JOIN medical_images AS m ON st.image_id = m.image_id
JOIN cases AS c ON m.case_id = c.case_id
WHERE (:user_id IS NULL OR c.user_id = :user_id)
GROUP BY st.severity
ORDER BY total_cases DESC;


-- 6. Case-wise medical image and analysis report
WITH segmentation_summary AS (
    SELECT
        s.image_id,
        COUNT(*) AS total_segmentation_results,
        STRING_AGG(DISTINCT s.model_name, ', ') AS model_names,
        ROUND(AVG(s.dice_score), 4) AS average_dice,
        ROUND(AVG(s.iou_score), 4) AS average_iou
    FROM segmentation_results AS s
    JOIN medical_images AS m ON s.image_id = m.image_id
    JOIN cases AS c ON m.case_id = c.case_id
    WHERE (:user_id IS NULL OR c.user_id = :user_id)
    GROUP BY s.image_id
),
stenosis_summary AS (
    SELECT
        st.image_id,
        COUNT(*) AS total_stenosis_results,
        STRING_AGG(DISTINCT st.severity, ', ') AS detected_severities,
        ROUND(AVG(st.confidence_score), 4) AS average_confidence,
        ROUND(AVG(st.estimated_diameter), 4) AS average_estimated_diameter
    FROM stenosis_results AS st
    JOIN medical_images AS m ON st.image_id = m.image_id
    JOIN cases AS c ON m.case_id = c.case_id
    WHERE (:user_id IS NULL OR c.user_id = :user_id)
    GROUP BY st.image_id
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
FROM cases AS c
JOIN medical_images AS m ON c.case_id = m.case_id
LEFT JOIN segmentation_summary AS s ON m.image_id = s.image_id
LEFT JOIN stenosis_summary AS st ON m.image_id = st.image_id
WHERE (:user_id IS NULL OR c.user_id = :user_id)
  AND (:case_id IS NULL OR c.case_id = :case_id)
ORDER BY c.case_id, m.image_id;


-- 7. Count images and analysis results for each case
SELECT
    c.case_id,
    c.case_name,
    COUNT(DISTINCT m.image_id) AS total_images,
    COUNT(DISTINCT s.segmentation_id) AS total_segmentation_results,
    COUNT(DISTINCT st.stenosis_id) AS total_stenosis_results
FROM cases AS c
LEFT JOIN medical_images AS m ON c.case_id = m.case_id
LEFT JOIN segmentation_results AS s ON m.image_id = s.image_id
LEFT JOIN stenosis_results AS st ON m.image_id = st.image_id
WHERE (:user_id IS NULL OR c.user_id = :user_id)
GROUP BY c.case_id, c.case_name
ORDER BY c.case_id;

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


-- 9. Verify project database tables
SELECT table_name
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
  AND (:table_name IS NULL OR table_name = LOWER(:table_name))
ORDER BY table_name, ordinal_position;


-- 11. Verify foreign-key data consistency
SELECT
    'Cases without valid users' AS check_name,
    COUNT(*) AS invalid_records
FROM cases AS c
LEFT JOIN users AS u ON c.user_id = u.id
WHERE u.id IS NULL

UNION ALL

SELECT
    'Images without valid cases',
    COUNT(*)
FROM medical_images AS m
LEFT JOIN cases AS c ON m.case_id = c.case_id
WHERE c.case_id IS NULL

UNION ALL

SELECT
    'Segmentation results without valid images',
    COUNT(*)
FROM segmentation_results AS s
LEFT JOIN medical_images AS m ON s.image_id = m.image_id
WHERE m.image_id IS NULL

UNION ALL

SELECT
    'Stenosis results without valid images',
    COUNT(*)
FROM stenosis_results AS st
LEFT JOIN medical_images AS m ON st.image_id = m.image_id
WHERE m.image_id IS NULL;


-- 12. Check existing medical image records visible to the user
SELECT
    m.image_id,
    m.case_id,
    m.image_name,
    m.image_path,
    m.image_format,
    m.image_size,
    m.uploaded_at
FROM medical_images AS m
JOIN cases AS c ON m.case_id = c.case_id
WHERE (:user_id IS NULL OR c.user_id = :user_id)
ORDER BY m.image_id;


-- 13. Insert medical image details after the backend verifies case ownership
INSERT INTO medical_images (
    case_id,
    image_name,
    image_path,
    image_format,
    image_size
)
VALUES (
    :case_id,
    :image_name,
    :image_path,
    :image_format,
    :image_size
);

-- 14. Insert segmentation results after the backend verifies image ownership
INSERT INTO segmentation_results (
    image_id,
    mask_path,
    overlay_path,
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
    :image_id,
    :mask_path,
    :overlay_path,
    :model_name,
    :dice_score,
    :iou_score,
    :processed_mask_path,
    :centerline_path,
    :model_version,
    :precision_score,
    :recall_score,
    :f1_score,
    :mean_vessel_diameter
);


-- 15. Retrieve segmentation results by image ID after image ownership verification
SELECT *
FROM segmentation_results
WHERE image_id = :image_id
ORDER BY segmentation_id;


-- 16. Retrieve segmentation results by case ID after case ownership verification
SELECT
    s.segmentation_id,
    s.image_id,
    m.case_id,
    s.mask_path,
    s.overlay_path,
    s.model_name,
    s.dice_score,
    s.iou_score,
    s.processed_mask_path,
    s.centerline_path,
    s.model_version,
    s.precision_score,
    s.recall_score,
    s.f1_score,
    s.mean_vessel_diameter,
    s.created_at
FROM segmentation_results AS s
JOIN medical_images AS m ON s.image_id = m.image_id
WHERE m.case_id = :case_id
ORDER BY s.segmentation_id;


-- 17. Retrieve the latest segmentation result visible to the user
SELECT s.*
FROM segmentation_results AS s
JOIN medical_images AS m ON s.image_id = m.image_id
JOIN cases AS c ON m.case_id = c.case_id
WHERE (:user_id IS NULL OR c.user_id = :user_id)
ORDER BY s.created_at DESC, s.segmentation_id DESC
LIMIT 1;


-- 18. Insert stenosis detection results after the backend verifies image ownership
INSERT INTO stenosis_results (
    image_id,
    location,
    severity,
    confidence_score,
    estimated_diameter
)
VALUES (
    :image_id,
    :location,
    :severity,
    :confidence_score,
    :estimated_diameter
);


-- 19. Retrieve stenosis results by image ID after image ownership verification
SELECT *
FROM stenosis_results
WHERE image_id = :image_id
ORDER BY stenosis_id;

-- 20. Retrieve stenosis results by case ID after case ownership verification
SELECT
    st.stenosis_id,
    st.image_id,
    m.case_id,
    st.location,
    st.severity,
    st.confidence_score,
    st.estimated_diameter,
    st.created_at
FROM stenosis_results AS st
JOIN medical_images AS m ON st.image_id = m.image_id
WHERE m.case_id = :case_id
ORDER BY st.stenosis_id;


-- 21. Filter stenosis results by severity
SELECT st.*
FROM stenosis_results AS st
JOIN medical_images AS m ON st.image_id = m.image_id
JOIN cases AS c ON m.case_id = c.case_id
WHERE LOWER(st.severity) = LOWER(:severity)
  AND (:user_id IS NULL OR c.user_id = :user_id)
ORDER BY st.stenosis_id;


-- 22. Sort stenosis results by confidence score
SELECT st.*
FROM stenosis_results AS st
JOIN medical_images AS m ON st.image_id = m.image_id
JOIN cases AS c ON m.case_id = c.case_id
WHERE (:user_id IS NULL OR c.user_id = :user_id)
ORDER BY
    CASE WHEN :descending THEN st.confidence_score END DESC NULLS LAST,
    CASE WHEN NOT :descending THEN st.confidence_score END ASC NULLS LAST;


-- 23. Retrieve the latest stenosis result visible to the user
SELECT st.*
FROM stenosis_results AS st
JOIN medical_images AS m ON st.image_id = m.image_id
JOIN cases AS c ON m.case_id = c.case_id
WHERE (:user_id IS NULL OR c.user_id = :user_id)
ORDER BY st.created_at DESC, st.stenosis_id DESC
LIMIT 1;


-- 24. Get total registered users visible to the user
SELECT COUNT(*) AS total_registered_users
FROM users
WHERE (:user_id IS NULL OR id = :user_id);


-- 25. Get total medical cases visible to the user
SELECT COUNT(*) AS total_medical_cases
FROM cases
WHERE (:user_id IS NULL OR user_id = :user_id);


-- 26. Get total uploaded medical images visible to the user
SELECT COUNT(*) AS total_uploaded_images
FROM medical_images AS m
JOIN cases AS c ON m.case_id = c.case_id
WHERE (:user_id IS NULL OR c.user_id = :user_id);


-- 27. Get total stenosis detections visible to the user
SELECT COUNT(*) AS total_stenosis_detections
FROM stenosis_results AS st
JOIN medical_images AS m ON st.image_id = m.image_id
JOIN cases AS c ON m.case_id = c.case_id
WHERE (:user_id IS NULL OR c.user_id = :user_id);


-- 28. Retrieve recent analysis activity visible to the user
SELECT
    m.image_id,
    m.image_name,
    'Segmentation' AS analysis_type,
    s.model_name AS model_or_severity,
    s.created_at AS analysis_date
FROM segmentation_results AS s
JOIN medical_images AS m ON s.image_id = m.image_id
JOIN cases AS c ON m.case_id = c.case_id
WHERE (:user_id IS NULL OR c.user_id = :user_id)

UNION ALL

SELECT
    m.image_id,
    m.image_name,
    'Stenosis' AS analysis_type,
    st.severity AS model_or_severity,
    st.created_at AS analysis_date
FROM stenosis_results AS st
JOIN medical_images AS m ON st.image_id = m.image_id
JOIN cases AS c ON m.case_id = c.case_id
WHERE (:user_id IS NULL OR c.user_id = :user_id)

ORDER BY analysis_date DESC
LIMIT 10;


