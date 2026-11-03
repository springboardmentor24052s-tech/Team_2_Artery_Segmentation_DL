
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

SELECT
    c.case_id,
    c.case_name,
    c.imaging_modality,
    c.target_vessel,
    m.image_id,
    m.image_name,
    s.model_name,
    s.dice_score,
    s.iou_score,
    st.severity,
    st.confidence_score,
    st.estimated_diameter
FROM cases c
JOIN medical_images m
    ON c.case_id = m.case_id
LEFT JOIN segmentation_results s
    ON m.image_id = s.image_id
LEFT JOIN stenosis_results st
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
