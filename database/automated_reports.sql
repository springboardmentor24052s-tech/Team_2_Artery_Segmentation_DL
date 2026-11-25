-- 1. SEGMENTATION PERFORMANCE REPORT
-- Purpose: Retrieves segmentation results and
-- performance metrics for report generation.
-- Includes model details and evaluation metrics.

CREATE OR REPLACE VIEW report_segmentation_details AS
SELECT
    segmentation_id,
    image_id,
    model_name,
    model_version,
    dice_score,
    iou_score,
    precision_score,
    recall_score,
    f1_score,
    mean_vessel_diameter,
    created_at
FROM segmentation_results;

-- 2. STENOSIS DETECTION REPORT
-- Purpose: Retrieves stenosis detection details
-- for generating stenosis analysis reports.
-- Includes location, severity and confidence.

CREATE OR REPLACE VIEW report_stenosis_details AS
SELECT
    stenosis_id,
    image_id,
    location,
    severity,
    confidence_score,
    estimated_diameter,
    created_at
FROM stenosis_results;


-- 3. CASE SUMMARY REPORT
-- Purpose: Retrieves case information along with
-- uploaded medical image counts.
-- Used for: Case summary report generation.
CREATE OR REPLACE VIEW report_case_summary AS
SELECT
    c.case_id,
    c.user_id,
    c.case_name,
    c.imaging_modality,
    c.target_vessel,
    c.status,
    c.created_at,
    COUNT(mi.image_id) AS total_images
FROM cases c
LEFT JOIN medical_images mi
    ON c.case_id = mi.case_id
GROUP BY
    c.case_id,
    c.user_id,
    c.case_name,
    c.imaging_modality,
    c.target_vessel,
    c.status,
    c.created_at;

-- 4. MODEL PERFORMANCE REPORT
-- Purpose: Calculates average performance metrics
-- for each segmentation model.
-- Used for: Model comparison report generation.

CREATE OR REPLACE VIEW report_model_performance AS
SELECT
    model_name,
    COUNT(*) AS total_predictions,
    ROUND(AVG(dice_score)::numeric, 4) AS average_dice,
    ROUND(AVG(iou_score)::numeric, 4) AS average_iou,
    ROUND(AVG(precision_score)::numeric, 4) AS average_precision,
    ROUND(AVG(recall_score)::numeric, 4) AS average_recall,
    ROUND(AVG(f1_score)::numeric, 4) AS average_f1
FROM segmentation_results
GROUP BY model_name;

-- 5. OVERALL PROJECT REPORT
-- Purpose: Provides an overall summary of project data.
-- Used for: Generating the complete project report.

CREATE OR REPLACE VIEW report_overall_summary AS
SELECT
    (SELECT COUNT(*) FROM users) AS total_users,
    (SELECT COUNT(*) FROM cases) AS total_cases,
    (SELECT COUNT(*) FROM medical_images) AS total_images,
    (SELECT COUNT(*) FROM segmentation_results) AS total_segmentations,
    (SELECT COUNT(*) FROM stenosis_results) AS total_stenosis_detections;

-- 6. REPORT GENERATION HISTORY
-- Purpose: Stores details of generated project reports.
-- Used for: Tracking report generation activities.

CREATE TABLE IF NOT EXISTS report_generation_history (
    report_id SERIAL PRIMARY KEY,
    report_type VARCHAR(100) NOT NULL,
    file_path TEXT,
    generation_status VARCHAR(50) DEFAULT 'Generated',
    generated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 7. CHECK AVAILABLE SEGMENTATION MODELS
-- Purpose: Identifies all segmentation models
-- stored in the database.
-- Used for: Model comparison verification.

SELECT
    model_name,
    COUNT(*) AS total_predictions
FROM segmentation_results
GROUP BY model_name
ORDER BY model_name;
