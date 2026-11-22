-- 1. DASHBOARD SUMMARY VIEW
-- Purpose: Displays overall project statistics
-- for dashboard summary cards.

CREATE OR REPLACE VIEW dashboard_summary AS
SELECT
    (SELECT COUNT(*) FROM users) AS total_users,
    (SELECT COUNT(*) FROM cases) AS total_cases,
    (SELECT COUNT(*) FROM medical_images) AS total_images,
    (SELECT COUNT(*) FROM segmentation_results)
        AS total_segmentations,
    (SELECT COUNT(*) FROM stenosis_results)
        AS total_stenosis_detections;


-- 2. CASE STATUS ANALYTICS VIEW
-- Purpose: Displays the distribution of cases
-- according to their current status.

CREATE OR REPLACE VIEW dashboard_case_status AS
SELECT
    status,
    COUNT(*) AS total_cases
FROM cases
GROUP BY status;


-- 3. STENOSIS SEVERITY ANALYTICS VIEW
-- Purpose: Displays the distribution of stenosis
-- detections based on severity levels.

CREATE OR REPLACE VIEW dashboard_stenosis_severity AS
SELECT
    severity,
    COUNT(*) AS total_detections
FROM stenosis_results
GROUP BY severity;


-- 4. SEGMENTATION MODEL PERFORMANCE VIEW
-- Purpose: Calculates average performance metrics
-- for comparing segmentation models.

CREATE OR REPLACE VIEW dashboard_model_performance AS
SELECT
    model_name,
    COUNT(*) AS total_predictions,
    ROUND(AVG(dice_score)::numeric, 4) AS avg_dice,
    ROUND(AVG(iou_score)::numeric, 4) AS avg_iou,
    ROUND(AVG(precision_score)::numeric, 4) AS avg_precision,
    ROUND(AVG(recall_score)::numeric, 4) AS avg_recall,
    ROUND(AVG(f1_score)::numeric, 4) AS avg_f1
FROM segmentation_results
GROUP BY model_name;


-- 5. VESSEL LOCATION STENOSIS ANALYTICS
-- Purpose: Analyzes stenosis detections by location
-- and severity for dashboard analytics.

CREATE OR REPLACE VIEW dashboard_vessel_analytics AS
SELECT
    location,
    severity,
    COUNT(*) AS total_detections,
    ROUND(AVG(confidence_score), 4) AS average_confidence
FROM stenosis_results
GROUP BY location, severity;


-- 6. MONTHLY CASE ANALYTICS
-- Purpose: Displays monthly case registration
-- statistics for dashboard trend charts.

CREATE OR REPLACE VIEW dashboard_monthly_cases AS
SELECT
    DATE_TRUNC('month', created_at)::date AS month,
    COUNT(*) AS total_cases
FROM cases
GROUP BY DATE_TRUNC('month', created_at);


-- =====================================================
-- 7. TEST DASHBOARD VIEWS
-- Purpose: Retrieves data from all dashboard views.
-- =====================================================

SELECT * FROM dashboard_summary;

SELECT * FROM dashboard_case_status;

SELECT * FROM dashboard_stenosis_severity;

SELECT * FROM dashboard_model_performance;

SELECT * FROM dashboard_vessel_analytics;

SELECT * FROM dashboard_monthly_cases;
