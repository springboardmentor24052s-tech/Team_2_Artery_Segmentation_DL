-- 1.Check existing indexes in the public schema
SELECT
    tablename,
    indexname,
    indexdef
FROM pg_indexes
WHERE schemaname = 'public'
ORDER BY tablename, indexname;

--2. Check table sizes in the public schema
SELECT
    tablename,
    pg_size_pretty(
        pg_total_relation_size(
            schemaname || '.' || tablename
        )
    ) AS total_size
FROM pg_tables
WHERE schemaname = 'public'
ORDER BY pg_total_relation_size(
    schemaname || '.' || tablename
) DESC;

-- 3. Check the number of records in each project table
SELECT 'users' AS table_name, COUNT(*) AS total_records FROM users
UNION ALL
SELECT 'cases', COUNT(*) FROM cases
UNION ALL
SELECT 'medical_images', COUNT(*) FROM medical_images
UNION ALL
SELECT 'segmentation_results', COUNT(*) FROM segmentation_results
UNION ALL
SELECT 'stenosis_results', COUNT(*) FROM stenosis_results
UNION ALL
SELECT 'explainability_results', COUNT(*) FROM explainability_results
UNION ALL
SELECT 'report_generation_history', COUNT(*) FROM report_generation_history;

--4. Test case retrieval performance by user ID

EXPLAIN ANALYZE
SELECT *
FROM cases
WHERE user_id = 1;

--5. Test medical image retrieval performance by case ID

EXPLAIN ANALYZE
SELECT *
FROM medical_images
WHERE case_id = 1;

--6.Test segmentation result retrieval performance

EXPLAIN ANALYZE
SELECT *
FROM segmentation_results
WHERE image_id = 2;

--7. Test stenosis result retrieval performance

EXPLAIN ANALYZE
SELECT *
FROM stenosis_results
WHERE image_id = 2;

-- 8. Test dashboard summary view performance

EXPLAIN ANALYZE
SELECT *
FROM dashboard_summary;

--9. Test dashboard case status view performance

EXPLAIN ANALYZE
SELECT *
FROM dashboard_case_status;

-- 10. Check current database connections

SELECT
    datname AS database_name,
    COUNT(*) AS total_connections,
    COUNT(*) FILTER (
        WHERE state = 'active'
    ) AS active_connections,
    COUNT(*) FILTER (
        WHERE state = 'idle'
    ) AS idle_connections
FROM pg_stat_activity
WHERE datname = 'artery_db'
GROUP BY datname;

-- 11. Check long-running active queries

SELECT
    pid,
    usename,
    state,
    query,
    NOW() - query_start AS query_duration
FROM pg_stat_activity
WHERE datname = 'artery_db'
  AND state = 'active'
  AND query_start IS NOT NULL
  AND NOW() - query_start > INTERVAL '5 minutes';

-- 12. Check total database size

SELECT
    current_database() AS database_name,
    pg_size_pretty(
        pg_database_size(current_database())
    ) AS total_database_size;