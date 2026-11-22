# ARTERY SEGMENTATION AND STENOSIS DETECTION USING ADVANCED DEEP LEARNING TECHNIQUES

## 1. Project Overview

The project **Artery Segmentation and Stenosis Detection Using Advanced Deep Learning Techniques** focuses on analyzing coronary angiography images to identify coronary arteries and detect stenosis (narrowing of blood vessels) using deep learning techniques.

The system aims to support medical image analysis through artery segmentation, stenosis detection, vessel measurements and visualization of analysis results.

This repository contains the database-related implementation and SQL operations developed as part of the project.

## 2. My Contribution – Database Development

My primary responsibility was PostgreSQL database design, SQL query development, database management, dashboard data preparation, automated report database support, security inspection and performance testing.

The database supports storing and managing:

- User registration and authentication information.
- Medical case details.
- Medical angiography image information.
- Artery segmentation results.
- Stenosis detection results.
- Explainability-related information.
- Automated report-generation history.

## 3. Technologies Used

| Technology | Purpose |
|---|---|
| PostgreSQL 18 | Relational database management |
| SQL | Database operations and query development |
| pgAdmin | Database administration |
| psql | Executing and testing SQL queries |
| Git | Version control |
| GitHub | Code management and collaboration |

## 4. Database Design and Implementation

The project database is named `artery_db`.

The following seven tables were implemented and maintained.

| Table Name | Description |
|---|---|
| users | Stores user registration information |
| cases | Stores medical case details |
| medical_images | Stores uploaded image information and file paths |
| segmentation_results | Stores artery segmentation results and evaluation metrics |
| stenosis_results | Stores stenosis detection information and severity |
| explainability_results | Stores explainability-related information |
| report_generation_history | Maintains report-generation history |

### 4.1 Database Relationships

- A user can have multiple medical cases.
- A case can contain multiple medical images.
- Medical images are associated with segmentation results.
- Medical images are associated with stenosis detection results.
- Explainability information is maintained separately.
- Report-generation history stores information about generated reports.

## 5. SQL Query Implementation

SQL queries were developed to perform database operations and retrieve project information.

### 5.1 Authentication Queries

Implemented queries for:

- Checking whether an email already exists.
- Retrieving user information for login verification.
- Viewing registered users.
- Updating user information.

### 5.2 Case and Image Management

Implemented queries for:

- Viewing all medical cases.
- Retrieving cases belonging to a user.
- Viewing uploaded medical images.
- Retrieving images using case ID.
- Updating case information.
- Deleting medical image records when required.

### 5.3 Segmentation and Stenosis Results

Implemented queries for:

- Retrieving segmentation results.
- Viewing Dice and IoU scores.
- Comparing segmentation model performance using average metrics.
- Retrieving stenosis detection results.
- Viewing stenosis severity distribution.
- Counting segmentation and stenosis records.

## 6. Dashboard Development – Database Support

Six SQL views were created to provide aggregated information for the project dashboard.

| View Name | Purpose |
|---|---|
| dashboard_summary | Displays overall counts of users, cases, images and AI results |
| dashboard_case_status | Displays case distribution according to status |
| dashboard_stenosis_severity | Displays stenosis severity distribution |
| dashboard_model_performance | Displays segmentation model performance metrics |
| dashboard_vessel_analytics | Provides vessel-related analytical information |
| dashboard_monthly_cases | Displays monthly case statistics |

These views prepare summarized database information for dashboard integration.

## 7. Automated Report Generation – Database Support

Database views and a report history table were implemented to support automated report generation.

### 7.1 Report Views

| View Name | Purpose |
|---|---|
| report_segmentation_details | Retrieves segmentation result details |
| report_stenosis_details | Retrieves stenosis detection details |
| report_case_summary | Provides case-related information |
| report_model_performance | Provides model performance metrics |
| report_overall_summary | Provides overall project statistics |

### 7.2 Report Generation History

The `report_generation_history` table was created to maintain report-related information.

It stores:

- Report ID
- Report type
- File path
- Generation status
- Generated timestamp

The database structure is prepared for backend report-generation integration. Actual PDF, CSV and JSON file generation requires backend implementation.

## 8. Stenosis Detection and Explainability Database Support

During this stage, the database was extended to support stenosis detection and Explainable AI (XAI).

### 8.1 Stenosis Results Table Enhancement

The `stenosis_results` table was enhanced with two additional columns:

```sql
ALTER TABLE stenosis_results
ADD COLUMN visualization_path VARCHAR(255),
ADD COLUMN annotated_image_path VARCHAR(255);
```

These columns store references to visualization and annotated image files.

### 8.2 Stenosis Data Management

SQL operations were implemented to insert, update and retrieve stenosis detection records.

A sample record was verified with the following information:

| Field | Sample Value |
|---|---|
| Stenosis ID | 1 |
| Image ID | 2 |
| Location | LAD - Proximal segment |
| Severity | Moderate |
| Confidence Score | 0.92 |
| Estimated Diameter | 1.8 mm |

### 8.3 Explainability Results

The `explainability_results` table was created to support explainability-related information.

Example file references:

- `results/stenosis/stenosis_visualization_1.png`
- `results/stenosis/annotated_image_1.png`

Only file paths are stored in the database, not the actual image files.

## 9. Database Security Testing

Database-level security testing was performed using PostgreSQL to inspect database permissions, access control and security configuration.

A total of 10 security checks were implemented and executed.

| No. | Security Check | Purpose |
|---|---|---|
| 1 | Database privilege check | Inspects database-level permissions |
| 2 | Schema privilege check | Checks schema access permissions |
| 3 | Table ownership check | Identifies table owners |
| 4 | PUBLIC schema access check | Inspects permissions granted to PUBLIC |
| 5 | Row Level Security check | Checks whether RLS is enabled or forced |
| 6 | Password encryption check | Inspects the configured password encryption method |
| 7 | SSL connection check | Checks SSL status of the current connection |
| 8 | Logging configuration check | Inspects database logging settings |
| 9 | Table access permissions | Checks table-level access permissions |
| 10 | Login roles and privileges | Inspects login roles and their privileges |

### 9.1 Security Testing Findings

| Security Setting | Observed Result |
|---|---|
| Password encryption setting | `scram-sha-256` |
| Current connection SSL | Disabled |
| Row Level Security | Not enabled on inspected tables |
| Table ownership | All seven tables owned by `postgres` |
| Login roles | Only `postgres` appeared in the login-role check |

These checks provided an overview of the current database security configuration and identified areas requiring further review.

**Note:** These activities represent database security inspection, not complete security hardening. No security configuration changes were made during these checks.

## 10. Database Performance Testing

Performance testing was performed to inspect database indexes, table sizes, record counts, query execution plans and database activity.

A total of 12 SQL checks were executed.

### 10.1 Performance Testing Activities

| No. | Test | Purpose |
|---|---|---|
| 1 | Existing index inspection | Identifies existing database indexes |
| 2 | Table size analysis | Checks storage size of each table |
| 3 | Record count analysis | Counts records in project tables |
| 4 | Case retrieval performance | Tests case retrieval by user ID |
| 5 | Medical image retrieval | Tests image retrieval by case ID |
| 6 | Segmentation retrieval | Tests segmentation retrieval by image ID |
| 7 | Stenosis retrieval | Tests stenosis retrieval by image ID |
| 8 | Dashboard summary performance | Checks dashboard summary execution |
| 9 | Dashboard case status performance | Checks case status view execution |
| 10 | Database connection monitoring | Checks active and idle connections |
| 11 | Long-running query inspection | Identifies queries running longer than five minutes |
| 12 | Database size inspection | Checks total database storage size |

### 10.2 Query Execution Results

The following execution times were recorded using PostgreSQL `EXPLAIN ANALYZE`.

| Query | Execution Time |
|---|---:|
| Case retrieval | 0.050 ms |
| Medical image retrieval | 0.060 ms |
| Segmentation retrieval | 0.081 ms |
| Stenosis retrieval | 0.080 ms |
| Dashboard summary | 0.196 ms |
| Dashboard case status | 0.133 ms |

### 10.3 Performance Testing Findings

- Eight existing indexes were identified.
- Seven project tables were inspected.
- All tested retrieval and dashboard queries executed successfully.
- No active queries running longer than five minutes were found.
- One active database connection was observed during connection monitoring.
- Total database size was 8,558 kB (approximately 8.36 MB).

The execution results provide a performance baseline for the current sample dataset. Additional load and stress testing would be required to evaluate performance under larger workloads.

## 11. SQL Files and Implementation Details

The following SQL files contain the database queries and implementations developed for this project.

| File Name | Description |
|---|---|
| [database_work.sql](database_work.sql) | Database creation, table definitions and general database operations |
| [auth_queries.sql](auth_queries.sql) | User authentication and user-related queries |
| [result_queries.sql](result_queries.sql) | Segmentation and stenosis result queries |
| [dashboard_queries.sql](dashboard_queries.sql) | Dashboard SQL views and analytical queries |
| [automated_reports.sql](automated_reports.sql) | Automated report views and report history table |
| [security_checks.sql](security_checks.sql) | Database security inspection queries |
| [performance.sql](performance.sql) | Database performance testing queries |

Each SQL file contains the corresponding database operations and testing queries.

## 12. Database Testing Summary

| Activity | Status |
|---|---|
| Database schema and table implementation | Completed |
| Authentication SQL queries | Completed |
| Case and image management queries | Completed |
| Segmentation and stenosis result queries | Completed |
| Dashboard SQL views | Completed |
| Automated report database support | Completed |
| Stenosis and explainability database support | Completed |
| Database security inspection | Completed |
| Database performance testing | Completed |

## 13. Conclusion

The database development work established the PostgreSQL foundation required for the Artery Segmentation and Stenosis Detection project.

The work included database table implementation, authentication queries, medical case and image management, segmentation and stenosis result storage, dashboard views, automated report database support, security inspection and performance testing.

The database is structured to support integration with the backend application and frontend dashboard.

Further integration and application-level testing will help validate the complete project workflow.

---

**Database:** artery_db  
**Database Technology:** PostgreSQL 18  
**Project:** Artery Segmentation and Stenosis Detection Using Advanced Deep Learning Techniques
