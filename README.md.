# ARTERY SEGMENTATION AND STENOSIS DETECTION USING ADVANCED DEEP LEARNING TECHNIQUES

## 1. Project Overview

The project **Artery Segmentation and Stenosis Detection Using Advanced Deep Learning Techniques** focuses on analyzing coronary angiography images to identify coronary arteries and detect stenosis (narrowing of blood vessels) using deep learning techniques.

The system aims to support medical image analysis by providing artery segmentation, stenosis detection, vessel measurements and visualization of analysis results.

This repository contains the database-related implementation and SQL operations developed as part of the project.

## 2. My Contribution – Database Development

My primary responsibility in the project is PostgreSQL database design, implementation, query development, database testing and performance analysis.

The database supports storing and managing:

- User registration and authentication information.
- Patient case details.
- Medical angiography images and file information.
- Artery segmentation results.
- Stenosis detection results.
- Explainability results.
- Automated report-generation history.

## 3. Technologies Used

| Technology | Purpose |
|---|---|
| PostgreSQL 18 | Relational database management |
| SQL | Database operations and query development |
| pgAdmin | Database administration |
| psql | Executing and testing SQL queries |
| Git and GitHub | Version control and collaboration |

## 4. Database Architecture

The project database is named `artery_db`.

The following tables were created and maintained to support the application's database requirements.

| Table Name | Description |
|---|---|
| users | Stores user registration information |
| cases | Stores medical case details |
| medical_images | Stores uploaded image information and file paths |
| segmentation_results | Stores artery segmentation results and model evaluation metrics |
| stenosis_results | Stores stenosis detection information and severity |
| explainability_results | Stores explainability-related result information |
| report_generation_history | Maintains report-generation history |

### 4.1 Database Relationships

- A user can have multiple medical cases.
- A case can contain multiple medical images.
- Medical images are associated with segmentation results.
- Medical images are associated with stenosis detection results.
- Explainability information is maintained separately.
- Report-generation history records report-related information.

## 5. Database Query Implementation

SQL queries were developed to perform database operations and retrieve project information.

### 5.1 Authentication Queries

Implemented database queries for:

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

SQL views were created to provide aggregated information for the project dashboard.

The following six views were implemented:

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

Database views and a history table were implemented to support automated report generation.

### 7.1 Report Views

| View Name | Purpose |
|---|---|
| report_segmentation_details | Retrieves segmentation result details |
| report_stenosis_details | Retrieves stenosis detection details |
| report_case_summary | Provides case-related information |
| report_model_performance | Provides model performance metrics |
| report_overall_summary | Provides overall project statistics |

### 7.2 Report Generation History

The `report_generation_history` table was created to maintain information about generated reports.

It stores:

- Report ID
- Report type
- File path
- Generation status
- Generated timestamp

The database structure is prepared for backend report-generation integration. Actual PDF, CSV and JSON generation requires backend implementation.

## 8. Stenosis Detection and Explainability Database Support

During this stage, the database was extended to support stenosis detection and Explainable AI (XAI).

### 8.1 Stenosis Results Table Enhancement

The `stenosis_results` table was enhanced with the following columns:

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

The database also stores references to visualization files.

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

### 9.1 Important Findings

- Password encryption setting: `scram-sha-256`.
- Current database connection: SSL disabled.
- Row Level Security: Not enabled on the inspected tables.
- All seven tables were owned by the `postgres` role.
- Only the `postgres` login role appeared in the login-role check.

These checks provided an overview of the current database security configuration and identified areas requiring further review.

**Note:** These activities represent database security inspection, not complete security hardening. No security configuration changes were made as part of these checks.

## 10. Database Performance Testing

Performance testing was performed to inspect database size, existing indexes, record counts, query execution plans and database activity.

A total of 12 SQL checks were executed.

### 10.1 Performance Testing Activities

| No. | Test | Purpose |
|---|---|---|
| 1 | Existing index inspection | Identifies existing database indexes |
| 2 | Table size analysis | Checks the storage size of each table |
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

### 10.3 Performance Observations

- Eight existing indexes were identified.
- Seven project tables were inspected.
- All tested retrieval and dashboard queries executed successfully.
- No active queries running longer than five minutes were found.
- One active database connection was observed during connection monitoring.
- Total database size was approximately 8.36 MB.

The execution results provide a baseline for the current sample dataset. Additional load testing would be required to evaluate performance under larger workloads.

## 11. Database Testing Summary

The following database-related activities were completed:

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

## 12. Repository Structure

The database-related SQL files are maintained in the GitHub repository.

```text
Team_2_Artery_Segmentation_DL/
│
├── README.md
├── LICENSE
│
└── database/
    ├── database_work.sql
    ├── auth_queries.sql
    ├── result_queries.sql
    ├── dashboard_queries.sql
    ├── automated_reports.sql
    ├── security_checks.sql
    └── performance_testing.sql
```

## 13. Conclusion

The database development work established the PostgreSQL foundation required for the Artery Segmentation and Stenosis Detection project.

The work included database table implementation, authentication queries, medical case and image management, segmentation and stenosis result storage, dashboard views, automated report database support, security inspection and performance testing.

The database is structured to support integration with the backend application and frontend dashboard.

Further integration and application-level testing will help validate the complete project workflow.

---

**Database Technology:** PostgreSQL 18  
**Database Name:** artery_db  
**Project:** Artery Segmentation and Stenosis Detection Using Advanced Deep Learning Techniques
