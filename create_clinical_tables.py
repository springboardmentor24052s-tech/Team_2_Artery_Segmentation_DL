import sys
sys.path.insert(0, "backend")
from app.db.session import SessionLocal
from sqlalchemy import text

db = SessionLocal()
try:
    # Create clinical tables (users already exists from ORM)
    tables = [
        """
        CREATE TABLE IF NOT EXISTS cases (
            case_id SERIAL PRIMARY KEY,
            user_id UUID REFERENCES users(id),
            case_name VARCHAR(150),
            imaging_modality VARCHAR(50),
            target_vessel VARCHAR(100),
            status VARCHAR(30) DEFAULT 'pending',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        """,
        """
        CREATE TABLE IF NOT EXISTS medical_images (
            image_id SERIAL PRIMARY KEY,
            case_id INTEGER REFERENCES cases(case_id),
            image_name VARCHAR(255),
            image_path TEXT,
            image_format VARCHAR(10),
            image_size BIGINT,
            uploaded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        """,
        """
        CREATE TABLE IF NOT EXISTS segmentation_results (
            segmentation_id SERIAL PRIMARY KEY,
            image_id INTEGER REFERENCES medical_images(image_id),
            mask_path TEXT,
            model_name VARCHAR(100),
            dice_score NUMERIC(6,5),
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        """,
        """
        CREATE TABLE IF NOT EXISTS stenosis_results (
            stenosis_id SERIAL PRIMARY KEY,
            image_id INTEGER REFERENCES medical_images(image_id),
            location TEXT,
            severity VARCHAR(30),
            confidence_score NUMERIC(6,5),
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        """,
        # Add missing segmentation evaluation columns
        """
        ALTER TABLE segmentation_results
        ADD COLUMN IF NOT EXISTS iou_score NUMERIC(6,5),
        ADD COLUMN IF NOT EXISTS processed_mask_path TEXT,
        ADD COLUMN IF NOT EXISTS centerline_path TEXT,
        ADD COLUMN IF NOT EXISTS model_version VARCHAR(50),
        ADD COLUMN IF NOT EXISTS precision_score NUMERIC(6,5),
        ADD COLUMN IF NOT EXISTS recall_score NUMERIC(6,5),
        ADD COLUMN IF NOT EXISTS f1_score NUMERIC(6,5),
        ADD COLUMN IF NOT EXISTS mean_vessel_diameter NUMERIC(10,4);
        """,
        # Add vessel measurement column
        """
        ALTER TABLE stenosis_results
        ADD COLUMN IF NOT EXISTS estimated_diameter NUMERIC(10,4);
        """,
    ]

    for sql in tables:
        try:
            db.execute(text(sql))
            db.commit()
            print(f"OK: {sql.strip()[:80]}...")
        except Exception as e:
            print(f"ERROR: {sql.strip()[:80]}... -> {e}")
            db.rollback()

    # Verify tables
    print('\n=== VERIFIED TABLES ===')
    result = db.execute(text("""
        SELECT table_name FROM information_schema.tables
        WHERE table_schema = 'public' AND table_type = 'BASE TABLE'
        ORDER BY table_name
    """))
    for row in result:
        print(f"  {row[0]}")

    # Verify FKs
    print('\n=== VERIFIED FOREIGN KEYS ===')
    result = db.execute(text("""
        SELECT tc.table_name, kcu.column_name, ccu.table_name AS referenced_table,
               ccu.column_name AS referenced_column
        FROM information_schema.table_constraints AS tc
        JOIN information_schema.key_column_usage AS kcu
            ON tc.constraint_name = kcu.constraint_name
        JOIN information_schema.constraint_column_usage AS ccu
            ON ccu.constraint_name = tc.constraint_name
        WHERE tc.constraint_type = 'FOREIGN KEY'
        AND tc.constraint_schema = 'public'
        ORDER BY tc.table_name
    """))
    for row in result:
        print(f"  {row[0]}.{row[1]} -> {row[2]}.{row[3]}")

finally:
    db.close()
