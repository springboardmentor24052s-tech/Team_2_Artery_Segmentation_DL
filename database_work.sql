
CREATE TABLE users (
    user_id SERIAL PRIMARY KEY,
    full_name VARCHAR(100),
    email VARCHAR(150) UNIQUE,
    password_hash TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE cases (
    case_id SERIAL PRIMARY KEY,
    user_id INTEGER REFERENCES users(user_id),
    case_name VARCHAR(150),
    imaging_modality VARCHAR(50),
    target_vessel VARCHAR(100),
    status VARCHAR(30) DEFAULT 'pending',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE medical_images (
    image_id SERIAL PRIMARY KEY,
    case_id INTEGER REFERENCES cases(case_id),
    image_name VARCHAR(255),
    image_path TEXT,
    image_format VARCHAR(10),
    image_size BIGINT,
    uploaded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE segmentation_results (
    segmentation_id SERIAL PRIMARY KEY,
    image_id INTEGER REFERENCES medical_images(image_id),
    mask_path TEXT,
    model_name VARCHAR(100),
    dice_score NUMERIC(6,5),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE stenosis_results (
    stenosis_id SERIAL PRIMARY KEY,
    image_id INTEGER REFERENCES medical_images(image_id),
    location TEXT,
    severity VARCHAR(30),
    confidence_score NUMERIC(6,5),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Add missing segmentation evaluation columns
ALTER TABLE segmentation_results
ADD COLUMN IF NOT EXISTS iou_score NUMERIC(6,5),
ADD COLUMN IF NOT EXISTS processed_mask_path TEXT,
ADD COLUMN IF NOT EXISTS centerline_path TEXT,
ADD COLUMN IF NOT EXISTS model_version VARCHAR(50);
ADD COLUMN IF NOT EXISTS precision_score NUMERIC(6,5),
ADD COLUMN IF NOT EXISTS recall_score NUMERIC(6,5),
ADD COLUMN IF NOT EXISTS f1_score NUMERIC(6,5),
ADD COLUMN IF NOT EXISTS mean_vessel_diameter NUMERIC(10,4);

-- Add vessel measurement column
ALTER TABLE stenosis_results
ADD COLUMN IF NOT EXISTS estimated_diameter NUMERIC(10,4);
