CREATE TABLE users (
    id UUID PRIMARY KEY,
    name VARCHAR(255) NOT NULL,
    email VARCHAR(255) NOT NULL UNIQUE,
    password_hash VARCHAR(512) NOT NULL,
    role VARCHAR(50) NOT NULL,
    is_active BOOLEAN NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE NOT NULL,
    updated_at TIMESTAMP WITH TIME ZONE NOT NULL
);

CREATE INDEX ix_users_email ON users (email);

CREATE TABLE refresh_tokens (
    id UUID PRIMARY KEY,
    user_id UUID NOT NULL,
    token_hash VARCHAR(64) NOT NULL UNIQUE,
    expires_at TIMESTAMP WITH TIME ZONE NOT NULL,
    revoked_at TIMESTAMP WITH TIME ZONE NULL,
    created_at TIMESTAMP WITH TIME ZONE NOT NULL,
    CONSTRAINT fk_refresh_tokens_user_id_users
        FOREIGN KEY (user_id) REFERENCES users (id) ON DELETE CASCADE
);

CREATE INDEX ix_refresh_tokens_token_hash ON refresh_tokens (token_hash);
CREATE INDEX ix_refresh_tokens_expires_at ON refresh_tokens (expires_at);
CREATE INDEX ix_refresh_tokens_user_id ON refresh_tokens (user_id);

CREATE TABLE cases (
    case_id SERIAL PRIMARY KEY,
    user_id UUID NULL REFERENCES users (id),
    case_name VARCHAR(150) NULL,
    imaging_modality VARCHAR(50) NULL,
    target_vessel VARCHAR(100) NULL,
    status VARCHAR(30) NULL,
    created_at TIMESTAMP WITHOUT TIME ZONE NULL
);

CREATE TABLE medical_images (
    image_id SERIAL PRIMARY KEY,
    case_id INTEGER NULL REFERENCES cases (case_id),
    image_name VARCHAR(255) NULL,
    image_path TEXT NULL,
    image_format VARCHAR(10) NULL,
    image_size BIGINT NULL,
    uploaded_at TIMESTAMP WITHOUT TIME ZONE NULL
);

CREATE TABLE segmentation_results (
    segmentation_id SERIAL PRIMARY KEY,
    image_id INTEGER NULL REFERENCES medical_images (image_id),
    mask_path TEXT NULL,
    model_name VARCHAR(100) NULL,
    dice_score NUMERIC(6,5) NULL,
    iou_score NUMERIC(6,5) NULL,
    processed_mask_path TEXT NULL,
    centerline_path TEXT NULL,
    model_version VARCHAR(50) NULL,
    precision_score NUMERIC(6,5) NULL,
    recall_score NUMERIC(6,5) NULL,
    f1_score NUMERIC(6,5) NULL,
    mean_vessel_diameter NUMERIC(10,4) NULL,
    created_at TIMESTAMP WITHOUT TIME ZONE NULL,
    overlay_path TEXT NULL
);
CREATE TABLE stenosis_results (
    stenosis_id SERIAL PRIMARY KEY,
    image_id INTEGER NULL REFERENCES medical_images (image_id),
    location TEXT NULL,
    severity VARCHAR(30) NULL,
    confidence_score NUMERIC(6,5) NULL,
    estimated_diameter NUMERIC(10,4) NULL,
    created_at TIMESTAMP WITHOUT TIME ZONE NULL
);
-- ==========================================
-- EXPLAINABLE AI RESULTS
-- ==========================================

-- Create Explainable AI results table
-- Stores Grad-CAM heatmap and explanation details.
CREATE TABLE IF NOT EXISTS explainability_results (
    explanation_id SERIAL PRIMARY KEY,
    image_id INTEGER REFERENCES medical_images(image_id),
    stenosis_id INTEGER REFERENCES stenosis_results(stenosis_id),
    method VARCHAR(100),
    heatmap_path TEXT,
    overlay_path TEXT,
    model_name VARCHAR(100),
    created_at TIMESTAMP WITHOUT TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- ==========================================
-- STENOSIS VISUALIZATION
-- ==========================================

-- Add column to store stenosis visualization image path.

ALTER TABLE stenosis_results
ADD COLUMN IF NOT EXISTS visualization_path TEXT;


-- Add column to store annotated angiography image path.

ALTER TABLE stenosis_results
ADD COLUMN IF NOT EXISTS annotated_image_path TEXT;
