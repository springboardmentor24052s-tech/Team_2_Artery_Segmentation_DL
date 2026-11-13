
-- 1. Check whether an email already exists
SELECT EXISTS (
    SELECT 1
    FROM users
    WHERE email = 'test@example.com'
) AS email_exists;

-- 2. Retrieve a user for login verification
SELECT id, name, email, password_hash
FROM users
WHERE email = 'test@example.com';

-- 3. View registered users (without password hashes)
SELECT id, name, email, created_at
FROM users
ORDER BY id;

-- 4. Update a user's name
UPDATE users
SET name = :full_name
WHERE id = :user_id;

-- 5. View all cases
SELECT
    case_id,
    user_id,
    case_name,
    imaging_modality,
    target_vessel,
    status,
    created_at
FROM cases
ORDER BY case_id;

-- 6. View all medical images
SELECT *
FROM medical_images
ORDER BY image_id;

-- 7. View segmentation results
SELECT *
FROM segmentation_results
ORDER BY segmentation_id;

--8. View stenosis results
SELECT *
FROM stenosis_results
ORDER BY stenosis_id;

-- 9. Retrieve cases by user ID
SELECT
    case_id,
    user_id,
    case_name,
    imaging_modality,
    target_vessel,
    status,
    created_at
FROM cases
WHERE user_id = 1
ORDER BY created_at DESC;


-- 10. Insert a new case
INSERT INTO cases (
    user_id,
    case_name,
    imaging_modality,
    target_vessel,
    status
)
VALUES (
    1,
    'Coronary Angiography Case',
    'Coronary Angiography',
    'LAD',
    'pending'
)
RETURNING *;


-- 11. Update existing case details
UPDATE cases
SET
    case_name = 'Updated Angiography Case',
    imaging_modality = 'Coronary Angiography',
    target_vessel = 'RCA',
    status = 'completed'
WHERE case_id = 1
RETURNING *;


-- 12. Retrieve a specific case using case ID
SELECT
    case_id,
    user_id,
    case_name,
    imaging_modality,
    target_vessel,
    status,
    created_at
FROM cases
WHERE case_id = 1;

-- 13. Retrieve medical images by case ID
SELECT
    image_id,
    case_id,
    image_name,
    image_path,
    image_format,
    image_size,
    uploaded_at
FROM medical_images
WHERE case_id = 1
ORDER BY uploaded_at DESC;

-- 14. Retrieve a specific medical image using image ID
SELECT
    image_id,
    case_id,
    image_name,
    image_path,
    image_format,
    image_size,
    uploaded_at
FROM medical_images
WHERE image_id = 2;

-- 15. Update medical image metadata
UPDATE medical_images
SET
    image_name = 'updated_angiography.png',
    image_format = 'PNG',
    image_size = 307200
WHERE image_id = 2
RETURNING *;

-- 16. Check whether medical image can be deleted
SELECT
    image_id,
    EXISTS (
        SELECT 1
        FROM segmentation_results s
        WHERE s.image_id = m.image_id
    ) AS has_segmentation_results,
    EXISTS (
        SELECT 1
        FROM stenosis_results st
        WHERE st.image_id = m.image_id
    ) AS has_stenosis_results
FROM medical_images m
WHERE image_id = 2;

-- 17. Delete a medical image when permitted
DELETE FROM medical_images
WHERE image_id = 2
RETURNING *;

-- 18. Verify medical image deletion
SELECT *
FROM medical_images
WHERE image_id = 2;