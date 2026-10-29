
-- 1. Check whether an email already exists
SELECT EXISTS (
    SELECT 1
    FROM users
    WHERE email = 'test@example.com'
) AS email_exists;

-- 2. Retrieve a user for login verification
SELECT user_id, full_name, email, password_hash
FROM users
WHERE email = 'test@example.com';

-- 3. View registered users (without password hashes)
SELECT user_id, full_name, email, created_at
FROM users
ORDER BY user_id;

-- 4. Update a user's name
UPDATE users
SET full_name = 'Updated Name'
WHERE user_id = 1;

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

-- 8. View stenosis results
SELECT *
FROM stenosis_results
ORDER BY stenosis_id;
