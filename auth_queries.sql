-- 1. Check whether an email exists
SELECT EXISTS (SELECT 1 FROM users WHERE email = :email) AS email_exists;


-- 2. Retrieve a user for login verification
SELECT id, name, email, password_hash FROM users WHERE email = :email;


-- 3. List registered users
SELECT id, name, email, created_at FROM users ORDER BY id;


-- 4. Update a user's name
UPDATE users SET name = :full_name WHERE id = :user_id;


-- 5. List cases with ownership filtering
SELECT case_id, user_id, case_name, imaging_modality, target_vessel, status, created_at
FROM cases
WHERE (:user_id IS NULL OR user_id = :user_id)
ORDER BY case_id;


-- 6. List medical images with ownership filtering
SELECT m.image_id, m.case_id, m.image_name, m.image_path, m.image_format, m.image_size, m.uploaded_at
FROM medical_images AS m
JOIN cases AS c ON m.case_id = c.case_id
WHERE (:user_id IS NULL OR c.user_id = :user_id)
ORDER BY m.image_id;


-- 7. List segmentation results with ownership filtering
SELECT s.*
FROM segmentation_results AS s
JOIN medical_images AS m ON s.image_id = m.image_id
JOIN cases AS c ON m.case_id = c.case_id
WHERE (:user_id IS NULL OR c.user_id = :user_id)
ORDER BY s.segmentation_id;


-- 8. List stenosis results with ownership filtering
SELECT st.*
FROM stenosis_results AS st
JOIN medical_images AS m ON st.image_id = m.image_id
JOIN cases AS c ON m.case_id = c.case_id
WHERE (:user_id IS NULL OR c.user_id = :user_id)
ORDER BY st.stenosis_id;
