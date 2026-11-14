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

-- 9. Retrieve cases for one specific user (UUID).
SELECT case_id, user_id, case_name, imaging_modality,
       target_vessel, status, created_at
FROM cases
WHERE user_id = :user_id
ORDER BY created_at DESC, case_id DESC;

-- 10. Insert a case for the authenticated user.
INSERT INTO cases (
    user_id, case_name, imaging_modality, target_vessel, status
)
VALUES (
    :user_id, :case_name, :imaging_modality, :target_vessel,
    COALESCE(:status, 'pending')
)
RETURNING case_id, user_id, case_name, imaging_modality,
          target_vessel, status, created_at;

-- 11. Update a case only when it belongs to the authenticated user.

UPDATE cases
SET case_name = COALESCE(:case_name, case_name),
    imaging_modality = COALESCE(:imaging_modality, imaging_modality),
    target_vessel = COALESCE(:target_vessel, target_vessel),
    status = COALESCE(:status, status)
WHERE case_id = :case_id
  AND user_id = :user_id
RETURNING case_id, user_id, case_name, imaging_modality,
          target_vessel, status, created_at;

-- 12. Retrieve one case with ownership enforcement.
SELECT case_id, user_id, case_name, imaging_modality,
       target_vessel, status, created_at
FROM cases
WHERE case_id = :case_id
  AND user_id = :user_id;

-- 13. Retrieve images for a case owned by the authenticated user.
SELECT m.image_id, m.case_id, m.image_name, m.image_path,
       m.image_format, m.image_size, m.uploaded_at
FROM medical_images AS m
JOIN cases AS c ON c.case_id = m.case_id
WHERE m.case_id = :case_id
  AND c.user_id = :user_id
ORDER BY m.uploaded_at DESC, m.image_id DESC;

-- 14. Retrieve one image with ownership enforcement.
SELECT m.image_id, m.case_id, m.image_name, m.image_path,
       m.image_format, m.image_size, m.uploaded_at
FROM medical_images AS m
JOIN cases AS c ON c.case_id = m.case_id
WHERE m.image_id = :image_id
  AND c.user_id = :user_id;

-- 15. Update image metadata only for an image owned by the user.
UPDATE medical_images AS m
SET image_name = COALESCE(:image_name, m.image_name),
    image_path = COALESCE(:image_path, m.image_path),
    image_format = COALESCE(:image_format, m.image_format),
    image_size = COALESCE(:image_size, m.image_size)
FROM cases AS c
WHERE m.case_id = c.case_id
  AND m.image_id = :image_id
  AND c.user_id = :user_id
RETURNING m.image_id, m.case_id, m.image_name, m.image_path,
          m.image_format, m.image_size, m.uploaded_at;

-- 16. Check whether an image has dependent analysis results before deletion.
SELECT
    m.image_id,
    EXISTS (
        SELECT 1 FROM segmentation_results AS s
        WHERE s.image_id = m.image_id
    ) AS has_segmentation_results,
    EXISTS (
        SELECT 1 FROM stenosis_results AS st
        WHERE st.image_id = m.image_id
    ) AS has_stenosis_results
FROM medical_images AS m
JOIN cases AS c ON c.case_id = m.case_id
WHERE m.image_id = :image_id
  AND c.user_id = :user_id;

-- 17. Delete an image only if it belongs to the user and has no dependent
DELETE FROM medical_images AS m
USING cases AS c
WHERE m.case_id = c.case_id
  AND m.image_id = :image_id
  AND c.user_id = :user_id
  AND NOT EXISTS (
      SELECT 1 FROM segmentation_results AS s
      WHERE s.image_id = m.image_id
  )
  AND NOT EXISTS (
      SELECT 1 FROM stenosis_results AS st
      WHERE st.image_id = m.image_id
  )
RETURNING m.image_id;

-- 18. Verify an image record (ownership-scoped).
SELECT m.image_id, m.case_id, m.image_name, m.image_path,
       m.image_format, m.image_size, m.uploaded_at
FROM medical_images AS m
JOIN cases AS c ON c.case_id = m.case_id
WHERE m.image_id = :image_id
  AND c.user_id = :user_id;

