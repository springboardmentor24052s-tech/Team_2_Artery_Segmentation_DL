from typing import List, Dict, Any, Optional
import uuid
from sqlalchemy.orm import Session
from sqlalchemy import text

from app.schemas.clinical_schemas import (
    CaseCreateRequest, CaseUpdateRequest,
    MedicalImageCreateRequest, MedicalImageUpdateRequest,
    SegmentationResultCreateRequest,
    StenosisResultCreateRequest
)

class SQLClinicalService:
    """
    Executes SQL queries for clinical data functionality.
    Reuses queries directly using parameterized SQL inputs.
    """

    # --- auth_queries.sql Queries ---

    @staticmethod
    def check_email_exists(db: Session, email: str) -> bool:
        """
        Executes Query 1 from auth_queries.sql:
        SELECT EXISTS (
            SELECT 1 FROM users WHERE email = :email
        ) AS email_exists;
        """
        sql = text("""
            SELECT EXISTS (
                SELECT 1
                FROM users
                WHERE email = :email
            ) AS email_exists;
        """)
        result = db.execute(sql, {"email": email.strip().lower()}).scalar()
        return bool(result)

    @staticmethod
    def get_user_by_email_for_verification(db: Session, email: str, user_id: Optional[uuid.UUID] = None) -> Optional[Dict[str, Any]]:
        """
        Executes Query 2 from auth_queries.sql:
        SELECT id, name, email, password_hash
        FROM users
        WHERE email = :email;

        Restricted to administrators (user_id is None).
        """
        if user_id is not None:
            return None
        sql = text("""
            SELECT id, name, email, password_hash
            FROM users
            WHERE email = :email;
        """)
        row = db.execute(sql, {"email": email.strip().lower()}).fetchone()
        if not row:
            return None
        return dict(row._mapping)

    @staticmethod
    def get_registered_users(db: Session, user_id: Optional[uuid.UUID] = None) -> List[Dict[str, Any]]:
        """
        Executes Query 3 from auth_queries.sql:
        SELECT id, name, email, created_at
        FROM users
        ORDER BY id;

        Restricted to administrators (user_id is None).
        """
        if user_id is not None:
            return []
        sql = text("""
            SELECT id, name, email, created_at
            FROM users
            ORDER BY id;
        """)
        result = db.execute(sql)
        return [dict(row._mapping) for row in result.fetchall()]

    @staticmethod
    def update_user_name(db: Session, user_id: Any, full_name: str, current_user_id: Optional[uuid.UUID] = None, is_admin: bool = False) -> bool:
        """
        Executes Query 4 from auth_queries.sql:
        UPDATE users
        SET name = :full_name
        WHERE id = :user_id;

        Only the user themselves or an administrator may update a name.
        """
        if not is_admin and current_user_id is not None and str(current_user_id) != str(user_id):
            return False
        sql = text("""
            UPDATE users
            SET name = :full_name
            WHERE id = :user_id;
        """)
        result = db.execute(sql, {"full_name": full_name.strip(), "user_id": user_id})
        db.commit()
        return result.rowcount > 0

    @staticmethod
    def get_all_cases(db: Session, user_id: Optional[uuid.UUID] = None) -> List[Dict[str, Any]]:
        """
        Executes Query 5 from auth_queries.sql:
        SELECT case_id, user_id, case_name, imaging_modality, target_vessel, status, created_at
        FROM cases ORDER BY case_id;

        Regular users see only their own cases; administrators see all.
        """
        sql = text("""
            SELECT
                case_id,
                user_id,
                case_name,
                imaging_modality,
                target_vessel,
                status,
                created_at
            FROM cases
            WHERE (:user_id IS NULL OR user_id = :user_id)
            ORDER BY case_id;
        """)
        result = db.execute(sql, {"user_id": str(user_id) if user_id else None})
        return [dict(row._mapping) for row in result.fetchall()]

    @staticmethod
    def get_all_medical_images(db: Session, user_id: Optional[uuid.UUID] = None) -> List[Dict[str, Any]]:
        """
        Executes Query 6 from auth_queries.sql:
        SELECT * FROM medical_images ORDER BY image_id;

        Regular users see only images from their own cases; administrators see all.
        """
        sql = text("""
            SELECT m.image_id, m.case_id, m.image_name, m.image_path, m.image_format, m.image_size, m.uploaded_at
            FROM medical_images m
            JOIN cases c ON m.case_id = c.case_id
            WHERE (:user_id IS NULL OR c.user_id = :user_id)
            ORDER BY m.image_id;
        """)
        result = db.execute(sql, {"user_id": str(user_id) if user_id else None})
        return [dict(row._mapping) for row in result.fetchall()]

    @staticmethod
    def get_all_segmentation_results(db: Session, user_id: Optional[uuid.UUID] = None) -> List[Dict[str, Any]]:
        """
        Executes Query 7 from auth_queries.sql:
        SELECT * FROM segmentation_results ORDER BY segmentation_id;

        Regular users see only results from their own images/cases; administrators see all.
        """
        sql = text("""
            SELECT s.*
            FROM segmentation_results s
            JOIN medical_images m ON s.image_id = m.image_id
            JOIN cases c ON m.case_id = c.case_id
            WHERE (:user_id IS NULL OR c.user_id = :user_id)
            ORDER BY s.segmentation_id;
        """)
        result = db.execute(sql, {"user_id": str(user_id) if user_id else None})
        return [dict(row._mapping) for row in result.fetchall()]

    @staticmethod
    def get_all_stenosis_results(db: Session, user_id: Optional[uuid.UUID] = None) -> List[Dict[str, Any]]:
        """
        Executes Query 8 from auth_queries.sql:
        SELECT * FROM stenosis_results ORDER BY stenosis_id;

        Regular users see only results from their own images/cases; administrators see all.
        """
        sql = text("""
            SELECT st.*
            FROM stenosis_results st
            JOIN medical_images m ON st.image_id = m.image_id
            JOIN cases c ON m.case_id = c.case_id
            WHERE (:user_id IS NULL OR c.user_id = :user_id)
            ORDER BY st.stenosis_id;
        """)
        result = db.execute(sql, {"user_id": str(user_id) if user_id else None})
        return [dict(row._mapping) for row in result.fetchall()]

    # --- result_queries.sql Queries ---

    @staticmethod
    def get_segmentation_count(db: Session, user_id: Optional[uuid.UUID] = None) -> int:
        """
        Executes Query 1 from result_queries.sql:
        SELECT COUNT(*) AS total_segmentation_results FROM segmentation_results;

        Regular users see only counts for their own results; administrators see all.
        """
        sql = text("""
            SELECT COUNT(*) AS total_segmentation_results
            FROM segmentation_results s
            JOIN medical_images m ON s.image_id = m.image_id
            JOIN cases c ON m.case_id = c.case_id
            WHERE (:user_id IS NULL OR c.user_id = :user_id);
        """)
        result = db.execute(sql, {"user_id": str(user_id) if user_id else None}).scalar()
        return int(result or 0)

    @staticmethod
    def get_stenosis_count(db: Session, user_id: Optional[uuid.UUID] = None) -> int:
        """
        Executes Query 2 from result_queries.sql:
        SELECT COUNT(*) AS total_stenosis_results FROM stenosis_results;

        Regular users see only counts for their own results; administrators see all.
        """
        sql = text("""
            SELECT COUNT(*) AS total_stenosis_results
            FROM stenosis_results st
            JOIN medical_images m ON st.image_id = m.image_id
            JOIN cases c ON m.case_id = c.case_id
            WHERE (:user_id IS NULL OR c.user_id = :user_id);
        """)
        result = db.execute(sql, {"user_id": str(user_id) if user_id else None}).scalar()
        return int(result or 0)

    @staticmethod
    def get_model_performance(db: Session, user_id: Optional[uuid.UUID] = None) -> List[Dict[str, Any]]:
        """
        Executes Query 3 from result_queries.sql:
        SELECT model_name, AVG(dice_score) AS average_dice_score
        FROM segmentation_results
        GROUP BY model_name
        ORDER BY average_dice_score DESC NULLS LAST;

        Regular users see only their own results; administrators see all.
        """
        sql = text("""
            SELECT
                s.model_name,
                AVG(s.dice_score) AS average_dice_score
            FROM segmentation_results s
            JOIN medical_images m ON s.image_id = m.image_id
            JOIN cases c ON m.case_id = c.case_id
            WHERE (:user_id IS NULL OR c.user_id = :user_id)
            GROUP BY s.model_name
            ORDER BY average_dice_score DESC NULLS LAST;
        """)
        result = db.execute(sql, {"user_id": str(user_id) if user_id else None})
        output = []
        for row in result.fetchall():
            m = row._mapping
            output.append({
                "model_name": m["model_name"] or "Unknown",
                "average_dice_score": float(m["average_dice_score"]) if m["average_dice_score"] is not None else None
            })
        return output

    # =====================================================================
    # Additional Clinical Operations (Parameterized SQL)
    # =====================================================================

    # --- Case Management ---

    @staticmethod
    def create_case(db: Session, case_in: CaseCreateRequest, user_id) -> Dict[str, Any]:
        """
        Inserts a new clinical case into cases table.
        user_id is derived from the authenticated user, not from the request body.
        """
        sql = text("""
            INSERT INTO cases (user_id, case_name, imaging_modality, target_vessel, status)
            VALUES (:user_id, :case_name, :imaging_modality, :target_vessel, :status);
        """)
        db.execute(sql, {
            "user_id": user_id,
            "case_name": case_in.case_name.strip(),
            "imaging_modality": case_in.imaging_modality,
            "target_vessel": case_in.target_vessel,
            "status": case_in.status or "pending"
        })
        db.commit()

        # Retrieve newly inserted row (using max case_id or exact lookup)
        fetch_sql = text("""
            SELECT case_id, user_id, case_name, imaging_modality, target_vessel, status, created_at
            FROM cases
            WHERE case_name = :case_name
            ORDER BY case_id DESC
            LIMIT 1;
        """)
        row = db.execute(fetch_sql, {"case_name": case_in.case_name.strip()}).fetchone()
        return dict(row._mapping)

    @staticmethod
    def get_case_by_id(db: Session, case_id: int, user_id: Optional[uuid.UUID] = None) -> Optional[Dict[str, Any]]:
        """
        Retrieves a single case by case_id.
        Regular users can only retrieve their own cases; administrators can retrieve any.
        """
        sql = text("""
            SELECT case_id, user_id, case_name, imaging_modality, target_vessel, status, created_at
            FROM cases
            WHERE case_id = :case_id
            AND (:user_id IS NULL OR user_id = :user_id);
        """)
        row = db.execute(sql, {"case_id": case_id, "user_id": str(user_id) if user_id else None}).fetchone()
        if not row:
            return None
        return dict(row._mapping)

    @staticmethod
    def get_cases_by_user_id(db: Session, user_id: Any) -> List[Dict[str, Any]]:
        """
        Retrieves cases created by a specific user_id.
        """
        sql = text("""
            SELECT case_id, user_id, case_name, imaging_modality, target_vessel, status, created_at
            FROM cases
            WHERE user_id = :user_id
            ORDER BY case_id;
        """)
        result = db.execute(sql, {"user_id": user_id})
        return [dict(row._mapping) for row in result.fetchall()]

    @staticmethod
    def update_case(db: Session, case_id: int, case_in: CaseUpdateRequest, user_id: Optional[uuid.UUID] = None) -> Optional[Dict[str, Any]]:
        """
        Updates case details dynamically using parameterized SQL.
        Regular users can only update their own cases; administrators can update any.
        """
        current = SQLClinicalService.get_case_by_id(db, case_id, user_id=user_id)
        if not current:
            return None

        update_clauses = []
        params: Dict[str, Any] = {"case_id": case_id}

        if case_in.case_name is not None:
            update_clauses.append("case_name = :case_name")
            params["case_name"] = case_in.case_name.strip()
        if case_in.imaging_modality is not None:
            update_clauses.append("imaging_modality = :imaging_modality")
            params["imaging_modality"] = case_in.imaging_modality.strip()
        if case_in.target_vessel is not None:
            update_clauses.append("target_vessel = :target_vessel")
            params["target_vessel"] = case_in.target_vessel.strip()
        if case_in.status is not None:
            update_clauses.append("status = :status")
            params["status"] = case_in.status.strip()

        if update_clauses:
            sql = text(f"""
                UPDATE cases
                SET {', '.join(update_clauses)}
                WHERE case_id = :case_id;
            """)
            db.execute(sql, params)
            db.commit()

        return SQLClinicalService.get_case_by_id(db, case_id, user_id=user_id)

    # --- Medical Image Management ---

    @staticmethod
    def insert_medical_image(db: Session, case_id: int, image_in: MedicalImageCreateRequest, user_id: Optional[uuid.UUID] = None) -> Dict[str, Any]:
        """
        Registers uploaded image metadata against a case.
        Regular users can only register images for their own cases.
        """
        case = SQLClinicalService.get_case_by_id(db, case_id, user_id=user_id)
        if not case:
            raise ValueError(f"Case with ID {case_id} not found or access denied.")

        sql = text("""
            INSERT INTO medical_images (case_id, image_name, image_path, image_format, image_size)
            VALUES (:case_id, :image_name, :image_path, :image_format, :image_size);
        """)
        db.execute(sql, {
            "case_id": case_id,
            "image_name": image_in.image_name.strip(),
            "image_path": image_in.image_path.strip(),
            "image_format": image_in.image_format,
            "image_size": image_in.image_size
        })
        db.commit()

        fetch_sql = text("""
            SELECT image_id, case_id, image_name, image_path, image_format, image_size, uploaded_at
            FROM medical_images
            WHERE case_id = :case_id AND image_name = :image_name
            ORDER BY image_id DESC
            LIMIT 1;
        """)
        row = db.execute(fetch_sql, {
            "case_id": case_id,
            "image_name": image_in.image_name.strip()
        }).fetchone()
        return dict(row._mapping)

    @staticmethod
    def get_image_by_id(db: Session, image_id: int, user_id: Optional[uuid.UUID] = None) -> Optional[Dict[str, Any]]:
        """
        Retrieves medical image metadata by image_id.
        Regular users can only retrieve images from their own cases.
        """
        sql = text("""
            SELECT m.image_id, m.case_id, m.image_name, m.image_path, m.image_format, m.image_size, m.uploaded_at
            FROM medical_images m
            JOIN cases c ON m.case_id = c.case_id
            WHERE m.image_id = :image_id
            AND (:user_id IS NULL OR c.user_id = :user_id);
        """)
        row = db.execute(sql, {"image_id": image_id, "user_id": str(user_id) if user_id else None}).fetchone()
        if not row:
            return None
        return dict(row._mapping)

    @staticmethod
    def get_images_by_case_id(db: Session, case_id: int, user_id: Optional[uuid.UUID] = None) -> List[Dict[str, Any]]:
        """
        Retrieves images associated with a case_id.
        Regular users can only retrieve images from their own cases.
        """
        case = SQLClinicalService.get_case_by_id(db, case_id, user_id=user_id)
        if not case:
            return []

        sql = text("""
            SELECT image_id, case_id, image_name, image_path, image_format, image_size, uploaded_at
            FROM medical_images
            WHERE case_id = :case_id
            ORDER BY image_id;
        """)
        result = db.execute(sql, {"case_id": case_id})
        return [dict(row._mapping) for row in result.fetchall()]

    @staticmethod
    def update_image_metadata(db: Session, image_id: int, image_in: MedicalImageUpdateRequest, user_id: Optional[uuid.UUID] = None) -> Optional[Dict[str, Any]]:
        """
        Updates medical image metadata.
        Regular users can only update images from their own cases.
        """
        current = SQLClinicalService.get_image_by_id(db, image_id, user_id=user_id)
        if not current:
            return None

        update_clauses = []
        params: Dict[str, Any] = {"image_id": image_id}

        if image_in.image_name is not None:
            update_clauses.append("image_name = :image_name")
            params["image_name"] = image_in.image_name.strip()
        if image_in.image_path is not None:
            update_clauses.append("image_path = :image_path")
            params["image_path"] = image_in.image_path.strip()
        if image_in.image_format is not None:
            update_clauses.append("image_format = :image_format")
            params["image_format"] = image_in.image_format.strip()
        if image_in.image_size is not None:
            update_clauses.append("image_size = :image_size")
            params["image_size"] = image_in.image_size

        if update_clauses:
            sql = text(f"""
                UPDATE medical_images
                SET {', '.join(update_clauses)}
                WHERE image_id = :image_id;
            """)
            db.execute(sql, params)
            db.commit()

        return SQLClinicalService.get_image_by_id(db, image_id, user_id=user_id)

    @staticmethod
    def delete_medical_image(db: Session, image_id: int, user_id: Optional[uuid.UUID] = None) -> bool:
        """
        Deletes image metadata when permitted (checks for dependent analysis results).
        Regular users can only delete images from their own cases.
        """
        # Verify ownership first
        image = SQLClinicalService.get_image_by_id(db, image_id, user_id=user_id)
        if not image:
            return False

        # Relationship retention rule: check if image has associated segmentation or stenosis results
        seg_check = text("SELECT COUNT(*) FROM segmentation_results WHERE image_id = :image_id;")
        sten_check = text("SELECT COUNT(*) FROM stenosis_results WHERE image_id = :image_id;")
        seg_count = db.execute(seg_check, {"image_id": image_id}).scalar() or 0
        sten_count = db.execute(sten_check, {"image_id": image_id}).scalar() or 0

        if seg_count > 0 or sten_count > 0:
            raise ValueError(
                f"Cannot delete image {image_id}: It has {seg_count} segmentation result(s) "
                f"and {sten_count} stenosis result(s) associated with it."
            )

        sql = text("DELETE FROM medical_images WHERE image_id = :image_id;")
        result = db.execute(sql, {"image_id": image_id})
        db.commit()
        return result.rowcount > 0

    @staticmethod
    def create_medical_image_record(
        db: Session,
        case_id: int,
        image_name: str,
        image_path: str,
        image_format: Optional[str],
        image_size: Optional[int],
        user_id: Optional[uuid.UUID] = None,
    ) -> Dict[str, Any]:
        """
        Creates a medical_images record after verifying case ownership.
        Regular users can only create images for their own cases.
        """
        case = SQLClinicalService.get_case_by_id(db, case_id, user_id=user_id)
        if not case:
            raise ValueError(f"Case with ID {case_id} not found or access denied.")

        sql = text("""
            INSERT INTO medical_images (case_id, image_name, image_path, image_format, image_size)
            VALUES (:case_id, :image_name, :image_path, :image_format, :image_size);
        """)
        db.execute(sql, {
            "case_id": case_id,
            "image_name": image_name,
            "image_path": image_path,
            "image_format": image_format,
            "image_size": image_size,
        })
        db.commit()

        fetch_sql = text("""
            SELECT image_id, case_id, image_name, image_path, image_format, image_size, uploaded_at
            FROM medical_images
            WHERE case_id = :case_id AND image_name = :image_name
            ORDER BY image_id DESC
            LIMIT 1;
        """)
        row = db.execute(fetch_sql, {
            "case_id": case_id,
            "image_name": image_name,
        }).fetchone()
        return dict(row._mapping)

    @staticmethod
    def create_segmentation_result_record(
        db: Session,
        image_id: int,
        mask_path: Optional[str],
        overlay_path: Optional[str],
        model_name: Optional[str],
        dice_score: Optional[float],
        iou_score: Optional[float],
        precision_score: Optional[float],
        recall_score: Optional[float],
        f1_score: Optional[float],
        mean_vessel_diameter: Optional[float],
        processed_mask_path: Optional[str],
        centerline_path: Optional[str],
        model_version: Optional[str],
        user_id: Optional[uuid.UUID] = None,
    ) -> Dict[str, Any]:
        """
        Creates a segmentation_results record after verifying image ownership.
        Regular users can only create results for images in their own cases.
        """
        image = SQLClinicalService.get_image_by_id(db, image_id, user_id=user_id)
        if not image:
            raise ValueError(f"Medical image with ID {image_id} not found or access denied.")

        sql = text("""
            INSERT INTO segmentation_results (
                image_id, mask_path, overlay_path, model_name, dice_score,
                iou_score, processed_mask_path, centerline_path,
                model_version, precision_score, recall_score,
                f1_score, mean_vessel_diameter
            ) VALUES (
                :image_id, :mask_path, :overlay_path, :model_name, :dice_score,
                :iou_score, :processed_mask_path, :centerline_path,
                :model_version, :precision_score, :recall_score,
                :f1_score, :mean_vessel_diameter
            );
        """)
        db.execute(sql, {
            "image_id": image_id,
            "mask_path": mask_path,
            "overlay_path": overlay_path,
            "model_name": model_name,
            "dice_score": dice_score,
            "iou_score": iou_score,
            "processed_mask_path": processed_mask_path,
            "centerline_path": centerline_path,
            "model_version": model_version,
            "precision_score": precision_score,
            "recall_score": recall_score,
            "f1_score": f1_score,
            "mean_vessel_diameter": mean_vessel_diameter,
        })
        db.commit()

        fetch_sql = text("""
            SELECT * FROM segmentation_results
            WHERE image_id = :image_id
            ORDER BY segmentation_id DESC
            LIMIT 1;
        """)
        row = db.execute(fetch_sql, {"image_id": image_id}).fetchone()
        return dict(row._mapping)

    @staticmethod
    def get_segmentation_result_by_id(
        db: Session,
        segmentation_id: int,
        user_id: Optional[uuid.UUID] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Retrieves a single segmentation result by ID with ownership check.
        """
        sql = text("""
            SELECT s.*
            FROM segmentation_results s
            JOIN medical_images m ON s.image_id = m.image_id
            JOIN cases c ON m.case_id = c.case_id
            WHERE s.segmentation_id = :segmentation_id
            AND (:user_id IS NULL OR c.user_id = :user_id);
        """)
        row = db.execute(sql, {
            "segmentation_id": segmentation_id,
            "user_id": str(user_id) if user_id else None,
        }).fetchone()
        if not row:
            return None
        return dict(row._mapping)

    # --- Segmentation Results ---

    @staticmethod
    def insert_segmentation_result(db: Session, image_id: int, seg_in: SegmentationResultCreateRequest, user_id: Optional[uuid.UUID] = None) -> Dict[str, Any]:
        """
        Inserts segmentation result for an image.
        Regular users can only insert results for images from their own cases.
        """
        image = SQLClinicalService.get_image_by_id(db, image_id, user_id=user_id)
        if not image:
            raise ValueError(f"Medical image with ID {image_id} not found or access denied.")

        sql = text("""
            INSERT INTO segmentation_results (
                image_id, mask_path, overlay_path, model_name, dice_score,
                iou_score, processed_mask_path, centerline_path,
                model_version, precision_score, recall_score,
                f1_score, mean_vessel_diameter
            ) VALUES (
                :image_id, :mask_path, :overlay_path, :model_name, :dice_score,
                :iou_score, :processed_mask_path, :centerline_path,
                :model_version, :precision_score, :recall_score,
                :f1_score, :mean_vessel_diameter
            );
        """)
        db.execute(sql, {
            "image_id": image_id,
            "mask_path": seg_in.mask_path,
            "overlay_path": seg_in.overlay_path,
            "model_name": seg_in.model_name,
            "dice_score": seg_in.dice_score,
            "iou_score": seg_in.iou_score,
            "processed_mask_path": seg_in.processed_mask_path,
            "centerline_path": seg_in.centerline_path,
            "model_version": seg_in.model_version,
            "precision_score": seg_in.precision_score,
            "recall_score": seg_in.recall_score,
            "f1_score": seg_in.f1_score,
            "mean_vessel_diameter": seg_in.mean_vessel_diameter
        })
        db.commit()

        fetch_sql = text("""
            SELECT * FROM segmentation_results
            WHERE image_id = :image_id
            ORDER BY segmentation_id DESC
            LIMIT 1;
        """)
        row = db.execute(fetch_sql, {"image_id": image_id}).fetchone()
        return dict(row._mapping)

    @staticmethod
    def get_segmentations_by_image_id(db: Session, image_id: int, user_id: Optional[uuid.UUID] = None) -> List[Dict[str, Any]]:
        """
        Retrieves segmentation results for a specific image_id.
        Regular users can only retrieve results from images in their own cases.
        """
        image = SQLClinicalService.get_image_by_id(db, image_id, user_id=user_id)
        if not image:
            return []

        sql = text("""
            SELECT * FROM segmentation_results
            WHERE image_id = :image_id
            ORDER BY segmentation_id;
        """)
        result = db.execute(sql, {"image_id": image_id})
        return [dict(row._mapping) for row in result.fetchall()]

    @staticmethod
    def get_segmentations_by_case_id(db: Session, case_id: int, user_id: Optional[uuid.UUID] = None) -> List[Dict[str, Any]]:
        """
        Retrieves segmentation results associated with a case_id via medical_images.
        Regular users can only retrieve results from their own cases.
        """
        case = SQLClinicalService.get_case_by_id(db, case_id, user_id=user_id)
        if not case:
            return []

        sql = text("""
            SELECT s.*
            FROM segmentation_results s
            JOIN medical_images m ON s.image_id = m.image_id
            WHERE m.case_id = :case_id
            ORDER BY s.segmentation_id;
        """)
        result = db.execute(sql, {"case_id": case_id})
        return [dict(row._mapping) for row in result.fetchall()]

    @staticmethod
    def get_latest_segmentation_result(db: Session, user_id: Optional[uuid.UUID] = None) -> Optional[Dict[str, Any]]:
        """
        Retrieves the most recent segmentation result.
        Regular users see only their own latest result; administrators see all.
        """
        sql = text("""
            SELECT s.*
            FROM segmentation_results s
            JOIN medical_images m ON s.image_id = m.image_id
            JOIN cases c ON m.case_id = c.case_id
            WHERE (:user_id IS NULL OR c.user_id = :user_id)
            ORDER BY s.created_at DESC, s.segmentation_id DESC
            LIMIT 1;
        """)
        row = db.execute(sql, {"user_id": str(user_id) if user_id else None}).fetchone()
        if not row:
            return None
        return dict(row._mapping)

    # --- Stenosis Results ---

    @staticmethod
    def insert_stenosis_result(db: Session, image_id: int, sten_in: StenosisResultCreateRequest, user_id: Optional[uuid.UUID] = None) -> Dict[str, Any]:
        """
        Inserts stenosis detection result for an image.
        Regular users can only insert results for images from their own cases.
        """
        image = SQLClinicalService.get_image_by_id(db, image_id, user_id=user_id)
        if not image:
            raise ValueError(f"Medical image with ID {image_id} not found or access denied.")

        sql = text("""
            INSERT INTO stenosis_results (image_id, location, severity, confidence_score, estimated_diameter)
            VALUES (:image_id, :location, :severity, :confidence_score, :estimated_diameter);
        """)
        db.execute(sql, {
            "image_id": image_id,
            "location": sten_in.location,
            "severity": sten_in.severity,
            "confidence_score": sten_in.confidence_score,
            "estimated_diameter": sten_in.estimated_diameter
        })
        db.commit()

        fetch_sql = text("""
            SELECT * FROM stenosis_results
            WHERE image_id = :image_id
            ORDER BY stenosis_id DESC
            LIMIT 1;
        """)
        row = db.execute(fetch_sql, {"image_id": image_id}).fetchone()
        return dict(row._mapping)

    @staticmethod
    def get_stenosis_by_image_id(db: Session, image_id: int, user_id: Optional[uuid.UUID] = None) -> List[Dict[str, Any]]:
        """
        Retrieves stenosis results for a specific image_id.
        Regular users can only retrieve results from images in their own cases.
        """
        image = SQLClinicalService.get_image_by_id(db, image_id, user_id=user_id)
        if not image:
            return []

        sql = text("""
            SELECT * FROM stenosis_results
            WHERE image_id = :image_id
            ORDER BY stenosis_id;
        """)
        result = db.execute(sql, {"image_id": image_id})
        return [dict(row._mapping) for row in result.fetchall()]

    @staticmethod
    def get_stenosis_by_case_id(db: Session, case_id: int, user_id: Optional[uuid.UUID] = None) -> List[Dict[str, Any]]:
        """
        Retrieves stenosis results associated with a case_id via medical_images.
        Regular users can only retrieve results from their own cases.
        """
        case = SQLClinicalService.get_case_by_id(db, case_id, user_id=user_id)
        if not case:
            return []

        sql = text("""
            SELECT st.*
            FROM stenosis_results st
            JOIN medical_images m ON st.image_id = m.image_id
            WHERE m.case_id = :case_id
            ORDER BY st.stenosis_id;
        """)
        result = db.execute(sql, {"case_id": case_id})
        return [dict(row._mapping) for row in result.fetchall()]

    @staticmethod
    def filter_stenosis_by_severity(db: Session, severity: str, user_id: Optional[uuid.UUID] = None) -> List[Dict[str, Any]]:
        """
        Filters stenosis results by severity (e.g. Mild, Moderate, Severe).
        Regular users see only their own results; administrators see all.
        """
        sql = text("""
            SELECT st.*
            FROM stenosis_results st
            JOIN medical_images m ON st.image_id = m.image_id
            JOIN cases c ON m.case_id = c.case_id
            WHERE LOWER(st.severity) = LOWER(:severity)
            AND (:user_id IS NULL OR c.user_id = :user_id)
            ORDER BY st.stenosis_id;
        """)
        result = db.execute(sql, {"severity": severity.strip(), "user_id": str(user_id) if user_id else None})
        return [dict(row._mapping) for row in result.fetchall()]

    @staticmethod
    def get_stenosis_sorted_by_confidence(db: Session, descending: bool = True, user_id: Optional[uuid.UUID] = None) -> List[Dict[str, Any]]:
        """
        Retrieves stenosis results sorted by confidence score.
        Regular users see only their own results; administrators see all.
        """
        order_dir = "DESC NULLS LAST" if descending else "ASC NULLS LAST"
        sql = text(f"""
            SELECT st.*
            FROM stenosis_results st
            JOIN medical_images m ON st.image_id = m.image_id
            JOIN cases c ON m.case_id = c.case_id
            WHERE (:user_id IS NULL OR c.user_id = :user_id)
            ORDER BY st.confidence_score {order_dir};
        """)
        result = db.execute(sql, {"user_id": str(user_id) if user_id else None})
        return [dict(row._mapping) for row in result.fetchall()]

    @staticmethod
    def get_latest_stenosis_result(db: Session, user_id: Optional[uuid.UUID] = None) -> Optional[Dict[str, Any]]:
        """
        Retrieves the most recent stenosis result.
        Regular users see only their own latest result; administrators see all.
        """
        sql = text("""
            SELECT st.*
            FROM stenosis_results st
            JOIN medical_images m ON st.image_id = m.image_id
            JOIN cases c ON m.case_id = c.case_id
            WHERE (:user_id IS NULL OR c.user_id = :user_id)
            ORDER BY st.created_at DESC, st.stenosis_id DESC
            LIMIT 1;
        """)
        row = db.execute(sql, {"user_id": str(user_id) if user_id else None}).fetchone()
        if not row:
            return None
        return dict(row._mapping)

    # --- Dashboard Analytics ---

    @staticmethod
    def get_dashboard_analytics(db: Session, user_id: Optional[uuid.UUID] = None) -> Dict[str, Any]:
        """
        Returns unified dashboard counts: total users, cases, images, stenosis detections,
        and recent analysis activity log corresponding to result_queries.sql Q24-Q28.

        Regular users see only their own data; administrators see all.
        """
        uid_param = str(user_id) if user_id else None

        users_count = db.execute(
            text("SELECT COUNT(*) AS total_registered_users FROM users WHERE (:user_id IS NULL OR id = :user_id);"),
            {"user_id": uid_param}
        ).scalar() or 0
        cases_count = db.execute(
            text("SELECT COUNT(*) AS total_medical_cases FROM cases WHERE (:user_id IS NULL OR user_id = :user_id);"),
            {"user_id": uid_param}
        ).scalar() or 0
        images_count = db.execute(
            text("""
                SELECT COUNT(*) AS total_uploaded_images
                FROM medical_images m
                JOIN cases c ON m.case_id = c.case_id
                WHERE (:user_id IS NULL OR c.user_id = :user_id);
            """),
            {"user_id": uid_param}
        ).scalar() or 0
        stenosis_count = db.execute(
            text("""
                SELECT COUNT(*) AS total_stenosis_detections
                FROM stenosis_results st
                JOIN medical_images m ON st.image_id = m.image_id
                JOIN cases c ON m.case_id = c.case_id
                WHERE (:user_id IS NULL OR c.user_id = :user_id);
            """),
            {"user_id": uid_param}
        ).scalar() or 0

        # Query recent activity (Query 28 from result_queries.sql)
        recent_activity_sql = text("""
            SELECT
                m.image_id,
                m.image_name,
                'Segmentation' AS analysis_type,
                s.model_name AS model_or_severity,
                s.created_at AS analysis_date
            FROM segmentation_results s
            JOIN medical_images m
                ON s.image_id = m.image_id
            JOIN cases c ON m.case_id = c.case_id
            WHERE (:user_id IS NULL OR c.user_id = :user_id)

            UNION ALL

            SELECT
                m.image_id,
                m.image_name,
                'Stenosis' AS analysis_type,
                st.severity AS model_or_severity,
                st.created_at AS analysis_date
            FROM stenosis_results st
            JOIN medical_images m
                ON st.image_id = m.image_id
            JOIN cases c ON m.case_id = c.case_id
            WHERE (:user_id IS NULL OR c.user_id = :user_id)

            ORDER BY analysis_date DESC
            LIMIT 10;
        """)

        recent_activity: List[Dict[str, Any]] = []
        for row in db.execute(recent_activity_sql, {"user_id": uid_param}).fetchall():
            m = row._mapping
            recent_activity.append({
                "image_id": m["image_id"],
                "image_name": m["image_name"],
                "analysis_type": m["analysis_type"],
                "model_or_severity": m["model_or_severity"],
                "analysis_date": m["analysis_date"]
            })

        return {
            "total_users": int(users_count),
            "total_cases": int(cases_count),
            "total_images": int(images_count),
            "total_stenosis_detections": int(stenosis_count),
            "recent_activity": recent_activity,
            "source": "sql_database"
        }

    # =========================================================================
    # --- result_queries.sql Queries 4 to 11 ---
    # =========================================================================

    @staticmethod
    def get_detailed_model_performance(db: Session, user_id: Optional[uuid.UUID] = None) -> List[Dict[str, Any]]:
        """
        Executes Query 4 from result_queries.sql:
        Complete model performance analytics with IoU, precision, recall, and F1.

        Regular users see only their own results; administrators see all.
        """
        sql = text("""
            SELECT
                s.model_name,
                COUNT(*) AS total_results,
                ROUND(AVG(s.dice_score), 4) AS average_dice,
                ROUND(AVG(s.iou_score), 4) AS average_iou,
                ROUND(AVG(s.precision_score), 4) AS average_precision,
                ROUND(AVG(s.recall_score), 4) AS average_recall,
                ROUND(AVG(s.f1_score), 4) AS average_f1
            FROM segmentation_results s
            JOIN medical_images m ON s.image_id = m.image_id
            JOIN cases c ON m.case_id = c.case_id
            WHERE (:user_id IS NULL OR c.user_id = :user_id)
            GROUP BY s.model_name
            ORDER BY average_dice DESC;
        """)
        results = []
        for row in db.execute(sql, {"user_id": str(user_id) if user_id else None}).fetchall():
            m = row._mapping
            results.append({
                "model_name": m["model_name"],
                "total_results": int(m["total_results"]),
                "average_dice": float(m["average_dice"]) if m["average_dice"] is not None else None,
                "average_iou": float(m["average_iou"]) if m["average_iou"] is not None else None,
                "average_precision": float(m["average_precision"]) if m["average_precision"] is not None else None,
                "average_recall": float(m["average_recall"]) if m["average_recall"] is not None else None,
                "average_f1": float(m["average_f1"]) if m["average_f1"] is not None else None,
            })
        return results

    @staticmethod
    def get_stenosis_severity_distribution(db: Session, user_id: Optional[uuid.UUID] = None) -> List[Dict[str, Any]]:
        """
        Executes Query 5 from result_queries.sql:
        Stenosis severity distribution with case counts and average confidence.

        Regular users see only their own results; administrators see all.
        """
        sql = text("""
            SELECT
                st.severity,
                COUNT(*) AS total_cases,
                ROUND(AVG(st.confidence_score), 4) AS average_confidence
            FROM stenosis_results st
            JOIN medical_images m ON st.image_id = m.image_id
            JOIN cases c ON m.case_id = c.case_id
            WHERE (:user_id IS NULL OR c.user_id = :user_id)
            GROUP BY st.severity
            ORDER BY total_cases DESC;
        """)
        results = []
        for row in db.execute(sql, {"user_id": str(user_id) if user_id else None}).fetchall():
            m = row._mapping
            results.append({
                "severity": m["severity"],
                "total_cases": int(m["total_cases"]),
                "average_confidence": float(m["average_confidence"]) if m["average_confidence"] is not None else None,
            })
        return results

    @staticmethod
    def get_case_analysis_report(db: Session, case_id: Optional[int] = None, user_id: Optional[uuid.UUID] = None) -> List[Dict[str, Any]]:
        """
        Executes Query 6 from result_queries.sql:
        Case-wise medical image and complete analysis report joining cases, images,
        and CTE-aggregated segmentation and stenosis summaries.

        Regular users see only their own cases; administrators see all.
        """
        is_sqlite = db.bind is not None and db.bind.dialect.name == "sqlite"
        agg_fn = "GROUP_CONCAT" if is_sqlite else "STRING_AGG"
        agg_delim = "" if is_sqlite else ", ', '"

        query_str = f"""
            WITH segmentation_summary AS (
                SELECT
                    s.image_id,
                    COUNT(*) AS total_segmentation_results,
                    {agg_fn}(DISTINCT s.model_name{agg_delim}) AS model_names,
                    ROUND(AVG(s.dice_score), 4) AS average_dice,
                    ROUND(AVG(s.iou_score), 4) AS average_iou
                FROM segmentation_results s
                JOIN medical_images m ON s.image_id = m.image_id
                JOIN cases c ON m.case_id = c.case_id
                WHERE (:user_id IS NULL OR c.user_id = :user_id)
                GROUP BY s.image_id
            ),
            stenosis_summary AS (
                SELECT
                    st.image_id,
                    COUNT(*) AS total_stenosis_results,
                    {agg_fn}(DISTINCT st.severity{agg_delim}) AS detected_severities,
                    ROUND(AVG(st.confidence_score), 4) AS average_confidence,
                    ROUND(AVG(st.estimated_diameter), 4) AS average_estimated_diameter
                FROM stenosis_results st
                JOIN medical_images m ON st.image_id = m.image_id
                JOIN cases c ON m.case_id = c.case_id
                WHERE (:user_id IS NULL OR c.user_id = :user_id)
                GROUP BY st.image_id
            )
            SELECT
                c.case_id,
                c.case_name,
                c.imaging_modality,
                c.target_vessel,
                m.image_id,
                m.image_name,
                COALESCE(s.total_segmentation_results, 0) AS total_segmentation_results,
                s.model_names,
                s.average_dice,
                s.average_iou,
                COALESCE(st.total_stenosis_results, 0) AS total_stenosis_results,
                st.detected_severities,
                st.average_confidence,
                st.average_estimated_diameter
            FROM cases c
            JOIN medical_images m
                ON c.case_id = m.case_id
            LEFT JOIN segmentation_summary s
                ON m.image_id = s.image_id
            LEFT JOIN stenosis_summary st
                ON m.image_id = st.image_id
            WHERE (:user_id IS NULL OR c.user_id = :user_id)
        """
        params: Dict[str, Any] = {"user_id": str(user_id) if user_id else None}
        if case_id is not None:
            query_str += " AND c.case_id = :case_id"
            params["case_id"] = case_id
        query_str += " ORDER BY c.case_id, m.image_id;"

        sql = text(query_str)
        results = []
        for row in db.execute(sql, params).fetchall():
            m = row._mapping
            results.append({
                "case_id": m["case_id"],
                "case_name": m["case_name"],
                "imaging_modality": m["imaging_modality"],
                "target_vessel": m["target_vessel"],
                "image_id": m["image_id"],
                "image_name": m["image_name"],
                "total_segmentation_results": int(m["total_segmentation_results"]),
                "model_names": m["model_names"],
                "average_dice": float(m["average_dice"]) if m["average_dice"] is not None else None,
                "average_iou": float(m["average_iou"]) if m["average_iou"] is not None else None,
                "total_stenosis_results": int(m["total_stenosis_results"]),
                "detected_severities": m["detected_severities"],
                "average_confidence": float(m["average_confidence"]) if m["average_confidence"] is not None else None,
                "average_estimated_diameter": float(m["average_estimated_diameter"]) if m["average_estimated_diameter"] is not None else None,
            })
        return results

    @staticmethod
    def get_case_summary_counts(db: Session, user_id: Optional[uuid.UUID] = None) -> List[Dict[str, Any]]:
        """
        Executes Query 7 from result_queries.sql:
        Count images and analysis results for each case.

        Regular users see only their own cases; administrators see all.
        """
        sql = text("""
            SELECT
                c.case_id,
                c.case_name,
                COUNT(DISTINCT m.image_id) AS total_images,
                COUNT(DISTINCT s.segmentation_id) AS total_segmentation_results,
                COUNT(DISTINCT st.stenosis_id) AS total_stenosis_results
            FROM cases c
            LEFT JOIN medical_images m
                ON c.case_id = m.case_id
            LEFT JOIN segmentation_results s
                ON m.image_id = s.image_id
            LEFT JOIN stenosis_results st
                ON m.image_id = st.image_id
            WHERE (:user_id IS NULL OR c.user_id = :user_id)
            GROUP BY c.case_id, c.case_name
            ORDER BY c.case_id;
        """)
        results = []
        for row in db.execute(sql, {"user_id": str(user_id) if user_id else None}).fetchall():
            m = row._mapping
            results.append({
                "case_id": m["case_id"],
                "case_name": m["case_name"],
                "total_images": int(m["total_images"]),
                "total_segmentation_results": int(m["total_segmentation_results"]),
                "total_stenosis_results": int(m["total_stenosis_results"]),
            })
        return results

    @staticmethod
    def diagnose_foreign_keys(db: Session) -> List[Dict[str, Any]]:
        """
        Executes Query 8 from result_queries.sql:
        Verify database foreign-key relationships via information_schema.
        """
        sql = text("""
            SELECT
                tc.table_name,
                kcu.column_name,
                ccu.table_name AS referenced_table,
                ccu.column_name AS referenced_column,
                tc.constraint_name
            FROM information_schema.table_constraints AS tc
            JOIN information_schema.key_column_usage AS kcu
                ON tc.constraint_name = kcu.constraint_name
                AND tc.constraint_schema = kcu.constraint_schema
            JOIN information_schema.constraint_column_usage AS ccu
                ON ccu.constraint_name = tc.constraint_name
                AND ccu.constraint_schema = tc.constraint_schema
            WHERE tc.constraint_type = 'FOREIGN KEY'
            AND tc.constraint_schema = 'public'
            ORDER BY tc.table_name;
        """)
        results = []
        for row in db.execute(sql).fetchall():
            m = row._mapping
            results.append({
                "table_name": str(m["table_name"]),
                "column_name": str(m["column_name"]),
                "referenced_table": str(m["referenced_table"]),
                "referenced_column": str(m["referenced_column"]),
                "constraint_name": str(m["constraint_name"]),
            })
        return results

    @staticmethod
    def diagnose_tables(db: Session) -> List[Dict[str, Any]]:
        """
        Executes Query 9 from result_queries.sql:
        Verify all project database tables via information_schema.tables.
        """
        sql = text("""
            SELECT
                table_name
            FROM information_schema.tables
            WHERE table_schema = 'public'
            AND table_type = 'BASE TABLE'
            AND table_name IN (
                'users',
                'cases',
                'medical_images',
                'segmentation_results',
                'stenosis_results'
            )
            ORDER BY table_name;
        """)
        results = []
        for row in db.execute(sql).fetchall():
            m = row._mapping
            results.append({
                "table_name": str(m["table_name"]),
            })
        return results

    @staticmethod
    def diagnose_columns(db: Session, table_name: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Executes Query 10 from result_queries.sql:
        Verify table columns and data types via information_schema.columns.
        """
        query_str = """
            SELECT
                table_name,
                column_name,
                data_type,
                is_nullable,
                column_default
            FROM information_schema.columns
            WHERE table_schema = 'public'
            AND table_name IN (
                'users',
                'cases',
                'medical_images',
                'segmentation_results',
                'stenosis_results'
            )
        """
        params: Dict[str, Any] = {}
        if table_name:
            query_str += " AND table_name = :table_name"
            params["table_name"] = table_name.strip().lower()
        query_str += " ORDER BY table_name, ordinal_position;"

        sql = text(query_str)
        results = []
        for row in db.execute(sql, params).fetchall():
            m = row._mapping
            results.append({
                "table_name": str(m["table_name"]),
                "column_name": str(m["column_name"]),
                "data_type": str(m["data_type"]),
                "is_nullable": str(m["is_nullable"]),
                "column_default": str(m["column_default"]) if m["column_default"] is not None else None,
            })
        return results

    @staticmethod
    def diagnose_fk_consistency(db: Session) -> List[Dict[str, Any]]:
        """
        Executes Query 11 from result_queries.sql:
        Verify foreign-key data consistency and orphan records check.
        """
        sql = text("""
            SELECT
                'Cases without valid users' AS check_name,
                COUNT(*) AS invalid_records
            FROM cases c
            LEFT JOIN users u
                ON c.user_id = u.id
            WHERE u.id IS NULL

            UNION ALL

            SELECT
                'Images without valid cases' AS check_name,
                COUNT(*) AS invalid_records
            FROM medical_images m
            LEFT JOIN cases c
                ON m.case_id = c.case_id
            WHERE c.case_id IS NULL

            UNION ALL

            SELECT
                'Segmentation results without valid images' AS check_name,
                COUNT(*) AS invalid_records
            FROM segmentation_results s
            LEFT JOIN medical_images m
                ON s.image_id = m.image_id
            WHERE m.image_id IS NULL

            UNION ALL

            SELECT
                'Stenosis results without valid images' AS check_name,
                COUNT(*) AS invalid_records
            FROM stenosis_results s
            LEFT JOIN medical_images m
                ON s.image_id = m.image_id
            WHERE m.image_id IS NULL;
        """)
        results = []
        for row in db.execute(sql).fetchall():
            m = row._mapping
            results.append({
                "check_name": str(m["check_name"]),
                "invalid_records": int(m["invalid_records"]),
            })
        return results

