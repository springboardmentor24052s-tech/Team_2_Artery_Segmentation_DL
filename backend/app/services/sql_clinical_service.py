from typing import List, Dict, Any, Optional
from sqlalchemy.orm import Session
from sqlalchemy import text

class SQLClinicalService:
    """
    Executes existing SQL queries identified in auth_queries.sql and result_queries.sql.
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
    def get_user_by_email_for_verification(db: Session, email: str) -> Optional[Dict[str, Any]]:
        """
        Executes Query 2 from auth_queries.sql:
        SELECT user_id, full_name, email, password_hash
        FROM users
        WHERE email = :email;
        """
        sql = text("""
            SELECT user_id, full_name, email, password_hash
            FROM users
            WHERE email = :email;
        """)
        row = db.execute(sql, {"email": email.strip().lower()}).fetchone()
        if not row:
            return None
        return dict(row._mapping)

    @staticmethod
    def get_registered_users(db: Session) -> List[Dict[str, Any]]:
        """
        Executes Query 3 from auth_queries.sql:
        SELECT user_id, full_name, email, created_at
        FROM users
        ORDER BY user_id;
        """
        sql = text("""
            SELECT user_id, full_name, email, created_at
            FROM users
            ORDER BY user_id;
        """)
        result = db.execute(sql)
        return [dict(row._mapping) for row in result.fetchall()]

    @staticmethod
    def update_user_name(db: Session, user_id: Any, full_name: str) -> bool:
        """
        Executes Query 4 from auth_queries.sql:
        UPDATE users
        SET full_name = :full_name
        WHERE user_id = :user_id;
        """
        sql = text("""
            UPDATE users
            SET full_name = :full_name
            WHERE user_id = :user_id;
        """)
        result = db.execute(sql, {"full_name": full_name.strip(), "user_id": user_id})
        db.commit()
        return result.rowcount > 0

    @staticmethod
    def get_all_cases(db: Session) -> List[Dict[str, Any]]:
        """
        Executes Query 5 from auth_queries.sql:
        SELECT case_id, user_id, case_name, imaging_modality, target_vessel, status, created_at
        FROM cases ORDER BY case_id;
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
            ORDER BY case_id;
        """)
        result = db.execute(sql)
        return [dict(row._mapping) for row in result.fetchall()]

    @staticmethod
    def get_all_medical_images(db: Session) -> List[Dict[str, Any]]:
        """
        Executes Query 6 from auth_queries.sql:
        SELECT * FROM medical_images ORDER BY image_id;
        """
        sql = text("""
            SELECT *
            FROM medical_images
            ORDER BY image_id;
        """)
        result = db.execute(sql)
        return [dict(row._mapping) for row in result.fetchall()]

    @staticmethod
    def get_all_segmentation_results(db: Session) -> List[Dict[str, Any]]:
        """
        Executes Query 7 from auth_queries.sql:
        SELECT * FROM segmentation_results ORDER BY segmentation_id;
        """
        sql = text("""
            SELECT *
            FROM segmentation_results
            ORDER BY segmentation_id;
        """)
        result = db.execute(sql)
        return [dict(row._mapping) for row in result.fetchall()]

    @staticmethod
    def get_all_stenosis_results(db: Session) -> List[Dict[str, Any]]:
        """
        Executes Query 8 from auth_queries.sql:
        SELECT * FROM stenosis_results ORDER BY stenosis_id;
        """
        sql = text("""
            SELECT *
            FROM stenosis_results
            ORDER BY stenosis_id;
        """)
        result = db.execute(sql)
        return [dict(row._mapping) for row in result.fetchall()]

    # --- result_queries.sql Queries ---

    @staticmethod
    def get_segmentation_count(db: Session) -> int:
        """
        Executes Query 1 from result_queries.sql:
        SELECT COUNT(*) AS total_segmentation_results FROM segmentation_results;
        """
        sql = text("""
            SELECT COUNT(*) AS total_segmentation_results
            FROM segmentation_results;
        """)
        result = db.execute(sql).scalar()
        return int(result or 0)

    @staticmethod
    def get_stenosis_count(db: Session) -> int:
        """
        Executes Query 2 from result_queries.sql:
        SELECT COUNT(*) AS total_stenosis_results FROM stenosis_results;
        """
        sql = text("""
            SELECT COUNT(*) AS total_stenosis_results
            FROM stenosis_results;
        """)
        result = db.execute(sql).scalar()
        return int(result or 0)

    @staticmethod
    def get_model_performance(db: Session) -> List[Dict[str, Any]]:
        """
        Executes Query 3 from result_queries.sql:
        SELECT model_name, AVG(dice_score) AS average_dice_score
        FROM segmentation_results
        GROUP BY model_name
        ORDER BY average_dice_score DESC NULLS LAST;
        """
        sql = text("""
            SELECT
                model_name,
                AVG(dice_score) AS average_dice_score
            FROM segmentation_results
            GROUP BY model_name
            ORDER BY average_dice_score DESC NULLS LAST;
        """)
        result = db.execute(sql)
        output = []
        for row in result.fetchall():
            m = row._mapping
            output.append({
                "model_name": m["model_name"] or "Unknown",
                "average_dice_score": float(m["average_dice_score"]) if m["average_dice_score"] is not None else None
            })
        return output
