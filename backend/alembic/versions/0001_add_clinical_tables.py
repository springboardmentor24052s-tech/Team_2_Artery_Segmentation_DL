"""Add auth and clinical tables

Revision ID: 0001_add_clinical_tables
Revises: 
Create Date: 2026-11-12 16:41:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = '0001_add_clinical_tables'
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Create auth tables (users and refresh_tokens)
    op.create_table(
        'users',
        sa.Column('id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('name', sa.String(length=255), nullable=False),
        sa.Column('email', sa.String(length=255), nullable=False),
        sa.Column('password_hash', sa.String(length=512), nullable=False),
        sa.Column('role', sa.String(length=50), nullable=False),
        sa.Column('is_active', sa.Boolean(), nullable=False),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('email')
    )
    op.create_index(op.f('ix_users_email'), 'users', ['email'], unique=False)

    op.create_table(
        'refresh_tokens',
        sa.Column('id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('user_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('token_hash', sa.String(length=64), nullable=False),
        sa.Column('expires_at', sa.DateTime(timezone=True), nullable=False),
        sa.Column('revoked_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(['user_id'], ['users.id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('token_hash')
    )
    op.create_index(op.f('ix_refresh_tokens_token_hash'), 'refresh_tokens', ['token_hash'], unique=False)
    op.create_index(op.f('ix_refresh_tokens_expires_at'), 'refresh_tokens', ['expires_at'], unique=False)
    op.create_index(op.f('ix_refresh_tokens_user_id'), 'refresh_tokens', ['user_id'], unique=False)

    # Create clinical tables
    op.create_table(
        'cases',
        sa.Column('case_id', sa.Integer(), nullable=False),
        sa.Column('user_id', postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column('case_name', sa.String(length=150), nullable=True),
        sa.Column('imaging_modality', sa.String(length=50), nullable=True),
        sa.Column('target_vessel', sa.String(length=100), nullable=True),
        sa.Column('status', sa.String(length=30), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=True),
        sa.ForeignKeyConstraint(['user_id'], ['users.id'], ),
        sa.PrimaryKeyConstraint('case_id')
    )
    op.create_table(
        'medical_images',
        sa.Column('image_id', sa.Integer(), nullable=False),
        sa.Column('case_id', sa.Integer(), nullable=True),
        sa.Column('image_name', sa.String(length=255), nullable=True),
        sa.Column('image_path', sa.Text(), nullable=True),
        sa.Column('image_format', sa.String(length=10), nullable=True),
        sa.Column('image_size', sa.BigInteger(), nullable=True),
        sa.Column('uploaded_at', sa.DateTime(), nullable=True),
        sa.ForeignKeyConstraint(['case_id'], ['cases.case_id'], ),
        sa.PrimaryKeyConstraint('image_id')
    )
    op.create_table(
        'segmentation_results',
        sa.Column('segmentation_id', sa.Integer(), nullable=False),
        sa.Column('image_id', sa.Integer(), nullable=True),
        sa.Column('mask_path', sa.Text(), nullable=True),
        sa.Column('model_name', sa.String(length=100), nullable=True),
        sa.Column('dice_score', sa.Numeric(precision=6, scale=5), nullable=True),
        sa.Column('iou_score', sa.Numeric(precision=6, scale=5), nullable=True),
        sa.Column('processed_mask_path', sa.Text(), nullable=True),
        sa.Column('centerline_path', sa.Text(), nullable=True),
        sa.Column('model_version', sa.String(length=50), nullable=True),
        sa.Column('precision_score', sa.Numeric(precision=6, scale=5), nullable=True),
        sa.Column('recall_score', sa.Numeric(precision=6, scale=5), nullable=True),
        sa.Column('f1_score', sa.Numeric(precision=6, scale=5), nullable=True),
        sa.Column('mean_vessel_diameter', sa.Numeric(precision=10, scale=4), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=True),
        sa.ForeignKeyConstraint(['image_id'], ['medical_images.image_id'], ),
        sa.PrimaryKeyConstraint('segmentation_id')
    )
    op.create_table(
        'stenosis_results',
        sa.Column('stenosis_id', sa.Integer(), nullable=False),
        sa.Column('image_id', sa.Integer(), nullable=True),
        sa.Column('location', sa.Text(), nullable=True),
        sa.Column('severity', sa.String(length=30), nullable=True),
        sa.Column('confidence_score', sa.Numeric(precision=6, scale=5), nullable=True),
        sa.Column('estimated_diameter', sa.Numeric(precision=10, scale=4), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=True),
        sa.ForeignKeyConstraint(['image_id'], ['medical_images.image_id'], ),
        sa.PrimaryKeyConstraint('stenosis_id')
    )


def downgrade() -> None:
    op.drop_table('stenosis_results')
    op.drop_table('segmentation_results')
    op.drop_table('medical_images')
    op.drop_table('cases')
    op.drop_index(op.f('ix_refresh_tokens_user_id'), table_name='refresh_tokens')
    op.drop_index(op.f('ix_refresh_tokens_expires_at'), table_name='refresh_tokens')
    op.drop_index(op.f('ix_refresh_tokens_token_hash'), table_name='refresh_tokens')
    op.drop_table('refresh_tokens')
    op.drop_index(op.f('ix_users_email'), table_name='users')
    op.drop_table('users')
