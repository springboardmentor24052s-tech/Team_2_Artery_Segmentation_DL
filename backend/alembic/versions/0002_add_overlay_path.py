"""Add overlay_path column to segmentation_results

Revision ID: 0002_add_overlay_path
Revises: 0001_add_clinical_tables
Create Date: 2026-11-13 01:57:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision: str = '0002_add_overlay_path'
down_revision: Union[str, None] = '0001_add_clinical_tables'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        'segmentation_results',
        sa.Column('overlay_path', sa.Text(), nullable=True),
    )


def downgrade() -> None:
    op.drop_column('segmentation_results', 'overlay_path')
