"""index ratings (movie_id, rating)

Revision ID: 25c52f5a2f1e
Revises: c0095fe1882a
Create Date: 2026-09-10 20:03:38.366174

"""
from alembic import op

# revision identifiers, used by Alembic.
revision = '25c52f5a2f1e'
down_revision = 'c0095fe1882a'
branch_labels = None
depends_on = None


def upgrade():
    # Covering index for the item side of recommendations: GROUP BY movie_id with
    # AVG(rating) is served entirely from the index. Plain CREATE INDEX -- SQLite
    # does this in place; batch_alter_table would copy the whole 32M-row table.
    op.create_index(
        op.f('ix_ratings_movie_id_rating'), 'ratings', ['movie_id', 'rating'], unique=False
    )


def downgrade():
    op.drop_index(op.f('ix_ratings_movie_id_rating'), table_name='ratings')
