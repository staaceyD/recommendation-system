import os

from dotenv import load_dotenv

load_dotenv()

from flask import Flask  # noqa: E402

from app.config import config_by_name  # noqa: E402
from app.extensions import db, migrate  # noqa: E402


def create_app(config_name=None):
    config_name = config_name or os.environ.get("FLASK_ENV", "development")

    app = Flask(__name__)
    app.config.from_object(config_by_name[config_name])

    db.init_app(app)
    migrate.init_app(app, db)

    from app import models  # noqa: F401
    from app.routes import register_routes

    register_routes(app)

    return app
