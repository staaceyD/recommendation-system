import os

from dotenv import load_dotenv
from flask import Flask

from app.config import config_by_name
from app.extensions import db, migrate

load_dotenv()


def create_app(config_name=None):
    config_name = config_name or os.environ.get("FLASK_ENV", "development")

    app = Flask(__name__)
    app.config.from_object(config_by_name[config_name])

    db.init_app(app)
    migrate.init_app(app, db)

    from app import models  # noqa: F401

    return app
