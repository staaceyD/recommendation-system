from app.routes.movies import bp as movies_bp
from app.routes.recommendations import bp as recommendations_bp


def register_routes(app):
    app.register_blueprint(movies_bp)
    app.register_blueprint(recommendations_bp)
