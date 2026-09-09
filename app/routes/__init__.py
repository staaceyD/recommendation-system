from app.routes.movies import bp as movies_bp


def register_routes(app):
    app.register_blueprint(movies_bp)
