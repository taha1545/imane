import logging
from flask import Flask

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_app():
    """Create and configure an instance of the Flask application."""
    app = Flask(__name__, 
                static_folder='../static', 
                template_folder='../templates')

    with app.app_context():
        # Register blueprints
        from . import routes
        app.register_blueprint(routes.bp)

    logger.info("Flask app created and configured.")
    return app