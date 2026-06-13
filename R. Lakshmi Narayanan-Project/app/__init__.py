import os
from flask import Flask
from app.routes import main
from app.database import init_db

def create_app():
    app = Flask(__name__,
                static_folder=os.path.join(os.path.dirname(__file__), 'static'),
                template_folder=os.path.join(os.path.dirname(__file__), 'templates'))
    app.register_blueprint(main)
    init_db()
    return app