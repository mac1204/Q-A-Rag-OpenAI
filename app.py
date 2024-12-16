from flask import Flask

from routes.api_routes import api_bp

# Create Flask app
app = Flask(__name__)

# Register blueprints
app.register_blueprint(api_bp, url_prefix='/api')

if __name__ == '__main__':
    app.run(debug=True)