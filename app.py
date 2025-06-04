from flask import Flask
from flasgger import Swagger
from api.routes import api_blueprint
from flask_cors import CORS

def create_app():
    app = Flask(__name__)
    CORS(app)

    swagger_config = {
        "headers": [],
        "specs": [
            {
                "endpoint": "apispec",
                "route": "/apispec.json",
                "rule_filter": lambda rule: True,
                "model_filter": lambda tag: True,
            }
        ],
        "static_url_path": "/flasgger_static",
        "swagger_ui": True,
        "specs_route": "/"
    }
    
    swagger_template = {
        "swagger": "2.0",
        "info": {
            "title": "API de Recomendação de Receitas",
            "description": "API para recomendação de receitas baseada em preferências de ingredientes",
            "version": "1.0.0"
        },
        "basePath": "/api",
        "schemes": ["http", "https"],
        "securityDefinitions": {
            "Bearer": {
                "type": "apiKey",
                "name": "Authorization",
                "in": "header",
                "description": "Digite 'Bearer ' seguido pelo token JWT, ex: Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9"
            }
        },
    }
    
    app.register_blueprint(api_blueprint, url_prefix='/api')
    
    Swagger(app, config=swagger_config, template=swagger_template)
    
    @app.route('/')
    def index():
        return """
        <h1>API de Recomendação de Receitas</h1>
        """
    
    return app

if __name__ == "__main__":
    app = create_app()
    app.run(debug=True, host='0.0.0.0', port=8080)