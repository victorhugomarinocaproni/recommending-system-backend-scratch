from flask import Blueprint, request, jsonify
from flasgger import swag_from
from services.knn_service import predict_favorite_recipes_knn 
from services.decision_tree_service import predict_favorite_recipes_random_forest
from services.naive_bayes_service import predict_favorite_recipes_naive_bayes

api_blueprint = Blueprint("api", __name__)

@api_blueprint.route("recommendation/knn", methods=["POST"])
@swag_from({
    'tags': ['Recomendações'],
    'summary': 'Retorna as 3 receitas mais recomendadas com base no algoritmo de KNN',
    'parameters': [
        {
            'name': 'body',
            'in': 'body',
            'required': True,
            'schema': {
                'type': 'object',
                'properties': {
                    'ingredients': {
                        'type': 'array',
                        'items': {'type': 'string'},
                        'example': ['limão', 'pimenta jalapeno', 'carne de boi']
                    }
                },
                'required': ['ingredients']
            },
            'description': 'Objeto contendo a lista de ingredientes preferidos do usuário'
        }
    ],
    'definitions': {
        'Recipe': {
            'type': 'object',
            'properties': {
                'name': {'type': 'string'},
                'ingredients': {'type': 'array', 'items': {'type': 'string'}},
                'instructions': {'type': 'string'},
                'score': {'type': 'number', 'format': 'float'}
            }
        }
    },
    'responses': {
        200: {
            'description': 'Lista das 3 receitas mais compatíveis',
            'schema': {
                'type': 'object',
                'properties': {
                    'recommendations': {
                        'type': 'array',
                        'items': {'$ref': '#/definitions/Recipe'}
                    }
                }
            }
        },
        400: {
            'description': 'Requisição inválida'
        },
        500: {
            'description': 'Erro interno do servidor'
        }
    }
})
def recommend_drinks_knn():
    try:
        data = request.get_json()
        ingredients = data.get('ingredients', [])
    
        if not data or 'ingredients' not in data:
            return jsonify({"error": "Parâmetro 'ingredients' é obrigatório e deve ser uma lista de strings."}), 400

        top_3_drinks = predict_favorite_recipes_knn(ingredients) 
        
        return jsonify({"recommendations": top_3_drinks}), 200

    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        import traceback
        print(f"Erro ao processar recomendação KNN: {str(e)}") 
        print(traceback.format_exc())
        return jsonify({"error": f"Erro interno: {str(e)}"}), 500
    
  

@api_blueprint.route("recommendation/random-forest", methods=["POST"])
@swag_from({
    'tags': ['Recomendações'],
    'summary': 'Retorna as 3 receitas mais recomendadas a partir de um modelo de árvore de decisão',
    'parameters': [
        {
            'name': 'body',
            'in': 'body',
            'required': True,
            'schema': {
                'type': 'object',
                'properties': {
                    'ingredients': {
                        'type': 'array',
                        'items': {'type': 'string'},
                        'example': ['limão', 'pimenta jalapeno', 'carne de boi']
                    }
                },
                'required': ['ingredients']
            },
            'description': 'Objeto contendo a lista de ingredientes preferidos do usuário'
        }
    ],
    'definitions': {
        'Recipe': {
            'type': 'object',
            'properties': {
                'name': {'type': 'string'},
                'ingredients': {'type': 'array', 'items': {'type': 'string'}},
                'instructions': {'type': 'string'},
                'score': {'type': 'number', 'format': 'float'}
            }
        }
    },
    'responses': {
        200: {
            'description': 'Lista das 3 receitas mais compatíveis',
            'schema': {
                'type': 'object',
                'properties': {
                    'recommendations': {
                        'type': 'array',
                        'items': {'$ref': '#/definitions/Recipe'}
                    }
                }
            }
        },
        400: {
            'description': 'Requisição inválida'
        },
        500: {
            'description': 'Erro interno do servidor'
        }
    }
})
def recommend_drinks_dt():
    try:
        data = request.get_json()
        ingredients = data.get('ingredients', [])
    
        if not ingredients:
            return jsonify({"error": "Parâmetro 'ingredients' é obrigatório e deve ser uma lista de strings."}), 400
        
        top_3_recipes = predict_favorite_recipes_random_forest(ingredients)
                
        return jsonify({"recommendations": top_3_recipes}), 200

    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        import traceback
        print(f"Erro ao processar recomendação: {str(e)}")
        print(traceback.format_exc())
        return jsonify({"error": f"Erro interno: {str(e)}"}), 500



@api_blueprint.route("recommendation/naive-bayes", methods=["POST"])
@swag_from({
    'tags': ['Recomendações'],
    'summary': 'Retorna as 3 receitas mais recomendadas com base no algoritmo de Naive Bayes',
    'parameters': [
        {
            'name': 'body',
            'in': 'body',
            'required': True,
            'schema': {
                'type': 'object',
                'properties': {
                    'ingredients': {
                        'type': 'array',
                        'items': {'type': 'string'},
                        'example': ['limão', 'pimenta jalapeno', 'carne de boi']
                    }
                },
                'required': ['ingredients']
            },
            'description': 'Objeto contendo a lista de ingredientes preferidos do usuário'
        }
    ],
    'definitions': {
        'Recipe': {
            'type': 'object',
            'properties': {
                'name': {'type': 'string'},
                'ingredients': {'type': 'array', 'items': {'type': 'string'}},
                'instructions': {'type': 'string'},
                'score': {'type': 'number', 'format': 'float'}
            }
        }
    },
    'responses': {
        200: {
            'description': 'Lista das 3 receitas mais compatíveis',
            'schema': {
                'type': 'object',
                'properties': {
                    'recommendations': {
                        'type': 'array',
                        'items': {'$ref': '#/definitions/Recipe'}
                    }
                }
            }
        },
        400: {
            'description': 'Requisição inválida'
        },
        500: {
            'description': 'Erro interno do servidor'
        }
    }
})
def recommend_recipes_naive_bayes():
    try:
        data = request.get_json()
        ingredients = data.get('ingredients', [])
    
        if not ingredients:
            return jsonify({"error": "Parâmetro 'ingredients' é obrigatório e deve ser uma lista de strings."}), 400

        top_3_recipes = predict_favorite_recipes_naive_bayes(ingredients)
        
        return jsonify({"recommendations": top_3_recipes}), 200

    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        import traceback
        print(f"Erro ao processar recomendação: {str(e)}")
        print(traceback.format_exc())
        return jsonify({"error": f"Erro interno: {str(e)}"}), 500