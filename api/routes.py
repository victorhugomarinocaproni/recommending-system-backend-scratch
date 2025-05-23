from flask import Blueprint, request, jsonify
from flasgger import swag_from
from services.knn_service import recommend_top_5_foods_knn, get_user_profile


from services.decision_tree_service import predict_favorite_recipes

api_blueprint = Blueprint("api", __name__)

@api_blueprint.route("knn/recommendation", methods=["POST"])
@swag_from({
    'tags': ['Recomendações'],
    'summary': 'Retorna as 5 receitas mais recomendadas com base no algoritmo de KNN',
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
            'description': 'Lista das 5 receitas mais compatíveis',
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
    
        if not ingredients:
            return jsonify({"error": "Parâmetro 'ingredients' é obrigatório e deve ser uma lista de strings."}), 400

        user_profile = get_user_profile(ingredients)
        
        top_3_drinks = recommend_top_5_foods_knn(user_profile)
        
        return jsonify({"recommendations": top_3_drinks}), 200

    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        import traceback
        print(f"Erro ao processar recomendação: {str(e)}")
        print(traceback.format_exc())
        return jsonify({"error": f"Erro interno: {str(e)}"}), 500
    
  

@api_blueprint.route("decision-tree/recommendation", methods=["POST"])
@swag_from({
    'tags': ['Recomendações'],
    'summary': 'Retorna as 5 receitas mais recomendadas a partir de um modelo de árvore de decisão',
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
            'description': 'Lista das 5 receitas mais compatíveis',
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
        
        top_3_recipes = predict_favorite_recipes(ingredients)
                
        return jsonify({"recommendations": top_3_recipes}), 200

    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        import traceback
        print(f"Erro ao processar recomendação: {str(e)}")
        print(traceback.format_exc())
        return jsonify({"error": f"Erro interno: {str(e)}"}), 500