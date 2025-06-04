import os
import sys
import pickle
import numpy as np

recipes = {
    "Quesadilla de Barbacoa": ["carne de boi", "queijo", "ovo", "mostarda", "pimenta jalapeno", "nirá"],
    "Taco Pescado Baja": ["carne de peixe", "repolho", "abacate", "tomate", "pimenta jalapeno", "ovo", "mostarda", "limão", "picles", "cebola", "coentro"],
    "Taco Camarão": ["camarão", "repolho", "ovo", "mostarda", "abacate", "tomate", "pimenta jalapeno", "coentro", "picles"],
    "Taco Barbacoa": ["carne de boi", "queijo", "cebola", "ciboullete", "coentro", "picles", "pimenta jalapeno"],
    "Gringa Coreana": ["tortilla", "milho", "queijo", "carne de porco", "acelga", "nabo", "picles", "limão", "cebola"],
    "Taco Birria": ["carne de boi", "queijo", "cebola", "coentro", "pimenta jalapeno", "carne"],
    "Taco La Flor": ["couve-flor", "feijão", "grão de bico", "ciboullete"],
    "Taco de Chorizo": ["carne de porco", "queijo", "coentro", "tomate", "cebola", "limão"],
    "Taco Al Pastor": ["abóbora", "alho", "carne de porco", "abacaxi", "canela", "salsa", "cebola", "coentro"],
    "Taco Lengua": ["cogumelo", "carne de boi", "castanha do pará", "tomate", "cebola", "limão"],
    "Taco Vegano": ["abacate", "tomate", "pimenta jalapeno", "coentro", "cogumelo", "salsa", "cebola", "nozes", "nirá", "gergilim"],
    "Taco Portenho": ["abóbora", "queijo", "carne de boi", "pimentão", "cebola", "salsa", "alho", "milho"],
    "Quesadilla de Cogumelos": ["queijo", "cogumelos", "nirá"],
    "Burrito": ["tortilla", "trigo", "carne de frango", "queijo", "feijão", "tomate", "cebola", "limão", "repolho", "ovo", "mostarda"],
    "Chimichanga": ["tortilla", "carne de boi", "queijo", "repolho", "ovo", "mostarda", "feijão", "tomate", "cebola", "limão", "abacate"],
    "Burrito Al Pastor": ["tortilla", "trigo", "carne de porco", "abacaxi", "queijo", "feijão", "tomate", "cebola", "limão", "repolho"]
}

def one_hot_encode_user_data(user_liked_ingredients, unique_ingredients):
    return [1 if ingredient in user_liked_ingredients else 0 for ingredient in unique_ingredients]

MODEL_FILE_PATH = "models/recipe_dt_classifier_model.pkl"

model = None

if not os.path.exists(MODEL_FILE_PATH):
    print("Modelo 'recipe_dt_classifier_model.pkl' não encontrado.")
    print("Execute antes o script 'decision_tree_classifier' para gerar e treinar o modelo.")
    sys.exit(1)

with open(MODEL_FILE_PATH, "rb") as model_file:
    loaded_data = pickle.load(model_file)
    
model = loaded_data["model"]
unique_ingredients = loaded_data["unique_ingredients"]
    
def predict_favorite_recipes_random_forest(user_liked_ingredients):
    if model is None:
        raise ValueError("Modelo não carregado. Verifique o caminho do arquivo do modelo.")
    
    one_hot_encoded_user_tastes = one_hot_encode_user_data(user_liked_ingredients, unique_ingredients)
    
    probs = model.predict_proba([one_hot_encoded_user_tastes])[0]
    
    top3_labels_index = np.argsort(-probs)[:3]
    top3_probs = -np.sort(-probs)[:3]
    top3_recipes = [model.classes_[i] for i in top3_labels_index]

    result = []
    for recipe_name, score in zip(top3_recipes, top3_probs):
        ingredients = recipes.get(recipe_name, [])
        result.append({
            "name": recipe_name,
            "ingredients": ingredients,
            "score": float(score),
            "instructions": ""
        })
    return result
    