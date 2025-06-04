import os
import sys
import pickle
import numpy as np

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
    
    # Debug: veja as probabilidades e as classes
    print("Probabilidades:", probs)
    print("Classes:", model.classes_)
    
    top3_labels_index = np.argsort(-probs)[:3]
    top3_probs = -np.sort(-probs)[:3]
    top3_recipes = [model.classes_[i] for i in top3_labels_index]
    
    # Debug: veja as receitas e suas probabilidades
    print("Top 3 receitas:", top3_recipes)
    print("Probabilidades top 3:", probs[top3_labels_index])
    
    return [
        {"recipe": recipe, "probability": float(prob)}
        for recipe, prob in zip(top3_recipes, probs[top3_labels_index])
    ]
    