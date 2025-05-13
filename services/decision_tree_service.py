import json
import os
import sys
from sklearn.ensemble import RandomForestClassifier

# =============================
# Constantes: 

DATA_FILE_PATH = "data/generated_users.json"

# =============================
# Funções:

def transform_ingredients_to_vector(preferred_ingredients):
    return [1 if ingredient in preferred_ingredients else 0 for ingredient in unique_ingredients]

def predict_favorite_recipe(preferred_ingredients):
    vector = transform_ingredients_to_vector(preferred_ingredients)    
    probabilities = model.predict_proba([vector])[0]
    top_5_indexes = probabilities.argsort()[-5:][::-1]
    top_5_recipes = [model.classes_[i] for i in top_5_indexes]
    return top_5_recipes

def one_hot_encode(ingredients, unique_ingredients):
    return [1 if ingredient in ingredients else 0 for ingredient in unique_ingredients]

# =============================
# "Main" do script:

if not os.path.exists(DATA_FILE_PATH):
    print("Arquivo 'generated_users.json' não encontrado.")
    print("Execute antes o script 'generate_users.py' para gerar os dados.")
    sys.exit(1)

with open(DATA_FILE_PATH, "r", encoding="utf-8") as dataset_file:
    user_profiles = json.load(dataset_file)
    
ingredients = set()

for user in user_profiles:
    ingredients.update(user["liked_ingredients"])
        
unique_ingredients = sorted(ingredients)

feature_vectors = []
target_labels = []

for user in user_profiles:
    user_features = one_hot_encode(user["liked_ingredients"], unique_ingredients)
    feature_vectors.append(user_features)
    target_labels.append(user["most_liked_recipe"])

model = RandomForestClassifier(random_state=42)
model.fit(feature_vectors, target_labels)

# =============================
# Executado apenas quando o script é chamado diretamente:

if __name__ == "__main__":
    
    example_user = [
        "pimenta jalapeno",
        "ovo",
        "mostarda",
        "limão",
        "picles",
        "cebola",
        "coentro",
        "abacaxi"
      ]
    
    suggestion = predict_favorite_recipe(example_user)
    
    print(f"Para o usuário que gosta de {example_user}, recomendamos: {suggestion}")
    