import json
import os
import sys
from sklearn.tree import DecisionTreeClassifier

# Constantes: 
DATA_FILE_PATH = "data/generated_users.json"

# =============================

# Funções:
def transform_ingredients_to_vector(preferred_ingredients):
    return [1 if ingredient in preferred_ingredients else 0 for ingredient in unique_ingredients]

def predict_favorite_drink(preferred_ingredients):
    vector = transform_ingredients_to_vector(preferred_ingredients)    
    return model.predict([vector])[0]

def one_hot_encode(ingredients, unique_ingredients):
    return [1 if ingredient in ingredients else 0 for ingredient in unique_ingredients]

# =============================

# "Main" do script:
if not os.path.exists(DATA_FILE_PATH):
    print("Arquivo 'generated_users.json' não encontrado.")
    print("Execute antes o script generate_users.py para gerar os dados.")
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
    target_labels.append(user["most_liked_drink"])

model = DecisionTreeClassifier(random_state=42)
model.fit(feature_vectors, target_labels)

# =============================

# Executado apenas quando o script é chamado diretamente:
if __name__ == "__main__":
    
    example_user = ["cachaça", "limão", "açúcar"]
    
    suggested_drink = predict_favorite_drink(example_user)
    
    print(f"Para o usuário que gosta de {example_user}, recomendamos: {suggested_drink}")
    