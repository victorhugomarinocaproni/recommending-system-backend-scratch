import json
import os
import sys
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

ingredients = sorted([
    "aperol", "espumante", "água com gás", "laranja", "rum", "limão", "hortelã",
    "açúcar", "tequila", "licor de laranja", "leite de coco", "abacaxi", "vodka",
    "licor de pêssego", "suco de laranja", "groselha", "gengibre", "vermouth",
    "angostura", "gin", "tônica", "cranberry", "campari", "cola", "suco de tomate",
    "molho inglês", "pimenta", "cachaça"
])

recipes = [
    {"aperol_spritz": ["aperol", "espumante", "água com gás", "laranja"]},
    {"mojito_tradicional": ["rum", "limão", "hortelã", "açúcar", "água com gás"]},
    {"margarita_facil": ["tequila", "licor de laranja", "limão"]},
    {"pina_colada": ["rum", "leite de coco", "abacaxi"]},
    {"sex_on_the_beach": ["vodka", "licor de pêssego", "suco de laranja", "groselha"]},
    {"moscow_mule": ["vodka", "gengibre", "limão", "água com gás"]},
    {"manhattan": ["whisky", "vermouth", "angostura"]},
    {"old_fashioned": ["whisky", "açúcar", "angostura", "laranja"]},
    {"gin_tonica_com_limao": ["gin", "tônica", "limão"]},
    {"cosmopolitan_perfeito": ["vodka", "licor de laranja", "limão", "cranberry"]},
    {"negroni": ["gin", "campari", "vermouth"]},
    {"cuba_libre": ["rum", "cola", "limão"]},
    {"bloody_mary": ["vodka", "suco de tomate", "limão", "molho inglês", "pimenta"]},
    {"caipirinha_de_limao": ["cachaça", "limão", "açúcar"]},
    {"daiquiri": ["rum", "limão", "açúcar"]}
]

DATASET_PATH = "data/generated_users.json"

if not os.path.exists(DATASET_PATH):
    print("File 'generated_users.json' not found.")
    print("You must run the 'generate_users.py' before running this script.")
    sys.exit(1)

def ingredients_to_vector(ingredient_list):
    return [1 if ing in ingredient_list else 0 for ing in ingredients]

with open(DATASET_PATH, "r", encoding="utf-8") as f:
    users_data = json.load(f)

X = []
y = []

recipe_vectors = {}
for recipe in recipes:
    name, ings = list(recipe.items())[0]
    recipe_vectors[name] = ingredients_to_vector(ings)

for user in users_data:
    likes = user["likes"]
    user_vector = ingredients_to_vector(likes)

    # Similaridade baseada na interseção de ingredientes
    max_similarity = -1
    best_recipe = None
    for name, recipe_vector in recipe_vectors.items():
        similarity = sum([a and b for a, b in zip(user_vector, recipe_vector)])
        if similarity > max_similarity:
            max_similarity = similarity
            best_recipe = name

    X.append(user_vector)
    y.append(best_recipe)

# Treinamento
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
clf = DecisionTreeClassifier()
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

print("Acurácia:", round(accuracy_score(y_test, y_pred) * 100, 2), "%")

def recommend_drink(user_ingredients):
    user_vector = ingredients_to_vector(user_ingredients)
    prediction = clf.predict([user_vector])
    return prediction[0]

new_user_likes = ["espumante", "laranja", "açúcar"]
recommended = recommend_drink(new_user_likes)
print(f"🥂 Bebida recomendada para {new_user_likes}: {recommended}")