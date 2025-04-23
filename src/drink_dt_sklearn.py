import json
import os
import sys
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

DATASET_PATH = "data/generated_users.json"

if not os.path.exists(DATASET_PATH):
    print("Arquivo 'generated_users.json' não encontrado.")
    print("Execute antes o script generate_users.py para gerar os dados.")
    sys.exit(1)

with open(DATASET_PATH, "r", encoding="utf-8") as f:
    users_data = json.load(f)

# Obter todos os ingredientes únicos de todos os usuários
all_ingredients = sorted(
    list({ing for user in users_data for ing in user["liked_ingredients"]})
)

# Criar vetores de entrada (X) e rótulos (y)
entry_data = []
labels = []

for user in users_data:
    user_vector = [1 if ingredient in user["liked_ingredients"] else 0 for ingredient in all_ingredients]
    entry_data.append(user_vector)
    labels.append(user["most_liked_drink"])

'''
=======================
TREINAMENTO E TESTE:
=======================
'''

clf = DecisionTreeClassifier(random_state=42)
clf.fit(entry_data, labels)

def vectorize_ingredients(user_ingredients):
    return [1 if ingredient in user_ingredients else 0 for ingredient in all_ingredients]

def recommend_drink(user_ingredients):
    user_vector = vectorize_ingredients(user_ingredients)
    prediction = clf.predict([user_vector])
    return prediction[0]

if __name__ == "__main__":
    test_users = [
        # ["rum", "limão", "hortelã", "açúcar"],  # Mojito tradicional
        # ["vodka", "cranberry", "limão", "licor de laranja"],  # Cosmopolitan
        # ["gin", "campari", "vermouth"],  # Negroni
        # ["rum", "leite de coco", "abacaxi"],  # Piña Colada
        ["cachaça", "limão", "açúcar"],  # Caipirinha
    ]

    for test in test_users:
        drink = recommend_drink(test)
        print(f"Para o usuário que gosta de {test}, recomendamos: {drink}")