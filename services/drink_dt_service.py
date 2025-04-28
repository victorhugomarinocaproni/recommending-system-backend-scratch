import json
import os
import sys
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

DATA_FILE_PATH = "data/generated_users.json"

if not os.path.exists(DATA_FILE_PATH):
    print("Arquivo 'generated_users.json' não encontrado.")
    print("Execute antes o script generate_users.py para gerar os dados.")
    sys.exit(1)

with open(DATA_FILE_PATH, "r", encoding="utf-8") as file:
    user_profiles = json.load(file)

unique_ingredients = sorted(
    list({ingredient for user in user_profiles for ingredient in user["liked_ingredients"]})
)

feature_vectors = []
target_labels = []

for user in user_profiles:
    user_features = [1 if ingredient in user["liked_ingredients"] else 0 for ingredient in unique_ingredients]
    feature_vectors.append(user_features)
    target_labels.append(user["most_liked_drink"])

model = DecisionTreeClassifier(random_state=42)
model.fit(feature_vectors, target_labels)

def transform_ingredients_to_vector(preferred_ingredients):
    return [1 if ingredient in preferred_ingredients else 0 for ingredient in unique_ingredients]

def predict_favorite_drink(preferred_ingredients):
    vector = transform_ingredients_to_vector(preferred_ingredients)    
    return model.predict([vector])[0]

if __name__ == "__main__":
    example_user = ["cachaça", "limão", "açúcar"],
    suggested_drink = predict_favorite_drink(example_user)
    print(f"Para o usuário que gosta de {example_user}, recomendamos: {suggested_drink}")
    
    
# Com esse modelo, ele faz um "predict" e retorna o nome da bebida mais recomendada para o usuário com base nos ingredientes que ele gosta.
# Mas e se o usuário não gostar da recomendação ? 
# Ele vai clicar para gerar outra bebida e o modelo vai continuar recomendando a mesma bebida... como resolver isso ?
# Uma solução seria criar um novo modelo que leve em consideração as bebidas que o usuário já não gostou.
# Isso pode ser feito criando um novo vetor de características que inclua as bebidas que o usuário não gosta e, 
# em seguida, treinar um novo modelo com esses dados.
        
 