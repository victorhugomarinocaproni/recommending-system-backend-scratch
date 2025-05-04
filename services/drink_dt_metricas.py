import json
import os
import sys
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

# Constantes
DATA_FILE_PATH = "data/generated_users.json"

# =============================

# Funções
def transform_ingredients_to_vector(preferred_ingredients):
    return [1 if ingredient in preferred_ingredients else 0 for ingredient in unique_ingredients]

def predict_favorite_drink(preferred_ingredients):
    vector = transform_ingredients_to_vector(preferred_ingredients)
    return model.predict([vector])[0]

def one_hot_encode(ingredients, unique_ingredients):
    return [1 if ingredient in ingredients else 0 for ingredient in unique_ingredients]

# =============================

# Leitura dos dados
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

# Divisão entre treino e teste
X_train, X_test, y_train, y_test = train_test_split(
    feature_vectors, target_labels, test_size=0.3, random_state=42
)

# Treinamento do modelo
model = DecisionTreeClassifier(random_state=42)
model.fit(X_train, y_train)

# Predição e métricas
y_pred = model.predict(X_test)

print("==== MÉTRICAS DE AVALIAÇÃO ====")
print(f"Acurácia: {accuracy_score(y_test, y_pred):.2f}")
print(f"Precisão macro: {precision_score(y_test, y_pred, average='macro'):.2f}")
print(f"Recall macro: {recall_score(y_test, y_pred, average='macro'):.2f}")
print(f"F1 Score macro: {f1_score(y_test, y_pred, average='macro'):.2f}")
print("\nMatriz de Confusão:")
print(confusion_matrix(y_test, y_pred))
print("\nRelatório de Classificação:")
print(classification_report(y_test, y_pred))

# =============================

# Executado apenas quando chamado diretamente
if __name__ == "__main__":
    example_user = ["vodka", "limão", "açúcar"]
    suggested_drink = predict_favorite_drink(example_user)
    print(f"\nPara o usuário que gosta de {example_user}, recomendamos: {suggested_drink}")
