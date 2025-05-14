from sklearn.model_selection import cross_validate
import numpy as np
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
    return model.predict([vector])[0]

def one_hot_encode(ingredients, unique_ingredients):
    return [1 if ingredient in ingredients else 0 for ingredient in unique_ingredients]

# =============================
# Leitura dos dados:

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
    target_labels.append(user["most_liked_recipe"])

model = RandomForestClassifier(random_state=42)

# Cross-validation com 5 folds
cv_results = cross_validate(
    model,
    feature_vectors,
    target_labels,
    cv=5,
    scoring=['accuracy', 'precision_macro', 'recall_macro', 'f1_macro'],
    return_train_score=False
)

# Exibição das médias das métricas
print("==== MÉTRICAS DE CROSS-VALIDATION (5 Folds) ====")
print(f"Acurácia média: {np.mean(cv_results['test_accuracy']):.2f}")
print(f"Precisão macro média: {np.mean(cv_results['test_precision_macro']):.2f}")
print(f"Recall macro médio: {np.mean(cv_results['test_recall_macro']):.2f}")
print(f"F1 Score macro médio: {np.mean(cv_results['test_f1_macro']):.2f}")
