import os
import json
import sys
import pickle
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_validate
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

def print_cross_validation_results(results):
    print()
    print("===== Resultados da Validação Cruzada Por Fold =====")
    print()
    print("Acurácia :", results['test_accuracy'])
    print("Precisão :", results['test_precision_weighted'])
    print("Revocação :", results['test_recall_weighted'])
    print("F1 :", results['test_f1_weighted'])
    print()
    print("===== Resultados da Validação Cruzada Média =====")
    print()
    print("Acurácia média:", results['test_accuracy'].mean())
    print("Precisão média:", results['test_precision_weighted'].mean())
    print("Revocação média:", results['test_recall_weighted'].mean())
    print("F1 média:", results['test_f1_weighted'].mean())
    print()
    print("====================================")
    print()
def print_evaluation_metrics(y_tested, y_predicted):
    
    confusion_matrix_result = confusion_matrix(y_tested, y_predicted)
    
    print("==== MÉTRICAS DE AVALIAÇÃO APÓS TREINAMENTO E PREDIÇÃO DO MODELO ====")
    print()
    print(f"Acurácia: {accuracy_score(y_tested, y_predicted):.2f}")
    print(f"Precisão weighted: {precision_score(y_tested, y_predicted, average='weighted'):.2f}")
    print(f"Recall weighted: {recall_score(y_tested, y_predicted, average='weighted'):.2f}")
    print(f"F1 Score weighted: {f1_score(y_tested, y_predicted, average='weighted'):.2f}")
    print()
    print("Matriz de Confusão: ")
    print(confusion_matrix_result)
    sns.heatmap(confusion_matrix_result, annot=True, fmt='d', cmap='Blues')
    plt.title("Matriz de Confusão")
    plt.xlabel("Predito")
    plt.ylabel("Real")
    plt.show()
    print()
    print("Relatório de Classificação: ")
    print(classification_report(y_tested, y_predicted, zero_division=0))
    print()
    print("====================================")

# =======================
# Constantes: 
# =======================
PCA_N_COMPONENTS = 0.95
ANALYSES_DIR = "analyses"
DATA_FILE_PATH = "data/generated_users.json"
MODEL_FILE_PATH = "models/knn_classifier_model.pkl"

fake_users_profiles = None

if not os.path.exists(DATA_FILE_PATH):
    print("Arquivo 'generated_users.json' não encontrado.")
    print("Execute antes o script generate_users.py para gerar os dados.")
    sys.exit(1)

with open(DATA_FILE_PATH, "r", encoding="utf-8") as dataset_file:
    fake_users_profiles = json.load(dataset_file)
    
all_ingredients = set()

for user in fake_users_profiles:
    all_ingredients.update(user["liked_ingredients"])
    
unique_ingredients = sorted(all_ingredients)

def one_hot_encode_fake_data(user_liked_ingredients, unique_ingredients):
    return [1 if ingredient in user_liked_ingredients else 0 for ingredient in unique_ingredients]

X = []
y = []

for user in fake_users_profiles:
    X.append(one_hot_encode_fake_data(user["liked_ingredients"], unique_ingredients))
    y.append(user["most_liked_recipe"])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

grid_params = {
    "n_neighbors": [3, 5, 7, 9, 11, 13, 15],
    "metric": ['cosine', 'euclidean', 'manhattan'],
    "weights": ['uniform', 'distance'],
}

model = KNeighborsClassifier()

gs = GridSearchCV(model, grid_params, cv=5, scoring='precision_weighted', n_jobs=-1)
gs.fit(X_train, y_train)

print("Melhores parâmetros: ", gs.best_params_)

model = gs.best_estimator_

cross_validation_results = cross_validate(
    model,
    X_train,
    y_train,
    cv=5,
    scoring=['accuracy', 'precision_weighted', 'recall_weighted', 'f1_weighted'],
    return_train_score=True
)

y_pred = model.predict(X_test)

print_cross_validation_results(cross_validation_results)
print_evaluation_metrics(y_test, y_pred)

# ==============================
# Plotagem dos Dados
# ==============================
scores = []
k_values = [3, 5, 7, 9, 11, 13, 15]
for k in k_values:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    
    score = accuracy_score(y_test, y_pred)  # Ou f1_score(...), etc
    scores.append(score)


os.makedirs("analyses", exist_ok=True)

plt.figure(figsize=(10, 6))
plt.plot(k_values, scores, marker='o', linestyle='-', color='b')
plt.title('Acurácia vs Valor de K no KNN')
plt.xlabel('Número de Vizinhos (K)')
plt.ylabel('Acurácia')
plt.xticks(k_values)
plt.grid(True)
plt.tight_layout()
plt.savefig("analyses/knn_accuracy_vs_k.png")
plt.show()

os.makedirs("models", exist_ok=True)

data_to_save = {
    "model": model,
    "unique_ingredients": unique_ingredients
}

with open(MODEL_FILE_PATH, "wb") as model_file:
    pickle.dump(data_to_save, model_file)
    
print(f"Modelo KNNClassifier e ingredientes únicos salvos com sucesso em '{MODEL_FILE_PATH}'!!!!")

"""
def recommend_top_3_foods_knn(user_profile):
    # Normalizar o perfil do usuário
    user_profile_normalized = scaler.transform([user_profile])

    # Aplicar PCA no perfil do usuário
    user_profile_pca = pca.transform(user_profile_normalized)

    # Faz o 'fit' do modelo KNN com os dados PCA
    knn_recommender = NearestNeighbors(n_neighbors=KNN_VALUE, metric=KNN_METRIC)
    knn_recommender.fit(X_pca) 

    distances, indices = knn_recommender.kneighbors(user_profile_pca)
    
    result = []
    for idx, i in enumerate(indices[0]):
        recipe_name = y[i]
        ingredients = recipes[recipe_name]
        similarity_score = 1 - distances[0][idx]

        result.append({
            "name": recipe_name,
            "ingredients": ingredients,
            "score": round(similarity_score, 2),
            "instructions": ""
        })
    
    return result
""" 

