import json
import os
import sys
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_validate
from sklearn.model_selection import train_test_split
import seaborn as sns
import matplotlib.pyplot as plt
import pickle
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)
from sklearn.tree import plot_tree 

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

def print_evaluation_metrics(y_tested, y_predicted, class_names=None):
    
    confusion_matrix_result = confusion_matrix(y_tested, y_predicted)
    
    print("==== MÉTRICAS DE AVALIAÇÃO APÓS TREINAMENTO E PREDIÇÃO DO MODELO ====")
    print()
    print(f"Acurácia: {accuracy_score(y_tested, y_predicted):.2f}")
    print(f"Precisão weighted: {precision_score(y_tested, y_predicted, average='weighted', zero_division=0):.2f}") 
    print(f"Recall weighted: {recall_score(y_tested, y_predicted, average='weighted', zero_division=0):.2f}") 
    print(f"F1 Score weighted: {f1_score(y_tested, y_predicted, average='weighted', zero_division=0):.2f}") 
    print()
    print("Matriz de Confusão: ")
    print(confusion_matrix_result)
    sns.heatmap(confusion_matrix_result, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title("Matriz de Confusão")
    plt.xlabel("Predito")
    plt.ylabel("Real")
    plt.show()
    print()
    print("Relatório de Classificação: ")
    print(classification_report(y_tested, y_predicted, zero_division=0, target_names=class_names))
    print()
    print("====================================")

DATA_FILE_PATH = "data/generated_users.json"
MODEL_FILE_PATH = "models/recipe_dt_classifier_model.pkl" 
ANALYSES_DIR = "analyses"

os.makedirs(ANALYSES_DIR, exist_ok=True) 

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
y_raw = []

for user in fake_users_profiles:
    X.append(one_hot_encode_fake_data(user["liked_ingredients"], unique_ingredients))
    y_raw.append(user["most_liked_recipe"])

from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y_raw)
class_names = label_encoder.classes_ 
feature_names = unique_ingredients 

X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.3, random_state=42
)

grid_params = {
    "n_estimators": [25, 50, 100, 115],
    'max_depth': [None, 5, 10], 
    'min_samples_leaf': [1, 5]
}

model = RandomForestClassifier(random_state=42)

gs = GridSearchCV(model, grid_params, cv=5, scoring='precision_weighted', n_jobs=-1) 

gs.fit(X_train, y_train) 

model = gs.best_estimator_

cross_validation_results = cross_validate(
    model,
    X_train,
    y_train,
    cv=5,
    scoring=['accuracy', 'precision_weighted', 'recall_weighted', 'f1_weighted'],
    return_train_score=False 
)

y_pred = model.predict(X_test) 

print_cross_validation_results(cross_validation_results)
print_evaluation_metrics(y_test, y_pred, class_names=class_names)

# ==============================
# PLOTAGEM DE UMA ÁRVORE DO RANDOM FOREST
# ==============================
if len(model.estimators_) > 0:
    tree_to_plot = model.estimators_[0] 

    plt.figure(figsize=(25, 15)) 
    plot_tree(
        tree_to_plot,
        feature_names=feature_names,
        class_names=class_names,
        filled=True,
        rounded=True,
        fontsize=8,
        max_depth=3
    )
    plt.title('Uma Árvore de Decisão do Random Forest (Primeira Árvore, Profundidade Máx. 3)', fontsize=15)
    plt.savefig(os.path.join(ANALYSES_DIR, "random_forest_single_tree.png"), dpi=300)
    plt.show()
    print(f"Uma árvore do Random Forest salva em '{ANALYSES_DIR}/random_forest_single_tree.png'")
else:
    print("RandomForestClassifier não possui estimadores. Verifique se o modelo foi treinado.")

os.makedirs("models", exist_ok=True)

data_to_save = {
    "model": model,
    "unique_ingredients": unique_ingredients,
    "label_encoder": label_encoder 
}

with open(MODEL_FILE_PATH, "wb") as model_file:
    pickle.dump(data_to_save, model_file)
    
print(f"Modelo Random Forest e ingredientes únicos salvos com sucesso em '{MODEL_FILE_PATH}'!!!!")