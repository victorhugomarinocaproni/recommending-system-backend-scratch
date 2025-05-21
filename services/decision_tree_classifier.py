import json
import os
import sys
from sklearn.ensemble import RandomForestClassifier
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

def print_cross_validation_results(results):
    print()
    print("===== Resultados da Validação Cruzada Por Fold =====")
    print()
    print("Acurácia :", results['test_accuracy'])
    print("Precisão :", results['test_precision_macro'])
    print("Revocação :", results['test_recall_macro'])
    print("F1 :", results['test_f1_macro'])
    print("Acurácia (treinamento):", results['train_accuracy'])
    print("Precisão (treinamento):", results['train_precision_macro'])
    print("Revocação (treinamento):", results['train_recall_macro'])
    print("F1 (treinamento):", results['train_f1_macro'])
    print()
    print("===== Resultados da Validação Cruzada Média =====")
    print()
    print("Acurácia média:", results['test_accuracy'].mean())
    print("Precisão média:", results['test_precision_macro'].mean())
    print("Revocação média:", results['test_recall_macro'].mean())
    print("F1 média:", results['test_f1_macro'].mean())
    print("Acurácia média (treinamento):", results['train_accuracy'].mean())
    print("Precisão média (treinamento):", results['train_precision_macro'].mean())
    print("Revocação média (treinamento):", results['train_recall_macro'].mean())
    print("F1 média (treinamento):", results['train_f1_macro'].mean())
    print()
    print("====================================")
    print()

def print_evaluation_metrics(y_tested, y_predicted):
    
    confusion_matrix_result = confusion_matrix(y_tested, y_predicted)
    
    print("==== MÉTRICAS DE AVALIAÇÃO APÓS TREINAMENTO E PREDIÇÃO DO MODELO ====")
    print()
    print(f"Acurácia: {accuracy_score(y_tested, y_predicted):.2f}")
    print(f"Precisão macro: {precision_score(y_tested, y_predicted, average='macro'):.2f}")
    print(f"Recall macro: {recall_score(y_tested, y_predicted, average='macro'):.2f}")
    print(f"F1 Score macro: {f1_score(y_tested, y_predicted, average='macro'):.2f}")
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

DATA_FILE_PATH = "data/generated_users.json"
MODEL_FILE_PATH = "models/recipe_dt_classifier_model.pkl"

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

model = RandomForestClassifier(random_state=42)

cross_validation_results = cross_validate(
    model,
    X_train,
    y_train,
    cv=5,
    scoring=['accuracy', 'precision_macro', 'recall_macro', 'f1_macro'],
    return_train_score=True
)

model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print_cross_validation_results(cross_validation_results)
print_evaluation_metrics(y_test, y_pred)

os.makedirs("models", exist_ok=True)

data_to_save = {
    "model": model,
    "unique_ingredients": unique_ingredients
}

with open(MODEL_FILE_PATH, "wb") as model_file:
    pickle.dump(data_to_save, model_file)
    
print(f"Modelo e ingredientes únicos salvos com sucesso em '{MODEL_FILE_PATH}'!!!!")