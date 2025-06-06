import os
import sys
import pandas as pd
import json
import pickle
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.naive_bayes import MultinomialNB
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
from sklearn.preprocessing import LabelEncoder 

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
MODEL_FILE_PATH = "models/naive_bayes_classifier.pkl"
ANALYSES_DIR = "analyses" 

os.makedirs(ANALYSES_DIR, exist_ok=True)

data = None

if not os.path.exists(DATA_FILE_PATH):
    print("Arquivo 'generated_users.json' não encontrado.")
    print("Execute antes o script generate_users.py para gerar os dados.")
    sys.exit(1)

with open(DATA_FILE_PATH, "r", encoding="utf-8") as dataset_file:
    data = json.load(dataset_file)
    
df = pd.DataFrame(data)
    
all_ingredients = set()

for liked_ingredients in df['liked_ingredients']:
    all_ingredients.update(liked_ingredients)
    
unique_ingredients = sorted(all_ingredients)

def one_hot_encode_fake_data(user_liked_ingredients, unique_ingredients):
    return [1 if ingredient in user_liked_ingredients else 0 for ingredient in unique_ingredients]

X = df['liked_ingredients'].apply(lambda ingredients: one_hot_encode_fake_data(ingredients, unique_ingredients)).tolist()
y_raw = df['most_liked_recipe'].tolist() 

label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y_raw) 
class_names = label_encoder.classes_ 

X_train, X_test, y_train_encoded, y_test_encoded = train_test_split( 
    X, y_encoded, test_size=0.3, random_state=42
)

params = {
    'alpha': [0.1, 0.5, 1.0, 2.0],
    'fit_prior': [True, False]
}

model = MultinomialNB()

gs = GridSearchCV(model, params, cv=5, scoring='precision_weighted', n_jobs=-1, return_train_score=True)

gs.fit(X_train, y_train_encoded) 

model = gs.best_estimator_

cross_validation_results = cross_validate(
    model,
    X_train,
    y_train_encoded,
    cv=5,
    scoring=['accuracy', 'precision_weighted', 'recall_weighted', 'f1_weighted'],
    return_train_score=True
)

y_pred_encoded = model.predict(X_test)

print_cross_validation_results(cross_validation_results)
print_evaluation_metrics(y_test_encoded, y_pred_encoded, class_names=class_names)

# ==============================
# Plotar resultados do GridSearchCV 
# ==============================
results_df = pd.DataFrame(gs.cv_results_)

plt.figure(figsize=(10, 6))
sns.lineplot(
    data=results_df,
    x='param_alpha',
    y='mean_test_score',
    hue='param_fit_prior',
    marker='o',
    dashes=False
)
plt.title('Precisão Ponderada (Validação) vs. Alpha e Fit_Prior (Naive Bayes)')
plt.xlabel('Alpha')
plt.ylabel('Precisão Ponderada (Validação)')
plt.xscale('log') 
plt.grid(True)
plt.legend(title='Fit Prior')
plt.tight_layout()
plt.savefig(os.path.join(ANALYSES_DIR, "naive_bayes_gridsearch_results.png"))
plt.show()

os.makedirs("models", exist_ok=True)

data_to_save = {
    "model": model,
    "unique_ingredients": unique_ingredients,
    "label_encoder": label_encoder 
}

with open(MODEL_FILE_PATH, "wb") as model_file:
    pickle.dump(data_to_save, model_file)
    
print(f"Modelo NaiveBayes, ingredientes únicos e LabelEncoder salvos com sucesso em '{MODEL_FILE_PATH}'!!!!")