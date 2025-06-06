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
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
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

def print_evaluation_metrics(y_tested, y_predicted, label_encoder_obj=None):
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
    
    class_names = label_encoder_obj.classes_ if label_encoder_obj else None
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
y_raw = [] 

for user in fake_users_profiles:
    X.append(one_hot_encode_fake_data(user["liked_ingredients"], unique_ingredients))
    y_raw.append(user["most_liked_recipe"])

label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y_raw) 

X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.3, random_state=42 
)

X_train_np = pd.DataFrame(X_train).values
X_test_np = pd.DataFrame(X_test).values

# ==============================
# PLOTAGEM DO PAIRPLOT ANTES DO PCA
# ==============================
os.makedirs(ANALYSES_DIR, exist_ok=True)

X_train_df_original = pd.DataFrame(X_train, columns=unique_ingredients) 

num_features_for_pairplot = 10
features_for_pairplot = X_train_df_original.columns[:num_features_for_pairplot].tolist()

pairplot_df = X_train_df_original[features_for_pairplot].copy()
pairplot_df['most_liked_recipe'] = y_train 
pairplot_df['most_liked_recipe_name'] = label_encoder.inverse_transform(y_train)

print("\nGerando PairPlot... Isso pode levar alguns minutos se o número de features for alto.")
pair_plot_grid = sns.pairplot(pairplot_df, hue='most_liked_recipe_name', diag_kind='kde')
pair_plot_grid.fig.suptitle(f'Pair Plot das Primeiras {num_features_for_pairplot} Features por Tipo de Receita (Pré-PCA)', y=1.02) 
plt.savefig(os.path.join(ANALYSES_DIR, "pair_plot_pre_pca.png"))
plt.show()
print("PairPlot gerado e salvo em 'analyses/pair_plot_pre_pca.png'")

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_np)
X_test_scaled = scaler.transform(X_test_np)

pca = PCA(n_components=PCA_N_COMPONENTS)

X_train_pca = pca.fit_transform(X_train_scaled)
X_test_pca = pca.transform(X_test_scaled)

print(f"\nNúmero original de features: {X_train_scaled.shape[1]}")
print(f"Número de componentes PCA selecionados para {PCA_N_COMPONENTS*100}% de variância: {pca.n_components_}")

os.makedirs(ANALYSES_DIR, exist_ok=True)
plt.figure(figsize=(10, 6))
plt.plot(range(1, len(pca.explained_variance_ratio_) + 1), pca.explained_variance_ratio_.cumsum(), marker='o', linestyle='--')
plt.title('Variância Explicada Acumulada por Componentes Principais')
plt.xlabel('Número de Componentes Principais')
plt.ylabel('Variância Explicada Acumulada')
plt.grid(True)
plt.axhline(y=PCA_N_COMPONENTS, color='r', linestyle='-', label=f'{PCA_N_COMPONENTS*100}% de Variância')
plt.axvline(x=pca.n_components_, color='g', linestyle='--', label=f'{pca.n_components_} Componentes para {PCA_N_COMPONENTS*100}%')
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(ANALYSES_DIR, "pca_explained_variance.png"))
plt.show()

grid_params = {
    "n_neighbors": [3, 5, 7, 9, 11, 13, 15],
    "metric": ['cosine', 'euclidean', 'manhattan'],
    "weights": ['uniform', 'distance'],
}

model = KNeighborsClassifier()

gs = GridSearchCV(model, grid_params, cv=5, scoring='precision_weighted', n_jobs=-1, return_train_score=True) 
gs.fit(X_train_pca, y_train)

print("Melhores parâmetros (com PCA): ", gs.best_params_)

model = gs.best_estimator_

cross_validation_results = cross_validate(
    model,
    X_train_pca,
    y_train,
    cv=5,
    scoring=['accuracy', 'precision_weighted', 'recall_weighted', 'f1_weighted'],
    return_train_score=True,
    error_score='raise'
)

y_pred = model.predict(X_test_pca)

print_cross_validation_results(cross_validation_results)
print_evaluation_metrics(y_test, y_pred, label_encoder)

k_values = [3, 5, 7, 9, 11, 13, 15]

results_df = pd.DataFrame(gs.cv_results_)

plot_df = results_df[[
    'param_n_neighbors',
    'param_metric',
    'param_weights',
    'mean_test_score' 
]].copy()

plot_df.rename(columns={
    'param_n_neighbors': 'Número de Vizinhos (n_neighbors)',
    'param_metric': 'Métrica',
    'param_weights': 'Peso',
    'mean_test_score': 'Score de Precisão Ponderada (Validação)'
}, inplace=True)

plot_df['Número de Vizinhos (n_neighbors)'] = pd.to_numeric(plot_df['Número de Vizinhos (n_neighbors)'])

plt.figure(figsize=(12, 8))
sns.lineplot(
    data=plot_df,
    x='Número de Vizinhos (n_neighbors)',
    y='Score de Precisão Ponderada (Validação)',
    hue='Métrica',
    style='Peso',
    markers=True,
    dashes=False,
    palette='deep'
)

plt.title('Comparação de Precisão Ponderada de Validação por N. Vizinhos, Métrica e Peso (com PCA)')
plt.xlabel('Número de Vizinhos (n_neighbors)')
plt.ylabel('Score de Precisão Ponderada (Validação)')
plt.xticks(k_values)
plt.grid(True)
plt.legend(title='Parâmetros', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.savefig(os.path.join(ANALYSES_DIR, "knn_test_precision_weighted_comparison_pca.png"))
plt.show()

os.makedirs("models", exist_ok=True)

data_to_save = {
    "model": model,
    "unique_ingredients": unique_ingredients,
    "pca": pca,
    "scaler": scaler,
    "label_encoder": label_encoder 
}

with open(MODEL_FILE_PATH, "wb") as model_file:
    pickle.dump(data_to_save, model_file)
    
print(f"Modelo KNNClassifier, PCA, scaler, label encoder e ingredientes únicos salvos com sucesso em '{MODEL_FILE_PATH}'!!!!")