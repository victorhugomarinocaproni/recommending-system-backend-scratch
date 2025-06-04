from sklearn.neighbors import NearestNeighbors
from pprint import pprint
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import Normalizer
import os

# =======================
# Constantes: 

KNN_VALUE = 3
KNN_METRIC = "cosine" 
KNN_WEIGHT = "uniform" # -> Não existe para NearestNeighbors, apenas para KNeighborsClassifier ou KNeighborsRegressor
PCA_N_COMPONENTS = 3
ANALYSES_DIR = "analyses"

recipes = {
    "Quesadilla de Barbacoa": ["carne de boi", "queijo", "ovo", "mostarda", "pimenta jalapeno", "nirá"],
    "Taco Pescado Baja": ["carne de peixe", "repolho", "abacate", "tomate", "pimenta jalapeno", "ovo", "mostarda", "limão", "picles", "cebola", "coentro"],
    "Taco Camarão": ["camarão", "repolho", "ovo", "mostarda", "abacate", "tomate", "pimenta jalapeno", "coentro", "picles"],
    "Taco Barbacoa": ["carne de boi", "queijo", "cebola", "ciboullete", "coentro", "picles", "pimenta jalapeno"],
    "Gringa Coreana": ["tortilla", "milho", "queijo", "carne de porco", "acelga", "nabo", "picles", "limão", "cebola"],
    "Taco Birria": ["carne de boi", "queijo", "cebola", "coentro", "pimenta jalapeno", "carne"],
    "Taco La Flor": ["couve-flor", "feijão", "grão de bico", "ciboullete"],
    "Taco de Chorizo": ["carne de porco", "queijo", "coentro", "tomate", "cebola", "limão"],
    "Taco Al Pastor": ["abóbora", "alho", "carne de porco", "abacaxi", "canela", "salsa", "cebola", "coentro"],
    "Taco Lengua": ["cogumelo", "carne de boi", "castanha do pará", "tomate", "cebola", "limão"],
    "Taco Vegano": ["abacate", "tomate", "pimenta jalapeno", "coentro", "cogumelo", "salsa", "cebola", "nozes", "nirá", "gergilim"],
    "Taco Portenho": ["abóbora", "queijo", "carne de boi", "pimentão", "cebola", "salsa", "alho", "milho"],
    "Quesadilla de Cogumelos": ["queijo", "cogumelos", "nirá"],
    "Burrito": ["tortilla", "trigo", "carne de frango", "queijo", "feijão", "tomate", "cebola", "limão", "repolho", "ovo", "mostarda"],
    "Chimichanga": ["tortilla", "carne de boi", "queijo", "repolho", "ovo", "mostarda", "feijão", "tomate", "cebola", "limão", "abacate"],
    "Burrito Al Pastor": ["tortilla", "trigo", "carne de porco", "abacaxi", "queijo", "feijão", "tomate", "cebola", "limão", "repolho"]
}

ingredient_set = set()
for ingredients in recipes.values():
    ingredient_set.update(ingredients)
    
unique_ingredients = sorted(list(ingredient_set))

def one_hot_encode_vector(recipe_ingredients, ingredients_list):
    return [1 if ingredient in recipe_ingredients else 0 for ingredient in ingredients_list]

X = []
y = []

for name, ingredients in recipes.items():
    X.append(one_hot_encode_vector(ingredients, unique_ingredients))
    y.append(name)
    
# Normalizar 
scaler = Normalizer()  
X_scaled = scaler.fit_transform(X)
    
# Aplicar o PCA
pca = PCA(n_components=PCA_N_COMPONENTS)
X_pca = pca.fit_transform(X_scaled)

def get_user_profile_one_hot_encoded(user_desired_ingredients):
    for ingredient in user_desired_ingredients:
        if ingredient not in unique_ingredients:
            print(f"Ingrediente inválido: {ingredient}. Ingredientes válidos: " + ", ".join(unique_ingredients))
            raise ValueError(f"Ingrediente inválido: {ingredient}. Ingredientes válidos: " + ", ".join(unique_ingredients))
    return one_hot_encode_vector(user_desired_ingredients, unique_ingredients)

# Coletando melhores parâmetros para o modelo KNN
grid_params = {
    "n_neighbors": [3, 5, 7, 9, 11, 13, 15],
    "metric": ['cosine', 'euclidean', 'manhattan'],
    "weights": ['uniform', 'distance']
}

knn_model = KNeighborsClassifier()

best_score = -1
best_params = None
results = []

for n_neighbors in grid_params["n_neighbors"]:
    for metric in grid_params["metric"]:
        for weight in grid_params["weights"]:
            model = KNeighborsClassifier(n_neighbors=n_neighbors, metric=metric)
            model.fit(X_pca, y)
            score = model.score(X_pca, y)
            # print(f"Parâmetros: n_neighbors={n_neighbors}, metric={metric}, weights={weight}, Score: {score:.4f}")
            results.append((n_neighbors, metric, weight, score))
            if score > best_score:
                best_score = score
                best_params = {"n_neighbors": n_neighbors, "metric": metric, "weights": weight}
                

print("Melhores parâmetros encontrados para KNN: ", best_params)

# ==============================
# Plotagem dos Dados
# ==============================

# Plotagem dos resultados do KNN
df = pd.DataFrame(results, columns=["n_neighbors", "metric", "weights", "score"])

os.makedirs("analyses", exist_ok=True)

plt.figure(figsize=(10,6))
sns.lineplot(data=df, x="n_neighbors", y="score", hue="metric", style="weights", markers=True)
plt.title("Comparação de Score por Número de Vizinhos (KNN)")
plt.xlabel("Número de Vizinhos (n_neighbors)")
plt.ylabel("Score de Acurácia (Treino)")
plt.legend(title="Métrica / Peso")
plt.grid(True)
plt.tight_layout()
plt.savefig("analyses/knn_scores.png")
plt.show()

# Análise de PCA
plt.figure(figsize=(8,5))
plt.plot(range(1, len(pca.explained_variance_ratio_)+1), 
         pca.explained_variance_ratio_.cumsum(), marker='o')
plt.title("Variância Explicada Acumulada por PCA")
plt.xlabel("Número de Componentes")
plt.ylabel("Variância Explicada Acumulada")
plt.axhline(y=0.95, color='r', linestyle='--', label="95% de Variância")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("analyses/pca_variancia_explicada.png")
plt.show()

# Plotagem 3D dos dados PCA
fig = plt.figure(figsize=(10,7))
ax = fig.add_subplot(111, projection='3d')

for i, label in enumerate(y):
    ax.scatter(X_pca[i, 0], X_pca[i, 1], X_pca[i, 2], label=label)

ax.set_xlabel("PC1")
ax.set_ylabel("PC2")
ax.set_zlabel("PC3")
plt.title("Receitas no Espaço PCA (3 Componentes)")
plt.tight_layout()
plt.savefig("analyses/receitas_pca_3d.png")
plt.show()

# Matriz de Similaridade Cosine
similarity_matrix = cosine_similarity(X)
plt.figure(figsize=(12,10))
sns.heatmap(similarity_matrix, xticklabels=y, yticklabels=y, cmap="YlGnBu", annot=False)
plt.title("Matriz de Similaridade entre Receitas (Cosine)")
plt.xticks(rotation=90)
plt.tight_layout()
plt.savefig("analyses/matriz_similaridade_cosine.png")
plt.show()

def recommend_top_5_foods_knn(user_profile):
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

# Execução de teste
if __name__ == "__main__":
    user_likes = ["carne de boi", "queijo", "cebola"]
    user_profile = get_user_profile_one_hot_encoded(user_likes)
    recommendations = recommend_top_5_foods_knn(user_profile)
    
    pprint(recommendations)
