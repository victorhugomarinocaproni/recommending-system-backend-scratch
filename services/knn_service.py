from sklearn.neighbors import NearestNeighbors
from pprint import pprint
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA

# =======================
# Constantes: 

KNN_VALUE = 5
N_COMPONENTS = 3

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
    
pca = PCA(n_components=N_COMPONENTS)
X_pca = pca.fit_transform(X)

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

for n_neighbors in grid_params["n_neighbors"]:
    for metric in grid_params["metric"]:
        for weight in grid_params["weights"]:
            model = KNeighborsClassifier(n_neighbors=n_neighbors, metric=metric)
            model.fit(X, y)
            score = model.score(X, y)
            print(f"Parâmetros: n_neighbors={n_neighbors}, metric={metric}, weights={weight}, Score: {score:.4f}")
            if score > best_score:
                best_score = score
                best_params = {"n_neighbors": n_neighbors, "metric": metric, "weights": weight}

print("Melhores parâmetros encontrados para KNN: ", best_params)

def recommend_top_5_foods_knn(user_profile):
    knn_recommender = NearestNeighbors(n_neighbors = KNN_VALUE, metric = 'cosine')
    knn_recommender.fit(X)
    
    distances, indices = knn_recommender.kneighbors([user_profile])
    
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
