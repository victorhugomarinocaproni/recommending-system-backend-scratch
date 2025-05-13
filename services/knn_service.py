from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors
import numpy as np
from pprint import pprint

# =======================
# Constantes: 

KNN_VALUE = 5


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
    
ingredients_list = sorted(list(ingredient_set))

def one_hot_encode_vector(recipe_ingredients, ingredients_list):
    return [1 if ingredient in recipe_ingredients else 0 for ingredient in ingredients_list]

recipe_profiles = []
recipe_names = []
labels = []

for name, ingredients in recipes.items():
    vector = one_hot_encode_vector(ingredients, ingredients_list)
    recipe_profiles.append(vector)
    recipe_names.append(name)

def get_user_profile(user_likes):
    for ingredient in user_likes:
        if ingredient not in ingredients_list:
            print(f"Ingrediente inválido: {ingredient}. Ingredientes válidos: " + ", ".join(ingredients_list))
            raise ValueError(f"Ingrediente inválido: {ingredient}. Ingredientes válidos: " + ", ".join(ingredients_list))
    return one_hot_encode_vector(user_likes, ingredients_list)

def recommend_top_5_foods_knn(user_profile):
    knn_recommender = NearestNeighbors(n_neighbors = KNN_VALUE, metric = 'cosine')
    knn_recommender.fit(recipe_profiles)
    
    distances, indices = knn_recommender.kneighbors([user_profile])
    
    result = []
    for idx, i in enumerate(indices[0]):
        recipe_name = recipe_names[i]
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
    user_profile = get_user_profile(user_likes)
    recommendations = recommend_top_5_foods_knn(user_profile)
    
    pprint(recommendations)
