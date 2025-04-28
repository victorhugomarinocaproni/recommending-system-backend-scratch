from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors, KNeighborsClassifier
import numpy as np

recipes = [
    {"aperol_spritz": ["aperol", "espumante", "água com gás", "laranja"]},
    {"mojito_tradicional": ["rum", "limão", "hortelã", "açúcar", "água com gás"]},
    {"margarita_facil": ["tequila", "licor de laranja", "limão"]},
    {"pina_colada": ["rum", "leite de coco", "abacaxi"]},
    {"sex_on_the_beach": ["vodka", "licor de pêssego", "suco de laranja", "groselha"]},
    {"moscow_mule": ["vodka", "gengibre", "limão", "água com gás"]},
    {"manhattan": ["whisky", "vermouth", "angostura"]},
    {"old_fashioned": ["whisky", "açúcar", "angostura", "laranja"]},
    {"gin_tonica_com_limao": ["gin", "tônica", "limão"]},
    {"cosmopolitan_perfeito": ["vodka", "licor de laranja", "limão", "cranberry"]},
    {"negroni": ["gin", "campari", "vermouth"]},
    {"cuba_libre": ["rum", "cola", "limão"]},
    {"bloody_mary": ["vodka", "suco de tomate", "limão", "molho inglês", "pimenta"]},
    {"caipirinha_de_limao": ["cachaça", "limão", "açúcar"]},
    {"daiquiri": ["rum", "limão", "açúcar"]}
]

ingredient_set = set()
for recipe in recipes:
    for ingredients in recipe.values():
        ingredient_set.update(ingredients)

ingredients_list = sorted(list(ingredient_set))

recipe_profiles = []
recipe_names = []
labels = [] 

def create_binary_profile_vector(recipe_ingredients, ingredients_list):
    return [1 if ingredient in recipe_ingredients else 0 for ingredient in ingredients_list]

for recipe in recipes:
    for name, ingredients in recipe.items():
        vector = create_binary_profile_vector(ingredients, ingredients_list)
        recipe_profiles.append(vector)
        recipe_names.append(name)
        labels.append(name.split("_")[0])  # Exemplo de label genérica baseada no nome da receita 

def get_user_profile(user_likes):
    
    for ingredient in user_likes:
        if ingredient not in ingredients_list:
            print(f"Ingrediente inválido: {ingredient}. Ingredientes válidos: " + ", ".join(ingredients_list))
            raise ValueError(f"Ingrediente inválido: {ingredient}. Ingredientes válidos: " + ", ".join(ingredients_list))
    
    return create_binary_profile_vector(user_likes, ingredients_list)

def recommend_top_3_drinks_knn(user_profile):
    knn_recommender = NearestNeighbors(n_neighbors=5, metric='cosine')
    knn_recommender.fit(recipe_profiles)
    
    distances, indices = knn_recommender.kneighbors([user_profile])
    
    result = []
    
    for idx, i in enumerate(indices[0]):
        drink_name = recipe_names[i]
        
        ingredients = []
        for recipe in recipes:
            if drink_name in recipe:
                ingredients = recipe[drink_name]
                break
        
        similarity_score = 1 - distances[0][idx]
        
        result.append({
            "name": drink_name.replace("_", " ").title(),
            "ingredients": ingredients,
            "score": round(similarity_score, 2),
            "instructions": ""
        })
    
    return result

# Testando a função com um exemplo:
#user_likes = ["limão", "rum", "hortelã"]
#user_profile = get_user_profile(user_likes)
#print(recommend_top_3_drinks_knn(user_profile))

