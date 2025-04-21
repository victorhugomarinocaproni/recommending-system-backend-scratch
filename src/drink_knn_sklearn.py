from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors, KNeighborsClassifier
import numpy as np

# Lista de receitas (nome + ingredientes principais)
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

# Criar lista de ingredientes únicos
ingredient_set = set()
for recipe in recipes:
    for ingredients in recipe.values():
        ingredient_set.update(ingredients)

ingredients_list = sorted(list(ingredient_set))

# Função para criar vetor binário de perfil
def create_profile_vector(recipe_ingredients, ingredients_list):
    return [1 if ingredient in recipe_ingredients else 0 for ingredient in ingredients_list]

# Criar vetores de perfil para cada receita
recipe_profiles = []
recipe_names = []
labels = []  # Para classificação: ex: tipo de drink (só como exemplo)

for recipe in recipes:
    for name, ingredients in recipe.items():
        vector = create_profile_vector(ingredients, ingredients_list)
        recipe_profiles.append(vector)
        recipe_names.append(name)
        labels.append(name.split("_")[0])  # Exemplo de label genérica

# Perfil do usuário
def get_user_profile(user_likes):
    return create_profile_vector(user_likes, ingredients_list)

user_likes = ["limão", "rum", "hortelã"]
user_profile = get_user_profile(user_likes)

# ============================
# RECOMENDAÇÃO COM KNN + DIST. COSSENO
# ============================
print("\n--- Recomendação com KNN (cosine) ---")
knn_recommender = NearestNeighbors(n_neighbors=5, metric='cosine')
knn_recommender.fit(recipe_profiles)
distances, indices = knn_recommender.kneighbors([user_profile])

for i in indices[0]:
    print(f"{recipe_names[i]} => Similaridade: {1 - distances[0][list(indices[0]).index(i)]:.2f}")

# ============================
# CLASSIFICAÇÃO COM KNN
# ============================
print("\n--- Classificação com KNN ---")
knn_classifier = KNeighborsClassifier(n_neighbors=3, metric='cosine')
knn_classifier.fit(recipe_profiles, labels)
predicted_label = knn_classifier.predict([user_profile])[0]

print(f"O perfil do usuário parece gostar de drinks do tipo: {predicted_label}")

'''
Expected Output:

mojito_tradicional => Similaridade: 0.77
cuba_libre => Similaridade: 0.67
daiquiri => Similaridade: 0.67
margarita_facil => Similaridade: 0.33
pina_colada => Similaridade: 0.33
'''
