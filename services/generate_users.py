import json
import os
import random
import numpy as np
import matplotlib.pyplot as plt

recipes = {
    "Quesadilla de Barbacoa": ["carne de boi", "queijo", "ovo", "mostarda", "pimenta jalapeno", "nirá"],
    "Taco Pescado Baja": ["carne de peixe", "repolho", "abacate", "tomate", "pimenta jalapeno", "ovo", "mostarda", "limão", "picles", "cebola"],
    "Taco Camarão": ["camarão", "repolho", "ovo", "mostarda", "abacate", "tomate", "pimenta jalapeno", "picles"],
    "Taco Barbacoa": ["carne de boi", "queijo", "cebola", "ciboullete", "picles", "pimenta jalapeno"],
    "Gringa Coreana": ["tortilla", "milho", "queijo", "carne de porco", "acelga", "nabo", "picles", "limão", "cebola"],
    "Taco Birria": ["carne de boi", "queijo", "cebola", "pimenta jalapeno", "carne"],
    "Taco La Flor": ["couve-flor", "feijão", "grão de bico", "ciboullete"],
    "Taco de Chorizo": ["carne de porco", "queijo", "tomate", "cebola", "limão"],
    "Taco Al Pastor": ["abóbora", "alho", "carne de porco", "abacaxi", "canela", "cebola"],
    "Taco Lengua": ["cogumelo", "carne de boi", "castanha do pará", "tomate", "cebola", "limão"],
    "Taco Vegano": ["abacate", "tomate", "pimenta jalapeno", "cogumelo", "cebola", "nozes", "nirá", "gergilim"],
    "Taco Portenho": ["abóbora", "queijo", "carne de boi", "pimentão", "cebola", "alho", "milho"],
    "Quesadilla de Cogumelos": ["queijo", "cogumelos", "nirá"],
    "Burrito": ["tortilla", "trigo", "carne de frango", "queijo", "feijão", "tomate", "cebola", "limão", "repolho", "ovo", "mostarda"],
    "Chimichanga": ["tortilla", "carne de boi", "queijo", "repolho", "ovo", "mostarda", "feijão", "tomate", "cebola", "limão", "abacate"],
    "Burrito Al Pastor": ["tortilla", "trigo", "carne de porco", "abacaxi", "queijo", "feijão", "tomate", "cebola", "limão", "repolho"]
}

unique_ingrediets = set()
for ingredients_list in recipes.values():
    for ingredient in ingredients_list:
        unique_ingrediets.add(ingredient)

all_ingredients = []
for item in unique_ingrediets:
    all_ingredients.append(item)
all_ingredients.sort()

recipe_names = []
for nome in recipes.keys():
    recipe_names.append(nome)

N_USERS = 300
mean = len(recipe_names) // 2
std_dev = len(recipe_names) / 4

recipe_indices = np.random.normal(loc=mean, scale=std_dev, size=N_USERS).astype(int)
recipe_indices = np.clip(recipe_indices, 0, len(recipe_names) - 1)

generated_users = []
for user_id in range(len(recipe_indices)):
    recipe_index = recipe_indices[user_id]
    recipe_name = recipe_names[recipe_index]
    ingredients = recipes[recipe_name]

    liked = []
    for ingredient in ingredients:
        if random.random() < 0.8:
            liked.append(ingredient)

    disliked_pool = []
    for ingredient in all_ingredients:
        if ingredient not in ingredients:
            disliked_pool.append(ingredient)

    num_extra = random.randint(0, 2)
    if num_extra > 0 and len(disliked_pool) >= num_extra:
        extra = random.sample(disliked_pool, k = num_extra)
        for ingredient in extra:
            liked.append(ingredient)

    user_data = {
        "id": user_id,
        "liked_ingredients": liked,
        "most_liked_recipe": recipe_name
    }
    generated_users.append(user_data)

os.makedirs("data", exist_ok=True)
with open("data/generated_users.json", "w", encoding="utf-8") as f:
    json.dump(generated_users, f, ensure_ascii=False, indent=2)

print(f"{N_USERS} usuários gerados em data/generated_users.json")

recipe_count = {}
for index in recipe_indices:
    recipe_name = recipe_names[index]
    if recipe_name in recipe_count:
        recipe_count[recipe_name] += 1
    else:
        recipe_count[recipe_name] = 1

ordered_counts = []
for name in recipe_names:
    if name in recipe_count:
        ordered_counts.append(recipe_count[name])
    else:
        ordered_counts.append(0)

plt.figure(figsize=(12, 6))
plt.bar(recipe_names, ordered_counts)
plt.xticks(rotation=45, ha='right')
plt.xlabel("Receitas")
plt.ylabel("Número de usuários que mais gostaram")
plt.title("Distribuição de usuários por receita mais curtida")
plt.tight_layout()
plt.show()