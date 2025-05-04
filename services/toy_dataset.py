import json
import os
import random

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
    {"daiquiri": ["rum", "limão", "açúcar"]},
    {"whiskey_sour": ["whisky", "limão", "açúcar", "clara de ovo"]},
    {"tequila_sunrise": ["tequila", "suco de laranja", "groselha"]}
]

# Converter recipes em dict com nome -> ingredientes
recipe_dict = {list(d.keys())[0]: list(d.values())[0] for d in recipes}

# Todos os ingredientes disponíveis para que possamos gerar os fake users
all_ingredients = sorted(set(ing for ingredients in recipe_dict.values() for ing in ingredients))

# jogar pro .env depois (quando criado) !!!
N_USERS = 300

generated_users = []

for user_id in range(N_USERS):
    drink = random.choice(list(recipe_dict.keys()))
    ingredients = recipe_dict[drink]

    # Simula gosto parcial com chance de 80% de manter o ingrediente da receita
    liked_ingredients = [ing for ing in ingredients if random.random() < 0.8]

    # Adiciona de 0 a 2 ingredientes aleatórios fora da receita, para que tenhamos um cenário mais "real" 
    disliked_pool = list(set(all_ingredients) - set(ingredients))
    liked_ingredients += random.sample(disliked_pool, k=random.randint(0, 2))

    generated_users.append({
        "id": user_id,
        "liked_ingredients": liked_ingredients,
        "most_liked_drink": drink
    })

os.makedirs("data", exist_ok=True)
with open("data/generated_users.json", "w", encoding="utf-8") as f:
    json.dump(generated_users, f, ensure_ascii=False, indent=2)

print(f"{N_USERS} usuários gerados em data/generated_users.json")