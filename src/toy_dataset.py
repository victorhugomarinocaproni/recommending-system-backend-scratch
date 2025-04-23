import numpy as np
import random
import json
import os

ingredients_list = sorted([
    "aperol", "espumante", "água com gás", "laranja", "rum", "limão", "hortelã", "açúcar",
    "tequila", "licor de laranja", "leite de coco", "abacaxi", "vodka", "licor de pêssego",
    "suco de laranja", "groselha", "gengibre", "vermouth", "angostura", "gin", "tônica",
    "cranberry", "campari", "cola", "suco de tomate", "molho inglês", "pimenta", "cachaça"
])

NUM_USERS = 100

def generate_users(num_users=NUM_USERS, seed=42):
    random.seed(seed)
    np.random.seed(seed)

    users = []

    for i in range(num_users):
        # Gera um número de ingredientes preferidos para o usuário (entre 2 e 7)
        num_likes = int(np.clip(np.random.normal(loc=4, scale=1.5), 2, 7))

        user_likes = random.sample(ingredients_list, num_likes)
        users.append({"id": i + 1, "likes": user_likes})

    return users

if __name__ == "__main__":
    users = generate_users()

    for idx, user in enumerate(users[:5]):  # Mostrar só os 5 primeiros
        print(f"Usuário {idx+1}: {user}")
        
    os.makedirs("data", exist_ok=True) 

    with open("data/generated_users.json", "w", encoding="utf-8") as f:
        json.dump(users, f, ensure_ascii=False, indent=4)

    print(f"\n✅ Gerados {len(users)} perfis de usuários!")
    print("Arquivo 'generated_users.json' salvo no diretório '/data' com sucesso.")