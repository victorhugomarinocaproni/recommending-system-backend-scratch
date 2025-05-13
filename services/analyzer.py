import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split

DATA_FILE = "data/generated_users.json"
OUTPUT_DIR = "visualizations"
os.makedirs(OUTPUT_DIR, exist_ok=True)

ingredient_list = sorted([
    "abacate","abóbora","abacaxi","acelga","alho","canela","carne","carne de boi",
    "carne de frango","carne de peixe","carne de porco","castanha do pará","cebola",
    "ciboullete","cogumelo","cogumelos","coentro","couve-flor","feijão","gergilim",
    "grão de bico","limão","milho","mostarda","nabo","nirá","nozes","ovo","picles",
    "pimentão","pimenta jalapeno","queijo","repolho","salsa","tomate","tortilla","trigo"
])

with open(DATA_FILE, "r", encoding="utf-8") as file:
    user_data = json.load(file)

features = []
labels = []

for user in user_data:
    selected_ingredients = user["liked_ingredients"]
    favorite_recipe = user["most_liked_recipe"]
    ingredient_vector = [1 if ingredient in selected_ingredients else 0 for ingredient in ingredient_list]
    features.append(ingredient_vector)
    labels.append(favorite_recipe)

features_train, features_test, labels_train, labels_test = train_test_split(
    features, labels, test_size=0.3, random_state=42
)

model = RandomForestClassifier(random_state=42)
model.fit(features_train, labels_train)

predictions = model.predict(features_test)
model_accuracy = accuracy_score(labels_test, predictions)
print(f"Acurácia do modelo: {model_accuracy:.2f}")

conf_matrix = confusion_matrix(labels_test, predictions, labels=model.classes_)
plt.figure(figsize=(12, 8))
sns.heatmap(conf_matrix, annot=True, fmt="d", xticklabels=model.classes_, yticklabels=model.classes_, cmap="Blues")
plt.title("Matriz de Confusão")
plt.xlabel("Predição")
plt.ylabel("Valor Real")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "confusion_matrix.png"))
plt.close()
print("Matriz de Confusão gerada com sucesso!")

prediction_distribution = pd.DataFrame({"recipe": predictions})
plt.figure(figsize=(10, 6))
sns.countplot(data=prediction_distribution, x="recipe", order=prediction_distribution["recipe"].value_counts().index)
plt.xticks(rotation=45)
plt.title("Distribuição das Receitas Recomendadas")
plt.ylabel("Frequência")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "recipe_distribution.png"))
plt.close()
print("Visualização da Distribuição dos Dados gerada com sucesso!")

classification_metrics = classification_report(
    labels_test, predictions, target_names=model.classes_, output_dict=True
)
metrics_df = pd.DataFrame(classification_metrics).transpose()
metrics_df.to_csv(os.path.join(OUTPUT_DIR, "classification_report.csv"), index=True)
print("Relatório de Classificação gerado com sucesso!")