import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split

DATA_FILE = "data/generated_users.json"
OUTPUT_DIR = "visualizations"
os.makedirs(OUTPUT_DIR, exist_ok=True)

ingredient_list = sorted([
    "aperol", "espumante", "água com gás", "laranja", "rum", "limão", "hortelã",
    "açúcar", "tequila", "licor de laranja", "leite de coco", "abacaxi", "vodka",
    "licor de pêssego", "suco de laranja", "groselha", "gengibre", "vermouth",
    "angostura", "gin", "tônica", "cranberry", "campari", "cola", "suco de tomate",
    "molho inglês", "pimenta", "cachaça"
])

with open(DATA_FILE, "r", encoding="utf-8") as file:
    user_data = json.load(file)

features = []
labels = []

for user in user_data:
    selected_ingredients = user["liked_ingredients"]
    favorite_drink = user["most_liked_drink"]
    ingredient_vector = [1 if ingredient in selected_ingredients else 0 for ingredient in ingredient_list]
    features.append(ingredient_vector)
    labels.append(favorite_drink)

features_train, features_test, labels_train, labels_test = train_test_split(
    features, labels, test_size=0.3, random_state=42
)

model = DecisionTreeClassifier(random_state=42)
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

plt.figure(figsize=(20, 10))
plot_tree(model, feature_names=ingredient_list, class_names=model.classes_, filled=True, max_depth=3)
plt.title("Árvore de Decisão (Profundidade até 3)")
plt.savefig(os.path.join(OUTPUT_DIR, "tree_visualization.png"))
plt.close()
print("Árvore de Decisão gerada com sucesso!")

prediction_distribution = pd.DataFrame({"drink": predictions})
plt.figure(figsize=(10, 6))
sns.countplot(data=prediction_distribution, x="drink", order=prediction_distribution["drink"].value_counts().index)
plt.xticks(rotation=45)
plt.title("Distribuição das Bebidas Recomendadas")
plt.ylabel("Frequência")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "drink_distribution.png"))
plt.close()
print("Visualização da Distribuição dos Dados gerada com sucesso!")

classification_metrics = classification_report(
    labels_test, predictions, target_names=model.classes_, output_dict=True
)
metrics_df = pd.DataFrame(classification_metrics).transpose()
metrics_df.to_csv(os.path.join(OUTPUT_DIR, "classification_report.csv"), index=True)
print("Relatório de Classificação gerado com sucesso!")