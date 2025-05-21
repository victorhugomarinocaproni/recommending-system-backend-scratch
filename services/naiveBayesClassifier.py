import pandas as pd
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

data = pd.read_json("data/generated_users.json")
df = pd.DataFrame(data)

mlb = MultiLabelBinarizer()
X = mlb.fit_transform(df['liked_ingredients'])

y = df['most_liked_recipe']

# Divide para treino/teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = MultinomialNB()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))

novo_usuario_ingredientes = ["carne de porco", "abacaxi", "limão", "cebola"]
entrada = mlb.transform([novo_usuario_ingredientes])
receita_prevista = model.predict(entrada)

print("Receita recomendada:", receita_prevista[0])
