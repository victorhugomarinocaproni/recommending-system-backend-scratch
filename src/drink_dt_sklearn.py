import random
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

ingredients = ["açúcar", "limão", "gelo", "hortelã", "pepino", "sal", "laranja", "abacaxi", "romã", "soda", 
               "morango", "framboesa", "canela", "noz-moscada", "xarope de açúcar", "grenadine", "laranja", 
               "coco", "café", "pêssego", "maracujá"]

beverages = {
    "Bloody Mary": ["vodka", "tomate", "limão", "pimenta", "açúcar", "sal", "gelo"],
    "Moscow Mule": ["vodka", "ginger beer", "limão", "gelo"],
    "Mai Tai": ["rum", "amaretto", "limão", "gelo", "grenadine"],
    "Margarita": ["tequila", "limão", "sal", "gelo"],
    "Piña Colada": ["rum", "abacaxi", "coco", "gelo"],
    "Cosmopolitan": ["vodka", "limão", "soda", "morango"],
    "Rum Punch": ["rum", "laranja", "abacaxi", "grenadine", "gelo"],
    "Mojito": ["rum", "hortelã", "açúcar", "gelo", "limão"],
    "Caipirinha": ["cachaça", "açúcar", "limão", "gelo"],
    "Tequila Sunrise": ["tequila", "laranja", "grenadine", "gelo"],
    "Negroni": ["gin", "vermouth", "campari", "gelo"],
    "Long Island Iced Tea": ["vodka", "rum", "gin", "tequila", "limão", "xarope de açúcar", "gelo"],
    "Sex on the Beach": ["vodka", "pêssego", "morango", "soda", "gelo"],
    "Daiquiri": ["rum", "limão", "açúcar", "gelo"]
}

data = []
target = []

for beverage, bev_ingredients in beverages.items():
    for _ in range(50):  
        random_ingredients = [1 if ingredient in bev_ingredients else 0 for ingredient in ingredients]
        data.append(random_ingredients)
        target.append(beverage)

X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.3)

clf = DecisionTreeClassifier()
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)

print("Acurácia:", accuracy_score(y_test, y_pred))

def recommend_drink(user_ingredients):
    user_vector = [1 if ingredient in user_ingredients else 0 for ingredient in ingredients]
    prediction = clf.predict([user_vector])
    return prediction[0]

user_input = ["pepino", "morango"]
recommended_drink = recommend_drink(user_input)
print(f"A bebida recomendada para os ingredientes {user_input} é: {recommended_drink}")
