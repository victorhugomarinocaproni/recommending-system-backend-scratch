# Projeto de Recomendação de Receitas 🍸

Este é um projeto simples de recomendação de receitas utilizando **KNN** e **similaridade do cosseno**, feito em **Python** com o auxílio das bibliotecas `scikit-learn` e `numpy`. Além disso, contém também scripts de recomendação a partir da utilização dos modelos de **RandomForestClassifier** e **NaiveBayesClassifier**. 

O projeto analisa os ingredientes que o usuário gosta e recomenda receitas semelhantes, além de tentar classificar o tipo de receita favorita do usuário.

---

## 📋 Tecnologias utilizadas
- Python 3.8 ou superior
- Bibliotecas: `scikit-learn`, `numpy`, `Flask`, `Flasgger`

---

# Como rodar o projeto localmente

## 1. Clone o repositório

Abra seu terminal e execute:

```
git clone https://github.com/victorhugomarinocaproni/recommending-system-backend-scratch.git
cd recomendation_project
```

## 2. Crie um ambiente virtual (VENV)

Windows:
```
python -m venv venv
```

Mac/Linux:
```
python3 -m venv venv
```

## 3. Ative o ambiente virtual (toda vez que quiser entrar no VENV, rode isso no terminal)
Windows
``` 
venv\Scripts\activate
```
Mac/Linux
```
source venv/bin/activate
```

## 4. Instale as dependências

### Com o VENV ativo:
```
pip install -r requirements.txt
```

## 5. Execute o projeto
* KNN para Recomendação de Receitas:
```
python service/knn_service.py
```
ou

* DecisionTree para Recomendação de Receitas:
```
python service/decision_tree_service.py
```
ou

* NaiveBayes para Recomendação de Receitas:
```
python service/naiveBayesClassifier.py
```
ou 

* Algoritmo para gerar os usuários fake (gerar um toy_dataset) 
```
python service/generate_users.py
```
Para maior organização do projeto, ao rodar o script generate_users, um diretório '/data' será criado automaticamente, onde o arquivo JSON gerado será colocado. Caso o diretório já exista, apenas o arquivo JSON será sobrescrito. Cada vez que o script generate_users é rodado, um novo JSON é gerado, sempre sobrescrevendo o antigo!

## 6. Como fechar (desativar) o ambiente virtual
Windows
```
venv\Scripts\deactivate.bat
```

Mac/Linux
```
deactivate
```

## 7. Rodar a Flask API
Executar:
```
python app.py
```
A API ficará disponível em: 
`http://0.0.0.0:8080` 
<br><br>
O Swagger estará em: 
`http://0.0.0.0:8080/docs/` 
<br>

## Como testar o Endpoint do Swagger
1. Clique no endpoint <strong>[POST]</strong> /recommendation
2. Clique em <strong>"Try it out"</strong>
3. No campo "ingredientes", envie uma lista de ingredientes que você quer testar, como por exemplo:
```
{
  "ingredients": ["limão", "pimenta jalapeno", "carne de boi"]
}
```
4. Clique em <strong>"Execute"</strong>
5. Você receberá como resposta uma Lista com as Top 5 Receitas mais condizentes com seu gosto
