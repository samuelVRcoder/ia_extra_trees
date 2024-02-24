from sklearn.ensemble import ExtraTreesClassifier as XT

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import LabelEncoder

from sklearn.metrics import accuracy_score as nota

import pandas as pd

tabela = pd.read_csv("clientes.csv")

codificador = LabelEncoder()

for i in tabela.columns:
    if tabela[i].dtype == "object" and i != "score_credito":
        tabela[i] = codificador.fit_transform(tabela[i])

x = tabela.drop(["score_credito", "id_cliente"], axis=1)

y = tabela["score_credito"]

x_treino, x_teste, y_treino, y_teste = train_test_split(x,y, test_size=0.3)

xt = XT()

xt.fit(x_treino, y_treino)

previsao = xt.predict(x_teste)

print(nota(previsao, y_teste))

