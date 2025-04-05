# ML_-regression_logistique

# 📊 Analyse des Données et Modélisation Logistique

Ce projet vise à analyser un jeu de données provenant des réseaux sociaux et à prédire si une personne achètera un produit en fonction de son âge et de son salaire estimé. Le modèle utilisé est la régression logistique.

## 🔧 Prérequis

Avant de lancer le projet, vous devez installer les bibliothèques suivantes :

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

## 📑 Chargement et exploration des données

Le jeu de données est importé et les premières lignes sont affichées pour avoir un aperçu des données.

```python
import pandas as pd

data = pd.read_csv("Social_Network_Ads.csv")
data.head()
```

Les données comprennent trois colonnes :
- `Age` : L'âge de l'utilisateur.
- `EstimatedSalary` : Le salaire estimé de l'utilisateur.
- `Purchased` : Si l'utilisateur a acheté ou non (0 pour non, 1 pour oui).

## 🔍 Exploration des données

Nous examinons les données pour identifier les doublons et vérifier les valeurs uniques dans la colonne `Purchased` :

```python
data.Purchased.value_counts()
```

Ensuite, un **scatterplot** est généré pour visualiser la répartition des données :

```python
import seaborn as sns

sns.scatterplot(x="Age", y="EstimatedSalary", data=data, hue="Purchased")
```

Nous affichons également une **matrice de corrélation** pour mieux comprendre les relations entre les variables :

```python
sns.heatmap(data.corr(), annot=True)
```

## 🧮 Préparation des données

Les données sont divisées en ensembles d'entraînement et de test :

```python
from sklearn.model_selection import train_test_split

X = data.drop(['Purchased'], axis=1)
y = data[['Purchased']]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=123)
```

Ensuite, les données sont standardisées pour améliorer les performances du modèle :

```python
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
```

## 📈 Entraînement du modèle

Nous utilisons la **régression logistique** pour entraîner le modèle sur les données d'entraînement :

```python
from sklearn.linear_model import LogisticRegression

lr = LogisticRegression()
lr.fit(X_train, y_train)
```

Nous évaluons ensuite la performance du modèle avec l'**accuracy score** et la **matrice de confusion** :

```python
from sklearn.metrics import accuracy_score, confusion_matrix

y_pred_train = lr.predict(X_train)
print('accuracy_score: ', accuracy_score(y_train, y_pred_train))
```

## 📊 Évaluation du modèle

Nous affichons le rapport de classification pour évaluer la performance par classe :

```python
from sklearn.metrics import classification_report

print(classification_report(y_train, y_pred_train))
```

Le modèle est également évalué sur l'ensemble de test :

```python
y_pred_test = lr.predict(X_test)
print('accuracy_score: ', accuracy_score(y_test, y_pred_test))
print(classification_report(y_test, y_pred_test))
```

## 🎯 Prédiction

Pour faire une prédiction, nous pouvons utiliser le modèle sur de nouvelles données :

```python
prediction = lr.predict(sc.transform([[30, 250000]]))
print(prediction)
```

## 💾 Sauvegarde du modèle

Enfin, pour enregistrer le modèle entraîné, vous pouvez utiliser **pickle** :

```python
import pickle
pickle.dump(lr, open('nom.pkl', 'wb'))
```

## 🔑 Conclusion

Le modèle de régression logistique permet de prédire si un utilisateur va acheter un produit en fonction de son âge et de son salaire. Nous avons optimisé les données et évalué la performance du modèle avant de faire une prédiction sur de nouvelles données.

---

🎉 **Bonne analyse et bonne prédiction !** 🎉
```

Ce fichier `README.md` décrit les étapes principales du projet, avec des emojis pour ajouter de la convivialité et de l'enthousiasme.
```

## ✍️ Auteur
Fouejio Francky Joël
