import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
import joblib
from sklearn.ensemble import GradientBoostingClassifier

# from google.colab import drive
# drive.mount('/content/drive')

# path = "/content/drive/MyDrive/Projetos"

path = "."
df_train = pd.read_csv(f'{path}/data/train.csv')
df_test = pd.read_csv(f'{path}/data/test.csv')

df_train.head()

df_train = df_train.drop('ID_code', axis=1)

X_train, y_train = df_train.drop('target', axis=1), df_train['target']

all(df_train.isnull().sum()) == False

pca = PCA(n_components=0.95)
X_reduced = pca.fit_transform(X_train)

print(X_reduced)

componentes = pca.components_
print(componentes)

joblib.dump(pca, f'{path}/models/pca_model.pkl')

# Treinamento
clf = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0,
    max_depth=1, random_state=0).fit(X_reduced, y_train)

# Previsão
X_test = df_test.drop("ID_code", axis=1)
X_test_reduced = pca.transform(X_test)
clf.predict(X_test_reduced)

joblib.dump(clf, f'{path}/models/gbc_model.pkl')