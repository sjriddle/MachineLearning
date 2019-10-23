import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns  # plot

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

import warnings
warnings.filterwarnings('ignore')

df = pd.read_csv('./disease_prediction/cv_train.csv', sep=';')
df.head()

df.drop('id', axis=1, inplace=True)
df.info()

df.describe()
sns.countplot(x='cardio', data=df, hue='gender', palette='rainbow')

plt.figure(figsize=(14,6))
plt.subplot(1,2,1)
sns.boxplot(x='cardio', y='height', data=df, palette='winter')
plt.subplot(1,2,2)
sns.boxplot(x='cardio', y='weight', data=df, palette='summer')

correlations = df.corr()['cardio'].drop('cardio')
print(correlations)


def feat_select(threshold):
    abs_cor = correlations.abs()
    features = abs_cor[abs_cor > threshold].index.tolist()
    return features


def model(mod,X_tr,X_te):
    mod.fit(X_tr,y_train)
    pred = mod.predict(X_te)
    print('Model score = ',mod.score(X_te,y_test)*100,'%')


msk = np.random.rand(len(df))<0.85
df_train_test = df[msk]
df_val = df[~msk]

X = df_train_test.drop('cardio',axis=1)
y = df_train_test['cardio']
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=70)

lr = LogisticRegression()
threshold = [0.001,0.002,0.005,0.01,0.05,0.1]
for i in threshold:
    print('\n',i)
    feature_i = feat_select(i)
    X_train_i = X_train[feature_i]
    X_test_i = X_test[feature_i]
    model(lr,X_train_i,X_test_i)