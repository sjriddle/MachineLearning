import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import warnings

warnings.filterwarnings("ignore")

df = pd.read_csv('./disease_prediction/cv_train.csv', sep=';')
df.head()

df['age'] = df['age'].map(lambda x : x // 365)

# Display correlations of metrics with cardiovascular disease
cardio_corr = df.corr()['cardio'].drop('cardio')

rnd = np.random.rand(len(df)) < 0.85
df_train_test = df[rnd]
df_val = df[~rnd]


X = df_train_test.drop('cardio',axis=1)
Y = df_train_test['cardio']
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.2,random_state=70)

lr = LogisticRegression()

def train_model(model,X_tr,X_te):
    model.fit(X_tr,Y_train)
    print('Model score = ',model.score(X_te,Y_test)*100,'%')

def attribute_dist(th):
    return cardio_corr.abs()[cardio_corr.abs() > th].index.tolist()

th = [0.001, 0.005, 0.01, 0.05, 0.1]
for i in th:
    print('\n', 'Given Threshold: ', float(i * 100), '%')
    feature_i = attribute_dist(i)
    X_train_i = X_train[feature_i]
    X_test_i = X_test[feature_i]
    train_model(lr, X_train_i, X_test_i)



