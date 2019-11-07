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

cardio_corr = df.corr()['cardio'].drop('cardio')

rnd = np.random.rand(len(df)) < 0.85
df_train_test = df[rnd]
df_val = df[~rnd]


x = df_train_test.drop('cardio',axis=1)
y = df_train_test['cardio']
x_train, x_score, y_train, y_score = train_test_split(x, y, test_size=0.15, random_state=75)

lr = LogisticRegression()

def train_model(model, x_tr, x_te):
    model.fit(x_tr,y_train)
    print('Model score = ', model.score(x_te, y_score)*100, '%')

def attribute_dist(th):
    return cardio_corr.abs()[cardio_corr.abs() > th].index.tolist()

th = [0.001, 0.005, 0.01, 0.05, 0.1]
for i in th:
    print('\n', 'Given Threshold: ', float(i * 100), '%')
    attr_i = attribute_dist(i)
    x_train_i = x_train[attr_i]
    x_score_i = x_score[attr_i]
    train_model(lr, x_train_i, x_score_i)




