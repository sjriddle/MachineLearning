from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings("ignore")

df = pd.read_csv('./disease_prediction/cv_train.csv', sep=';')
df.head()

# Change age to years
df['age'] = df['age'].map(lambda x : x // 365)

# Store correlation table in memory
cardio_corr = df.corr()['cardio'].drop('cardio')

rnd = np.random.rand(len(df)) < 0.85
df_train_test = df[rnd]
df_val = df[~rnd]

# Declare training/scoring data
x = df_train_test.drop('cardio',axis=1)
y = df_train_test['cardio']
x_train, x_score, y_train, y_score = train_test_split(x, y, test_size=0.15, random_state=75)

# Declare Logistic Regression from sklearn
lr = LogisticRegression()

# Function that fits and scores a specific model
def train_model(model, x_tr, x_te):
    model.fit(x_tr,y_train)
    print('Model score = ', model.score(x_te, y_score)*100, '%')

# Declare different thrsholds and output model score
th = [0.001, 0.005, 0.01, 0.05, 0.1]
for i in th:
    print('\n', 'Model Threshold: ', float(i * 100), '%')
    attr_i = cardio_corr.abs()[cardio_corr.abs() > i].index.tolist()
    x_train_i = x_train[attr_i]
    x_score_i = x_score[attr_i]
    train_model(lr, x_train_i, x_score_i)

scale = StandardScaler()
scale.fit(x_train)
x_train_scaled = scale.transform(x_train)
x_train_ = pd.DataFrame(x_train_scaled, columns=df.columns[:-1])

scale.fit(x_score)
x_score_scaled = scale.transform(x_score)
x_score_ = pd.DataFrame(x_score_scaled, columns=df.columns[:-1])


# Find the best thresholds
for i in th:
    feature = cardio_corr.abs()[cardio_corr.abs() > i].index.tolist()
    x_train_k = x_train_[feature]
    x_score_k = x_score_[feature]
    err = []
    for j in range(1,30):
        knn = KNeighborsClassifier(n_neighbors=j)
        knn.fit(x_train_k, y_train)
        pred_j = knn.predict(x_score_k)
        err.append(np.mean(y_score != pred_j))

    plt.figure(figsize=(10,6))
    plt.plot(range(1,30),err)
    plt.xlabel('K-Value')
    plt.ylabel('Error')

# Attribute input set to 0.05 for best result
feat_final = cardio_corr.abs()[cardio_corr.abs() > 0.05].index.tolist()
print(feat_final)

# Scaling data
x_train = x_train_[feat_final]
x_val = np.asanyarray(df_val[feat_final])
y_val = np.asanyarray(df_val['cardio'])

scale.fit(x_val)
x_val_scaled = scale.transform(x_val)
x_val_ = pd.DataFrame(x_val_scaled, columns=df_val[feat_final].columns)

# K-NN with k=15
knn = KNeighborsClassifier(n_neighbors=15)
knn.fit(x_train, y_train)
pred = knn.predict(x_val_)

# Confusion Matrix
print('Confusion Matrix =\n',confusion_matrix(y_val,pred))
print('\n',classification_report(y_val,pred))

# Logistic regression
lr.fit(x_train,y_train)
pred = lr.predict(x_val_)

# Confusion Matrix
print('Confusion Matrix =\n',confusion_matrix(y_val, pred))
print('\n',classification_report(y_val, pred))
