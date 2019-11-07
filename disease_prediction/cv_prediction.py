import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib import rcParams
import os

df = pd.read_csv('./cv_train.csv', sep=';')
df.head()

rcParams['figure.figsize'] = 12, 8
df['years'] = (df['age'] // 365).astype('int')
sns.countplot(x='years', hue='cardio', data=df, palette="Set1")

df['BMI'] = df['weight'] / df['height'] / df['height'] * 10000
sns.boxplot(x='cardio', y='BMI', data=df)
plt.ylim(10, 50)
plt.show()