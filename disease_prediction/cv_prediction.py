import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib import rcParams

df = pd.read_csv('./cv_train.csv', sep=';')
df.head()

rcParams['figure.figsize'] = 12, 8
df['years'] = (df['age'] // 365).astype('int')
sns.countplot(x='years', hue='cardio', data=df, palette="Set1")

df['BMI'] = df['weight'] / df['height'] / df['height'] * 10000
sns.boxplot(x='cardio', y='BMI', data=df)
plt.ylim(10, 50)
plt.show()