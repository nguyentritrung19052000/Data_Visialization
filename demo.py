import numpy as np
import pandas  as pd
import seaborn as sns
import matplotlib.pyplot as plt

df=pd.read_csv('healthcare-dataset-stroke-data.csv')
df.head(3)
df.describe()
df.isnull().sum()
plt.figure(figsize=(10,8))
sns.heatmap(df.corr(),annot=True)
plt.show()

df=df.dropna().reset_index(drop=True)
df.isnull().sum()
df=df.drop(columns=['id'])
plt.figure(figsize=(12,8))
df.boxplot()
plt.show()

for i in df.select_dtypes(include=np.number).columns:
    sns.boxplot(df[i])
    plt.show()
df['hypertension'] = df['hypertension'].astype(object)
df['heart_disease'] = df['heart_disease'].astype(object)
df['stroke'] = df['stroke'].astype(object)
df_int=df.select_dtypes(include=np.number)
df_cat=df.select_dtypes(exclude=np.number)
df_int.head()
df_cat.head()
for i in df_cat:
    sns.countplot(df[i])
    plt.show()

pd.crosstab(df['ever_married'],df['stroke']).plot(kind='bar',stacked=True)
plt.show()
pd.crosstab(df['work_type'],df['stroke']).plot(kind='bar',stacked=True)
plt.show()
plt.figure(figsize=(12,8))
sns.kdeplot(df[df['stroke']==0]['age'],shade=True,label='no_stroke')
sns.kdeplot(df[df['stroke']==1]['age'],shade=True,label='stroke')
plt.xlabel('Age')
plt.title('Stroke Density vs Age')

plt.legend()
plt.show()
plt.figure(figsize=(12,8))
sns.kdeplot(df[df['stroke']==0]['bmi'],shade=True,label='no_stroke')
sns.kdeplot(df[df['stroke']==1]['bmi'],shade=True,label='stroke')
plt.legend()
plt.title('Stroke Density vs BMI ')
plt.show()

#trích xuất đặc trưng
X=pd.get_dummies(df,columns=df_cat.columns,drop_first=True).iloc[:,:-2]
y=pd.to_numeric(df['stroke'])
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,stratify=y,random_state=8)
from imblearn.over_sampling import SMOTE
smt = SMOTE(random_state=8)
X_train_sm, y_train_sm = smt.fit_resample(X_train, y_train)

from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report
knnsm=KNeighborsClassifier()
knnsm.fit(X_train_sm,y_train_sm)
print('Train:',knnsm.score(X_train_sm,y_train_sm))
print('Test:',knnsm.score(X_test,y_test))
y_pred_knnsm=knnsm.predict(X_test)

print(classification_report(y_test,y_pred_knnsm))