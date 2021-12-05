import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split

np.random.seed(1)

dataset_dir = r'F:\Google Drive\umich\eecs545_machine_learning\final_project'

def exam_data_load(df, target, id_name="", null_name=""):
    if id_name == "":
        df = df.reset_index().rename(columns={"index": "id"})
        id_name = 'id'
    else:
        id_name = id_name
    
    if null_name != "":
        df[df == null_name] = np.nan
    
    X_train, X_test = train_test_split(df, test_size=0.2, shuffle=True, random_state=2021)
    y_train = X_train[[id_name, target]]
    X_train = X_train.drop(columns=[id_name, target])
    y_test = X_test[[id_name, target]]
    X_test = X_test.drop(columns=[id_name, target])
    return X_train, X_test, y_train, y_test 
    
df = pd.read_csv(os.path.join(dataset_dir, 'adult.csv'))
X_train, X_test, y_train, y_test = exam_data_load(df, target='income', null_name='?')

X_train.shape, X_test.shape, y_train.shape, y_test.shape

from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder, OneHotEncoder
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score

X_train.info()
X_train.isnull().sum()

print(X_train['workclass'].unique())
print(X_train['occupation'].unique())
print(X_train['native.country'].unique())

# check the number of values in each category
print(X_train['workclass'].value_counts())
print(X_test['workclass'].value_counts())

print(X_train['occupation'].value_counts())
print(X_test['occupation'].value_counts())

print(X_train['native.country'].value_counts())
print(X_test['native.country'].value_counts())

# Since not a few numbers are missing, we cannot ignore them
# For the workclass and the native country, almost every value is concentrated to the most frequent category, so replace it by the mode of each category
# However, for occupation, let's treat the missing values as another category, 'unknown'

X_train['workclass'].fillna(X_train['workclass'].mode()[0], inplace=True)
X_train['native.country'].fillna(X_train['native.country'].mode()[0], inplace=True)
X_train['occupation'].fillna('unknown', inplace=True)

X_test['workclass'].fillna(X_test['workclass'].mode()[0], inplace=True)
X_test['native.country'].fillna(X_test['native.country'].mode()[0], inplace=True)
X_test['occupation'].fillna('unknown', inplace=True)

X_train.isnull().sum()

# check if the education and education.num matches
def make_tuple(x):
    return (x['education'], x['education.num'])

edu = X_train[['education', 'education.num']].apply(make_tuple, axis=1)
print(edu.unique())

# Since education.num represents the education perfectly, drop the education feature
X_train.drop(['education'], axis=1, inplace=True)
X_test.drop(['education'], axis=1, inplace=True)

# Training Set
cat_features = ['workclass', 'marital.status', 'occupation', 'relationship', 'race', 'sex', 'native.country']
for cat_col in cat_features:
    le = LabelEncoder()
    le.fit(X_train[cat_col])
    
    X_train[cat_col] = le.transform(X_train[cat_col])
    X_test[cat_col] = le.transform(X_test[cat_col])

X_train.head()

# Test Set - >50k: 1, <=50k: 0
onehot = OneHotEncoder()
onehot.fit(y_train[['income']])
y_train['income'] = onehot.transform(y_train[['income']]).toarray()[:, 1]
y_test['income'] = onehot.transform(y_test[['income']]).toarray()[:, 1]
X_train.describe()

num_features = ['age', 'fnlwgt', 'capital.gain', 'capital.loss', 'hours.per.week']

# check skewness
for num_col in num_features:
    print(num_col, 'skewness: %.4f' % (X_train[num_col].skew()))

# Transform every variable to logarithmic scale
for num_col in num_features[:-1]:
    if 0 in list(X_train[num_col]):
        scaled = np.log1p(X_train[num_col])
    else:
        scaled = np.log(X_train[num_col])
    
    print(num_col, 'skewness: %.4f' % (scaled.skew()))

# Transform
for num_col in num_features[:-1]:
    if 0 in list(X_train[num_col]):
        X_train[num_col] = np.log1p(X_train[num_col])
    else:
        X_train[num_col] = np.log(X_train[num_col])
        
    if 0 in list(X_test[num_col]):
        X_test[num_col] = np.log1p(X_test[num_col])
    else:
        X_test[num_col] = np.log(X_test[num_col])

X_train.head(3)

# And standardize the numerical features
for num_col in num_features:
    std = StandardScaler()
    std.fit(X_train[[num_col]])
    X_train[num_col] = std.transform(X_train[[num_col]]).flatten()
    X_test[num_col] = std.transform(X_test[[num_col]]).flatten()

X_train.head(3)

# save dataframes
X_train.to_csv('./x_train_adult.csv')
X_test.to_csv('./x_test_adult.csv')
y_train.to_csv('./y_train_adult.csv')
y_test.to_csv('./y_test_adult.csv')