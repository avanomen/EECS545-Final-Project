import pandas as pd
import os
import numpy as np
import xgboost2

# load higgs boson dataset
print('loading dataset...')
higgs_dir = r'F:\Google Drive\umich\eecs545_machine_learning\final_project\HIGGS'
higgs_df = pd.read_csv(os.path.join(higgs_dir, 'HIGGS.csv'))

test_n = 500000
train_n = 50000
use_top_features = True

Y = higgs_df.iloc[:,0]
X = higgs_df.iloc[:,1:]
n = higgs_df.shape[0]
print('total num: {}'.format(n))
print('train samples: {}'.format(train_n))

if use_top_features:
    x_train_df = X.iloc[0:train_n,-7:]
else:
    x_train_df = X.iloc[0:train_n,:]
y_train_df = Y.iloc[0:train_n]
print(x_train_df.shape)
print(y_train_df.shape)
x_test_df = X.iloc[-test_n:, :]
y_test_df = Y.iloc[-test_n:]

model_higgs = xgboost2.XGBclassifier()
model_higgs.fit(x_train_df.values, y_train_df.values, min_leaf_samples=50, max_depth=5)

pred = model_higgs.predict(x_test_df.values)

acc = np.sum(pred == y_test_df.values)/len(pred)
print('test acc: {}'.format(acc))