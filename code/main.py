import pandas as pd
import preprocessing as pre
import os
import warnings
warnings.filterwarnings('ignore')

# import datasets
path_train = os.path.join('..', 'data', 'dcard-post-train.csv')
path_test = os.path.join('..', 'data', 'dcard-post-test.csv')
df_train = pd.read_csv(path_train)
df_test = pd.read_csv(path_test)

# preprocess data
X_train, X_test, y_train, y_test = pre.preproc_data(df_train, df_test)