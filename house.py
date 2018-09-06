# -*- coding: utf-8 -*-
#%%
import tensorflow as tf
import pandas as pd

train_set = pd.read_csv('./data/train-1.csv')
train_set = train_set.drop(['Id'],axis=1)
#%%
description = train_set.describe(include = 'all')
corr = train_set.corr()
#%%
from math import isnan
numeric_keys = []
categorical_keys = []
date_keys = []
for column in train_set.columns:
    if description.loc["count"][column] < 100:
        train_set = train_set.drop([column],axis=1)
    elif ("Yr" in column) or ("Year" in column):
        train_set[column] = train_set[column].fillna(description.loc["mean"][column])
        date_keys.append(column)
    elif isnan(description.loc["mean"][column]):
        train_set[column] = train_set[column].fillna(description.loc["top"][column])
        categorical_keys.append(column)
    else:
        train_set[column] = train_set[column].fillna(description.loc["mean"][column])
        numeric_keys.append(column)
numeric_keys.remove('SalePrice')
#%%
description = train_set.describe(include='all')
corr = train_set.corr()
#%%
from sklearn.model_selection import train_test_split
labels = train_set.pop('SalePrice')
features = train_set
X_train,X_test,y_train,y_test = train_test_split(features,labels,test_size=0.2)
variance = y_test.var()
train_input_fn = tf.estimator.inputs.pandas_input_fn(X_train,y_train,shuffle=True,num_epochs=5000,batch_size=128)
eval_input_fn = tf.estimator.inputs.pandas_input_fn(X_test,y_test,shuffle=True)
#%% 
feature_column = []
for key in categorical_keys:
    feature = tf.feature_column.categorical_column_with_hash_bucket(key,description.loc["unique"][key])
    feature = tf.feature_column.embedding_column(feature,dimension=4)
    feature_column.append(feature)
for key in numeric_keys:
    feature = tf.feature_column.numeric_column(key,dtype=tf.float64)
    feature_column.append(feature)
for key in date_keys:
    original = tf.feature_column.numeric_column(key)
    feature = tf.feature_column.bucketized_column(original,[description.loc["25%"][key],description.loc["50%"][key],description.loc["75%"][key]])
    feature = tf.feature_column.indicator_column(feature)
    feature_column.append(feature)

def create_crossed_column(c1,c2,c1_boundary,c2_boundary):
    c1_bucketized_column = tf.feature_column.bucketized_column(tf.feature_column.numeric_column(c1),c1_boundary)
    c2_bucketized_column = tf.feature_column.bucketized_column(tf.feature_column.numeric_column(c2),c2_boundary)
    return tf.feature_column.embedding_column(
                tf.feature_column.crossed_column([c1_bucketized_column,c2_bucketized_column],20),
                dimension=4
            )

feature_column.append(create_crossed_column('TotRmsAbvGrd','GrLivArea',[5,6,7],[500,1130,1464,1776,2500]))
feature_column.append(create_crossed_column('1stFlrSF','TotalBsmtSF',[882,1087,1391,1700],[330,795,991,1298,1700]))
feature_column.append(create_crossed_column('GarageArea','GarageCars',[1,2,3],[334,480,576]))
run_config = tf.estimator.RunConfig(
        save_checkpoints_steps=500,
        save_summary_steps=500,
        model_dir="./model/",
        keep_checkpoint_max=None
)
optimizer = tf.train.AdamOptimizer(learning_rate=0.01)
estimator = tf.estimator.DNNRegressor(
        feature_columns=feature_column,
        hidden_units=[100,120,120,100],
        optimizer=optimizer,
        dropout=0.6,
        batch_norm=True,
        config=run_config
)

train_spec = tf.estimator.TrainSpec(input_fn=train_input_fn,max_steps=20000)
eval_spec = tf.estimator.EvalSpec(input_fn=eval_input_fn,throttle_secs=40,start_delay_secs=40)
shutil.rmtree("./model/",ignore_errors=True)
#%%
tf.estimator.train_and_evaluate(estimator,train_spec,eval_spec,)
