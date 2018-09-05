# -*- coding: utf-8 -*-
#%%
import tensorflow as tf
import pandas as pd
import shutil

train_set = pd.read_csv('./data/train.csv')
test_set = pd.read_csv('./data/test.csv')
#%%
description = train_set.describe(include = 'all')
corr = train_set.corr()
#%%
train_set = train_set.drop(['Name','PassengerId','Ticket'],axis=1)
test_set = test_set.drop(['Name','PassengerId','Ticket'],axis=1)
#%%
from sklearn.model_selection import train_test_split
train_set['Embarked'] = train_set['Embarked'].fillna('S')
train_set['Cabin'] = train_set['Cabin'].fillna('N')
train_set['Cabin'] = train_set['Cabin'].apply(lambda x: x[0])
test_set['Embarked'] = test_set['Embarked'].fillna('S')
train_set = train_set.fillna(0)
test_set = test_set.fillna(0)
description = train_set.describe(include = 'all')
corr = train_set.corr()
#%%
features = train_set[['Pclass','Sex','Age','SibSp','Parch','Fare','Embarked','Cabin']]
labels = train_set.pop('Survived')
X_train,X_test,y_train,y_test = train_test_split(features,labels,test_size=0.2)
train_input_fn = tf.estimator.inputs.pandas_input_fn(X_train,y_train,shuffle=True,num_epochs=10000)
eval_input_fn = tf.estimator.inputs.pandas_input_fn(X_test,y_test,shuffle=True)
#%% 
pClass_column = tf.feature_column.categorical_column_with_identity('Pclass',4)
pClass_column = tf.feature_column.indicator_column(pClass_column)
sex_column = tf.feature_column.categorical_column_with_vocabulary_list('Sex',['male','female'])
sex_column = tf.feature_column.indicator_column(sex_column)
age_column = tf.feature_column.numeric_column('Age')
sibSp_column = tf.feature_column.numeric_column('SibSp')
parch_column = tf.feature_column.numeric_column('Parch')
fare_column = tf.feature_column.numeric_column('Fare')
embarked_column = tf.feature_column.categorical_column_with_vocabulary_list('Embarked',['C','Q','S'])
embarked_column = tf.feature_column.indicator_column(embarked_column)
cabin_column = tf.feature_column.categorical_column_with_vocabulary_list('Cabin',['N','C','E','G','D','A','B','F','T'])
cabin_column = tf.feature_column.embedding_column(cabin_column,dimension=3)
run_config = tf.estimator.RunConfig(
        save_checkpoints_steps=500,
        save_summary_steps=500,
        model_dir="./model/",
        keep_checkpoint_max=None
)
optimizer = tf.train.AdamOptimizer(learning_rate=0.0001)
estimator = tf.estimator.DNNClassifier(
        feature_columns=[age_column,pClass_column,sex_column,sibSp_column,fare_column,parch_column,embarked_column,cabin_column],
        hidden_units=[100,120,140,140,140,140,140,120,100],
        optimizer=optimizer,
        model_dir="./model/",
        dropout=0.45,
        batch_norm=True,
        config=run_config
)
train_spec = tf.estimator.TrainSpec(input_fn=train_input_fn,max_steps=50000)
eval_spec = tf.estimator.EvalSpec(input_fn=eval_input_fn,throttle_secs=20,start_delay_secs=20)
shutil.rmtree("./model/",ignore_errors=True)
#%%
tf.estimator.train_and_evaluate(estimator,train_spec,eval_spec)
