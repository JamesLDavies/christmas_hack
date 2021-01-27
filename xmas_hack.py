# -*- coding: utf-8 -*-
"""
Created on Fri Dec 18 13:36:08 2020

@author: daviej9

Usage:
* pip3 install -r requirements.txt
* streamlit run xmas_hack.py

Functions:
    * _load_data
    * _find_distinct_values
    * _scaling
    * _train_test_split
    * _predict
    * _visualise_features_to_behaviour
    * run    
    
Classes:
    None
"""

import numpy as np
import pandas as pd
import plotly.express as px

import streamlit as st

from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report

# Other models
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

from rich.console import Console
console = Console()

st.set_page_config(
        page_title='James Davies Naughty or Nice ML Dashboard',
        layout='wide'
        )
    
st.sidebar.title('James Davies Naughty or Nice ML Dashboard')
st.sidebar.write('This is a Machine Learning dashboard')


def _load_data(path):
    """
    load data
    
    Args:
        path (string)
        
    Returns:
        pandas dataframe: 
    """
    return pd.read_csv(path, header=0)


def _scaling(series_name):
    """
    _scaling
    
    Args:
        series_name
        
    Returns:
        sc_transform
    """
    sc = preprocessing.OrdinalEncoder()
    sc_fit = sc.fit(series_name)   
    sc_transform = sc_fit.transform(series_name)
    return sc_transform
  

def _train_test_split(df, target_value):
    """
    _train_test_split
    
    Args:
        df
        
    Returns:
        X_train
        X_test
        y_train
        y_test
    """
    test_size = st.sidebar.slider(
        label='Select the proportion of the input data to use for testing',
        min_value=0.05,
        max_value=0.95,
        step=0.05,
        value=0.3
        )
    feature_names = df.columns[np.where(df.columns != target_value)]
    X = df[feature_names].values
    y = df[target_value].values.ravel()

    (X_train, X_test,
     y_train, y_test) = train_test_split(X, y, test_size=test_size,
                                         random_state=42)
    
    return X_train, X_test, y_train, y_test
  
  
def _predict(model, test):
  """
  Predict whether someone is naughy or nice
  
  Args:
    model
    test
  """
  X_test = _scaling(test)
  y_pred = model.predict(X_test[:len(test)])
  
  print(f"Predictions: {y_pred}")
  
  return y_pred

      
def _visualise_features_to_behaviour(df, target_col):
  """
  Visualise the data
  
  Args:
    df (Pandas DataFrame)
    target_col (String)
    
  Returns:
    None
  """
  x_col = st.selectbox(label='What do you want to plot?',
                       options=sorted(df.columns.drop(target_col)))
  st.plotly_chart(
      px.histogram(df, x=x_col, barmode='group', color=target_col),
      use_container_width=True,)   
    
    
def _k_neighbor_model(df, X_train, y_train, X_test, y_test):
  """
  docstring
  """
  best_score = 0
  max_k = int(len(df) / 2)
  for k in range(1, max_k):
    clf = KNeighborsClassifier(n_neighbors=k
                              ).fit(X_train, y_train)
    score = clf.score(X_test, y_test)
    
    if score > best_score:
      best_score = score
      model = clf
      optimum_k = k
    
  print(f"Optimum k: {optimum_k}")
  return model


def _MLP_classifier(X_train, y_train):
  """
  docstring
  """
  parameter_space = {
    'hidden_layer_sizes': [(50,50,50), (50,100,50), (100,)],
    'activation': ['tanh', 'relu'],
    'solver': ['sgd', 'adam'],
    'alpha': [0.0001, 0.05],
    'learning_rate': ['constant','adaptive'],
    'random_state': [0, 1000000]
 }
  
  mlp = MLPClassifier(max_iter=1000000)

  clf = GridSearchCV(mlp, parameter_space, n_jobs=-1, cv=3)
  clf.fit(X_train, y_train)
  
  # Best paramete set
  print('Best parameters found:\n', clf.best_params_)

  # All results
#  means = clf.cv_results_['mean_test_score']
#  stds = clf.cv_results_['std_test_score']
#  for mean, std, params in zip(means, stds, clf.cv_results_['params']):
#      print("%0.3f (+/-%0.03f) for %r" % (mean, std * 2, params))
#  
#  clf = MLPClassifier(activation='relu', alpha=0.0001,
#                      hidden_layer_sizes=(100,), learning_rate='constant',
#                      random_state=0, solver='sgd'
#                      ).fit(X_train, y_train) 
  return clf
  
    

def run(path, verbose=0):
    """
    Main ENTRYPOINT function
    
    Args:
        path(String): path to raw data
        verbose(Int, optional): 0 is default
        
    Returns:
        results_dict (dict)
    """
    raw_df = _load_data(path)
    
    st.write('Head of Input Data:')
    st.dataframe(raw_df.head())
    
    target_col = st.sidebar.selectbox(label='Select target column:',
                                      options=sorted(raw_df.columns))
    
    _visualise_features_to_behaviour(raw_df, target_col)
       
    df = raw_df
    
    # train and test split
    (X_train, X_test,
     y_train, y_test) =  _train_test_split(df, target_col)
    
    
    # scaling
    X_train = _scaling(X_train)
    X_test = _scaling(X_test)
    
    # Choose model
    # TODO: Add arg functions
    model_dict = {
        'KNeighborsClassifier': {
            'model': KNeighborsClassifier,
            'args': None
            },
        'DecisionTreeClassifier': {
            'model': DecisionTreeClassifier,
            'args': None
            },
        'LogisticRegression': {
            'model': LogisticRegression,
            'args': None
            },
        'GaussianNB': {
            'model': GaussianNB,
            'args': None
            },
        'SVC': {
            'model': SVC,
            'args': None
            },
        'MLPClassifier': {
            'model': MLPClassifier,
            'args': None
            }
        }
        
    model_choice = st.sidebar.selectbox(
        'Please choose the ML model you would like to use:',
        sorted(model_dict.keys())
        )
    
    clf = model_dict[model_choice]['model']().fit(X_train, y_train)
    
    clf.predict_proba(X_test[:1])
    clf.predict(X_test[:5, :])
    
    score = clf.score(X_test, y_test)
    report = classification_report(y_test, clf.predict(X_test))
    
    st.write(f'Score: {score}')
    st.text('Report: \n' + report)
    
    # TODO: Add this to the dashboard
    # Predict whether someone is naughty or nice
    test = [['blue', 'human', 'red', 'tall', 'alive'],
            ['brown', 'human', 'bald', 'tall', 'alive']]
    
    # TODO: Write a function to find unique vals and put this in a dict
    test_data = {}
    if target_col != 'target':
        target = st.selectbox('target', ['naughty', 'nice'])
        test_data['target'] = target
        
    if target_col != 'eye_colour':
        eye_colour = st.selectbox('eye_colour', ['blue', 'green', 'brown'])
        test_data['eye_colour'] = eye_colour
        
    if target_col != 'race':
        race = st.selectbox('race', ['human', 'android', 'mutant'])
        test_data['race'] = race
        
    if target_col != 'hair_colour':
        hair_colour = st.selectbox('hair_colour', ['bald', 'black', 'blonde', 'grey'])
        test_data['hair_colour'] = hair_colour
        
    if target_col != 'height':
        height = st.selectbox('height', ['tall', 'medium', 'small'])
        test_data['height'] = height
        
    if target_col != 'status':
        status = st.selectbox('status', ['alive', 'dead'])
        test_data['status'] = status
        
    st.write(f"TEST DATA: {test_data}")
    
    test_data_2 = [list(test_data.values()),
                   ['brown', 'human', 'bald', 'tall', 'alive']]
    prediction = _predict(clf, test_data_2)
    st.write(f"PREDICTION: {prediction}")
        
    y_pred = _predict(clf, test)
    
    
    results_dict = {'raw_data': raw_df}
    
    return results_dict, score, report



    
if __name__ == "__main__":
  
    path = "christmas_hack_naughty_or_nice.csv"
    
    verbose = 0
    party = False
    results, score, report = run(path, verbose)
    
    # Party?!
    if party:
      for i in range(20):
        console.print(":tada:", f"SCORE = {score} so PARTY!", ":tada:")
        i = i + 1
    
    console.print(":sunglasses:", f"SCORE = {score}", ":sunglasses:")
    console.print(":sunglasses:", "Report", ":sunglasses:")
    print(report)
    