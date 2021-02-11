# -*- coding: utf-8 -*-
"""
Created on Fri Dec 18 13:36:08 2020

@author: daviej9

Usage:
* pip3 install -r requirements.txt
* streamlit run xmas_hack.py

Functions:
    * _load_data
    * _scaling
    * _train_test_split
    * _predict
    * _visualise_features_to_behaviour
    * args_KNeighborsClassifier
    * args_DecisionTreeClassifier
    * args_LogisticRegression
    * args_GaussianNB
    * args_SVC
    * args_MLPClassifier
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
from sklearn.metrics import classification_report

# Other models
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC


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
    

def args_KNeighborsClassifier():
    """
    Options for arguments
    
    Args:
        None
        
    Returns:
        Dict: keys(String) - arg
              values - argument in the form of a streamlit option
    """
    return {
        'n_neighbors': st.sidebar.slider(
            label='Number of neighbours:',
            min_value=1,
            max_value=20,
            step=1
            ),
        'weights': st.sidebar.selectbox(
            label='weights',
            options=['uniform', 'distance']
            ),
        'algorithm': st.sidebar.selectbox(
            label='algorithm',
            options=['auto', 'ball_tree', 'kd_tree', 'brute']
            ),
        }


def args_DecisionTreeClassifier():
    """
    Options for arguments
    
    Args:
        None
        
    Returns:
        Dict: keys(String) - arg
              values - argument in the form of a streamlit option
    """
    return {}


def args_LogisticRegression():
    """
    Options for arguments
    
    Args:
        None
        
    Returns:
        Dict: keys(String) - arg
              values - argument in the form of a streamlit option
    """
    return {
        'penalty': st.sidebar.selectbox(
            label='penalty',
            options=['l2', 'l1', 'elasticnet', 'none']
            ),
        'dual': st.sidebar.selectbox(
            label='dual',
            options=[False, True]
            ),
        'solver': st.sidebar.selectbox(
            label='solver',
            options=['lbfgs', 'newton-cg', 'liblinear', 'sag', 'saga']
            ),
        }


def args_GaussianNB():
    """
    Options for arguments
    
    Args:
        None
        
    Returns:
        Dict: keys(String) - arg
              values - argument in the form of a streamlit option
    """
    return {}


def args_SVC():
    """
    Options for arguments
    
    Args:
        None
        
    Returns:
        Dict: keys(String) - arg
              values - argument in the form of a streamlit option
    """
    return {
        'kernel': st.sidebar.selectbox(
            label='kernel',
            options=['rbf', 'linear', 'poly', 'sigmoid', 'precomputed']
            ),
        }


def args_MLPClassifier():
    """
    Options for arguments
    
    Args:
        None
        
    Returns:
        Dict: keys(String) - arg
              values - argument in the form of a streamlit option
    """
    return {
        'activation': st.sidebar.selectbox(
            label='activation',
            options=['relu', 'identity', 'logistic', 'tanh']
            ),
        'solver': st.sidebar.selectbox(
            label='solver',
            options=[ 'adam', 'lbfgs', 'sgd']
            ),
        'learning_rate': st.sidebar.selectbox(
            label='learning_rate',
            options=['constant', 'invscaling', 'adaptive']
            )
        }
  
    
def run(path):
    """
    Main ENTRYPOINT function
    
    Args:
        path(String): path to raw data
        
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
    model_dict = {
        'KNeighborsClassifier': {
            'model': KNeighborsClassifier,
            'args': args_KNeighborsClassifier
            },
        'DecisionTreeClassifier': {
            'model': DecisionTreeClassifier,
            'args': args_DecisionTreeClassifier
            },
        'LogisticRegression': {
            'model': LogisticRegression,
            'args': args_LogisticRegression
            },
        'GaussianNB': {
            'model': GaussianNB,
            'args': args_GaussianNB
            },
        'SVC': {
            'model': SVC,
            'args': args_SVC
            },
        'MLPClassifier': {
            'model': MLPClassifier,
            'args': args_MLPClassifier
            }
        }
        
    model_choice = st.sidebar.selectbox(
        'Please choose the ML model you would like to use:',
        sorted(model_dict.keys())
        )
    
    model_args = model_dict[model_choice]['args']()
    
    model = model_dict[model_choice]['model'](**model_args)
    
    clf = model.fit(X_train, y_train)
    
    clf.predict_proba(X_test[:1])
    clf.predict(X_test[:5, :])
    
    score = clf.score(X_test, y_test)
    report = classification_report(y_test, clf.predict(X_test))
    
    st.write(f'Score: {score}')
    st.text('Report: \n' + report)
    
    # TODO: This could be improved
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
        hair_colour = st.selectbox('hair_colour',
                                   ['bald', 'black', 'blonde', 'grey'])
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
    
    results_dict = {'raw_data': raw_df}
    
    return results_dict, score, report



    
if __name__ == "__main__":
    path = "christmas_hack_naughty_or_nice.csv"
    results, score, report = run(path)
    