# -*- coding: utf-8 -*-
"""
Created on Fri Dec 18 13:36:08 2020

@author: daviej9

Functions:
    * _load_data
    * _find_distinct_values
    * _scaling
    * _feature_selection
    * _train_test_split
    * _visualise
    * run    
    
Classes:
    None
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Powerpoint suggested libraries
from sklearn import preprocessing
from sklearn.feature_selection import VarianceThreshold
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report

from rich.console import Console
console = Console()


def _load_data(path):
    """
    load data
    
    Args:
        path (string)
        
    Returns:
        pandas dataframe: 
    """
    return pd.read_csv(path, header=0)


def _find_distinct_values(df):
    """
    Create a dict of unique values in each col in df
    
    Args:
        df
        
    Returns:
        distinct_dict
    """
    distinct_dict = {}
    for col in df.columns:
        distinct_dict[col] = df[col].unique()
    return distinct_dict


def _scaling(series_name):
    """
    _scaling
    
    Args:
        series_name
        
    Returns:
    """
    sc = preprocessing.OrdinalEncoder()
    sc_fit = sc.fit(series_name)   
    sc_transform = sc_fit.transform(series_name)
    return sc_transform
    
    
def _feature_selection(df):
    """
    _feature_selection
    
    Args:
        df
    Returns:
        df
    """
    # Kirsty messages: - do for all categorical columns
    le = preprocessing.LabelEncoder()
    le.fit(df['target'])
    return le


def _train_test_split(df):
    """
    _train_test_split
    
    Args:
        df
        
    Returns:
        stuff
    """
    #train_test_split(x, y, test_size=0.3)
    target_value = "target"
    feature_names = df.columns[np.where(df.columns != target_value)]
    X = df[feature_names].values
    y = df[target_value].values.ravel()

    (X_train, X_test,
     y_train, y_test) = train_test_split(X, y, test_size = 0.3,
                                         random_state=42)
    
    return X_train, X_test, y_train, y_test
  

def _visualise(clf, X_test, y_test):
  """
  _visualise
  """
  y_pred = clf.predict(X_test)
  plt.scatter(y_test, y_pred)
  plt.xlabel("True values")
  plt.ylabel("Predictions")
  


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
    
    
    if verbose == 1:
      print("Input Data:")
      print(raw_df.head())
      distinct_dict = _find_distinct_values(raw_df)
      print(distinct_dict)
        
        
#    # feature selection
#    le =  _feature_selection(raw_df)
    
       
    df = raw_df
   
    
    # train and test split
    (X_train, X_test,
     y_train, y_test) =  _train_test_split(df)
    
    
    # scaling
    X_train = _scaling(X_train)
    X_test = _scaling(X_test)
    # y scaling breaks :(
    #y_train = _scaling(y_train)
    #y_test = _scaling(y_test)
    
    
    clf = MLPClassifier(hidden_layer_sizes=(100, ),
                        activation='relu', solver='adam',
                        alpha=0.0001, batch_size='auto',
                        learning_rate='constant', learning_rate_init=0.001,
                        power_t=0.5, max_iter=10000000000000000000,
                        shuffle=True,
                        random_state=6, tol=0.0001, verbose=False,
                        warm_start=False, momentum=0.9, nesterovs_momentum=True,
                        early_stopping=False, validation_fraction=0.1,
                        beta_1=0.9, beta_2=0.999, epsilon=1e-08,
                        n_iter_no_change=10, max_fun=15000
                        ).fit(X_train, y_train)
    clf.predict_proba(X_test[:1])
    clf.predict(X_test[:5, :])
    score = clf.score(X_test, y_test)
    print(clf)
    print(score)
    
    #_visualise(clf, X_test, y_test)
    
    report = classification_report(y_test, clf.predict(X_test))
    print(report)
    
    
    results_dict = {'raw_data': raw_df}
    
    return results_dict, score, report

    
if __name__ == "__main__":    
    #path = "H:\My Documents\christmas_hack\christmas_hack_naughty_or_nice.csv"
    path = "/home/cdsw/christmas_hack_naughty_or_nice.csv"
    
    verbose = 0
    party = True
    results, score, report = run(path, verbose)
    
    # Party?!
    if party:
      for i in range(20):
        console.print(":tada:", f"SCORE = {score} so PARTY!", ":tada:")
        i = i + 1
    
    console.print(":sunglasses:", "Report", ":sunglasses:")
    print(report)
    