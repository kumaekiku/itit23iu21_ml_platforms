#!/usr/bin/env python
# coding: utf-8

import numpy as np
import pandas as pd

from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn.metrics import confusion_matrix, roc_auc_score
from sklearn.model_selection import train_test_split, GridSearchCV, cross_validate
from sklearn.preprocessing import StandardScaler, OneHotEncoder, FunctionTransformer, MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

class NBWrapper:
    def __init__(self, model, target, num, cat):
        cat_tf = Pipeline(steps=[('onehot', OneHotEncoder(sparse_output=False, handle_unknown='ignore'))])
        num_tf = 'passthrough'
        
        if isinstance(model, MultinomialNB):
            num_tf = Pipeline(steps=[
                ('shift', FunctionTransformer(lambda X: X-X.min()+1e-10, validate=True)),
            ])
        elif isinstance(model, GaussianNB):
            num_tf = Pipeline(steps=[('scaler', StandardScaler())])
            
        preprocessor = ColumnTransformer(transformers=[
            ('num', num_tf, num),
            ('cat', cat_tf, cat)
        ])
        
        self.model = Pipeline([
            ('preprocessor', preprocessor),
            ('classifier', model)
        ])

    def cross_validate(self, X, y, cv=10, n_jobs=-1, scoring='roc_auc'):
        result = cross_validate(
            self.model,
            X=X, y=y, cv=cv,
            scoring=scoring,
            n_jobs=n_jobs,
        )
        
        return np.mean(result['test_score'])

    def grid_search(self, X, y, param_grid, cv=10, n_jobs=-1, scoring='roc_auc'):
        search = GridSearchCV(
            self.model,
            param_grid,
            cv=cv,
            scoring=scoring,
            n_jobs=n_jobs,
        )

        search.fit(X, y)
        return search.best_params_