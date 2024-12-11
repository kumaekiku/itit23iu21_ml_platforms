#!/usr/bin/env python
# coding: utf-8

from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB, CategoricalNB
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder, FunctionTransformer, MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from IPython.display import display

class NBVariants:
    def __init__(self, df, target, models:dict=None, categorical=None, numerical=None):
        self.df = df
        self.target = target
        self.models = models or {
            'gaussian': GaussianNB(),
            'multinomial': MultinomialNB(),
            'bernoulli': BernoulliNB(),
        }
        self.cat_f = categorical
        self.num_f = numerical
        
        self.scaler = StandardScaler()
        self.encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')

    def f_fit(self, X_in):
        """
        Fit features to scaler and encoder
        """
        if self.num_f: self.scaler.fit(X_in[self.num_f])
        if self.cat_f: self.encoder.fit(X_in[self.cat_f])

    def f_transform(self, X_in):
        """
        Apply scaler and encoder to transform features
        """
        X_num = self.scaler.transform(X_in[self.num_f])
        X_cat = self.encoder.transform(X_in[self.cat_f])
        return np.column_stack([X_num, X_cat])
        
    def train_models(self, test_size=0.2, random_state=None):
        """
        Train multiple NB models
        """
        df_train, df_val = train_test_split(self.df, test_size=test_size, random_state=random_state)

        for data in [df_train, df_val]:
            data.reset_index(drop=True)

        y_train = df_train[self.target].values
        y_val = df_val[self.target].values

        X_train = df_train.drop(columns=[self.target])
        X_val = df_val.drop(columns=[self.target])

        self.f_fit(X_train)
        X_train = self.f_transform(X_train)
        X_val = self.f_transform(X_val)

        results = {}
        for name, model in self.models.items():
            if name == 'multinomial':
                # Ensure all values are non-negative for MultinomialNB
                X_train = X_train - X_train.min()
                
            model.fit(X_train, y_train)
            y_pred = model.predict(X_val)
            report = classification_report(y_pred, y_val, zero_division=True)

            print(f"---{name.capitalize()}---")
            print(report)
            
            results[name] = {
                'report': classification_report(y_pred, y_val, output_dict=True, zero_division=True),
                'cm': confusion_matrix(y_pred, y_val)
            }
            
        return results
        
    def compare(self, results):
        """
        Plot model performance to make comparisons
        """
        model_types = list(self.models.keys())
        accuracy = [dict(results[name]['report'])['accuracy'] for name in model_types]

        fig, axes = plt.subplots(1, 4, figsize=(20, 5))
        
        axes[0].bar(model_types, accuracy)
        axes[0].set_title('Accuracy Comparison')
        axes[0].set_ylim(0, 1)
        for i, v in enumerate(accuracy):
            axes[0].text(i, v+0.01, f'{v:.3f}', ha='center')

        for i, (name, result) in enumerate(results.items()):
            sns.heatmap(result['cm'], annot=True, fmt='d', ax=axes[i+1])
            axes[i+1].set_title(f"{name.capitalize()}")
            
        plt.tight_layout()
        plt.show()
    
class ParamsTuning:
    def __init__(self, df, target, numerical, categorical):
        self.df = df
        self.num = numerical
        self.cat = categorical
        self.target = target
        
    def grid_search(self, model, param_grid, 
                    n_splits=10, n_jobs=-1, n_iter = 50,
                    scoring='accuracy', verbose=0):
        
        # initialize pipeline
        self.__create_pipeline(model)

        # get features and target
        y = self.df[self.target].values
        X = self.df.drop(columns=[self.target])

        # create a grid search
        search = GridSearchCV(
            self.pipeline,
            param_grid,
            cv=n_splits,
            scoring=scoring,
            n_jobs=n_jobs,
            verbose=verbose,
        )

        # fit the values to grid search
        search.fit(X, y)
        return search

    def __create_pipeline(self, model) -> None:
        """
        Features transforming
        """
        if isinstance(model, MultinomialNB):
            num_transformer = Pipeline(steps=[
                ('shift', FunctionTransformer(lambda X: X-X.min()+1e-10, validate=True)),
                ('scaler', MinMaxScaler()) # recommended for Multinomial NB
            ])
        else:
            num_transformer = Pipeline(steps=[('scaler', StandardScaler())])
            
        cat_transformer = Pipeline(steps=[('onehot', OneHotEncoder(sparse_output=False, handle_unknown='ignore'))])

        f_preprocessor = ColumnTransformer(
            transformers=[
                ('num', num_transformer, self.num),
                ('cat', cat_transformer, self.cat)
            ]
        )

        self.pipeline = Pipeline([
            ('preprocessor', f_preprocessor),
            ('classifier', model)
        ])