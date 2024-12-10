#!/usr/bin/env python
# coding: utf-8

from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB, CategoricalNB
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from IPython.display import display

class NBVariants:
    def __init__(self, df, target, categorical=None, numerical=None):
        self.models = {
            'gaussian': GaussianNB(),
            'multinomial': MultinomialNB(),
            'bernoulli': BernoulliNB(),
        }
        self.df = df
        self.target = target
        self.cat_f = categorical
        self.num_f = numerical
        self.scaler = StandardScaler()
        self.encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')

    def __f_fit(self, X_in):
        """
        Fit scaler and encoder
        """
        if self.num_f: self.scaler.fit(X_in[self.num_f])
        if self.cat_f: self.encoder.fit(X_in[self.cat_f])

    def __f_transform(self, X_in):
        """
        Apply scaler and encoder to transform X variable
        """
        X_num = self.scaler.transform(X_in[self.num_f])
        X_cat = self.encoder.transform(X_in[self.cat_f])
        return np.column_stack([X_num, X_cat])
        
    def getXy(self, data):
        """
        Obtain corresponding X and y for a dataframe
        """
        self.__f_fit(data.drop(columns=[self.target]))
        X = self.__f_transform(data.drop(columns=[self.target]))
        y = data[self.target].values
        return X, y
        
    def train_models(self, test_size=0.2, random_state=None):
        """
        Train multiple NB models
        """
        df_train, df_test = train_test_split(self.df, test_size=test_size, random_state=random_state)
        X_train, y_train = self.getXy(df_train)
        X_val, y_val = self.getXy(df_train)

        results = {}
        for name, model in self.models.items():
            if name == 'multinomial':
                # Ensure all values are non-negative for MultinomialNB
                X_train = X_train - X_train.min()
            else:
                X_train = X_train
                
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