# -*- coding: utf-8 -*-
"""
@author: Anthony J. Vasquez
email: avasque1@jh.edu
phone: 315-920-8877

Class: Introduction to Machine Learning

Description: All functions needed to preprocess data
"""
import pandas as pd
import numpy as np

class PreprocessData:
    '''
    Preprocess dataframe
    '''
    def __init__(self, data, values_to_replace, values_to_change, dataset_name,
                 discretize_data, quantization_number, standardize_data, remove_orig_cat_col,
                 category_list, mode):
        print('Preprocess data initialized')
        self.data = data
        self.values_to_replace = values_to_replace ##list of values to replace
        self.values_to_change = values_to_change ##list of values to change Ex. 5more -> 5
        self.dataset_name = dataset_name ## name of dataset
        self.discretize_data = discretize_data ##discretize data flag
        self.quantization_number = quantization_number ##number to quantize data to
        self.standardize_data = standardize_data ##standardize data flag
        self.remove_orig_cat_col = remove_orig_cat_col ##remove the original categorical data
        self.category_list = category_list ##list of categorical data
        self.mode = mode

    ##drop rows containing cells with values values_to_drop
    def dropRowsBasedOnListValues(self):
        self.col_names = list(self.data)
        for col_name in self.col_names:
            self.data = self.data[self.data[col_name].isin(self.values_to_replace) == False]
        self.data = self.data.reset_index(drop=True) ##reset index for any dropped rows
        return self.data

    def changeValues(self):
        keys = list(self.values_to_change)
        for col_name in self.col_names:
            for key in keys:
                self.data.loc[self.data[col_name] == key, col_name] = self.values_to_change[key]
        return self.data

    ##replace containing cells with values values_to_drop
    def encodeData(self):
        for col_name in self.col_names:
            if col_name in self.category_list:
                dummies = pd.get_dummies(self.data[col_name], drop_first=True)
                if self.remove_orig_cat_col:
                    self.data = pd.concat([self.data, dummies], axis = 1)
                    print('Encoded Column: ', col_name)
        self.data = self.data.drop(self.category_list, axis=1) ##delete original categorical data
        return self.data
    
    def createUniqueAttribNames(self):
        ##make sure all attributes have a unique name
        new_col_names = []
        col_names = list(self.data)
        for idx, col_name in enumerate(col_names):
            if col_name != 'target':
                new_col_name = col_name + '_' + str(idx)
                new_col_names.append(new_col_name)
        
        ##extract target 
        target = self.data['target']
        self.data = self.data.drop('target', axis=1)
        ##rename cols
        self.data.columns = new_col_names
        ##add target back in
        self.data = pd.concat([self.data, target], axis = 1)
                
        return self.data

    def convertDatatypes(self):
        columns = list(self.data)
        for col in columns:
            if col not in self.category_list and col != 'target':  
                self.data[col] = self.data[col].astype('float64')
        return self.data

    def discretizeData(self):
        if self.discretize_data:
            for col_name in self.col_names:
                if col_name not in self.category_list:
                    self.data[col_name] = pd.qcut(self.data[col_name], q=self.quantization_number)
                    print('Quantized: ', col_name)
        return self.data

    def standardizeData(self):
        if self.mode != 'classification':
            col_names = list(self.data)
            if self.standardize_data:
                for col_name in col_names:
                    if col_name != 'target':
                        column_mean = np.mean(self.data[col_name])
                        column_std = np.mean(self.data[col_name])
                        for i in range(len(self.data[col_name])):
                            z = (self.data[col_name][i] - column_mean) / column_std
                            self.data[col_name][i] = z
        return self.data