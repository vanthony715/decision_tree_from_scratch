#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Anthony J. Vasquez
email: avasque1@jh.edu
phone: 315-920-8877

Class: Introduction to Machine Learning

Description: Data Loaders
"""
import pandas as pd

class LoadCsvData:
    '''
    Loads standard data from a csv file
    '''
    def __init__(self, data_path, names_path, dataset_name):
        print('LoadCsvData Initialized')
        self.data_path = data_path ##path to data folder
        self.names_path = names_path ##path to names file
        self.dataset_name = dataset_name ##name of the dataset
    
    ##loads names from text file
    def loadNamesFromText(self):
        self.col_names = list(pd.read_csv(self.names_path)) ##assumes first line contains columns
        return self.col_names
    
    ##load the data
    def loadData(self):
        self.data = pd.read_csv(self.data_path, names=self.col_names)
        return self.data
    

