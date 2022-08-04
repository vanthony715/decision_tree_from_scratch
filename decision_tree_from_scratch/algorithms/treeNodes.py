# -*- coding: utf-8 -*-
"""
@author: Anthony J. Vasquez
email: avasque1@jh.edu
phone: 315-920-8877

Class: Introduction to Machine Learning

Description: Create, traverse, and delete nodes
"""


    
class Node:
    '''
    Creates a node and stores partitioned data This node is used for classification only
    '''
    def __init__(self, parent, data, col_name):
        self.parent = parent ##self.name of the parent node or none for root node
        self.data = data ##dataframe of pre-partitioned data
        self.col_name = col_name ##max gain column for classification and min mse for regression
        
    def sortData(self):
        ##these are the unique values of the max gain column
        branches = list(self.data[self.col_name].unique())
        self.branch_dict = {'branch_name':[], 'branch_data':[], 'col_name': []}
        for _, branch in enumerate(branches):
            ##partition data according to value
            branch_data = self.data[(self.data[self.col_name] == branch)]
            ##drop the derived column from the data
            self.branch_dict['branch_data'].append(branch_data.drop(self.col_name, axis = 1))
            self.branch_dict['branch_name'].append(branch)
            self.branch_dict['col_name'].append(self.col_name)
        return self.branch_dict ##which is the new data

