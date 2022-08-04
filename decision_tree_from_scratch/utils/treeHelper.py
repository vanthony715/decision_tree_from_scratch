# -*- coding: utf-8 -*-
"""
@author: Anthony J. Vasquez
email: avasque1@jh.edu
phone: 315-920-8877

Class: Introduction to Machine Learning

Description: This module uses the CART algorithm to build regression trees
"""

import pandas as pd

class TreeHelper:
    '''
    CART Method Implementation
    '''
    def __init__(self, data):
        self.data = data ##train data
        
    def getUniqueBranches(self, col):
        
        '''Returns unique values given the data and the col'''
        
        return self.data[col].unique()

    def partData(self, column, branch):
        
        '''Takes data, a column, and a value and Returns a partitioned dataframe'''
        
        return self.data[self.data[column] == branch]
        
    def getColCount(self, partitioned_data):
        
        '''Takes partitioned data and returns col count for unique data'''
        
        return partitioned_data.count()[0]
    
    def getUniqueTargs(self, partitioned_data):
        
        '''Take partitioned data and returns a list of unique target values'''
        
        return list(partitioned_data['target'].unique())
    
    def getUniqueTargCount(self, partitioned_data, targ_unique_val):
        
        ''' Takes partitioned data and returns dictionary of target value counts'''
        
        return partitioned_data[partitioned_data.target == targ_unique_val].count()
    
    def calcMean(self, partitioned_data):
        
        '''calculates the mean value of target values in a partitioned dataset'''
        
        return partitioned_data['target'].mean()
    
    def calcSquaredError(self, target_value, partition_mean):
        
        '''Calculate the squared error given an observation and the partition mean'''
        
        return (target_value - partition_mean)**2
    
    
    def calcWeightedMSE(self, partition_len, dataset_len, MSE):
        
        '''Calculates the weighted mse given the partition and dataset lengths caclulated mse'''
        
        return (partition_len / dataset_len)*MSE
    
    def getMinMSE(self, mse_dict):
        
        '''Gets the minimum column mse score and returns the associated column'''
        
        min_mse = min(mse_dict['weighted_mse'])
        min_mse_idx = mse_dict['weighted_mse'].index(min_mse)
        return mse_dict['col'][min_mse_idx]
    
    def getLeaves(self, col_name, leaf_df):
        
        '''Given col_name, and leaf_df, returns leaf decision df'''
        
        ##seperate only the minimum mse column values
        leaf_df = leaf_df[leaf_df['parent'] == col_name]
        ##get unique branch values
        unique_vals = leaf_df['branch_name'].unique()
        leaf_list = []
        ##iterate through unique branch values and take the mean to be the decision
        for unique_val in unique_vals:
            df = leaf_df[leaf_df['branch_name'] == unique_val]
            targ_mean = round(df['decision'].mean(), 3)
            leaf_dict = {'parent': [df.iloc[0]['parent']], 'branch_name': [unique_val], 'decision': [targ_mean]}
            leaf = pd.DataFrame(leaf_dict)
            leaf_list.append(leaf)
        leaves = pd.concat(leaf_list)
        leaves.reset_index(inplace = True, drop = True)
        return leaves
    
    def printTree(self, node):
        try:
            parents = node.leaf_dict['parent']
            branches = node.leaf_dict['branch_name']
            leaves = node.leaf_dict['decision']
            print('\n-------------------- T r e e --------------------')
            print('-------------------------------------------------')
            for idx, leaf in enumerate(leaves):
                print('  parent:  ', parents[idx], 'branch:  ', branches[idx], '  leaf:  ', leaf)
        except:
            pass
        
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    # def calcRatioSquared(self, targ_cnt, column_cnt):
        
    #     '''Caclculate Gini Ratio given the length of a column and a target cnt'''
        
    #     return (targ_cnt / column_cnt)**2
    
    # def calcWeightedGini(self, partition_len, dataset_len, gini):
        
    #     '''Calculates the weighted gini given the partition and dataset lengths with caclulated gini'''
        
    #     return (partition_len / dataset_len)*gini
    
    # def getMinGini(self, gini_dict):
        
    #     '''Gets the minimum column gini score and returns the associated column'''
        
    #     min_gini = min(gini_dict['weighted_gini'])
    #     min_gini_idx = gini_dict['weighted_gini'].index(min_gini)
    #     return gini_dict['col'][min_gini_idx]
        
        
        
            