# -*- coding: utf-8 -*-
"""
@author: Anthony J. Vasquez
email: avasque1@jh.edu
phone: 315-920-8877

Class: Introduction to Machine Learning

Description:Tests data
"""

from random import sample
import pandas as pd

class TestID3:
    '''
    Tests the ID3 Derived Decision tree
    '''
    def __init__(self, nodes, data):
        self.nodes = nodes ##this is a nodes dict
        self.data = data ##this is the test data
        
    def test(self):
        ##get parents
        parents = self.nodes['node_name']
        ##get columns of test data
        columns = list(self.data)
        ##counts for accuracy
        correct_preds = 0
        
        ##get each row in the test data
        for i in range(len(self.data['target'])):
            row = self.data.iloc[i]
            ##get the truth value for the test data
            truth = row['target']
            
            ##get each attribute from test set
            for column in columns:
                ##see if the columns are in the parent list
                if column in parents:
                    ##find the node location of the parent
                    parent_idx = self.nodes['node_name'].index(column)
                    ##get the unique value of the test data given a row and a column
                    branch = row[column]
                    
                    ##check if the unique test value is in leaves
                    if branch in self.nodes['node'][parent_idx].leaf_dict['branch_name']:
                        ##find where the branch is in the leaf_dict
                        branch_idx = self.nodes['node'][parent_idx].leaf_dict['branch_name'].index(branch)
                        ##find the decision given the branch idx
                        decision = self.nodes['node'][parent_idx].leaf_dict['decision'][branch_idx]
                        
                        ##find whether the prediction is truth
                        if decision == truth:
                            correct_preds += 1
                        ##assuming that the column, branch and decision were found once, then break
                        break
                else:
                    ##take a random sample of possible 
                    try:
                        rand_branch = sample(self.nodes['node'][parent_idx].leaf_dict['branch_name'], 1)
                        rand_decision = sample(rand_branch, 1)[0]

                        if rand_decision == truth:
                            correct_preds += 1
                    except:
                        pass
        return correct_preds / len(self.data['target'])
        
        
class TestCART:
    '''
    Tests the CART Derived Decision Tree
    '''
    def __init__(self, nodes, data):
        self.nodes = nodes ##this is a nodes dict
        self.data = data ##this is the test data
        
    def test(self):
        ##get parents
        parents = self.nodes['node_name']
        ##get columns of test data
        columns = list(self.data)
        ##track predictions
        self.metrics_dict = {'truth': [], 'prediction': []}
        
        ##get each row in the test data
        for i in range(len(self.data['target'])):
            row = self.data.iloc[i]
            ##get the truth value for the test data
            truth = row['target']
            
            ##get each attribute from test set
            for column in columns:
                ##see if the columns are in the parent list
                if column in parents:
                    ##find the node location of the parent
                    parent_idx = self.nodes['node_name'].index(column)
                    ##get the unique value of the test data given a row and a column
                    branch = row[column]
                    
                    ##check if the unique test value is in leaves
                    try:
                        if branch in self.nodes['node'][parent_idx].leaf_dict['branch_name']:
                            ##find where the branch is in the leaf_dict
                            branch_idx = self.nodes['node'][parent_idx].leaf_dict['branch_name'].index(branch)
                            ##find the decision given the branch idx
                            decision = self.nodes['node'][parent_idx].leaf_dict['decision'][branch_idx]
                            ##keep track of predictions
                            self.metrics_dict['truth'].append(truth)
                            self.metrics_dict['prediction'].append(decision)
                            
                            ##for demo purposes
                            print('\n----------Regression Tree Traversal----------')
                            print('-------------------------------------------------')
                            print('Truth: ', truth)
                            print('Parent: ', column)
                            print('Branch: ', branch)
                            print('Prediction at leaf: ', decision)
                            
                            break
                    except:
                        print('No Leaves in this Node: ', column)
                else:
                    ##take make a random prediction if not captured in tree 
                    try:
                        rand_branch = sample(self.nodes['node'][parent_idx].leaf_dict['branch_name'], 1)
                        rand_decision = sample(rand_branch, 1)[0]
                        self.metrics_dict['prediction'].append(rand_decision)
                        self.metrics_dict['truth'].append(truth)
                        break
                    except:
                        pass
    
    def calculateMSE(self):
        '''calculates mse given a dictionary with truth and prediction data'''
        squared_error = 0
        for idx, pred in enumerate(self.metrics_dict['prediction']):
            truth = self.metrics_dict['truth'][idx]
            #sum the squared error
            squared_error = squared_error + (truth - pred) ** 2
        mse = squared_error / len(self.metrics_dict['truth'])
        print('\n******************************************')
        print('\Mean Squared Error: ', mse)
        print('\n******************************************')
        return mse
    
class CountDecisionNodes:
    
    '''counts Nodes'''
    
    def __init__(self, node_dict):
        self.node_dict = node_dict
        
    def countNodes(self):
        decisions = []
        for i in range(len(self.node_dict['node'])):
            try:
                num_dec_nodes = len(self.node_dict['node'][i].leaf_dict['decision'])
                decisions.append(num_dec_nodes)
            except:
                pass
        dec_cnt = sum(decisions)
        return dec_cnt
