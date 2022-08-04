#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Anthony J. Vasquez
email: avasque1@jh.edu
phone: 315-920-8877

Class: Introduction to Machine Learning

Description: Project 3 - Main
"""
##standard python libraries
import os
import sys
import warnings
import argparse
import time
import gc

import pandas as pd
import numpy as np

##preprocess pipeline
from utils.dataLoaders import LoadCsvData
from utils.preprocess import PreprocessData
from utils.splitData import SplitData
from utils.splitDataClassless import SplitDataClassless

##preprocess pipeline
from utils.test import TestID3
from utils.test import TestCART
from utils.test import CountDecisionNodes
from utils.treeHelper import TreeHelper
from utils.holdout import HoldOut

##algorithms
from algorithms.id3 import ID3
from algorithms.treeNodes import Node
from algorithms.prune import ReducedErrorPrune

##turn off all warnings
if not sys.warnoptions:
    warnings.simplefilter("ignore")

gc.collect()

##command line arguments
parser = argparse.ArgumentParser()
parser.add_argument('--data_folder_name', type = str ,default = '/data',
                    help='Name of the folder where the data and names files are located'),

parser.add_argument('--dataset_name', type = str ,default = '/machine',
                    help='Name of the folder where the data and names files are located'),

parser.add_argument('--namespath', type = str , default = 'data/machine',
                    help='Path to dataset names'),

parser.add_argument('--discretize_data', type = bool ,default = False,
                    help='Should dataset be discretized?'),

parser.add_argument('--quantization_number', type = int ,default = 2,
                    help='If discretized, then quantization number'),

parser.add_argument('--standardize_data', type = bool , default = True,
                    help='Should data be standardized?'),

parser.add_argument('--k_folds', type = int , default = 5,
                    help='Number of folds for k-fold validation'),

parser.add_argument('--stratified', type = bool, default = False,
                    help='split the dataset evenly based on classes'),

parser.add_argument('--min_examples', type = int , default = 1,
                    help='Drop classes with less examples then this value'),

parser.add_argument('--remove_orig_cat_col', type = bool , default = True,
                    help='Remove the original categorical columns for data encoding'),

parser.add_argument('--holdout_percent', type = float , default = 0.20,
                    help='Holdout set to prune trees'),

parser.add_argument('--early_stopping_criteria', type = int , default = [2, 4, 6, 8],
                    help='Stop building tree flag'),

parser.add_argument('--prune_depth', type = int , default = 4,
                    help='Bottom up prune depth for classification tree pruning'),

args = parser.parse_args()
## =============================================================================
##                                  MAIN
## =============================================================================
if __name__ == "__main__":
    ##start timer
    tic = time.time()
## =============================================================================
##                              PATHS / ARGUMENTS
## =============================================================================
    ##define paths
    cwd = os.getcwd().replace('\\', '/') ##get current working directory
    data_folder_name = cwd + args.data_folder_name
    datapath = data_folder_name + args.dataset_name + '.data'
    namespath = data_folder_name + args.dataset_name + '.names'
    dataset_name = args.dataset_name
## =============================================================================
##                                  PREPROCESS
## ============================================================================
    
    ##list of categorical data columns per dataset
    car = ['buying', 'maint','lug_boot', 'safety']
    abalone = ['sex']
    forestfires= ['month', 'day']
    practice = ['gender', 'travel', 'income_level']
    
    if dataset_name == '/car':
        category_list = car
    elif dataset_name == '/abalone':
        category_list = abalone
    elif dataset_name == '/forestfires':
        category_list = forestfires
    elif dataset_name == '/practice':
        category_list = practice
    else:
        category_list = []
    
    classification_list = ['car','breast-cancer-wisconsin','house-votes-84', 'practice']
    regression_list = ['abalone', 'forestfires', 'machine', 'iris']

    if args.dataset_name.split('/', 1)[1] in classification_list:
        mode = 'classification'
        algorithm = 'ID3'
        test_dict = {'testset': [], 'dataset': [], 'score': [], 
                 'score_pruned': [], 'decision_node_cnt': [],
                 'decision_node_cnt_pruned': [], 'stop_criteria': []}
        
    elif args.dataset_name.split('/', 1)[1] in regression_list:
        mode = 'regression'
        algorithm = 'CART'
        test_dict = {'testset': [], 'dataset': [], 'score': [], 
                 'decision_node_cnt': [], 'stop_criteria': []}
    else:
        'None'
        
    print('Dataset: ', args.dataset_name.split('/', 1)[1])
    print('Train mode: ', mode)
    
    print('\n******************** ML Pipeline Started ********************')
    ##define tuple of values to drop from dataframe
    values_to_replace = ('na', 'NA', 'nan', 'NaN', 'NAN', '?', ' ')
    # values_to_change = {'place_holder':0}
    values_to_change = {'5more':6, 'more': 5}

    # ##load data
    load_data_obj = LoadCsvData(datapath, namespath, dataset_name)
    names = load_data_obj.loadNamesFromText() ##load names from text
    data = load_data_obj.loadData() ##data to process

    ##preprocess pipeline
    proc_obj = PreprocessData(data, values_to_replace, values_to_change,
                              args.dataset_name, args.discretize_data, 
                              args.quantization_number, args.standardize_data, 
                              args.remove_orig_cat_col, category_list, mode)
    proc_obj.dropRowsBasedOnListValues() ##replaces values from list
    proc_obj.changeValues() ##changes values from values_to_change list
    proc_obj.encodeData() ##encodes data
    proc_obj.createUniqueAttribNames()
    proc_obj.convertDatatypes()
    proc_obj.standardizeData() ##standardizes data
    df_encoded = proc_obj.discretizeData() ##discretizes dat
    
    if mode == 'regression':
        encoded_targs = {'target': []}
        for i in range(len(data['target'])):
            mean = np.mean(df_encoded.target)
            std = np.std(df_encoded.target)
            z = (df_encoded.target[i] - mean) / std
            encoded_targs['target'].append(z)
        df_encoded = df_encoded.drop('target', axis = 1)
        encoded_targs = pd.DataFrame(encoded_targs)
        df_encoded['target'] = encoded_targs
        
        ##round the data before split
        df_encoded = round(df_encoded, 3)
            
    ##holdout data for pruning
    hold_obj = HoldOut(df_encoded, args.holdout_percent)
    df_encoded, holdout = hold_obj.holdout()

    if args.stratified and mode == 'classification':
        split_obj = SplitData(df_encoded, args.k_folds, args.min_examples)
        split_obj.removeSparseClasses() ##removes classes that do not meet the min_examples criteria
        split_obj.countDataClasses() ##counts data classes
        split_obj.splitPipeline() ##start of the stratefied k-fold validation split
        train_test_sets = split_obj.createTrainSets() ##k train and test sets returned as a dictionary
        train_columns = split_obj.getTrainColumns() ##gets all columns but target
    else:
        split_obj = SplitDataClassless(df_encoded, args.k_folds, args.min_examples)
        split_obj.splitPipeline() ##start of the stratefied k-fold validation split
        train_test_sets = split_obj.createTrainSets() ##k train and test sets returned as a dictionary
        train_columns = split_obj.getTrainColumns() ##gets all columns but target
   
# ## =============================================================================
# ##                                  TRAIN
# ## =============================================================================
    ##no need to early stop for classification
    if len(args.early_stopping_criteria) and mode == 'classification':
        early_stopping_criteria = [0]
    else:
        early_stopping_criteria = args.early_stopping_criteria
    
    #pruning for classification
    for depth in range(1, args.prune_depth):
        print('Prune Depth: ', depth)
        
        for stop_criteria in args.early_stopping_criteria:
            
            for k in range(len(train_test_sets['train_set'])):
            # for k in range(0, 2):
                ##get train data and labels
                X_train = train_test_sets['train_set'][k].loc[:, train_columns]
                y_train = train_test_sets['train_set'][k]['target']
                ##get test data and labels
                X_test = train_test_sets['test_set'][k].loc[:, train_columns]
                y_test = train_test_sets['test_set'][k]['target']
                ##indicate the train test iteration
                print('\nTrain/Test Set: ', k)
                ##instantiate naive classifier model
                X_train['target'] = y_train ##combine labels and data for knn
                X_test['target'] = y_test
                ##keep track of each attributes unique values
                node_dict = {'node_name': [], 'node': []} 
                
                if algorithm == 'ID3':
                    flag = len(list(X_train)) - 1
                    
                    while flag:
                        ##name of node
                        if len(node_dict['node']) == 0:
                            parent_name = 'None'
                            data_list = [X_train]
                        else:
                            parent_name = node_dict['node_name'][-1]
                            data_list = node_dict['node'][-1].branch_dict['branch_data']
                        
                        for idx, d in enumerate(data_list):
                            ##kickoff id3 algorithm
                            id3_obj = ID3(d)
                            ##get the unique counts for all values in the dominate attribute
                            id3_obj.getCnts()
                            ##calculate the dataset entropy
                            id3_obj.calcDatasetEntropy()
                            
                            ##walk through id3 algorithm once
                            max_gain, max_gain_col, leaf_dict = id3_obj.id3()
                            if max_gain != 0:
                                ##create node_object
                                node_obj = Node(parent_name, X_train, max_gain_col)
                                ##sort the child data in the node
                                node_obj.sortData()
                                ##append leaves to node
                                node_obj.leaf_dict = leaf_dict
                                ##append node to dictionary
                                node_dict['node_name'].append(max_gain_col)
                                node_dict['node'].append(node_obj)
                                ##instance of tree object
                                # tree_obj = TreeHelper(d)
                                # #print the tree
                                # tree_obj.printTree(node_obj)
                    
                        flag -= 1
                    ##test Tree algorithm
                    test_obj = TestID3(node_dict, X_test)
                    ##test the ID3 tree
                    accuracy = test_obj.test()
                    print('Accuracy: ', accuracy)
                    
                    ##get the decision node count
                    cnt_obj = CountDecisionNodes(node_dict)
                    dec_cnt = cnt_obj.countNodes()
                    print('Node Count Before Pruning: ', dec_cnt)
                    
                    ##instantiate prune object
                    prune_obj = ReducedErrorPrune(node_dict, holdout, depth)
                    pruned_node_dict = prune_obj.prune()
                    
                    # ##test Tree algorithm
                    # test_obj = TestID3(pruned_node_dict, X_test)
                    # ##test the ID3 tree
                    # pruned_accuracy = test_obj.test()
                    # print('Pruned Accuracy: ', pruned_accuracy)
                    
                    # ##get the decision node count
                    cnt_obj = CountDecisionNodes(pruned_node_dict)
                    dec_cnt_pruned = cnt_obj.countNodes()
                    print('Node Count After Pruning: ', dec_cnt_pruned)
                    
                    # ##instance of tree object
                    # print('\nTree after pruning: ')
                    # tree_obj = TreeHelper(d)
                    # #print the tree
                    # tree_obj.printTree(pruned_node_dict)
                    
                    # test_dict['dataset'].append(args.dataset_name.split('/', 1)[1])
                    # test_dict['score'].append(accuracy)
                    # test_dict['score_pruned'].append(pruned_accuracy)
                    # test_dict['decision_node_cnt'].append(dec_cnt)
                    # test_dict['decision_node_cnt_pruned'].append(dec_cnt_pruned)
                    # test_dict['stop_criteria'].append(depth)
                    # test_dict['testset'].append(k)
                    
                else:
                    ##keep track of each attributes unique values
                    node_dict = {'node_name': [], 'node': []} 
                    flag = len(list(X_train)) - 1
                    
                    while flag:
                        ##keep track of leaf dataframes
                        leaf_list = []
                        ##name of node
                        if len(node_dict['node']) == 0:
                            parent_name = 'None'
                            data_list = [X_train]
                        else:
                            parent_name = node_dict['node_name'][-1]
                            data_list = node_dict['node'][-1].branch_dict['branch_data']
                    
                        col_mse = {'col': [], 'weighted_mse': []}
                        for idx, d in enumerate(data_list):
                            ##keep track of leav
                            leaf_dict = {'parent': [], 'branch_name': [], 'decision': [], 'square_error': []}
                            ##keep track of mse calculation 
                            weighted_mse_list = []
                            
                            for col in list(d):
                                if col != 'target':
                                    ##get unique values in each col
                                    ##instance of tree object
                                    tree_obj = TreeHelper(d)
                                    branches = tree_obj.getUniqueBranches(col)
                                    already_logged = []
                                    
                                    ##partition the data
                                    branch_cnt = 0
                                    for _, branch in enumerate(branches):
                                        # print('\nbranch: ', branch)
                                        partition = tree_obj.partData(col, branch)
                                        ##count the partitioned column
                                        partition_cnt = tree_obj.getColCount(partition)
                                        ##get the target values for this partition
                                        unique_targs = tree_obj.getUniqueTargs(partition)
                                        ##count each of the unique targs in partition
                                        squared_errors = []
                                        
                                        if args.early_stopping_criteria:
                                            branch_cnt += 1
                                            if branch_cnt <= stop_criteria:
                                                pass
                                            else:
                                                break
                    
                                        for _, unique_targ in enumerate(unique_targs):
                                            targ_cnt = tree_obj.getUniqueTargCount(partition, unique_targ)[0]
                                            ##calculate the ratios squared and append to dictionary
                                            mean = tree_obj.calcMean(partition)
                                            ##calculate Squared Error
                                            square_error = tree_obj.calcSquaredError(unique_targ, mean)
                                            ##append squared error for MSE calculation
                                            squared_errors.append(square_error)
                                            
                                            if square_error == 0:
                                                leaf_dict['parent'].append(col)
                                                leaf_dict['branch_name'].append(branch)
                                                leaf_dict['decision'].append(unique_targ)
                                                leaf_dict['square_error'].append(square_error)
                                        
                                        ##dictionary of dataframe leaves
                                        leaf_df = pd.DataFrame(leaf_dict)
                                        leaf_list.append(leaf_df)
                                        
                                        ##calculate MSE
                                        mse = (1 / len(partition['target'])) * sum(squared_errors)
                                        ##calculate weighted gini
                                        weighted_mse = tree_obj.calcWeightedMSE(len(partition['target']), 
                                                                          len(X_train['target']), mse)
                                        ##append values to dictionary
                                        weighted_mse_list.append(weighted_mse)
                                    
                                    ##append to dictionary to calculate the minimum gini
                                    col_mse['col'].append(col)
                                    col_mse['weighted_mse'].append(sum(weighted_mse_list))
                                    
                                    ##get the column with the lowest mse
                                    min_mse_col = tree_obj.getMinMSE(col_mse)
                                    
                            ##get the column with the lowest mse
                            min_mse_col = tree_obj.getMinMSE(col_mse)
                            # print('min_mse_col: ', min_mse_col)
                            ##create a node for the min mse column
                            node_obj =  Node(parent_name, d, min_mse_col)
                            ##get branch_dict
                            node_obj.sortData()
                        
                        try:
                            ##add leaf data to node_obj
                            all_leaves = pd.concat(leaf_list)
                            ##get only the leaves associated with the min_mse_col
                            leaves = tree_obj.getLeaves(min_mse_col, all_leaves)
                            leaves = leaves.reset_index()
                            # ##attach leaves to node_object
                            node_obj.leaf_dict = leaves.to_dict('list')
                            ##print the tree
                            # tree_obj.printTree(node_obj)
                            
                        except:
                            print('No Leaf Nodes in: ', min_mse_col)
                        
                        ##append node to dictionary
                        node_dict['node_name'].append(min_mse_col)
                        node_dict['node'].append(node_obj)
                        ##set all leaves and leaves to zero
                        leaves = 0
                        all_leaves = 0
                        flag = flag - 1
                    
                    ##test Tree algorithm
                    test_obj = TestCART(node_dict, X_test)
                    test_obj.test()
    
                    ##get the decision node count
                    cnt_obj = CountDecisionNodes(node_dict)
                    dec_cnt = cnt_obj.countNodes()
                    
                    ##append to dict
                    test_dict['dataset'].append(args.dataset_name.split('/', 1)[1])
                    test_dict['score'].append(test_obj.calculateMSE())
                    test_dict['decision_node_cnt'].append(dec_cnt)
                    test_dict['stop_criteria'].append(stop_criteria)
                    test_dict['testset'].append(k)
                
    test_df = pd.DataFrame(test_dict)
    test_df.to_csv(args.dataset_name.split('/', 1)[1] + '_results.csv', mode='a')
    toc = time.time()
    tf = round((toc - tic), 2)
    print('Total Time: ', tf)