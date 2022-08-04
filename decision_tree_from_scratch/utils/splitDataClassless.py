# -*- coding: utf-8 -*-
"""
@author: Anthony J. Vasquez
email: avasque1@jh.edu
phone: 315-920-8877

Class: Introduction to Machine Learning

Description: Functions to split data into k-fold train/test sets
"""
import random
import math
import pandas as pd

class SplitDataClassless:
    '''
    splits classless data into k-folds for cross validation
    '''
    def __init__(self, data, k_folds, min_examples):
        print('Split Data Initialized')
        self.data = data
        self.k_folds = k_folds

    def splitPipeline(self):
        self.columns = list(self.data) ##dataset columns
        self.df_dict = self.data.to_dict(orient = 'list') ##convert to dictionary for ease of use
        self.split_df_list = [] ##list of k-fold dataframes
        self.used_id_list = []

        ##calc number of samples per fold and per class rounded down
        self.num_samples_per_fold = math.trunc((len(self.data['target']) / (self.k_folds)))

        ##add unique id to current df_dict
        def uniqueID():
            self.df_dict['id'] = []
            for i in range(len(self.df_dict[self.columns[0]])):
                self.df_dict['id'].append(i)
            return self.df_dict
        uniqueID() ##execute inplace

        ##create a new k-fold dict with an id column for bookkeeping
        def clearDict():
            self.split_dict = {} ##create dictionary for holding randomly sampled data
            for column in self.columns:
                self.split_dict[column] = []

        ##module randomly samples dataset dictionary
        def randomSample():
            ##flag to stop fold sampling
            flag = self.num_samples_per_fold
            while flag:
                success = 0
                sample_id = random.sample(self.df_dict['id'], 1)
                sample_id_index = self.df_dict['id'].index(sample_id[0])
                try:
                    for column in self.columns:
                        self.split_dict[column].append(self.df_dict[column][sample_id_index]) ##apend to split dictionary
                        self.df_dict[column].pop(sample_id_index) ##pop from df_dict
                        success = 1
                except:
                    pass
                if success:
                    flag -= 1
            split_df = pd.DataFrame(self.split_dict)
            self.split_df_list.append(split_df)
            # return self.split_df_list

        ##creates k dictionaries for each fold
        for k in range(self.k_folds):
            clearDict() ## start with cleared dictionary
            randomSample()## sample from data
        # return self.split_df_list

    def createTrainSets(self):
        k_used_list = [] #keep track of the random indices
        k_train_test_sets = {'train_set':[], 'test_set': []}

        flag = len(self.split_df_list) ##each fold needs to be a test set once
        while flag:
            train_df = pd.DataFrame() #empty dataframe to concat folds to create train set
            random_index = random.sample(range(len(self.split_df_list)), 1)
            temp_list = self.split_df_list.copy()
            if random_index[0] not in k_used_list:
                temp_list.pop(random_index[0])
                k_used_list.append(random_index[0])
                test_set = self.split_df_list[random_index[0]] ##randomly selected test set
                train_df = pd.concat(temp_list, ignore_index=True) ##new train set
                k_train_test_sets['train_set'].append(train_df) ## append train set to dictionary
                k_train_test_sets['test_set'].append(test_set) ## append test set to dictionary
                del train_df ##clear out train_df for qa
                flag -= 1
        return k_train_test_sets

    ##module gets the trainable columns if not otherwise specified
    def getTrainColumns(self):
        train_column_list = []
        for column in self.columns:
            if column != 'target':
                train_column_list.append(column)
        return train_column_list


