# -*- coding: utf-8 -*-
"""
@author: Anthony J. Vasquez
email: avasque1@jh.edu
phone: 315-920-8877

Class:Introduction to Machine Learning

Description: This Method Creates a Holdout Test set Prior to the K-fold split
"""

class HoldOut:
    '''
    Creates a holdout set for tree pruning
    '''
    def __init__(self, data, holdout_percent):
        self.data = data
        self.holdout_percent = holdout_percent 
    
    def holdout(self):
        num_samples = int(len(self.data['target']) * self.holdout_percent)
        holdout = self.data.sample(n = num_samples)
        ##remove sampled from data
        indices = [i for i in holdout.index]
        self.data = self.data.drop(index = indices)
        ##reset indices
        self.data.reset_index(drop = True)
        holdout = holdout.reset_index(drop = True)
        return self.data, holdout
    