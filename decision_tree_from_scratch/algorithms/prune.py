# -*- coding: utf-8 -*-
"""
@author: Anthony J. Vasquez
email: avasque1@jh.edu
phone: 315-920-8877

Class: Introduction to Machine Learning

Description: Reduced Error Pruning for Classification
"""
import numpy as np
class ReducedErrorPrune:
    '''
    Method takes a copy of the node dictionary and prunes from the bottom up
    The output is a modified node dictionary to compare to the original one
    '''
    def __init__(self, nodes, data, prune_depth):
        self.nodes = nodes ##copy of node_dict
        self.prune_depth = prune_depth ##how far up to prune
                    
    def prune(self):
        ##start from this value and work
        prune_depth = -1 * np.arange(2, self.prune_depth + 2)
        for depth in prune_depth:
            try:
                print('Pruning Subtree: ', self.nodes['node'][depth].leaf_dict)
                del self.nodes['node'][depth]
                del self.nodes['node_name'][depth]
            except:
                pass
        return self.nodes