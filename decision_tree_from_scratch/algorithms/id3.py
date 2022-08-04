# -*- coding: utf-8 -*-
"""
@author: Anthony J. Vasquez
email: avasque1@jh.edu
phone: 315-920-8877

Class: Introduction to Machine Learning

Description: ID3 Module
"""

import math

class ID3:
    
    '''ID3 Module'''
        
    def __init__(self, data):
        self.data = data ##input data
     
    ##get unique attrib values
    def getCnts(self):
        ##get dominate attributes unique values
        self.targ_unique_vals = self.data['target'].unique()
        
        ##get dominate attributes unique count
        self.targ_cnt = self.data['target'].count()
        
    def calcDatasetEntropy(self):
        ##calculate dominate attribute's enrtopy
        e_cnt = 0
        for targ_unique_val in self.targ_unique_vals:
            
            ##get the count of each value in the dominate attribute values
            dom_attrib_unique_cnt = self.data[self.data['target'] == targ_unique_val].count()[0]
            
            if self.targ_cnt != 0 and dom_attrib_unique_cnt != 0 and dom_attrib_unique_cnt != self.targ_cnt:
                e = -1 * (dom_attrib_unique_cnt / self.targ_cnt) * math.log(dom_attrib_unique_cnt / self.targ_cnt, 10)
            else:
                e = 0
            
            e_cnt = e_cnt + e
        
        self.targ_entropy = e_cnt
    
    ##gets the target values at an attribute's unique value
    def id3(self):
        ##get col namees
        cols = list(self.data)
        
        ##keep track of cols and associated gain
        gain_dict = {'gain': [], 'col': []}
        
        ##leaf_dict takes leaf candidates and the ones with parent of highest gain are used
        leaf_dict = {'parent': [], 'branch_name': [], 'decision': []}
        
        ##iterate through columns
        for col in cols:
            
            if col != 'target':
                
                ##unique values at attrib
                unique_col_vals = self.data[col].unique()
                
                ##iterate over unique vals
                avg_info = 0
                for unique_col_val in unique_col_vals:
                    
                    #iterate through dominate attributes unique vals
                    e_cnt = 0
                    for targ_unique_val in self.targ_unique_vals:
                
                        ##get dom attrib values at unique col values
                        df = self.data[(self.data[col] == unique_col_val) & (self.data['target'] == targ_unique_val)]
                        # print('df: ', df)
                        
                        ##get the dom attrib count at for each unique col val
                        targ_cnt_at_unique_value = df.count()[0]
                        
                        ##get unique value count for the current column
                        unique_val_cnt = self.data[self.data[col] == unique_col_val].count()[0]
                        
                        ##check that none of the values in the denominator or numerator are not == 0
                        if unique_val_cnt != 0 and targ_cnt_at_unique_value != 0 and unique_val_cnt != targ_cnt_at_unique_value:
                            ##calculate entropies 
                            e = -1*(targ_cnt_at_unique_value / unique_val_cnt * math.log(targ_cnt_at_unique_value / unique_val_cnt, 10))
                        else:
                            e = 0
                        
                        if targ_cnt_at_unique_value == unique_val_cnt and (targ_cnt_at_unique_value / unique_val_cnt) != 0:
                            leaf_dict['parent'].append(col)
                            leaf_dict['branch_name'].append(unique_col_val)
                            leaf_dict['decision'].append(targ_unique_val)
                        
                        ##sum the entropies
                        e_cnt = e_cnt + e

                        ##calculate avg information gain
                        info = (unique_val_cnt / self.targ_cnt) * e_cnt
                    
                    avg_info = avg_info + info
                    
                    # if e_cnt != 0 and avg_info != 0:
                        # print('Entropy: ', round(e_cnt, 4))
                        # print('Average Info: ', round(avg_info, 4))
                
                gain = self.targ_entropy - avg_info
                
                # if gain != 0:    
                    # print('Gain: ', round(gain, 4))
                
                ##only append to gain dict if gain is not zero
                gain_dict['col'].append(col)
                gain_dict['gain'].append(gain)
                
        max_gain = max(gain_dict['gain'])
        max_gain_idx = gain_dict['gain'].index(max_gain)
        max_gain_col = gain_dict['col'][max_gain_idx]
        
        # if max_gain != 0:
        #     print('max_gain_col: ', max_gain_col)
        #     print('max_gain: ', round(max_gain, 4))
        
        ##remove leaf_dict values that are not associated with max_gain_col
        for idx, column in enumerate(leaf_dict['parent']):
            if column != max_gain_col or max_gain == 0:
                del leaf_dict['parent'][idx]
                del leaf_dict['branch_name'][idx]
                del leaf_dict['decision'][idx]
            
        return max_gain, max_gain_col, leaf_dict