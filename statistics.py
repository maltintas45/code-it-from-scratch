#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 27 22:20:53 2018

@author: heysem

This file contains statistical calculations such as mean, standart derivation, variance, covariance etc.

input: the dataset whose columns are features and rows are instances as numpy ndarray
output: the relevant result

Refferences:
    Maximum Likelihood Estimator for Variance is Biased: Proof ,Dawen Liang, Carnegie Mellon University, dawenl@andrew.cmu.edu
"""
import numpy as np
from sklearn.datasets import load_iris


def mean(dataset):
    return np.sum(dataset,  axis=0)/float(dataset.shape[0])

def variance(dataset,biased=False):
    #biased : int, optional “Delta Degrees of Freedom”: the divisor used in the calculation is N - biased, where N represents the number of elements. By default biased is zero.
    return np.sum((dataset-mean(dataset))**2,axis=0)/float(dataset.shape[0]-float(biased))

def standard_deviation(dataset,biased=False):
    return np.sqrt(variance(dataset,biased))

def covariance_matrix(dataset,biased=False):
    return np.dot((dataset-mean(dataset)).T,(dataset-mean(dataset)))/float(dataset.shape[0]-float(biased))

def corelation_matrix(dataset,biased=False):
    return covariance_matrix(dataset,biased)/(np.dot(standard_deviation(dataset)[np.newaxis].T,standard_deviation(dataset)[np.newaxis]))


def test_statistics(verbose=True):
    iris_data = load_iris()
    dataset=iris_data.data
    #dataset=np.concatenate(iris_data.data,iris_data.target)
    
    assert all(np.equal(mean(dataset),np.mean(dataset, axis=0)))==True
    assert all(np.equal(variance(dataset),np.var(dataset, axis=0)))==True
    assert all(np.equal(standard_deviation(dataset),np.std(dataset, axis=0)))==True    
    #assert np.testing.assert_array_almost_equal(covariance_matrix(dataset),np.cov(dataset.T))        
    #assert np.testing.assert_array_equal(corelation_matrix(dataset),np.corrcoef(dataset.T))
    
    if(verbose):
        print "our results"
        print "\tmean\n",mean(dataset)
        print "\tvariance\n",variance(dataset,False)
        print "\tstandart derivaiton\n",standard_deviation(dataset)
        print "\tcovariance matrix\n",covariance_matrix(dataset)
        print "\tcorrealtion matrix\n",corelation_matrix(dataset)
        
        print "numpy results"
        print "\tmean\n",np.mean(dataset, axis=0)
        print "\tvariance\n",np.var(dataset,axis=0)
        print "\tstandart derivaiton\n",np.std(dataset,axis=0)
        print "\tcovariance matrix\n",np.cov(dataset.T)        
        print "\tcorrealtion matrix\n",np.corrcoef(dataset.T)
        
#test_statistics()
