#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 28 19:34:05 2018

@author: heysem

Naive Bayes Classifier
"""
import numpy as np
import sys
import statistics as st

class NaiveBayes:
    def __init__(self):
        self._continues_columns=None
        self._categorical_columns=None
        self._prior_probabilities=None
        self._likelihood_probabilities=None
        self._likelihood_probability_arguments=None
        self.epsilon=sys.float_info.epsilon
        
    def normal_distribution_probability(self,x,mean,standard_deviation):
        return np.exp(-((x-mean)**2/(self.epsilon+2*standard_deviation**2).astype(float)))/(self.epsilon+standard_deviation*np.sqrt(2*np.pi)).astype(float)

    def fit(self,train_data_features,train_data_labels):
        assert np.ndarray==type(train_data_features)
        assert np.ndarray==type(train_data_labels)
        assert train_data_features.shape[0]==len(train_data_labels)
        #labels will be used to calculate Prior P(X) and Likelihood P(X|C) probabilies
        #lets find the indices of labels
        """ ! we assume that label vector contains categorical values"""
        instances_by_label={}
        for label in set(train_data_labels):  
            instances_by_label[label]= np.where(train_data_labels==label)[0]
        #lets find Prior probability
        prior_probabilities={} # a dicitinary (key: label, value: prior)
        for label in set(train_data_labels):
            prior_probabilities[label]=len(instances_by_label[label])/float(len(train_data_labels))
        #features will be used to calculate Likelihood P(X|C) probabilities 
        categorical_columns=[]
        continuous_columns=[]
        for i in range(len(train_data_features[0])) :
            try:
                _=[float(feature_value) for feature_value in set(train_data_features[:,i])]
                continuous_columns.append(i)
                #if it continue the feature column has continues values
                # so we deal with a distribution
            except:
                #if it drops here the feature column has categorical values
                categorical_columns.append(i)
        #continuous features
        cont_mat=train_data_features[:,continuous_columns].astype(float)
        likelihood_probability_arguments={}# a dicitinary (key: label, value: [mean,standard deviation])
        for label in set(train_data_labels):
            likelihood_probability_arguments[label]=[st.mean(cont_mat[instances_by_label[label],:]),st.standard_deviation(cont_mat[instances_by_label[label],:])]       
        #categorical features        
        cate_mat=train_data_features[:,categorical_columns].astype(str)
        likelihood_probabilities={}# a dicitinary (key: label, value: [mean,standard deviation])
        for label in set(train_data_labels):
            sub_train_data_features=train_data_features[instances_by_label[label],:]
            for i in range(cate_mat.shape[1]):
                feature_column=sub_train_data_features[:,i]
                for feature_value in set(feature_column):
                    likelihood_tag=feature_value+"|"+label
                    likelihood_probabilities[likelihood_tag]=len(np.where(feature_column==feature_value)[0])/float(feature_column)       
        #prepare package that will return trainining 
        self._continues_columns=continuous_columns
        self._categorical_columns=categorical_columns
        self._prior_probabilities=prior_probabilities
        self._likelihood_probabilities=likelihood_probabilities
        self._likelihood_probability_arguments=likelihood_probability_arguments
        package=[]
        package.append(continuous_columns)
        package.append(categorical_columns)
        package.append(prior_probabilities)
        package.append(likelihood_probabilities)
        package.append(likelihood_probability_arguments)
        return package
    
    def predict(self,test_data_features):
        #lets find posterior probability P(C|X)=P(C)*P(X|C)/P(X)
        posterior_probabilities={}        
        for label in self._prior_probabilities.keys():
            #continuous features
            sub_test_data_features=test_data_features[:,self._continues_columns]
            multiplication_of_likelihoods_cont=np.prod(self.normal_distribution_probability(sub_test_data_features,self._likelihood_probability_arguments[label][0],self._likelihood_probability_arguments[label][1]),axis=1)
            #categorical features
            sub_test_data_features=test_data_features[:,self._categorical_columns]
            array_of_multiplication_of_likelihoods_cate=np.ones(test_data_features.shape[0])
            for i in range(sub_test_data_features.shape[1]):
                multiplication_of_likelihoods_cate=1
                for feature_value in sub_test_data_features[:,i]:
                    likelihood_tag=feature_value+"|"+label
                    multiplication_of_likelihoods_cate*=(self._likelihood_probabilities[likelihood_tag]+self.epsilon)
                array_of_multiplication_of_likelihoods_cate.append(array_of_multiplication_of_likelihoods_cate)
            #array_of_multiplication_of_likelihoods_cate=np.array(array_of_multiplication_of_likelihoods_cate)[np.newaxis].T
            posterior_probabilities[label]=self._prior_probabilities[label]*multiplication_of_likelihoods_cont*array_of_multiplication_of_likelihoods_cate*array_of_multiplication_of_likelihoods_cate
        #find max P(C)*P(X|C) for all instance
        y_pred=[]
        for j in range(test_data_features.shape[0]):
            y_pred.append(max(posterior_probabilities, key=lambda i: posterior_probabilities[i][j]))
        return y_pred



def test_naive_bayes():
    from sklearn import datasets
    iris = datasets.load_iris()
    from sklearn.naive_bayes import GaussianNB
    gnb = GaussianNB()
    y_pred = gnb.fit(iris.data, iris.target).predict(iris.data)
    print("Scikit-learn GaussianNB\t: Number of mislabeled points out of a total %d points : %d"
          % (iris.data.shape[0],(iris.target != y_pred).sum()))
    
    nb=NaiveBayes()
    nb.fit(iris.data,iris.target)
    y_pred=nb.predict(iris.data)
    print("Scratch Naive Bayes\t: Number of mislabeled points out of a total %d points : %d"
          % (iris.data.shape[0],iris.data.shape[0]-sum(np.equal(iris.target, y_pred))))
    
    
test_naive_bayes()