import matplotlib
import matplotlib.pyplot as plt
import random;
import scipy 
import scipy.io as sio
import pandas as pd
import numpy as np
from sklearn.cross_validation import train_test_split
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_fscore_support
from sklearn import tree
from sklearn.neighbors.nearest_centroid import NearestCentroid
from sklearn.linear_model import SGDClassifier
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.linear_model import RandomizedLogisticRegression
from pandas import ExcelWriter
from sklearn.linear_model import RandomizedLasso
from sklearn import linear_model
from sklearn.feature_selection import f_regression
from sklearn.metrics import matthews_corrcoef
from scipy.stats.stats import pearsonr

class MLforML():

    def __init__(self, test_size = 0.30, cross_val = False, N_population = 1):
        self.test_size = test_size
        self.cross_val = cross_val
        self.N_population = N_population

    def get_indiv(self, data):
        gene = []
        gene.append(randint(0,4))
        gene.append(randint(0,4))
        gene.append(randint(0,4))
        return gene

    # apply Regression
    def build_models(predictors, responses, modelNo):
        if(modelNo==0):
            # Linear Regression
            model = linear_model.LinearRegression();
            modelName = "Linear Regression";
        if(modelNo==1):
            # Ridge Regression
            model = linear_model.RidgeCV(alphas = (0.1,0.1,10));
            modelName = "Ridge Regression";
        if(modelNo==2):
            # lasso Regression
            model = linear_model.MultiTaskLassoCV(eps=0.001, n_alphas=100, alphas=(0.1,0.1,10));
            modelName = "Lasso Regression";

        model.fit(predictors, responses);
        predictions = model.predict(predictors);
        Result = {};
        Result['modelName'] = modelName;
        Result['predictions'] = predictions;
        Result['model'] = model;
        Result['Corr'] = pearsonr(predictions,responses)[0][0];
        return Result;

    def feature_selection_tech(predictors, responses, test_predictors, selectFeatTech):
        if(selectFeatTech==0):
            t=int(predictors.shape[1]*0.40);
            t=40;
            model = SelectKBest(chi2, k=t).fit(predictors, responses);
            predictors_new = model.transform(predictors);
            predictors_test_new = model.transform(test_predictors);
            indices = model.get_support(indices=True);
        if(selectFeatTech==1):
            randomized_logistic = RandomizedLogisticRegression();
            model = randomized_logistic.fit(predictors, responses);
            predictors_new = model.transform(predictors);
            predictors_test_new = model.transform(test_predictors);
            indices = model.get_support(indices=True);
        return predictors_new, predictors_test_new, indices;

    def feature_selection_regression(predictors, responses, test_predictors, selectFeatTech):
        if(selectFeatTech==0):        
            chk = int(predictors.shape[1]*0.40);
            # have fixed the value of how many features are to be selected as of now.
            model = SelectKBest(f_regression, k=25);
            model = model.fit(predictors, responses[0]);
            predictors_new = model.transform(predictors);
            predictors_test_new = model.transform(test_predictors);
            indices = model.get_support(indices=True);
            print "SelectKBest -> "+str(len(indices));
        if(selectFeatTech==1):
            model = RandomizedLasso(alpha='aic', scaling=0.3, sample_fraction=0.60, n_resampling=200, selection_threshold=0.15);
            model = model.fit(predictors, responses[0]);
            predictors_new = model.transform(predictors);
            predictors_test_new = model.transform(test_predictors);
            indices = model.get_support(indices=True);
            print "Randomized Lasso -> "+str(len(indices));
        return predictors_new, predictors_test_new, indices;


    if __name__ == "__main__":
    	predictors = pd.read_csv('predictors.csv');
    	responses = pd.read_csv('responses.csv');

    	Result = []; 
    	for i in range(0,3):
    	    temp = build_models(predictors, responses, i);
    	    Result.append([temp['modelName'],temp['Corr']]);
    	Result = pd.DataFrame(Result);
    	print Result;