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
    def __init__(self, test_size = 0.30, cross_val = False, N_population):
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


if __name__ == "__main__":
	predictors = pd.read_csv('predictors.csv');
	responses = pd.read_csv('responses.csv');

	Result = []; 
	for i in range(0,3):
	    temp = build_models(predictors, responses, i);
	    Result.append([temp['modelName'],temp['Corr']]);
	Result = pd.DataFrame(Result);
	print Result;