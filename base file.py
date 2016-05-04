
# coding: utf-8

# ### Loading all data and libraries

# In[8]:

get_ipython().magic(u'matplotlib inline')
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


# In[7]:

predictors = pd.read_csv('predictors.csv');
responses = pd.read_csv('responses.csv');


# ### Data Stratification | Feature Selection | Data Replication | Model Building | Classification & Regression
# ----

# In[22]:

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


# In[26]:

Result = []; 
for i in range(0,3):
    temp = build_models(predictors, responses, i);
    Result.append([temp['modelName'],temp['Corr']]);
Result = pd.DataFrame(Result);
print Result;


# In[20]:




# In[78]:

def get_data(selectData):
    if(selectData == 0):
        features = feat_matrix[0,0];
        labels = labels_matrix[0,0];
    if(selectData == 2):
        features = feat_matrix[0,2];
        labels = labels_matrix[0,2];
    if(selectData == 4):
        features = feat_matrix[0,4];
        labels = labels_matrix[0,4];
    features = pd.DataFrame(features);
    labels = pd.DataFrame(labels);
    lab = labels[labels[0]!= -1.0]
    feat = features[labels[0]!= -1.0]
    feat = pd.DataFrame(feat.values);
    return lab, feat;

def get_data_whole(selectData):
    features = pd.concat([pd.DataFrame(feat_matrix[0,selectData]),pd.DataFrame(feat_matrix_sty[0,selectData])],axis=1);
    labels = labels_matrix[0,selectData];
    labels = pd.DataFrame(labels);
    lab = labels[labels[0]!= -1.0]
    feat = features[labels[0]!= -1.0]
    feat = pd.DataFrame(feat.values);
    lab = pd.DataFrame(lab.values);
    keys_mat = test['feat_keys'][0][selectData];
    keys_sty = test['feat_keys_sty'][0];
    final_keys = np.append(keys_mat, keys_sty);
    return lab, feat, final_keys;

def apply_Model(temp_data, selectModel):
    data = {};
    data['X_train_ceil'] = temp_data['X_train_ceil'];
    data['X_test_ceil'] = temp_data['X_test_ceil'];
    data['y_train_ceil'] = temp_data['y_train_ceil'];
    data['y_test_ceil'] = temp_data['y_test_ceil'];
    data['ind_ceil'] = temp_data['ind_ceil'] 
    
    # feature selection for the floor
    data['X_train_floor'] = temp_data['X_train_floor'];
    data['X_test_floor'] = temp_data['X_test_floor'];
    data['y_train_floor'] = temp_data['y_train_floor'] ;
    data['y_test_floor'] = temp_data['y_test_floor'];
    data['ind_floor'] = temp_data['ind_floor'];
    
    if(selectModel==1):
        print "OneVsRest";
        classifier_floor = OneVsRestClassifier(LinearSVC(random_state=0)).fit(data['X_train_floor'], data['y_train_floor'])
        classifier_ceil = OneVsRestClassifier(LinearSVC(random_state=0)).fit(data['X_train_ceil'], data['y_train_ceil'])
    if(selectModel==2):
        print "Decision Tree";
        classifier_floor = tree.DecisionTreeClassifier().fit(data['X_train_floor'], data['y_train_floor'])
        classifier_ceil = tree.DecisionTreeClassifier().fit(data['X_train_ceil'], data['y_train_ceil'])
    if(selectModel==3):
        print "Nearest Centroid";
        classifier_floor = NearestCentroid().fit(data['X_train_floor'], np.ravel(data['y_train_floor']));
        classifier_ceil = NearestCentroid().fit(data['X_train_ceil'], np.ravel(data['y_train_ceil']));
    if(selectModel==4):
        print "SGD Classifier";
        classifier_floor = SGDClassifier(loss="hinge", penalty="l2").fit(data['X_train_floor'], np.ravel(data['y_train_floor']));
        classifier_ceil = SGDClassifier(loss="hinge", penalty="l2").fit(data['X_train_ceil'], np.ravel(data['y_train_ceil']));
    
    train_predict_floor = classifier_floor.predict(data['X_train_floor']);
    conf_mat_floor = confusion_matrix(train_predict_floor,data['y_train_floor']);
    train_predict_ceil = classifier_ceil.predict(data['X_train_ceil']);
    conf_mat_ceil = confusion_matrix(train_predict_ceil, data['y_train_ceil'])
    
    y_predict_ceil = classifier_ceil.predict(data['X_test_ceil'])
    result_ceil = confusion_matrix(y_predict_ceil,data['y_test_ceil'])
    
    y_predict_floor = classifier_floor.predict(data['X_test_floor'])
    result_floor = confusion_matrix(y_predict_floor,data['y_test_floor'])
    
    precision_floor, recall_floor, _, _ = precision_recall_fscore_support(data['y_test_floor'], y_predict_floor)
    precision_ceil, recall_ceil, _, _ = precision_recall_fscore_support(data['y_test_ceil'], y_predict_ceil)
    
    acc_ceil = result_ceil.trace()*100/result_ceil.sum();
    acc_ceil_train = conf_mat_ceil.trace()*100/conf_mat_ceil.sum();
    acc_floor = result_floor.trace()*100/result_floor.sum();
    acc_floor_train = conf_mat_floor.trace()*100/conf_mat_floor.sum();
    
    data['acc_ceil_train'] = acc_ceil_train;
    data['acc_floor_train'] = acc_floor_train;
    
    return data, acc_ceil, acc_floor, recall_ceil, recall_floor, result_ceil, result_floor, conf_mat_floor, conf_mat_ceil;

def apply_regression_model(X_train, y_train, X_test, y_test, indices, selectModel):
    Result = {};
    Result['X_train'] = X_train;
    Result['y_train'] = y_train; 
    Result['X_test'] = X_test;
    Result['y_test'] = y_test;
    Result['indices'] = indices;
    if(selectModel==0):
        print "Linear Regression";
        model = linear_model.LinearRegression();
        model.fit(X_train, y_train);
        predictions = model.predict(X_test);
        predictions_train = model.predict(X_train);
    if(selectModel==1):
        print "Ridge Regression";
        model = linear_model.RidgeCV(alphas = (0.1,0.1,10));
        model.fit(X_train, y_train);
        predictions = model.predict(X_test);
        predictions_train = model.predict(X_train);
    if(selectModel==2):
        print "Lasso Regression";
        model = linear_model.MultiTaskLassoCV(eps=0.001, n_alphas=100, alphas=(0.1,0.1,10));
        model.fit(X_train, y_train);
        predictions = model.predict(X_test);
        predictions_train = model.predict(X_train);
    Result['predictions'] = predictions;
    Result['model'] = model;
    Result['predictions_train'] = predictions_train;
    return Result;

def get_data_allModel(lab, feat, selectFeatTech):
    X_train, X_test, y_train, y_test = train_test_split(feat, lab, test_size=0.3, random_state=42);
    X_train, y_train = replicate_data(X_train, y_train);
    X_test, y_test = replicate_data(X_test, y_test);
    X_train, X_test , indices = feature_selection_regression(X_train, y_train, X_test, selectFeatTech);
    return X_train, y_train, X_test, y_test, indices;

def get_data_allModel_class(lab, feat, selectFeatTech):
    lab_ceil, lab_floor = get_lab_ceil(lab);
    data = {};
    chk = True;
    # just to make sure that there is no case where only 2 classes are there in training or testing set
    while(chk):
        X_train_ceil, X_test_ceil, y_train_ceil, y_test_ceil = train_test_split(feat, lab_ceil, test_size=0.3, random_state=10);
        if(len(pd.unique(y_train_ceil[0].ravel()))==3 & len(pd.unique(y_test_ceil[0].ravel()))==3):
            chk = False;
    data['X_train_before'] = X_train_ceil;
    data['y_train_before'] = y_train_ceil;
    data['X_test_before'] = X_test_ceil;
    data['y_test_before'] = y_test_ceil;
    X_train_ceil, y_train_ceil = replicate_data(X_train_ceil, y_train_ceil);
    X_test_ceil, y_test_ceil = replicate_data(X_test_ceil, y_test_ceil);
    data['X_train_after'] = X_train_ceil;
    data['y_train_after'] = y_train_ceil;
    data['X_test_after'] = X_test_ceil;
    data['y_test_after'] = y_test_ceil;
    
    # feature Selection for the ceil    
    X_train_ceil, X_test_ceil, indices_ceil = feature_selection_tech(X_train_ceil,y_train_ceil[0],X_test_ceil,selectFeatTech)
    data['X_train_ceil'] = X_train_ceil;
    data['X_test_ceil'] = X_test_ceil;
    data['y_train_ceil'] = y_train_ceil;
    data['y_test_ceil'] = y_test_ceil;
    data['ind_ceil'] = indices_ceil;
    
    # just to make sure that there is no case where only 2 classes are there in training or testing set
    chk = True;
    while(chk):
        X_train_floor, X_test_floor, y_train_floor, y_test_floor = train_test_split(feat, lab_floor, test_size=0.3, random_state=10)
        if(len(pd.unique(y_train_floor[0].ravel()))==3 & len(pd.unique(y_test_floor[0].ravel()))==3):
            chk = False;
    X_train_floor, y_train_floor = replicate_data(X_train_floor, y_train_floor);
    X_test_floor, y_test_floor = replicate_data(X_test_floor, y_test_floor);
    
    # feature selection for the floor
    X_train_floor, X_test_floor, indices_floor = feature_selection_tech(X_train_floor,y_train_floor[0],X_test_floor,selectFeatTech)
    data['X_train_floor'] = X_train_floor;
    data['X_test_floor'] = X_test_floor;
    data['y_train_floor'] = y_train_floor;
    data['y_test_floor'] = y_test_floor;
    data['ind_floor'] = indices_floor;
    return data;

def get_lab_ceil(lab):
    lab_ceil = np.ceil(lab);
    lab_floor = np.floor(lab);
    return lab_ceil, lab_floor;

def replicate_data(x_val, y_val):
    ind1 = y_val[y_val[0]==1].shape[0];
    ind2 = y_val[y_val[0]==2].shape[0];
    ind3 = y_val[y_val[0]==3].shape[0];
    max_val = max(ind1,ind2,ind3);
    # we are increasing all the data to the 90% of the maximum of the class that is present.
    cnt = int(max_val*1);
    if(ind1>0 & ind1!=max_val):
        new_cnt = cnt-ind1;
        pred = x_val[y_val[0]==1];
        resp = y_val[y_val[0]==1];
        for x in range(0,new_cnt):
            pos = np.random.randint(pred.shape[0]);
            val = pred.iloc[[pos]];
            res = resp.iloc[[pos]];
            x_val = pd.concat([x_val,val]);
            y_val = pd.concat([y_val,res]);
    
    if(ind2>0 & ind2!=max_val):
        new_cnt = cnt-ind2;
        pred = x_val[y_val[0]==2];
        resp = y_val[y_val[0]==2];
        for x in range(0,new_cnt):
            pos = np.random.randint(pred.shape[0]);
            val = pred.iloc[[pos]];
            res = resp.iloc[[pos]];
            x_val = pd.concat([x_val,val]);
            y_val = pd.concat([y_val,res]);    
    
    if(ind3>0 & ind3!=max_val):
        new_cnt = cnt-ind3;
        pred = x_val[y_val[0]==3];
        resp = y_val[y_val[0]==3];
        for x in range(0,new_cnt):
            pos = np.random.randint(pred.shape[0]);
            val = pred.iloc[[pos]];
            res = resp.iloc[[pos]];
            x_val = pd.concat([x_val,val]);
            y_val = pd.concat([y_val,res]);
    return x_val, y_val;

# feature selection techniques
#X_train_floor, X_test_floor, indices = feature_selection_tech(X_train_floor,y_train_floor[0],X_test_floor,selectFeatTech)
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


# ### Classification and Regression Functions. The next cell contains their iterative versions.
# ----

# In[79]:

def get_classification_results():
    # Final Cell to get all the classification results.
    presentFeatTech = 0;
    Result_dump = [];
    features_selected_dump = [];
    cnt = 0;
    feat_tech = ['SelectK','Randomized Lasso'];
    mod_tech = ['OneVsRest','Decision Trees','Nearest Centroid','SGD Classifier'];
    index_dict = {};
    for dataSet in range(0,5,2):
        lab, feat, final_keys = get_data_whole(dataSet)
        # Selecting data once for all of the models.
        for presentFeatTech in range(0,2):
            temp_data = get_data_allModel_class(lab, feat, presentFeatTech);
            for modelNo in range(1,4):
                index_dict[cnt] = mod_tech[modelNo-1];
                cnt = cnt+1;
                data0, acc_ceil, acc_floor, recall_ceil, recall_floor, result_ceil, result_floor, conf_mat_floor, conf_mat_ceil = apply_Model(temp_data,modelNo);
                sel_features_floor = final_keys[data0['ind_floor']];
                sel_features_floor = [x[0] for x in sel_features_floor];
                sel_features_ceil = final_keys[data0['ind_ceil']];
                sel_features_ceil = [str(x[0]) for x in sel_features_ceil];
                Result_dump.append([data0['acc_ceil_train'],acc_ceil, conf_mat_ceil,result_ceil,len(sel_features_ceil),sel_features_ceil,
                                    data0['acc_floor_train'],acc_floor, conf_mat_floor, result_floor,len(sel_features_floor),sel_features_floor]);
    A = pd.DataFrame(Result_dump)
    A.rename(columns={0:'Train-Acc-Ceil',1:'Test-Acc-Ceil',2:'Train-Conf-Ceil',3:'Test-Conf-Ceil',4:'FeatNo-Ceil',5:'Selected Features-Ceil',
                  6:'Train-Acc-Floor',7:'Test-Acc-Floor',8:'Train-Conf-Floor',9:'Test-Conf-Floor',10:'FeatNo-Floor',11:'Selected Features-Floor'}, inplace=True)
    A.rename(index=index_dict, inplace=True)
    return A;

def get_regression_results():
    # Final Regression result dumpings.
    import warnings
    warnings.filterwarnings('ignore')
    # For all of the models and all of the feature selection techniques
    Result = {};
    final_stats = [];
    mod_tech = ['linear','Ridge','Lasso'];
    mod_dist = {};
    cnt = 0;
    for dataSelect in range(0,5,2):
        lab, feat, final_keys = get_data_whole(dataSelect)
        print "\n"
        print 'Java-sID->'+str(test['sids'][0][dataSelect])
        Result[dataSelect] = {};
        Result[dataSelect]['final_keys'] = final_keys;
        for presentFeatRegress in range(0,2):
            Result[dataSelect][presentFeatRegress] = {};
            # Just to select one data for all of the models. So that the result of models are comparable
            X_train, y_train, X_test, y_test, indices = get_data_allModel(lab, feat, presentFeatRegress);
            for modelNo in range(0,3):
                Result[dataSelect][presentFeatRegress][modelNo] = apply_regression_model(X_train, y_train, X_test, y_test, indices, modelNo);
                present_data = Result[dataSelect][presentFeatRegress][modelNo];
                Corr_train = pearsonr(Result[dataSelect][presentFeatRegress][modelNo]['y_train'].values, Result[dataSelect][presentFeatRegress][modelNo]['predictions_train'])[0];
                Corr_test = pearsonr(Result[dataSelect][presentFeatRegress][modelNo]['y_test'].values, Result[dataSelect][presentFeatRegress][modelNo]['predictions'])[0];
                selected_feat = final_keys[present_data['indices']];
                selected_feat = [str(x[0]) for x in selected_feat];
                final_stats.append([Corr_train[0], Corr_test[0],len(selected_feat),selected_feat]);
                mod_dist[cnt] = mod_tech[modelNo];
                cnt = cnt+1;
    A = pd.DataFrame(final_stats);
    A.rename(columns={0:'Corr-Train',1:'Corr-Test',2:'FeatNo',3:'Features Selected'},inplace=True);
    A.rename(index=mod_dist, inplace=True)
    return A;


# In[80]:

def get_classification_iterate_results(iterate_times):
    for i in range(0,iterate_times):
        if(i==0):
            val = get_classification_results();
            val = val[['Train-Acc-Ceil','Test-Acc-Ceil','FeatNo-Ceil','Train-Conf-Ceil','Test-Conf-Ceil','Train-Acc-Floor','Test-Acc-Floor','FeatNo-Floor','Train-Conf-Floor','Test-Conf-Floor']];
        else:
            temp_val = get_classification_results();
            temp_val = temp_val[['Train-Acc-Ceil','Test-Acc-Ceil','FeatNo-Ceil','Train-Conf-Ceil','Test-Conf-Ceil','Train-Acc-Floor','Test-Acc-Floor','FeatNo-Floor','Train-Conf-Floor','Test-Conf-Floor']];
            val = val + temp_val;
    val = val/iterate_times;
    return val;

def get_regression_iterate_results(iterate_times):
    for i in range(0,iterate_times):
        if(i==0):
            val = get_regression_results();
            val = val[['Corr-Train','Corr-Test','FeatNo']];
        else:
            temp_val = get_regression_results();
            temp_val = temp_val[['Corr-Train','Corr-Test','FeatNo']];
            val = val+temp_val;
    val = val/iterate_times;
    return val;


# In[81]:

#Result_class = get_classification_iterate_results(10);
#Result_regression = get_regression_iterate_results(10);


# In[82]:

if __name__ == '__main__':
    iterate_items = 10;
    if(iterate_items):
        Result_class = get_classification_iterate_results(iterate_items);
        #Result_regression = get_regression_iterate_results(iterate_items);
        Result_class.to_csv("Result_class_iterate10.csv");
        #Result_regression.to_csv("Result_regress_iterate10.csv");


# In[8]:

val.keys()

