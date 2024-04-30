# In[0]
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio 
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import SelectFromModel
from xgboost import XGBRegressor 
from sklearn import svm
import xgboost as xgb
from sklearn.svm import SVR 
from scipy.stats import pearsonr
from sklearn.feature_selection import mutual_info_regression as MIR
import warnings 
import seaborn as sns
import csv
import math
from numpy import savetxt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.model_selection import LeaveOneOut
from sklearn.model_selection import cross_val_score
from sklearn.metrics import make_scorer
from numpy import mean
from numpy import absolute
import h5py

warnings.filterwarnings("ignore")

# from google.colab import files
# from google.colab import drive
# drive.mount('/content/drive')

# In[]:
    
    
# file_path = 'Baqiatollah_All_Data.h5'


# # with pd.HDFStore(file_path, mode='r') as store:
    
# #     All_Features_Loaded = store['All Features']  
# #     Time_Domain_Features_Loaded = store['Time Domain Features']
# #     Other_Features_Loaded = store['Other Features']  
# #     Frequency_Features_Loaded = store['Frequency Features'] 
    
# #     Blood_Glucose_Loaded = store['Blood Glucose']
# #     Systolic_Pressure_Loaded = store['Systolic Pressure']  
# #     Diastolic_Pressure_Loaded = store['Diastolic Pressure'] 
    
# #     Bad_ID_Loaded = store['Bad ID']
    
    
# with pd.HDFStore(file_path, mode='r') as store:
    
#     BP_All = store['Blood Pressure All Features']  
#     BP_TimeDom = store['Blood Pressure Time Domain Features']
#     BP_Others = store['Blood Pressure Other Features']  
#     BP_Freq = store['Blood Pressure Frequency Features'] 
    
#     # All_Features_Loaded = store['Blood Glucose All Features']  
#     # Time_Domain_Features_Loaded = store['Blood Glucose Time Domain Features']
#     # Other_Features_Loaded = store['Blood Glucose Other Features']  
#     # Frequency_Features_Loaded = store['Blood Glucose Frequency Features'] 
    
#     # Blood_Glucose_Loaded = store['Blood Glucose']
#     Systolic_Pressure_Loaded = store['Systolic Pressure']  
#     Diastolic_Pressure_Loaded = store['Diastolic Pressure'] 
    
#     Bad_ID_Loaded = store['Bad ID']
# choose "prebaqi" and "baqi"
which_data = "baqi"

if which_data == "prebaqi":
    file_path = '65.h5'
    with pd.HDFStore(file_path, mode='r') as store:
        
        BP_All = store['Blood Pressure All Features']  
        BP_TimeDom = store['Blood Pressure Time Domain Features']
        BP_Others = store['Blood Pressure Other Features']  
        BP_Freq = store['Blood Pressure Frequency Features'] 
        
        # All_Features_Loaded = store['Blood Glucose All Features']  
        # Time_Domain_Features_Loaded = store['Blood Glucose Time Domain Features']
        # Other_Features_Loaded = store['Blood Glucose Other Features']  
        # Frequency_Features_Loaded = store['Blood Glucose Frequency Features'] 
        
        Blood_Glucose_Loaded = store['Blood Glucose']
        Systolic_Pressure_Loaded = store['Systolic Pressure']  
        Diastolic_Pressure_Loaded = store['Diastolic Pressure'] 
        
        Bad_ID_Loaded = store['Bad ID']
        
    file_path = 'azad.h5'
    with pd.HDFStore(file_path, mode='r') as store:
        
        BP_All = pd.concat([store['Blood Pressure All Features'], BP_All],axis=0)  
        BP_TimeDom = pd.concat([store['Blood Pressure Time Domain Features'], BP_TimeDom],axis=0) 
        BP_Others = pd.concat([store['Blood Pressure Other Features'], BP_Others],axis=0)  
        BP_Freq = pd.concat([store['Blood Pressure Frequency Features'], BP_Freq],axis=0)  
        
        # All_Features_Loaded = pd.concat([store['Blood Glucose All Features'],All_Features_Loaded],axis=0) 
        # Time_Domain_Features_Loaded = pd.concat([store['Blood Glucose Time Domain Features'],Time_Domain_Features_Loaded],axis=0)
        # Other_Features_Loaded = pd.concat([store['Blood Glucose Other Features'], Other_Features_Loaded],axis=0)  
        # Frequency_Features_Loaded = pd.concat([store['Blood Glucose Frequency Features'],Frequency_Features_Loaded],axis=0)
        
        Blood_Glucose_Loaded = pd.concat([store['Blood Glucose'], Blood_Glucose_Loaded],axis=0)
        Systolic_Pressure_Loaded =  pd.concat([store['Systolic Pressure'], Systolic_Pressure_Loaded], axis=0)
        Diastolic_Pressure_Loaded = pd.concat([store['Diastolic Pressure'], Diastolic_Pressure_Loaded],axis=0)
        
        Bad_ID_Loaded = store['Bad ID']
        
    file_path = 'ziyaee.h5'
    with pd.HDFStore(file_path, mode='r') as store:
        
        BP_All = pd.concat([store['Blood Pressure All Features'], BP_All],axis=0)  
        BP_TimeDom = pd.concat([store['Blood Pressure Time Domain Features'], BP_TimeDom],axis=0) 
        BP_Others = pd.concat([store['Blood Pressure Other Features'], BP_Others],axis=0)  
        BP_Freq = pd.concat([store['Blood Pressure Frequency Features'], BP_Freq],axis=0) 
        
        # All_Features_Loaded = pd.concat([store['Blood Glucose All Features'],All_Features_Loaded],axis=0) 
        # Time_Domain_Features_Loaded = pd.concat([store['Blood Glucose Time Domain Features'],Time_Domain_Features_Loaded],axis=0)
        # Other_Features_Loaded = pd.concat([store['Blood Glucose Other Features'], Other_Features_Loaded],axis=0)  
        # Frequency_Features_Loaded = pd.concat([store['Blood Glucose Frequency Features'],Frequency_Features_Loaded],axis=0)
        
        Blood_Glucose_Loaded = pd.concat([store['Blood Glucose'], Blood_Glucose_Loaded],axis=0)
        Systolic_Pressure_Loaded =  pd.concat([store['Systolic Pressure'], Systolic_Pressure_Loaded], axis=0)
        Diastolic_Pressure_Loaded = pd.concat([store['Diastolic Pressure'], Diastolic_Pressure_Loaded],axis=0)
        
        Bad_ID_Loaded = store['Bad ID']
        
        
elif which_data == "baqi":
    
    with pd.HDFStore('V5_normaltored.h5', mode='r') as store:
        
        BP_All = store['Blood Pressure All Features']  
        BP_TimeDom = store['Blood Pressure Time Domain Features']
        BP_Others = store['Blood Pressure Other Features']  
        BP_Freq = store['Blood Pressure Frequency Features'] 
        
        
        #All_Features_Loaded = store['Blood Glucose All Features']  
        # Time_Domain_Features_Loaded = store['Blood Glucose Time Domain Features']
        # dfraw = store['Blood Pressure All Features']  
        Blood_Glucose_Loaded = store['Blood Glucose']
        # yraw = store['Diastolic Pressure']
        Systolic_Pressure_Loaded = store['Systolic Pressure']
        Diastolic_Pressure_Loaded = store['Diastolic Pressure']

else:
    print("choose dataset")

# In[2]: Load Dataset from Matlab


X = np.array(BP_All) #Load Features
X_time_dom = np.array(BP_TimeDom)
X_freq = np.array(BP_Freq)
X_others = np.array(BP_Others)


df = BP_All #Convert Feature Matrix into Pandas DataFrame 
df = df.reset_index(drop=True)
df = df.drop(35, axis=0)
df_names = df.columns

df_time_dom = BP_TimeDom
df_time_dom = df_time_dom.reset_index(drop=True)
df_time_dom = df_time_dom.drop(35, axis=0)
df_time_dom_names = df_time_dom.columns

df_freq = BP_Freq
df_freq = df_freq.reset_index(drop=True)
df_freq = df_freq.drop(35, axis=0)
df_freq_names = df_freq.columns

df_others = BP_Others
df_others = df_others.reset_index(drop=True)
df_others = df_others.drop(35, axis=0)
df_others_names = df_others.columns
 
obsv = X.shape[0] #Number of Feature Matrix Rows(Observations) 
feat = X.shape[1] #Number of Feature Matrix Columns(Features) 
 
# y_glc = np.array(Blood_Glucose_Loaded) #Load Target Matrix for Blood Glucose 

y_sys = (np.array(Systolic_Pressure_Loaded)) #Load Target Matrix for Blood Pressure Systolic 
y_dia = np.array(Diastolic_Pressure_Loaded) #Load Target Matrix for Blood Pressure Diastolic
y_sys = np.delete(y_sys,35)
y_dia = np.delete(y_dia,35)
# In[3]: Cleaning Dataset 
 
# Removing NaN & inf Values from Dataset 
df = df.replace([np.inf, -np.inf], np.nan) 
for i in range(df.shape[1]): 
    df.iloc[:,i] = df.iloc[:,i].fillna(df.iloc[:,i].mean()) 
df_time_dom = df_time_dom.replace([np.inf, -np.inf], np.nan)
for i in range(df_time_dom.shape[1]):
    df_time_dom.iloc[:,i] = df_time_dom.iloc[:,i].fillna(df_time_dom.iloc[:,i].mean())

df_freq = df_freq.replace([np.inf, -np.inf], np.nan)
for i in range(df_freq.shape[1]):
    df_freq.iloc[:,i] = df_freq.iloc[:,i].fillna(df_freq.iloc[:,i].mean())

df_others = df_others.replace([np.inf, -np.inf], np.nan)
for i in range(df_others.shape[1]):
    df_others.iloc[:,i] = df_others.iloc[:,i].fillna(df_others.iloc[:,i].mean())

# Remove Glucose Outlier
# glc_outlier = np.where(y_glc > 200)[0]
# y_glc = np.delete(y_glc,glc_outlier,axis=0)
# y_sys = np.delete(y_sys,glc_outlier,axis=0)
# y_dia = np.delete(y_dia,glc_outlier,axis=0)

# df_temp = np.asarray(df)
# df_temp = np.delete(df_temp,glc_outlier,axis=0)
# df = pd.DataFrame(df_temp)
# df_cln = df 
# X_cln = np.asarray(df_cln) 

# df_temp = np.asarray(df_time_dom)
# df_temp = np.delete(df_temp,glc_outlier,axis=0)
# df_time_dom = pd.DataFrame(df_temp)
# X_time_dom = np.asarray(df_time_dom)

# df_temp = np.asarray(df_freq)
# df_temp = np.delete(df_temp,glc_outlier,axis=0)
# df_freq = pd.DataFrame(df_temp)
# X_freq = np.asarray(df_freq)

# df_temp = np.asarray(df_others)
# df_temp = np.delete(df_temp,glc_outlier,axis=0)
# df_others = pd.DataFrame(df_temp)
# X_others = np.asarray(df_others)   

# Removing Duplicated Columns
# df_uni = df.T.drop_duplicates().T 
# X_uni = np.unique(X, axis=1, return_index=True, return_inverse=True, return_counts=True) 
# indicies_uni = np.asarray(X_uni[1]) 
# indicies_uni = np.sort(indicies_uni)

# Deliver Cleaned Dataset as Output 
df_cln = df 
X_cln = np.asarray(df_cln) 
# X_cln = np.delete(X_cln,35)
X_time_dom = np.asarray(df_time_dom)
# X_time_dom = np.delete(X_time_dom,35)
X_freq = np.asarray(df_freq)
# X_freq = np.delete(X_freq,35)
X_others = np.asarray(df_others)
# X_others = np.delete(X_others,35)
# df_cln = pd.DataFrame(X_cln)

# In[]: Useful Functions
   
def mape(y_pred,y_true):
    y_pred = np.asarray(y_pred)
    y_true = np.asarray(y_true)
    mape = (abs((y_true - y_pred)/y_true).mean())*100
    return(mape)

def NMAPE(y_true, y_pred):
    y_pred = np.asarray(y_pred)
    y_true = np.asarray(y_true)
    return -(1 - np.mean(np.abs((y_true - y_pred) / y_true)) * 100)    

def max_min(list_func,type_func):
    target_number,largest_number,smallest_number = 0,0,0

    if(type_func == 'Max'):
        largest_number = list_func[0]    
        for number in list_func:
            if number > largest_number:
                largest_number = number
        target_number = largest_number
    
    if(type_func == 'Min'):
        smallest_number = list_func[0]    
        for number in list_func:
            if number < smallest_number:
                smallest_number = number
        target_number = smallest_number
        
    return(target_number)

def common_elements(list1, list2):
    list1 = list(list1)
    list2 = list(list2)
    return [element for element in list1 if element in list2] 

# In[2]: Computing Model Score for Glucose by XGBoost
nRun = 20 
importances = 0 
for i in range(nRun): 
    # sel = SelectFromModel(RandomForestRegressor(n_estimators = 1000, max_features="sqrt")) 
    sel = SelectFromModel(XGBRegressor()) 
    # sel = SelectFromModel(SVR(kernel="rbf")) 
    sel.fit(df_cln, y_sys) 
    coefs = sel.estimator_.feature_importances_
    # coefs = sel.estimator_.coef_ 
    importances = importances + coefs/nRun
    
# In[]: PCA 
    
def pca_features(input_data,selected_features,k):
    
    if(selected_features == 'Null'):
        input_data = input_data
    else:
        input_data = input_data[:,selected_features]

    scaler = StandardScaler()
    scaler.fit(input_data)
    scaled_input_data = scaler.transform(input_data)
    
    pca = PCA(n_components = k)
    pca.fit(scaled_input_data)
    data_pca = pca.transform(scaled_input_data)
    return(data_pca)
   
     
# In[]: 
 
# compute Fature-Target statistical test 
corr_fsys, corr_fdia, mir_fsys, mir_fdia = [], [], [], [] 
# Mutal information computation 
def MIR_func(input1,input2): 
    mir = [] 
    for i in range(0,5): 
        mir_score = MIR(input1,input2) 
        mir.append(mir_score) 
    mir_mean = (mir[0]+mir[1]+mir[2]+mir[3]+mir[4])/5 
    mir_mean = pd.Series(mir_mean, index = input1.columns) 
    return(mir_mean) 
# Correlation computation 
def linear_corr_func(input1,input2): 
    corr_list = [] 
    for i in range(0,input1.shape[1]): 
        corr = pearsonr(input1.iloc[:,i],input2)[0] 
        corr_list.append(abs(corr)) 
    corr_arr = np.asarray(corr_list).astype(None)
    corr_arr = corr_arr.ravel()
    corr_series = pd.Series(corr_arr, index = input1.columns) 
    return(corr_series) 
 
importances = pd.Series(importances, index = df_cln.columns) 
     
# In[]: 
     
# inputs: 
#    X: pandas.DataFrame, features 
#    y: pandas.Series, target variable 
#    K: number of features to select 
#    fraction_type: num & denom type 
 
def mrmr_fraction(X,y,k,fraction_type): #inputs are dataframe, target vector, number of features you want and score type(availabe for MODELSCORE, MIQ, CORR) 
    # initialize matrix 
    corr = pd.DataFrame(.00001, index = X.columns, columns =X.columns) 
     
    # initialize list of selected features and list of excluded features 
    selected = [] 
    not_selected = X.columns.to_list() 
     
    # specify the num type 
    if(fraction_type == 'CORR'): 
        F = linear_corr_func(X,y) 
    elif(fraction_type == 'MODELSCORE'): 
        F = importances 
    elif(fraction_type == 'MIQ'): 
        F = MIR_func(X,y) 
    else: 
        return('Correct your fraction type1') 
     
    # repeat for K times 
    for i in range(k): 
         
        if(fraction_type == 'CORR'): 
        # compute statistical test between the last selected feature and all the (currently) excluded features 
            if i > 0: 
                last_selected = selected[-1] 
                corr.loc[not_selected, last_selected] = X[not_selected].corrwith(X[last_selected]).abs().clip(.00001) 
         
        elif(fraction_type == 'MIQ'): 
        # compute statistical test between the last selected feature and all the (currently) excluded features 
            if i > 0: 
                last_selected = selected[-1] 
                corr.loc[not_selected, last_selected] = MIR_func(X[not_selected],X[last_selected]).abs().clip(.00001) 
        elif(fraction_type == 'MODELSCORE'): 
        # compute statistical test between the last selected feature and all the (currently) excluded features 
            if i > 0: 
                last_selected = selected[-1] 
                corr.loc[not_selected, last_selected] = X[not_selected].corrwith(X[last_selected]).abs().clip(.00001) 
        else: 
            print('Correct your fraction type2') 
             
        # compute score for all the (currently) excluded features 
        score = F.loc[not_selected] / corr.loc[not_selected, selected].mean(axis = 1).fillna(.00001) 
         
        # find best feature, add it to selected and remove it from not_selected 
        best = score.index[score.argmax()] 
        selected.append(best) 
        not_selected.remove(best) 
    return(np.asarray(selected))
     


# In[]
# function of model
def model_imp(inputfeatures,target,model_type):   
    
    if(model_type == 'xgboost'):
        nRun = 3
        importances = 0
        for i in range(nRun):
            sel = SelectFromModel(XGBRegressor())
            sel.fit(inputfeatures,target)
            coefs = sel.estimator_.feature_importances_
            importances = importances + coefs/nRun
            
    elif(model_type == 'svm'):
        sel = SelectFromModel(SVR(kernel="linear"))
        sel.fit(inputfeatures,target)
        coefs = sel.estimator_.coef_
        importances = coefs
    else:
        print('Correct your model type') 
    return importances  
  
# In[]: 
def choose_embedded(X,y,model,atleast,zarib): 
    importances = model_imp(X, y, model) 
     
    if(model == 'svm'): 
        importances = abs(importances) 
        importances = importances.reshape(X.shape[1],) 
         
    importances_sorted = importances[np.argsort(importances)[::-1]] 
    indices = np.argsort(importances)[::-1] 
    chosen_ind = indices[0:atleast] 
     
    for i in range(atleast+1,importances.shape[0]-1): 
        if (importances_sorted[i] > importances_sorted[i-1]*zarib): 
            chosen_ind = np.append(chosen_ind,indices[i]) 
        else: 
            break        
    return(chosen_ind)

# In[]:
    
def model(X,y,features,modeltype):
    
    X_chosen = X[:,features]
    
    # define model 
    if(modeltype == 'XGBRegressor()'):
        model = XGBRegressor()        
    if(modeltype == 'SVR(kernel="rbf")'):
        model = SVR(kernel="rbf")
    if(modeltype == 'SVR(kernel="poly")'):
        model = SVR(kernel="poly")
    # if(modeltype == 'KNeighborsRegressor(n_neighbors=2)'):
    #     model = KNeighborsRegressor(n_neighbors=2)
    if(modeltype == 'SVR(kernel="linear")'):
        model = SVR(kernel="linear")
    # if(modeltype == 'XGBoostDecoder(max_depth=3,num_round=200,eta=0.3,gpu=-1)')
    #     model=XGBoostDecoder(max_depth=3,num_round=200,eta=0.3,gpu=-1)
    
    predicted_train, predicted_test, mape_predicted, error_train = [],[],[],[]
    # error_train, r2_score_train, error_test, r2_score_test = [],[],[],[]
    
    for j in range(0,10):
        
        train_size = int(0.7 * X_chosen.shape[0])
        train_idx = np.random.choice(range(X_chosen.shape[0]), size=train_size, replace=False)
        valid_idx = np.array(list(set(range(X_chosen.shape[0])) - set(train_idx)))
        X_train, X_test = X_chosen[train_idx,:], X_chosen[valid_idx,:]
        y_train, y_test = y[train_idx], y[valid_idx]
        
        # X_train, X_test, y_train, y_test = X_chosen[C,:],X_chosen[D,:], y[C],y[D]
        
        # fit model 
        model.fit(X_train, y_train)
        predict_train = model.predict(X_train)
        predict_test = model.predict(X_test)
        
        predicted_train.append(predict_train)
        predicted_test.append(predict_test)
        
        error_train.append(mape(y_train,predicted_train[j]))
        # r2_score_train.append(r2_score(y_train,predicted_train[j]))
        
        # error_test.append(mean_squared_error(y_test,predicted_test[j]))
        # r2_score_test.append(r2_score(y_test,predicted_test[j]))
        
        mape_predict = mape(predict_test,y[valid_idx])
        mape_predicted.append(mape_predict)
    
    mean_mape = np.mean(mape_predicted)
        
    return(predicted_test,mean_mape,error_train)


# In[]:
    
def SBS(X,y,features,modeltype):
    
    features = np.asanyarray(features)
    features_komaki = features
    sbs_features,mapes_out,smallest_mapes = [],[],[]
    
    for j in range(0,features.shape[0]-1):
        
        mapes = []
        for i in range(0,features_komaki.shape[0]):
                      
            featureselected = np.delete(features_komaki,np.where(features_komaki==features_komaki[i]),axis=0)
            modelout = model(X,y,list(featureselected),modeltype)
            mapes.append(modelout[1])
                    
        smallest_mape = max_min(mapes,'Min')
        smallest_mapes.append(smallest_mape)
        mapes = np.asarray(mapes)
        features_komaki = np.delete(features_komaki,np.where(mapes==smallest_mape),axis=0)
        sbs_features.append(features_komaki)
        mapes_out.append(mapes)
        smallest_mape_total = max_min(smallest_mapes,'Min')
        # print(largest_mape,features_komaki)
        # best_feature_subset = sbs_features[np.where(smallest_mapes==smallest_mape_total)]
        best_feature_subset = sbs_features[np.where(smallest_mapes==smallest_mape_total)[0][0]]
        # best_feature_subset_names = name[best_feature_subset]
        # breakpoint()
    return(sbs_features,mapes_out,smallest_mapes,smallest_mape_total,best_feature_subset)

# In[]:

#make scorer from custome function
nmape_scorer = make_scorer(mape,greater_is_better=False)

def SBS_Package(df_data,y,selected_features,model_type):
    
    if(model_type == 'XGBRegressor()'):
        model = XGBRegressor()
    elif(model_type == 'SVR linear'):
        model = SVR(kernel="linear")
    elif(model_type == 'SVR(kernel="rbf")'):
        model = SVR(kernel="rbf")
    elif(model_type == 'SVR(kernel="poly")'):
        model = SVR(kernel="poly")
    # else:
    #     return('Correct your Model Type')
        
    X_featureselected = df_data.iloc[:,selected_features]    
    X_featureselected = X_featureselected.to_numpy()
    # number_of_sample = X_featureselected.shape[0]
    number_of_features = X_featureselected.shape[1]
          
    cv = LeaveOneOut()
    selected_features = np.asarray(selected_features)
    
    # tedad = []
    error = []
    selected_feat = []
    for ft in range(1,number_of_features-1):
        nfeat = number_of_features - ft
        sbs = SequentialFeatureSelector(model, n_features_to_select=nfeat,cv=cv,direction ='backward',scoring = nmape_scorer)    
        sbs.fit(X_featureselected, y)
        selected_features = selected_features[sbs.get_support()]
        selected_feat.append(selected_features)
        # print(sbs.get_support())
        X_featureselected = sbs.transform(X_featureselected)
        scores = cross_val_score(model,X_featureselected, y, scoring = nmape_scorer,
                             cv=cv, n_jobs=-1)
        rmse = mean(absolute(scores))
        error.append(rmse)
        # tedad.append(nfeat)        
    sbs_slc_features = selected_feat[np.where(error == min(error))[0][0]]
    return sbs_slc_features,error,selected_feat

# In[]:
    
# glc_mat_before  =  sio.loadmat('F:/NBP/PPG/Code/3/Azad/Before_SBS_Glucose_Selected_Features.mat') 

# mrmr_glc_all_embedded = glc_mat_before['mrmr_selected_glc_features1'].ravel().transpose()
# mrmr_glc_cln2 = glc_mat_before['mrmr_selected_glc_features2'].ravel().transpose()

# embd_glc_all = glc_mat_before['glc_all_selected'].ravel().transpose()
# embd_glc_morph = glc_mat_before['glc_morph_selected'].ravel().transpose()
# embd_glc_freq = glc_mat_before['glc_freq_selected'].ravel().transpose()
# embd_glc_others = glc_mat_before['glc_others_selected'].ravel().transpose() 

# glc_mat = sio.loadmat('F:/NBP/PPG/Code/3/Azad/Glucose_Features_Selected.mat')

# sbs_cln_glc1 = glc_mat['sbs_glc_cln_best_feature_subset1'].ravel().transpose()
# sbs_cln_glc2 = glc_mat['sbs_glc_cln_best_feature_subset2'].ravel().transpose()
# sbs_morph_glc = glc_mat['sbs_glc_morph_best_feature_subset'].ravel().transpose()
# sbs_freq_glc = glc_mat['sbs_glc_freq_best_feature_subset'].ravel().transpose()
# sbs_others_glc = glc_mat['sbs_glc_others_best_feature_subset'].ravel().transpose()

# In[]: 
 
# mrmr_glc_cln1 = mrmr_fraction(df_cln, y_glc, 60, 'MIQ') 
# mrmr_glc_cln2_test = mrmr_fraction(df_cln, y_glc, 20, 'MODELSCORE')        
mrmr_sys_cln = mrmr_fraction(df_cln, y_sys, 60, 'CORR')  
mrmr_sys_morph = mrmr_fraction(df_time_dom, y_sys, 30, 'CORR')
# mrmr_dia_cln  = mrmr_fraction(df_cln, y_dia, 60, 'CORR')
# mrmr_dia_morph = mrmr_fraction(df_time_dom, y_dia, 30, 'CORR')    

print('done mrmr')

# In[]

# SYS  
input_features = df_cln.loc[:,mrmr_sys_cln]
embd_sys_all = choose_embedded(input_features,y_sys,'svm',20,1)
input_features = df_cln.loc[:,mrmr_sys_morph]
embd_sys_morph = choose_embedded(input_features,y_sys,'svm',20,1)
embd_sys_freq = choose_embedded(df_freq,y_sys,'svm',20,1)
embd_sys_others = choose_embedded(df_others,y_sys,'svm',20,1)

# # DIA
# input_features = df_cln.loc[:,mrmr_dia_cln]
# embd_dia_all = choose_embedded(input_features,y_dia,'svm',20,1)
# input_features = df_cln.loc[:,mrmr_dia_morph]
# embd_dia_morph = choose_embedded(input_features,y_dia,'svm',20,1)
# embd_dia_freq = choose_embedded(df_freq,y_dia,'svm',20,1)
# embd_dia_others = choose_embedded(df_others,y_dia,'svm',20,1)

# # GLUCOSE
# input_features = df_cln.loc[:,mrmr_glc_cln1]
# embd_glc_all = choose_embedded(input_features,y_glc,'svm',20,1)
# embd_glc_morph = choose_embedded(df_time_dom,y_glc,'svm',20,1)
# embd_glc_freq = choose_embedded(df_freq,y_glc,'svm',20,1)
# embd_glc_others = choose_embedded(df_others,y_glc,'svm',20,1)

print('done embedded')

# In[]:

# sbs_cln_sys = SBS(X_cln,y_sys,embd_sys_all,'SVR(kernel="linear")')
# sbs_morph_sys = SBS(X_time_dom,y_sys,embd_sys_morph,'SVR(kernel="linear")')
# sbs_freq_sys = SBS(X_freq,y_sys,embd_sys_freq,'SVR(kernel="linear")')
# sbs_others_sys = SBS(X_others,y_sys,embd_sys_others,'SVR(kernel="linear")')

sbs_cln_sys_xgb = SBS(X_cln,y_sys,embd_sys_all,'XGBRegressor()')
# breakpoint()
sbs_morph_sys_xgb = SBS(X_time_dom,y_sys,embd_sys_morph,'XGBRegressor()')
sbs_freq_sys_xgb = SBS(X_freq,y_sys,embd_sys_freq,'XGBRegressor()')
sbs_others_sys_xgb = SBS(X_others,y_sys,embd_sys_others,'XGBRegressor()')

M_sys = { #'sbs_sys_cln_feature_subset' : sbs_cln_sys[0], 'sbs_sys_cln_mapes' : sbs_cln_sys[2], 'sbs_sys_cln_best_mape' : sbs_cln_sys[3], 'sbs_sys_cln_best_feature_subset' : sbs_cln_sys[4],
#        'sbs_sys_morph_feature_subset' : sbs_morph_sys[0], 'sbs_sys_morph_mapes' : sbs_morph_sys[2], 'sbs_sys_morph_best_mape' : sbs_morph_sys[3], 'sbs_sys_morph_best_feature_subset' : sbs_morph_sys[4],
#        'sbs_sys_freq_feature_subset' : sbs_freq_sys[0], 'sbs_sys_freq_mapes' : sbs_freq_sys[2], 'sbs_sys_freq_best_mape' : sbs_freq_sys[3], 'sbs_sys_freq_best_feature_subset' : sbs_freq_sys[4],
#        'sbs_sys_others_feature_subset' : sbs_others_sys[0], 'sbs_sys_others_mapes' : sbs_others_sys[2], 'sbs_sys_others_best_mape' : sbs_others_sys[3], 'sbs_sys_others_best_feature_subset' : sbs_others_sys[4]}
        'sbs_sys_cln_feature_subset_xgb' : sbs_cln_sys_xgb[0], 'sbs_sys_cln_mapes_xgb' : sbs_cln_sys_xgb[2], 'sbs_sys_cln_best_mape_xgb' : sbs_cln_sys_xgb[3], 'sbs_sys_cln_best_feature_subset_xgb' : sbs_cln_sys_xgb[4],
        'sbs_sys_morph_feature_subset_xgb' : sbs_morph_sys_xgb[0], 'sbs_sys_morph_mapes_xgb' : sbs_morph_sys_xgb[2], 'sbs_sys_morph_best_mape_xgb' : sbs_morph_sys_xgb[3], 'sbs_sys_morph_best_feature_subset_xgb' : sbs_morph_sys_xgb[4],
        'sbs_sys_freq_feature_subset_xgb' : sbs_freq_sys_xgb[0], 'sbs_sys_freq_mapes_xgb' : sbs_freq_sys_xgb[2], 'sbs_sys_freq_best_mape_xgb' : sbs_freq_sys_xgb[3], 'sbs_sys_freq_best_feature_subset_xgb' : sbs_freq_sys_xgb[4],
        'sbs_sys_others_feature_subset_xgb' : sbs_others_sys_xgb[0], 'sbs_sys_others_mapes_xgb' : sbs_others_sys_xgb[2], 'sbs_sys_others_best_mape_xgb' : sbs_others_sys_xgb[3], 'sbs_sys_others_best_feature_subset_xgb' : sbs_others_sys_xgb[4]}

sio.savemat('Systolic_Selected_Features_prebaqi_svm.mat', M_sys)


# sbs_cln_dia = SBS(X_cln,y_dia,embd_dia_all,'SVR(kernel="linear")')
# sbs_morph_dia = SBS(X_time_dom,y_dia,embd_dia_morph,'SVR(kernel="linear")')
# sbs_freq_dia = SBS(X_freq,y_dia,embd_dia_freq,'SVR(kernel="linear")')
# sbs_others_dia = SBS(X_others,y_dia,embd_dia_others,'SVR(kernel="linear")')  

# sbs_cln_dia_xgb = SBS(X_cln,y_dia,embd_dia_all,'XGBRegressor()',df_names)
# sbs_morph_dia_xgb = SBS(X_time_dom,y_dia,embd_dia_morph,'XGBRegressor()',df_time_dom_names)
# sbs_freq_dia_xgb = SBS(X_freq,y_dia,embd_dia_freq,'XGBRegressor()',df_freq_names)
# sbs_others_dia_xgb = SBS(X_others,y_dia,embd_dia_others,'XGBRegressor()',df_others_names)  

# M_dia = { 'sbs_dia_cln_feature_subset' : sbs_cln_dia[0], 'sbs_dia_cln_mapes' : sbs_cln_dia[2], 'sbs_dia_cln_best_mape' : sbs_cln_dia[3], 'sbs_dia_cln_best_feature_subset' : sbs_cln_dia[4],
#        'sbs_dia_morph_feature_subset' : sbs_morph_dia[0], 'sbs_dia_morph_mapes' : sbs_morph_dia[2], 'sbs_dia_morph_best_mape' : sbs_morph_dia[3], 'sbs_dia_morph_best_feature_subset' : sbs_morph_dia[4],
#        'sbs_dia_freq_feature_subset' : sbs_freq_dia[0], 'sbs_dia_freq_mapes' : sbs_freq_dia[2], 'sbs_dia_freq_best_mape' : sbs_freq_dia[3], 'sbs_dia_freq_best_feature_subset' : sbs_freq_dia[4],
#        'sbs_dia_others_feature_subset' : sbs_others_dia[0], 'sbs_dia_others_mapes' : sbs_others_dia[2], 'sbs_dia_others_best_mape' : sbs_others_dia[3], 'sbs_dia_others_best_feature_subset' : sbs_others_dia[4]}
#         # 'sbs_dia_cln_feature_subset_xgb' : sbs_cln_dia_xgb[0], 'sbs_dia_cln_mapes_xgb' : sbs_cln_dia_xgb[2], 'sbs_dia_cln_best_mape_xgb' : sbs_cln_dia_xgb[3], 'sbs_dia_cln_best_feature_subset_xgb' : sbs_cln_dia_xgb[4],'sbs_dia_cln_best_feature_subset_xgb_names' : sbs_cln_dia_xgb[5],
#         # 'sbs_dia_morph_feature_subset_xgb' : sbs_morph_dia_xgb[0], 'sbs_dia_morph_mapes_xgb' : sbs_morph_dia_xgb[2], 'sbs_dia_morph_best_mape_xgb' : sbs_morph_dia_xgb[3], 'sbs_dia_morph_best_feature_subset_xgb' : sbs_morph_dia_xgb[4], 'sbs_dia_morph_best_feature_subset_xgb_names' : sbs_morph_dia_xgb[5],
#         # 'sbs_dia_freq_feature_subset_xgb' : sbs_freq_dia_xgb[0], 'sbs_dia_freq_mapes_xgb' : sbs_freq_dia_xgb[2], 'sbs_dia_freq_best_mape_xgb' : sbs_freq_dia_xgb[3], 'sbs_dia_freq_best_feature_subset_xgb' : sbs_freq_dia_xgb[4], 'sbs_dia_freq_best_feature_subset_xgb_names' : sbs_freq_dia_xgb[5],
#         # 'sbs_dia_others_feature_subset_xgb' : sbs_others_dia_xgb[0], 'sbs_dia_others_mapes_xgb' : sbs_others_dia_xgb[2], 'sbs_dia_others_best_mape_xgb' : sbs_others_dia_xgb[3], 'sbs_dia_others_best_feature_subset_xgb' : sbs_others_dia_xgb[4], 'sbs_dia_others_best_feature_subset_xgb_names' : sbs_others_dia_xgb[5]}

# sio.savemat('diatolic_Selected_Features_prebaqi_svm.mat', M_dia)   


# sbs_cln_glc1 = SBS(X_cln,y_glc,embd_glc_all,'SVR(kernel="linear")')
# sbs_cln_glc2 = SBS(X_cln,y_glc,mrmr_glc_cln2_test,'SVR(kernel="linear")')
# sbs_morph_glc = SBS(X_time_dom,y_glc,embd_glc_morph,'SVR(kernel="linear")')
# sbs_freq_glc = SBS(X_freq,y_glc,embd_glc_freq,'SVR(kernel="linear")')
# sbs_others_glc = SBS(X_others,y_glc,embd_glc_others,'SVR(kernel="linear")')

# sbs_cln_glc1_xgb = SBS(X_cln,y_glc,embd_glc_all,'SVR(kernel="linear")')
# sbs_cln_glc2_xgb = SBS(X_cln,y_glc,np.asarray([df_cln.columns.get_loc(col) for col in mrmr_glc_cln2_test]),'SVR(kernel="linear")')
# sbs_morph_glc_xgb = SBS(X_time_dom,y_glc,embd_glc_morph,'XGBRegressor()')
# sbs_freq_glc_xgb = SBS(X_freq,y_glc,embd_glc_freq,'XGBRegressor()')
# sbs_others_glc_xgb = SBS(X_others,y_glc,embd_glc_others,'XGBRegressor()')

# M_glc = { #'sbs_glc_cln_feature_subset1' : sbs_cln_glc1[0], 'sbs_glc_cln_mapes1' : sbs_cln_glc1[2], 'sbs_glc_cln_best_mape1' : sbs_cln_glc1[3], 'sbs_glc_cln_best_feature_subset1' : sbs_cln_glc1[4],
# #        'sbs_glc_cln_feature_subset2' : sbs_cln_glc2[0], 'sbs_glc_cln_mapes2' : sbs_cln_glc2[2], 'sbs_glc_cln_best_mape2' : sbs_cln_glc2[3], 'sbs_glc_cln_best_feature_subset2' : sbs_cln_glc2[4],
# #        'sbs_glc_morph_feature_subset' : sbs_morph_glc[0], 'sbs_glc_morph_mapes' : sbs_morph_glc[2], 'sbs_glc_morph_best_mape' : sbs_morph_glc[3], 'sbs_glc_morph_best_feature_subset' : sbs_morph_glc[4],
# #        'sbs_glc_freq_feature_subset' : sbs_freq_glc[0], 'sbs_glc_freq_mapes' : sbs_freq_glc[2], 'sbs_glc_freq_best_mape' : sbs_freq_glc[3], 'sbs_glc_freq_best_feature_subset' : sbs_freq_glc[4],
# #        'sbs_glc_others_feature_subset' : sbs_others_glc[0], 'sbs_glc_others_mapes' : sbs_others_glc[2], 'sbs_glc_others_best_mape' : sbs_others_glc[3], 'sbs_glc_others_best_feature_subset' : sbs_others_glc[4],
#        'sbs_glc_cln_feature_subset1_xgb' : sbs_cln_glc1_xgb[0], 'sbs_glc_cln_mapes1_xgb' : sbs_cln_glc1_xgb[2], 'sbs_glc_cln_best_mape1_xgb' : sbs_cln_glc1_xgb[3], 'sbs_glc_cln_best_feature_subset1_xgb' : sbs_cln_glc1_xgb[4],
#        'sbs_glc_cln_feature_subset2_xgb' : sbs_cln_glc2_xgb[0], 'sbs_glc_cln_mapes2_xgb' : sbs_cln_glc2_xgb[2], 'sbs_glc_cln_best_mape2_xgb' : sbs_cln_glc2_xgb[3], 'sbs_glc_cln_best_feature_subset2_xgb' : sbs_cln_glc2_xgb[4],
#        'sbs_glc_morph_feature_subset_xgb' : sbs_morph_glc_xgb[0], 'sbs_glc_morph_mapes_xgb' : sbs_morph_glc_xgb[2], 'sbs_glc_morph_best_mape_xgb' : sbs_morph_glc_xgb[3], 'sbs_glc_morph_best_feature_subset_xgb' : sbs_morph_glc_xgb[4],
#        'sbs_glc_freq_feature_subset_xgb' : sbs_freq_glc_xgb[0], 'sbs_glc_freq_mapes_xgb' : sbs_freq_glc_xgb[2], 'sbs_glc_freq_best_mape_xgb' : sbs_freq_glc_xgb[3], 'sbs_glc_freq_best_feature_subset_xgb' : sbs_freq_glc_xgb[4],
#        'sbs_glc_others_feature_subset_xgb' : sbs_others_glc_xgb[0], 'sbs_glc_others_mapes_xgb' : sbs_others_glc_xgb[2], 'sbs_glc_others_best_mape_xgb' : sbs_others_glc_xgb[3], 'sbs_glc_others_best_feature_subset_xgb' : sbs_others_glc_xgb[4]}

# sio.savemat('glocuse_Selected_Features.mat', M_glc)

# In[]:
    

# best_features_sys = {
#     'cln_sys': df.iloc[:, sbs_cln_sys[4]].columns.tolist(),
#     'morph_sys': df.iloc[:, sbs_morph_sys[4]].columns.tolist(),
#     'freq_sys': df.iloc[:, sbs_freq_sys[4]].columns.tolist(),
#     'others_sys': df.iloc[:, sbs_others_sys[4]].columns.tolist(),
#     'cln_sys_xgb': df.iloc[:, sbs_cln_sys_xgb[4]].columns.tolist(),
#     'morph_sys_xgb': df.iloc[:, sbs_morph_sys_xgb[4]].columns.tolist(),
#     'freq_sys_xgb': df.iloc[:, sbs_freq_sys_xgb[4]].columns.tolist(),
#     'others_sys_xgb': df.iloc[:, sbs_others_sys_xgb[4]].columns.tolist()
# }


# best_features_dia = {
#     'cln_dia': df.iloc[:, sbs_cln_dia[4]].columns.tolist(),
#     'morph_dia': df.iloc[:, sbs_morph_dia[4]].columns.tolist(),
#     'freq_dia': df.iloc[:, sbs_freq_dia[4]].columns.tolist(),
#     'others_dia': df.iloc[:, sbs_others_dia[4]].columns.tolist(),
#     'cln_dia_xgb': df.iloc[:, sbs_cln_dia_xgb[4]].columns.tolist(),
#     'morph_dia_xgb': df.iloc[:, sbs_morph_dia_xgb[4]].columns.tolist(),
#     'freq_dia_xgb': df.iloc[:, sbs_freq_dia_xgb[4]].columns.tolist(),
#     'others_dia_xgb': df.iloc[:, sbs_others_dia_xgb[4]].columns.tolist()
# }


# best_features_glc = {
#     # 'cln_glc1': df.iloc[:, sbs_cln_glc1[4]].columns.tolist(),
#     # 'cln_glc2': df.iloc[:, sbs_cln_glc2[4]].columns.tolist(),
#     # 'morph_glc': df.iloc[:, sbs_morph_glc[4]].columns.tolist(),
#     # 'freq_glc': df.iloc[:, sbs_freq_glc[4]].columns.tolist(),
#     # 'others_glc': df.iloc[:, sbs_others_glc[4]].columns.tolist(),
#     'cln_glc1_xgb': df.iloc[:, sbs_cln_glc1_xgb[4]].columns.tolist(),
#     'cln_glc2_xgb': df.iloc[:, sbs_cln_glc2_xgb[4]].columns.tolist(),
#     'morph_glc_xgb': df.iloc[:, sbs_morph_glc_xgb[4]].columns.tolist(),
#     'freq_glc_xgb': df.iloc[:, sbs_freq_glc_xgb[4]].columns.tolist(),
#     'others_glc_xgb': df.iloc[:, sbs_others_glc_xgb[4]].columns.tolist()
#}












     









