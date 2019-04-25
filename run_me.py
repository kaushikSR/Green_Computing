# Import python modules
import numpy as np
import pandas as pd
import sklearn.neighbors as skln
from sklearn.model_selection import KFold
import sklearn.tree as skt
import matplotlib.pyplot as plt
import sklearn.linear_model as ll
import time
from sklearn.model_selection import cross_val_score, StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV


# Compute MAE
def compute_error(y_hat, y):
    # mean absolute error
    return np.abs(y_hat - y).mean()

############################################################################

#Function to perform k-fold cross validation for decision tree and KNN
    #Returns error for each K and the time taken
def kfoldCV(k,X,Y,model):
    startTime = time.time()
    folds = KFold(n_splits=k)
    error = 0
    for train_index, test_index in folds.split(train_x):
        trainFeat, testFeat = X[train_index], X[test_index]
        trainResp, testResp = Y[train_index], Y[test_index]
        model.fit(trainFeat,trainResp)
        error += compute_error(model.predict(testFeat),testResp) 
    return (error/k,1000*(time.time() - startTime))

#function to train Ridge and Lasso models.
    #Chooses the best model and returns the predicted output 
def trainRL(train_x,train_y,test_x,k,reg_constants):
    lin_kf = KFold(n_splits=k)
    lin_errorTable_r = {}
    lin_errorTable_l = {}
    for val in reg_constants:
        error_r = 0
        error_l = 0
        for train_index, test_index in lin_kf.split(train_x):
            #print(train_index,test_index)
            trainFeat, testFeat = train_x[train_index], train_x[test_index]
            #print(trainFeat)
            trainResp, testResp = train_y[train_index], train_y[test_index]
            y_hat_ridge = ll.Ridge(val).fit(trainFeat,trainResp).predict(testFeat)
            y_hat_lasso = ll.Lasso(val).fit(trainFeat,trainResp).predict(testFeat)
            error_r += compute_error(y_hat_ridge,testResp)
            error_l += compute_error(y_hat_lasso,testResp)
        lin_errorTable_r[val]=error_r/k
        lin_errorTable_l[val]=error_l/k
    print('Out of sample error for each Ridge Linear Model \n')
    print('{:<10} : {:<10}'.format('alpha', 'error'))
    for r, err in lin_errorTable_r.items():
        print('{:<10} : {:<10}'.format(r, err))
    print('\nOut of sample error for each Lasso Linear Model \n')
    print('{:<10} : {:<10}'.format('alpha', 'error'))
    for l, err in lin_errorTable_l.items():
        print('{:<10} : {:<10}'.format(l, err))
    best_r = min(lin_errorTable_r, key = lin_errorTable_r.get)
    best_l = min(lin_errorTable_l, key = lin_errorTable_l.get)
    
    if lin_errorTable_r[best_r]<lin_errorTable_l[best_l]:
        print('\nModel with lowest estimated out of sample error is Ridge with Alpha = ',best_r, ' with an error of ',lin_errorTable_r[best_r])
        predicted_y = ll.Ridge(best_r).fit(train_x,train_y).predict(test_x)
    else:
        print('\nModel with lowest estimated out of sample error is Lasso with Alpha = ',best_l, ' with an error of ',lin_errorTable_l[best_l])
        predicted_y = ll.Lasso(best_l).fit(train_x,train_y).predict(test_x)
        
    return predicted_y
            
#function to train the decision tree and KNN models and predict the test data output            
def train(tr_x,tr_y,te_x,hyperParameters,model,k):
    errorTable = {}
    time = {}
    select = {1:"skt.DecisionTreeRegressor(max_depth= ",2:"skln.KNeighborsRegressor(n_neighbors= "}
    for i in hyperParameters:
        modelName = select.get(model)
        modelSelected = eval(modelName + str(i) + ')')
        errorTable[i],time[i]= kfoldCV(k,tr_x,tr_y,modelSelected)
    print('Out of sample error for each model\n')
    print('{:<10} : {:<10}'.format('val', 'error'))
    for val, err in errorTable.items():
        print('{:<10} : {:<10}'.format(val, err))
    best_h = min(errorTable, key = errorTable.get)
    print('\nModel with lowest estimated out of sample error has hyperparameter = ',best_h, ', k =',k,' with a training error of ',errorTable[best_h])
    modelTest = eval(modelName + str(best_h) + ')')
    predicted_y = modelTest.fit(tr_x,tr_y).predict(te_x)
    return (predicted_y, time)
############################################################################

print('\n##############################################################################\n')
      
combined_features = pd.read_pickle('C:/Users/kaush/Documents/Training/combined_features.pkl')
features = pd.read_pickle('C:/Users/kaush/Documents/Training/features.pkl')
#print(combined_features.shape)

def fix_orientation(x):
    if x >= 0 and x < 90:
        return x+180
    elif x > 270 and x <= 360:
        return x-180
    else: return x
    
combined_features = combined_features[combined_features.orientation != 'NaN'].reset_index(drop = True)
#print(combined_features['orientation'])
combined_features = combined_features[combined_features.orientation != 'no'].reset_index(drop = True)
#print(combined_features['orientation'])
combined_features = combined_features[combined_features.orientation != 'not visible'].reset_index(drop = True)
#print(combined_features['orientation'])
combined_features = combined_features[combined_features.orientation != '194,46'].reset_index(drop = True)
combined_features = combined_features[combined_features.orientation != '361'].reset_index(drop = True)
#combined_features = combined_features.apply(pd.to_numeric, errors='ignore')
#combined_features['orientation'] = combined_features['orientation'].map(fix_orientation)
#combined_features.replace('194,46',194.46)
#print(combined_features['orientation'])
combined_features = combined_features[pd.notnull(combined_features['perimeter'])]
combined_features = combined_features[pd.notnull(combined_features['area'])]
combined_features = combined_features[pd.notnull(combined_features['circleness'])]
combined_features = combined_features[pd.notnull(combined_features['r_min'])]
combined_features = combined_features[pd.notnull(combined_features['r_max'])]
combined_features = combined_features[pd.notnull(combined_features['r_mean'])]
combined_features = combined_features[pd.notnull(combined_features['r_variance'])]
combined_features = combined_features[pd.notnull(combined_features['r_skewness'])]
combined_features = combined_features[pd.notnull(combined_features['r_kurtosis'])]
combined_features = combined_features[pd.notnull(combined_features['g_min'])]
combined_features = combined_features[pd.notnull(combined_features['g_max'])]
combined_features = combined_features[pd.notnull(combined_features['g_mean'])]
combined_features = combined_features[pd.notnull(combined_features['g_variance'])]
combined_features = combined_features[pd.notnull(combined_features['g_skewness'])]
combined_features = combined_features[pd.notnull(combined_features['g_kurtosis'])]
combined_features = combined_features[pd.notnull(combined_features['b_min'])]
combined_features = combined_features[pd.notnull(combined_features['b_max'])]
combined_features = combined_features[pd.notnull(combined_features['b_mean'])]
combined_features = combined_features[pd.notnull(combined_features['b_variance'])]
combined_features = combined_features[pd.notnull(combined_features['b_skewness'])]
combined_features = combined_features[pd.notnull(combined_features['b_kurtosis'])]
combined_features = combined_features[pd.notnull(combined_features['area_pixels'])]
combined_features = combined_features[pd.notnull(combined_features['orientation'])]



#print("value",setni)
#print(combined_features.shape)
#print(combined_features)
#print(combined_features.tail())
excluded = ['panel_class', 'image_id', 'segment_id', 'centroid', 'coords','area_pixels','orientation' ]
X_cols = [col for col in combined_features.columns if col not in excluded] 

ss = StandardScaler()  #To normalize the data
'''Standardization of a dataset is a common requirement for many machine learning estimators:
they might behave badly if the individual features do not more or less look like standard normally 
distributed data (e.g. Gaussian with 0 mean and unit variance).
If a feature has a variance that is orders of magnitude larger that others, 
it might dominate the objective function and make the estimator unable to learn from other features correctly as expected.
'''
X = combined_features[X_cols]
Xnew = features[X_cols]
Xnew = ss.fit_transform(Xnew)
#print(X.head())
Xn = ss.fit_transform(X) #Transforms X to have a mean value = 0 and Standard deviation = 1
#included = ['area_pixels','orientation']
#Y_cols = [col for col in combined_features.columns if col in included]
y_area = combined_features[['area_pixels']].reset_index(drop = True)
y_or = combined_features[['orientation']].reset_index(drop = True)

#print(y)
# =============================================================================
# #print(y.head())
# print(1 - np.mean(y)) #percent of zeroes
# y_base = max(np.mean(y), 1 - np.mean(y)) 
# # Split data into a training and testing set
# 
# =============================================================================
# =============================================================================
X_train_a, X_test_a, y_train_a, y_test_a = train_test_split(Xn, y_area, test_size=0.3)
X_train_o, X_test_o, y_train_o, y_test_o = train_test_split(Xn, y_or, test_size=0.3)
#print(X_train.shape, X_test.shape)
#print(y_train.shape, y_test.shape)

y_test_a = y_test_a.reset_index(drop = True)
y_test_o = y_test_o.reset_index(drop = True)
y_train_a = y_train_a.reset_index(drop = True)
y_train_o = y_train_o.reset_index(drop = True)

#print(y_test.shape)
y_test_a = y_test_a[pd.notnull(y_test_a['area_pixels'])]
y_test_o = y_test_o[pd.notnull(y_test_o['orientation'])]
y_test_a = y_test_a.apply(pd.to_numeric, errors='ignore')
y_test_o = y_test_o.apply(pd.to_numeric, errors='ignore')
y_train_a = y_train_a.apply(pd.to_numeric, errors='ignore')
y_train_o = y_train_o.apply(pd.to_numeric, errors='ignore')
#print(y_test)
#print(X_train)
#print(y_train)
# =============================================================================
X_train_a = np.array(X_train_a)
X_train_o = np.array(X_train_o)
X_test_a = np.array(X_test_a)
X_test_o = np.array(X_test_o)
y_train_a = np.array(y_train_a)
y_train_o = np.array(y_train_o)
y_test_a = np.array(y_test_a)
y_test_o = np.array(y_test_o)
#print(y_test.dtype)
# =============================================================================

#X_test.fillna(X_test.mean())
#X_train.fillna(X_train.mean())
#np.any(np.isnan(X_train))
#np.any(np.isnan(y_train))
#np.any(np.isnan(X_test))
#np.any(np.isnan(y_test))

#np.all(np.isfinite(X_train))
#np.all(np.isfinite(X_test))
#np.all(np.isfinite(y_train))
#np.all(np.isfinite(y_test))
# =============================================================================
# # =============================================================================
lm = ll.LinearRegression()

model_a = lm.fit(X_train_a, y_train_a)
predictions_a = lm.predict(Xnew)
print(predictions_a)

# =============================================================================
# print(predictions_a - y_test_a)
# print(compute_error(predictions_a,y_test_a))
# 
# plt.scatter(y_test_a, predictions_a)
# plt.xlabel('true Values area')
# plt.ylabel('Predictions area')
# 
# print('Score: area ', model_a.score(X_test_a, y_test_a))
# 
# predicted_y = trainRL(X_train_a,y_train_a,X_test_a,5,[1e-6,1e-4,1e-2,1,10])
# print("test error for area ",compute_error(predicted_y,y_test_a))
# 
# for i in range(y_test_a.shape[0]):
#     print(predicted_y[i],y_test_a[i])
# =============================================================================
# =============================================================================
# =============================================================================
# model_o = lm.fit(X_train_o, y_train_o)
# predictions_o = lm.predict(X_test_o)
# print(predictions_o - y_test_o)
# print(compute_error(predictions_o,y_test_o))
# plt.scatter(y_test_o, predictions_o)
# plt.xlabel('true Values orientation')
# plt.ylabel('Predictions orientation')
# print('Score: orientation ', model_o.score(X_test_o, y_test_o))
# 
# 
# 
# predicted_y = trainRL(X_train_o,y_train_o,X_test_o,5,[1e-6,1e-4,1e-2,1,10])
# print("test error for orientation ",compute_error(predicted_y,y_test_o))
# 
# 
# for i in range(y_test_o.shape[0]):
#     print(predicted_y[i],y_test_o[i])
# =============================================================================
# =============================================================================

# # =============================================================================
# =============================================================================
# =============================================================================
# X_train = X_train.reset_index(drop = True)
# y_train = y_train.reset_index(drop = True)
# X_test = X_test.reset_index(drop = True)
# y_test = y_test.reset_index(drop = True)
# print(X_train.columns)
# 
# trainFeat,testFeat,trainResp,testResp = train_test_split(X_train,y_train)
# =============================================================================
# =============================================================================
# lin_kf = KFold(n_splits=5)
# for train_index, test_index in lin_kf.split(X_train):
#             #print(train_index,test_index)
#             trainFeat, testFeat = X_train[train_index], X_train[test_index]
#             #print(trainFeat)
#             trainResp, testResp = y_train[train_index], y_train[test_index]
# 
# =============================================================================
#print(X_train[46])
# =============================================================================
#predicted_y = trainRL(X_train,y_train,X_test,5,[1e-6,1e-4,1e-2,1,10])
      
# =============================================================================
# =============================================================================
# train_x, train_y, test_x = read_data_air_foil()
# print('Train=', train_x.shape)
# print('Test=', test_x.shape)
# =============================================================================
# 
# #decision-tree
# print('\nQUESTION 1: DECISION TREES\n')
# =============================================================================
# print(X_train_o.shape)
# predicted_y_tree,D_timeCV = train(X_train_o,y_train_o,X_test_o,[3,6,9,12,15],1,5)
# print("test error for orientation ",compute_error(predicted_y_tree,y_test_o))
# 
# for i in range(y_test_o.shape[0]):
#     print(predicted_y_tree[i],y_test_o[i])
# =============================================================================

# #plt.plot([i for i in D_timeCV.keys()],[j for j in D_timeCV.values()])
# #plt.title('Training Time with 5-fold CV for different depths')
# #plt.xlabel('Depth of Decision Tree')
# #plt.ylabel('Time(milliseconds)')
# #plt.savefig("../Figures/AirFoil_CV_trainingTimeGraph.png", format = 'png')
# #plt.show()
# #kaggle.kaggleize(predicted_y,'../Predictions/AirFoil/decisionTree.csv')
# 
# ##KNN
# print('\nQUESTION 3: KNN\n')
# predicted_y,K_timeCV = train(train_x,train_y,test_x,[3,5,10,20,25],2,5)
# #kaggle.kaggleize(predicted_y,'../Predictions/AirFoil/knn.csv')
# 
# #Linear Regression
# print('\nQUESTION 4: Lasso/Ridge\n')
# predicted_y = trainRL(train_x,train_y,test_x,5,[1e-6,1e-4,1e-2,1,10])
# #kaggle.kaggleize(predicted_y,'../Predictions/AirFoil/ridgeLasso.csv')
# 
# #print('\nQUESTION 5: Best Model\n')
# #Best Model
# #for k in range(4,15):
# #    predicted_y,D_timeCV = train(train_x,train_y,test_x,np.arange(1,40,2),1,k)
# #    kaggle.kaggleize(predicted_y,'../Predictions/AirFoil/best'+str(k)+'.csv')
# =============================================================================

#print('\n##############################################################################\n')

# =============================================================================
# train_x, train_y, test_x = read_data_air_quality()
# print('Train=', train_x.shape)
# print('Test=', test_x.shape)
# 
# #decision-tree
# print('\nQUESTION 1: DECISION TREES\n')
# predicted_y,D_timeCV = train(train_x,train_y,test_x,[20,25,30,35,40],1,5)
# #plt.plot([i for i in D_timeCV.keys()],[j for j in D_timeCV.values()])
# #plt.title('Training Time with 5-fold CV for different depths')
# #plt.xlabel('Depth of Decision Tree')
# #plt.ylabel('Time(milliseconds)')
# ##plt.savefig("../Figures/AirQuality_CV_trainingTimeGraph.png", format = 'png')
# #plt.show()
# #kaggle.kaggleize(predicted_y,'../Predictions/AirQuality/decisionTree.csv')
# 
# #KNN
# print('\nQUESTION 3: KNN\n')
# predicted_y,timeCV = train(train_x,train_y,test_x,[3,5,10,20,25],2,5)
# #kaggle.kaggleize(predicted_y,'../Predictions/AirQuality/knn.csv')
# 
# #Linear Regression
# print('\nQUESTION 4: Lasso/Ridge\n')
# predicted_y = trainRL(train_x,train_y,test_x,5,[1e-4,1e-2,1,10])
# #kaggle.kaggleize(predicted_y,'../Predictions/AirQuality/ridgeLasso.csv')
# 
# #print('\nQUESTION 5: Best Model\n')
# #Best Model
# #for k in range(4,10):
#     #predicted_y,D_timeCV = train(train_x,train_y,test_x,np.arange(3,18,1),1,k)
#     #kaggle.kaggleize(predicted_y,'../Predictions/AirQuality/best'str(k)+'.csv')
# =============================================================================
