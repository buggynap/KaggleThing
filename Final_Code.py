import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Lasso
import xgboost as xgb
from sklearn.linear_model import Ridge, RidgeCV, ElasticNet, LassoCV, LassoLarsCV
from sklearn.cross_validation import cross_val_score
from sklearn.cross_validation import cross_val_score
from sklearn.metrics import make_scorer, mean_squared_error
from itertools import product, chain
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import Ridge, RidgeCV, ElasticNet, LassoCV, LassoLarsCV
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn import decomposition
from collections import Counter


def rmsle(y, y_pred):
     return np.sqrt((( (np.log1p(y_pred*price_scale)- np.log1p(y*price_scale)) )**2).mean())

def rmse_cv(model):
    rmse= np.sqrt(-cross_val_score(model, X_train, y, scoring="mean_squared_error", cv = 5))
    return(rmse)

def rmse_cv(model, X, y):
     return (cross_val_score(model, X, y, scoring=scorer)).mean()


def poly(X):
    areas = ['LotArea', 'TotalBsmtSF', 'GrLivArea', 'GarageArea', 'BsmtUnfSF']
    # t = [s for s in X.axes[1].get_values() if s not in areas]
    t = chain(qu_list.axes[1].get_values(), ['OverallQual', 'OverallCond', 'ExterQual', 'ExterCond', 'BsmtCond', 'GarageQual', 'GarageCond','KitchenQual', 'HeatingQC', 'bad_heating', 'MasVnrType_Any', 'SaleCondition_PriceDown', 'Reconstruct','ReconstructAfterBuy', 'Build.eq.Buy'])
    for a, t in product(areas, t):
        x = X.loc[:, [a, t]].prod(1)
        x.name = a + '_' + t
        yield x


# Read train data in panda
train = pd.read_csv("train.csv")
# Read test data in panda
test = pd.read_csv("test.csv")

# Concat the data of train and test
completeData = pd.concat((train.loc[:,'MSSubClass':'SaleCondition'], test.loc[:,'MSSubClass':'SaleCondition']), ignore_index=True)
# Where there is a NaN/null place some valid constant value
x = completeData.loc[np.logical_not(completeData["LotFrontage"].isnull()), "LotArea"]
y = completeData.loc[np.logical_not(completeData["LotFrontage"].isnull()), "LotFrontage"]
t = (x <= 25000) & (y <= 150)
p = np.polyfit(x[t], y[t], 1)
completeData.loc[completeData['LotFrontage'].isnull(), 'LotFrontage'] = np.polyval(p, completeData.loc[completeData['LotFrontage'].isnull(), 'LotArea'])
completeData.loc[completeData.Alley.isnull(), 'Alley'] = 'NoAlley'
completeData.loc[completeData.MasVnrType.isnull(), 'MasVnrType'] = 'None' # no good
completeData.loc[completeData.MasVnrType == 'None', 'MasVnrArea'] = 0
completeData.loc[completeData.BsmtQual.isnull(), 'BsmtQual'] = 'NoBsmt'
completeData.loc[completeData.BsmtCond.isnull(), 'BsmtCond'] = 'NoBsmt'
completeData.loc[completeData.BsmtExposure.isnull(), 'BsmtExposure'] = 'NoBsmt'
completeData.loc[completeData.BsmtFinType1.isnull(), 'BsmtFinType1'] = 'NoBsmt'
completeData.loc[completeData.BsmtFinType2.isnull(), 'BsmtFinType2'] = 'NoBsmt'
completeData.loc[completeData.BsmtFinType1=='NoBsmt', 'BsmtFinSF1'] = 0
completeData.loc[completeData.BsmtFinType2=='NoBsmt', 'BsmtFinSF2'] = 0
completeData.loc[completeData.BsmtFinSF1.isnull(), 'BsmtFinSF1'] = completeData.BsmtFinSF1.median()
completeData.loc[completeData.BsmtQual=='NoBsmt', 'BsmtUnfSF'] = 0
completeData.loc[completeData.BsmtUnfSF.isnull(), 'BsmtUnfSF'] = completeData.BsmtUnfSF.median()
completeData.loc[completeData.BsmtQual=='NoBsmt', 'TotalBsmtSF'] = 0
completeData.loc[completeData.FireplaceQu.isnull(), 'FireplaceQu'] = 'NoFireplace'
completeData.loc[completeData.GarageType.isnull(), 'GarageType'] = 'NoGarage'
completeData.loc[completeData.GarageFinish.isnull(), 'GarageFinish'] = 'NoGarage'
completeData.loc[completeData.GarageQual.isnull(), 'GarageQual'] = 'NoGarage'
completeData.loc[completeData.GarageCond.isnull(), 'GarageCond'] = 'NoGarage'
completeData.loc[completeData.BsmtFullBath.isnull(), 'BsmtFullBath'] = 0
completeData.loc[completeData.BsmtHalfBath.isnull(), 'BsmtHalfBath'] = 0
completeData.loc[completeData.KitchenQual.isnull(), 'KitchenQual'] = 'TA'
completeData.loc[completeData.MSZoning.isnull(), 'MSZoning'] = 'RL'
completeData.loc[completeData.Utilities.isnull(), 'Utilities'] = 'AllPub'
completeData.loc[completeData.Exterior1st.isnull(), 'Exterior1st'] = 'VinylSd'
completeData.loc[completeData.Exterior2nd.isnull(), 'Exterior2nd'] = 'VinylSd'
completeData.loc[completeData.Functional.isnull(), 'Functional'] = 'Typ'
completeData.loc[completeData.SaleCondition.isnull(), 'SaleCondition'] = 'Normal'
completeData.loc[completeData.SaleCondition.isnull(), 'SaleType'] = 'WD'
completeData.loc[completeData['PoolQC'].isnull(), 'PoolQC'] = 'NoPool'
completeData.loc[completeData['Fence'].isnull(), 'Fence'] = 'NoFence'
completeData.loc[completeData['MiscFeature'].isnull(), 'MiscFeature'] = 'None'
completeData.loc[completeData['Electrical'].isnull(), 'Electrical'] = 'SBrkr'
completeData.loc[completeData['GarageArea'].isnull(), 'GarageArea'] = completeData.loc[completeData['GarageType']=='Detchd', 'GarageArea'].mean()
completeData.loc[completeData['GarageCars'].isnull(), 'GarageCars'] = completeData.loc[completeData['GarageType']=='Detchd', 'GarageCars'].median()
# Convert the categorical data to the numerical data
completeData = completeData.replace({'Utilities': {'AllPub': 1, 'NoSeWa': 0, 'NoSewr': 0, 'ELO': 0},
                             'Street': {'Pave': 1, 'Grvl': 0 },
                             'FireplaceQu': {'Ex': 5, 'Gd': 4, 'TA': 3, 'Fa': 2,'Po': 1,'NoFireplace': 0 },
                             'Fence': {'GdPrv': 2, 'GdWo': 2, 'MnPrv': 1, 'MnWw': 1,'NoFence': 0},
                             'ExterQual': {'Ex': 5, 'Gd': 4, 'TA': 3, 'Fa': 2,'Po': 1},
                             'ExterCond': {'Ex': 5, 'Gd': 4, 'TA': 3, 'Fa': 2,'Po': 1},
                             'BsmtQual': {'Ex': 5, 'Gd': 4, 'TA': 3, 'Fa': 2,'Po': 1,'NoBsmt': 0},
                             'BsmtExposure': {'Gd': 3, 'Av': 2, 'Mn': 1,'No': 0,'NoBsmt': 0},
                             'BsmtCond': {'Ex': 5, 'Gd': 4, 'TA': 3, 'Fa': 2,'Po': 1,'NoBsmt': 0},
                             'GarageQual': {'Ex': 5, 'Gd': 4, 'TA': 3, 'Fa': 2,'Po': 1,'NoGarage': 0},
                             'GarageCond': {'Ex': 5, 'Gd': 4, 'TA': 3, 'Fa': 2,'Po': 1,'NoGarage': 0},
                             'KitchenQual': {'Ex': 5, 'Gd': 4, 'TA': 3, 'Fa': 2,'Po': 1},
                             'Functional': {'Typ': 0,'Min1': 1,'Min2': 1,'Mod': 2,'Maj1': 3,'Maj2': 4,'Sev': 5,'Sal': 6}
                            })
newer_dwelling = completeData.MSSubClass.replace({20: 1, 30: 0, 40: 0, 45: 0,50: 0, 60: 1,70: 0,75: 0,80: 0,85: 0,90: 0,120: 1,150: 0,160: 0,180: 0,190: 0})
newer_dwelling.name = 'newer_dwelling'
completeData = completeData.replace({'MSSubClass': {20: 'SubClass_20', 30: 'SubClass_30', 40: 'SubClass_40', 45: 'SubClass_45',50: 'SubClass_50', 60: 'SubClass_60',70: 'SubClass_70',75: 'SubClass_75',80: 'SubClass_80',85: 'SubClass_85',90: 'SubClass_90',120: 'SubClass_120',150: 'SubClass_150',160: 'SubClass_160',180: 'SubClass_180',190: 'SubClass_190'}})

# The idea is good quality should rise price, poor quality - reduce price
overall_poor_qu = completeData.OverallQual.copy()
overall_poor_qu = 5 - overall_poor_qu
overall_poor_qu[overall_poor_qu<0] = 0
overall_poor_qu.name = 'overall_poor_qu'

overall_good_qu = completeData.OverallQual.copy()
overall_good_qu = overall_good_qu - 5
overall_good_qu[overall_good_qu<0] = 0
overall_good_qu.name = 'overall_good_qu'

overall_poor_cond = completeData.OverallCond.copy()
overall_poor_cond = 5 - overall_poor_cond
overall_poor_cond[overall_poor_cond<0] = 0
overall_poor_cond.name = 'overall_poor_cond'

overall_good_cond = completeData.OverallCond.copy()
overall_good_cond = overall_good_cond - 5
overall_good_cond[overall_good_cond<0] = 0
overall_good_cond.name = 'overall_good_cond'

exter_poor_qu = completeData.ExterQual.copy()
exter_poor_qu[exter_poor_qu<3] = 1
exter_poor_qu[exter_poor_qu>=3] = 0
exter_poor_qu.name = 'exter_poor_qu'

exter_good_qu = completeData.ExterQual.copy()
exter_good_qu[exter_good_qu<=3] = 0
exter_good_qu[exter_good_qu>3] = 1
exter_good_qu.name = 'exter_good_qu'

exter_poor_cond = completeData.ExterCond.copy()
exter_poor_cond[exter_poor_cond<3] = 1
exter_poor_cond[exter_poor_cond>=3] = 0
exter_poor_cond.name = 'exter_poor_cond'

exter_good_cond = completeData.ExterCond.copy()
exter_good_cond[exter_good_cond<=3] = 0
exter_good_cond[exter_good_cond>3] = 1
exter_good_cond.name = 'exter_good_cond'

bsmt_poor_cond = completeData.BsmtCond.copy()
bsmt_poor_cond[bsmt_poor_cond<3] = 1
bsmt_poor_cond[bsmt_poor_cond>=3] = 0
bsmt_poor_cond.name = 'bsmt_poor_cond'

bsmt_good_cond = completeData.BsmtCond.copy()
bsmt_good_cond[bsmt_good_cond<=3] = 0
bsmt_good_cond[bsmt_good_cond>3] = 1
bsmt_good_cond.name = 'bsmt_good_cond'

garage_poor_qu = completeData.GarageQual.copy()
garage_poor_qu[garage_poor_qu<3] = 1
garage_poor_qu[garage_poor_qu>=3] = 0
garage_poor_qu.name = 'garage_poor_qu'

garage_good_qu = completeData.GarageQual.copy()
garage_good_qu[garage_good_qu<=3] = 0
garage_good_qu[garage_good_qu>3] = 1
garage_good_qu.name = 'garage_good_qu'

garage_poor_cond = completeData.GarageCond.copy()
garage_poor_cond[garage_poor_cond<3] = 1
garage_poor_cond[garage_poor_cond>=3] = 0
garage_poor_cond.name = 'garage_poor_cond'

garage_good_cond = completeData.GarageCond.copy()
garage_good_cond[garage_good_cond<=3] = 0
garage_good_cond[garage_good_cond>3] = 1
garage_good_cond.name = 'garage_good_cond'

kitchen_poor_qu = completeData.KitchenQual.copy()
kitchen_poor_qu[kitchen_poor_qu<3] = 1
kitchen_poor_qu[kitchen_poor_qu>=3] = 0
kitchen_poor_qu.name = 'kitchen_poor_qu'

kitchen_good_qu = completeData.KitchenQual.copy()
kitchen_good_qu[kitchen_good_qu<=3] = 0
kitchen_good_qu[kitchen_good_qu>3] = 1
kitchen_good_qu.name = 'kitchen_good_qu'

qu_list = pd.concat((overall_poor_qu, overall_good_qu, overall_poor_cond, overall_good_cond, exter_poor_qu,exter_good_qu, exter_poor_cond, exter_good_cond, bsmt_poor_cond, bsmt_good_cond, garage_poor_qu,garage_good_qu, garage_poor_cond, garage_good_cond, kitchen_poor_qu, kitchen_good_qu), axis=1)

bad_heating = completeData.HeatingQC.replace({'Ex': 0, 'Gd': 0, 'TA': 0, 'Fa': 1,'Po': 1})
bad_heating.name = 'bad_heating'
                                          
MasVnrType_Any = completeData.MasVnrType.replace({'BrkCmn': 1,'BrkFace': 1,'CBlock': 1,'Stone': 1,'None': 0})
MasVnrType_Any.name = 'MasVnrType_Any'

SaleCondition_PriceDown = completeData.SaleCondition.replace({'Abnorml': 1,'Alloca': 1,'AdjLand': 1,'Family': 1,'Normal': 0,'Partial': 0})
SaleCondition_PriceDown.name = 'SaleCondition_PriceDown'

X = pd.get_dummies(completeData, sparse=True)
X = X.fillna(0)
X = X.drop('MSSubClass_SubClass_160', axis=1)
X = X.drop('RoofMatl_ClyTile', axis=1)
X = X.drop('MSZoning_C (all)', axis=1)
X = X.drop('Condition2_PosN', axis=1)

XP = pd.concat(poly(X), axis=1)
X = pd.concat((X, XP), axis=1)
X_train = X[:train.shape[0]]
X_test = X[train.shape[0]:]

# Checking for the outliers
y = np.log1p(train.SalePrice)
outliers_id = np.array([524, 1299])
outliers_id = outliers_id - 1 # id starts with 1, index starts with 0
X_train = X_train.drop(outliers_id)
y = y.drop(outliers_id)
scorer = make_scorer(mean_squared_error, False)

# XBoost model
dtrain = xgb.DMatrix(X_train, label = y)
dtest = xgb.DMatrix(X_test)
params = {"max_depth":2, "eta":0.1}
model = xgb.cv(params, dtrain,  num_boost_round=500, early_stopping_rounds=100)
model_xgb = xgb.XGBRegressor(n_estimators=360, max_depth=2, learning_rate=0.1) #the params were tuned using xgb.cv
model_xgb.fit(X_train, y)  
xgb_preds = np.expm1(model_xgb.predict(X_test))
preds = xgb_preds
solution = pd.DataFrame({"id":test.Id, "SalePrice":preds})
col = solution.columns
solution[[col[0], col[1]]] = solution[[col[1], col[0]]]
solution.to_csv("model2.csv", index = False)

#  Lasso model
# Setting alphas
alphas = [1e-4, 5e-4, 1e-3, 5e-3]
cv_lasso = [rmse_cv(Lasso(alpha = alpha, max_iter=60000), X_train, y) for alpha in alphas]
pd.Series(cv_lasso, index = alphas).plot()
model_lasso = Lasso(alpha=5e-4, max_iter=50000).fit(X_train, y)
coef = pd.Series(model_lasso.coef_, index = X_train.columns).sort_values()
imp_coef = pd.concat([coef.head(10), coef.tail(10)])
imp_coef.plot(kind = "barh")
p_pred = np.expm1(model_lasso.predict(X_train))
p = np.expm1(model_lasso.predict(X_test))
solution = pd.DataFrame({"id":test.Id, "SalePrice":p}, columns=['id', 'SalePrice'])
solution.to_csv("model1.csv", index = False)


# using the other models
# Random Forest
# model_randomForest = RandomForestClassifier(n_estimators=300).fit(X_train, y)
# pred_randomForest = np.expm1(model_randomForest.predict(X_test))

#Neural Networks
# model_NN = MLPClassifier(algorithm='l-bfgs', alpha=1e-5, hidden_layer_sizes=(10,2), random_state = 1).fit(X_train, y)
# model_NN = MLPClassifier(alpha=1e-5, hidden_layer_sizes=(10,2), random_state = 1).fit(X_train, y)
# pred_NN = np.expm1(model_NN.predict(X_test))

#Adaboost Tree Classifier
# model_Adaboost = AdaBoostClassifier(n_estimators=300).fit(X_train, y)
# pred_Adaboost = np.expm1(model_Adaboost.predict(X_test))

#Gradient Boosting Classifier
# model_Boosting = GradientBoostingClassifier(n_estimators=300).fit(X_train, y)
# pred_Boosting = np.expm1(model_Boosting.predict(X_test))

#Bagging Classifier
# @model_Bagging = BaggingClassifier(n_estimators=300).fit(X_train, y)
# pred_Bagging = np.expm1(model_Bagging.predict(X_test))

# Using th ridge model
# model_Ridge = Ridge().fit(X_train, y)
# pred_Ridge = np.expm1(model_Ridge.predict(X_test))

# p =  0.2 * pred_xgb + 0.7 * pred_lasso + 0.1 * pred_Ridge

# last value
#p =  0.23 * pred_xgb + 0.77 * pred_lasso

# Boosting of both models
csv1 = pd.read_csv("./model1.csv")
csv2 = pd.read_csv("./model2.csv")
#csv1['SalePrice'] = 0.62*csv1['SalePrice']+0.38*csv2['id']
csv1['SalePrice'] = 0.65*csv1['SalePrice']+0.35*csv2['id']
# Store the final result
csv1.to_csv("final_last.csv",index=False)