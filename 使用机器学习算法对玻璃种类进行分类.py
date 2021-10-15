# -*- encoding = utf-8 -*-
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.base import TransformerMixin
from sklearn.preprocessing import (FunctionTransformer, StandardScaler)
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from scipy.stats import boxcox
from sklearn.model_selection import (train_test_split, KFold , StratifiedKFold,
                                     cross_val_score, GridSearchCV,
                                     learning_curve, validation_curve)
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from collections import Counter
import warnings

from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression

from xgboost import (XGBClassifier, plot_importance)
from sklearn.svm import SVC
from sklearn.ensemble import (RandomForestClassifier, AdaBoostClassifier,
                              ExtraTreesClassifier, GradientBoostingClassifier)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from time import time

warnings.filterwarnings('ignore')
sns.set_style('whitegrid')

#读数据
import pandas as pd
df = pd.read_csv(r'glass.csv')
features = df.columns[:-1].tolist()
# print(features)
# print(df.shape)
# print(df.head(15))
# print(df.dtypes)
# print(df.describe())
# print(df['Type'].value_counts())
#
# for feat in features:
#     skew = df[feat].skew() #偏度；偏度大于0，重尾在右边，偏度等于0时，数据是正态分布
#     sns.distplot(df[feat], kde= False, label='Skew = %.3f' %(skew), bins=30)
#     plt.legend(loc='best')
#     plt.show()

#
def outlier_hunt(df):
    """
    选出错误数据 标准是如果该条数据含有两个及以上的偏离值特大的值，
    就将该数据选出
    """
    outlier_indices = []

    for col in df.columns.tolist():
        Q1 = np.percentile(df[col], 25)
        Q3 = np.percentile(df[col], 75)
        IQR = Q3 - Q1
        outlier_step = 1.5 * IQR
        outlier_list_col = df[(df[col] < Q1 - outlier_step) | (df[col] > Q3 + outlier_step)].index
        outlier_indices.extend(outlier_list_col)

    outlier_indices = Counter(outlier_indices)
    # print(outlier_indices.items())
    multiple_outliers = list(k for k, v in outlier_indices.items() if v > 2)
    # print(multiple_outliers)
    return multiple_outliers

#
# print('The dataset contains %d observations with more than 2 outliers'
#       % (len(outlier_hunt(df[features]))))

# plt.figure(figsize=(8,6))
# sns.boxplot(orient='v',data=df[features])
# plt.show()
#
# plt.figure(figsize=(8,8))
# sns.pairplot(df[features],palette='coolwarm')
# plt.show()
#
# corr = df[features].corr()
# plt.figure(figsize=(16,16))
# sns.heatmap(corr, cbar = True,  square = True, annot=True, fmt= '.2f',annot_kws={'size': 15},
#            xticklabels= features, yticklabels= features, alpha = 0.7,   cmap= 'coolwarm')
# plt.show()
# print(df.info())
# #把不合格的数据删除
outlier_indices = outlier_hunt(df[features])
print(outlier_indices)
df = df.drop(outlier_indices).reset_index(drop=True)
print(df.shape)
#
# for feat in features:
#     skew = df[feat].skew()
#     sns.distplot(df[feat], kde=False, label='Skew = %.3f' %(skew), bins=30)
#     plt.legend(loc='best')
#     plt.show()
#
# print(df['Type'].value_counts())
# sns.countplot(df['Type'])
# plt.show()
#
# X = df[features]
# y = df['Type']
# seed = 7
# test_size = 0.2
#
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = test_size , random_state = seed)
#
#
# features_boxcox = []
# for feature in features:
#     bc_transformed, _ = boxcox(df[feature]+1)  #加1，保证值大于0，然后可以用boxcox使数据正态化
#     features_boxcox.append(bc_transformed)
# features_boxcox = np.column_stack(features_boxcox)
# df_bc = pd.DataFrame(data=features_boxcox, columns=features)
# df_bc['Type'] = df['Type']
# print(df_bc.describe())
#
#
# for feature in features:
#     fig, ax = plt.subplots(1,2,figsize=(7,3.5))
#     ax[0].hist(df[feature], color='blue', bins=30, alpha=0.3, label='Skew = %s' %(str(round(df[feature].skew(),3))) )
#     ax[0].set_title(str(feature))
#     ax[0].legend(loc=0)
#     ax[1].hist(df_bc[feature], color='red', bins=30, alpha=0.3, label='Skew = %s' %(str(round(df_bc[feature].skew(),3))) )
#     ax[1].set_title(str(feature)+' after a Box-Cox transformation')
#     ax[1].legend(loc=0)
#     plt.show()
#
# for feature in features:
#     delta = np.abs( df_bc[feature].skew() / df[feature].skew() )
#     if delta < 1.0 :
#         print('Feature %s is less skewed after a Box-Cox transform' %(feature))
#     else:
#         print('Feature %s is more skewed after a Box-Cox transform'  %(feature))
#
#     model_importances = XGBClassifier()
#     start = time()
#     model_importances.fit(X_train, y_train)
#     print('Elapsed time to train XGBoost  %.3f seconds' % (time() - start))
#     plot_importance(model_importances)
#     plt.show()
#
#     pca = PCA(random_state=seed)
#     pca.fit(X_train)
#     var_exp = pca.explained_variance_ratio_
#     cum_var_exp = np.cumsum(var_exp)
#     plt.figure(figsize=(8, 6))
#     plt.bar(range(1, len(cum_var_exp) + 1), var_exp, align='center', label='individual variance explained', \
#             alpha=0.7)
#     plt.step(range(1, len(cum_var_exp) + 1), cum_var_exp, where='mid', label='cumulative variance explained', \
#              color='red')
#     plt.ylabel('Explained variance ratio')
#     plt.xlabel('Principal components')
#     plt.xticks(np.arange(1, len(var_exp) + 1, 1))
#     plt.legend(loc='center right')
#     plt.show()
#
#     for i, sum in enumerate(cum_var_exp):
#         print("PC" + str(i + 1), "累计方差: %.3f% %" % (cum_var_exp[i] * 100))
#
# n_components = 5
# pipelines = []
# n_estimators = 200
#
# pipelines.append( ('SVC',
#                    Pipeline([
#                               ('sc', StandardScaler()),
# #                               ('pca', PCA(n_components = n_components, random_state=seed ) ),
#                              ('SVC', SVC(random_state=seed))])
#                    )
#                   )
#
#
# pipelines.append(('KNN',
#                   Pipeline([
#                               ('sc', StandardScaler()),
# #                             ('pca', PCA(n_components = n_components, random_state=seed ) ),
#                             ('KNN', KNeighborsClassifier()) ])))
# pipelines.append( ('RF',
#                    Pipeline([
#                               ('sc', StandardScaler()),
# #                              ('pca', PCA(n_components = n_components, random_state=seed ) ),
#                              ('RF', RandomForestClassifier(random_state=seed, n_estimators=n_estimators)) ]) ))
#
#
# pipelines.append( ('Ada',
#                    Pipeline([
#                               ('sc', StandardScaler()),
# #                              ('pca', PCA(n_components = n_components, random_state=seed ) ),
#                     ('Ada', AdaBoostClassifier(random_state=seed,  n_estimators=n_estimators)) ]) ))
#
# pipelines.append( ('ET',
#                    Pipeline([
#                               ('sc', StandardScaler()),
# #                              ('pca', PCA(n_components = n_components, random_state=seed ) ),
#                              ('ET', ExtraTreesClassifier(random_state=seed, n_estimators=n_estimators)) ]) ))
# pipelines.append( ('GB',
#                    Pipeline([
#                              ('sc', StandardScaler()),
# #                             ('pca', PCA(n_components = n_components, random_state=seed ) ),
#                              ('GB', GradientBoostingClassifier(random_state=seed)) ]) ))
#
# pipelines.append( ('LR',
#                    Pipeline([
#                               ('sc', StandardScaler()),
# #                               ('pca', PCA(n_components = n_components, random_state=seed ) ),
#                              ('LR', LogisticRegression(random_state=seed)) ]) ))
#
# results, names, times  = [], [] , []
# num_folds = 10
# scoring = 'accuracy'
#
# for name, model in pipelines:
#     start = time()
#     kfold = StratifiedKFold(n_splits=num_folds,)
#     cv_results = cross_val_score(model, X_train, y_train, cv=kfold,
#                                  scoring = scoring,
#                                 n_jobs=-1)
#     t_elapsed = time() - start
#     results.append(cv_results)
#     names.append(name)
#     times.append(t_elapsed)
#     msg = "%s: %f (+/- %f) performed in %f seconds" % (name, 100*cv_results.mean(),
#                                                        100*cv_results.std(), t_elapsed)
#     print(msg)
#
# fig = plt.figure(figsize=(12,8))
# fig.suptitle("Algorithms comparison")
# ax = fig.add_subplot(1,1,1)
# plt.boxplot(results)
# ax.set_xticklabels(names)
# plt.show()
#
# pipe_rfc = Pipeline([
#                       ('scl', StandardScaler()),
#                     ('rfc', RandomForestClassifier(random_state=seed, n_jobs=-1) )])
#
#
# param_grid_rfc =  [ {
#     'rfc__n_estimators': [100, 200,300,400],
#     'rfc__max_features':[0.05 , 0.1],
#     'rfc__max_depth': [None, 5],
#     'rfc__min_samples_split': [0.005, 0.01],
#     }]
# #kfold = StratifiedKFold(n_splits=num_folds, random_state= seed)
# kfold = StratifiedKFold(n_splits=num_folds)
# grid_rfc = GridSearchCV(pipe_rfc, param_grid= param_grid_rfc, cv=kfold, scoring=scoring, verbose= 1, n_jobs=-1)
#
# start = time()
# grid_rfc = grid_rfc.fit(X_train, y_train)
# end = time()
#
# print("RFC grid search took %.3f seconds" %(end-start))
#
# print('-------Best score----------')
# print(grid_rfc.best_score_ * 100.0)
# print('-------Best params----------')
# print(grid_rfc.best_params_)
#
#
# def plot_learning_curve(train_sizes, train_scores, test_scores, title, alpha=0.1):
#     train_mean = np.mean(train_scores, axis=1)
#     train_std = np.std(train_scores, axis=1)
#     test_mean = np.mean(test_scores, axis=1)
#     test_std = np.std(test_scores, axis=1)
#     plt.plot(train_sizes, train_mean, label='train score', color='blue', marker='o')
#     plt.fill_between(train_sizes, train_mean + train_std,
#                      train_mean - train_std, color='blue', alpha=alpha)
#     plt.plot(train_sizes, test_mean, label='test score', color='red', marker='o')
#     plt.fill_between(train_sizes, test_mean + test_std, test_mean - test_std, color='red', alpha=alpha)
#     plt.title(title)
#     plt.xlabel('Number of training points')
#     plt.ylabel('Accuracy')
#     plt.grid(ls='--')
#     plt.legend(loc='best')
#     plt.show()
#
#
# def plot_validation_curve(param_range, train_scores, test_scores, title, alpha=0.1):
#     train_mean = np.mean(train_scores, axis=1)
#     train_std = np.std(train_scores, axis=1)
#     test_mean = np.mean(test_scores, axis=1)
#     test_std = np.std(test_scores, axis=1)
#     plt.plot(param_range, train_mean, label='train score', color='blue', marker='o')
#     plt.fill_between(param_range, train_mean + train_std,
#                      train_mean - train_std, color='blue', alpha=alpha)
#     plt.plot(param_range, test_mean, label='test score', color='red', marker='o')
#     plt.fill_between(param_range, test_mean + test_std, test_mean - test_std, color='red', alpha=alpha)
#     plt.title(title)
#     plt.grid(ls='--')
#     plt.xlabel('Parameter value')
#     plt.ylabel('Accuracy')
#     plt.legend(loc='best')
#     plt.show()
#

import warnings
warnings.filterwarnings("ignore")
# plt.figure(figsize=(9,6))
#
# train_sizes, train_scores, test_scores = learning_curve(
#               estimator= grid_rfc.best_estimator_ , X= X_train, y = y_train,
#                 train_sizes=np.arange(0.1,1.1,0.1), cv= 10,  scoring='accuracy', n_jobs= - 1)
#
# plot_learning_curve(train_sizes, train_scores, test_scores, title='Learning curve for RFC')
#
#
