# %% Imports
# add this logic to stop getting deprication warnings
def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn
warnings.filterwarnings("ignore")

import pandas as pd
import os
import numpy as np
from tqdm import trange

from sklearn.pipeline import Pipeline
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

from sklearn.model_selection import GridSearchCV
from sklearn.utils import shuffle
from sklearn.metrics import recall_score, accuracy_score, precision_score
from sklearn.feature_selection import RFECV
from sklearn.metrics import confusion_matrix


# %% classifiers logic
def list_explainable_classifiers():
    return [
        ('Nearest Neighbors', KNeighborsClassifier(), {'n_neighbors': [2,3,4,5]}),
        ('Linear SVM', SVC(kernel="linear", C=5, random_state=seed), {'C': [0.025,0.05,0.1,1], 'class_weight': ['balanced', None]}),
        ('RBF SVM', SVC(gamma=2, C=1, random_state=seed), {'gamma': ['scale','auto', 0.1, 1, 10], 'C': [0.01,0.1,1,10], 'class_weight': ['balanced', None]}),
        ('Decision Tree', DecisionTreeClassifier(random_state=seed), {'max_depth': [3, 5, None], 'class_weight': ['balanced', None]}),
        ('Naive Bayes', GaussianNB(), {}),
        ('Logistic Regression', LogisticRegression(random_state=seed), {'penalty': ['l1', 'l2'], 'fit_intercept': [True, False], 'class_weight': ['balanced', None]})
    ]


def list_advanced_classifiers():
    return [
        ('Random Forest', RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1, random_state=seed), {'max_depth':[5, 10], 'n_estimators':[5, 10, 15], 'max_features':[2,5,10]}),
        ('Neural Net', MLPClassifier(alpha=1, random_state=seed), {'alpha':[0.1, 1, 2, 5]}),
        ('AdaBoost', AdaBoostClassifier(random_state=seed), {}),
        ('LDA', LinearDiscriminantAnalysis(), {}),
        ('QDA', QuadraticDiscriminantAnalysis(), {}),
        ('GaussianProcess', GaussianProcessClassifier(kernel=1.0*RBF(1.0), warm_start=True, random_state=seed), {}),
        ('Gradient Boosting', GradientBoostingClassifier(max_features=None, random_state=seed), {'learning_rate': [0.05, 0.1], 'subsample': [0.5, 0.75, 1], 'max_depth': [3, 5], 'n_estimators': [75, 100]})
    ]


# standart classifier search over predefined ranges of parameters
def classifier_search(classifiers, df_train, df_train_label, df_test, df_test_label, df_res, use_cv=True, cv_folds=5, compute_train_metrics=False):
    opt_clf_list = []
    pred_cols = []

    for name, clf, params in classifiers:
        opt_clf = {}
        opt_clf['name'] = name
        if use_cv:
            # , scoring='recall'
            clf_search = GridSearchCV(clf, params, cv=cv_folds, scoring='recall')
            clf_search.fit(df_train, df_train_label)

            predictions = clf_search.best_estimator_.predict(df_test)
            train_predictions = clf_search.best_estimator_.predict(df_train)

            df_res[name] = predictions
            pred_cols.append(name)

            opt_clf['clf'] = clf_search.best_estimator_
            opt_clf['params'] = clf_search.best_params_

        else:
            clf.fit(df_train, df_train_label)
            predictions = clf.predict(df_test)

            clf_search = clf
            opt_clf['clf'] = clf

        tn, fp, fn, tp = confusion_matrix(df_test_label, predictions).ravel()
        opt_clf['accuracy']    = accuracy_score(df_test_label, predictions)
        opt_clf['recall']      = recall_score(df_test_label, predictions)
        opt_clf['precision']   = precision_score(df_test_label, predictions)
        opt_clf['sensitivity'] = tp / (tp+fn)
        opt_clf['specificity'] = tn / (tn+fp)

        if compute_train_metrics:
            train_tn, train_fp, train_fn, train_tp = confusion_matrix(df_train_label, train_predictions).ravel()
            opt_clf['train_accuracy'] = accuracy_score(df_train_label, train_predictions)
            opt_clf['train_recall'] = recall_score(df_train_label, train_predictions)
            opt_clf['train_precision'] = precision_score(df_train_label, train_predictions)
            opt_clf['train_sensitivity'] = train_tp / (train_tp+train_fn)
            opt_clf['train_specificity'] = train_tn / (train_tn+train_fp)


        opt_clf_list.append(opt_clf)

    return opt_clf_list


# ensemble of classifiers
def classifier_ensemble(df_train, df_train_label, df_test, df_test_label, df_res):
    # ensemble of previous models
    name = "ensemble"
    ens_clf = {}
    ens_clf['name'] = 'ensemble'

    sums = df_test.replace({False: 0, True: 1}).values[:, 3:].sum(1)
    predictions = [i > 0 for i in sums.values]
    df_res['ensemble'] = predictions

    tn, fp, fn, tp = confusion_matrix(df_test_label, predictions).ravel()
    ens_clf['accuracy']    = accuracy_score(df_test_label, predictions)
    ens_clf['recall']      = recall_score(df_test_label, predictions)
    ens_clf['precision']   = precision_score(df_test_label, predictions)
    ens_clf['sensitivity'] = tp / (tp+fn)
    ens_clf['specificity'] = tn / (tn+fp)

    return ens_clf, df_res


# printing an output of classification results
def print_output(opt_clf_list, show_training=True):
    for item in opt_clf_list:
        if 'params' in item:
            print(item['params'])
        print('{0:25} accuracy = {1:<6.3};  recall = {2:<6.3}; precision = {3:<6.3} '.format(item['name'], item['accuracy'], item['recall'], item['precision']))
        if show_training:
            print('{0:25} accuracy = {1:<6.3};  recall = {2:<6.3}; precision = {3:<6.3} '.format('   on training data', item['train_accuracy'], item['train_recall'], item['train_precision']))
        print('')


#%% stratified train/test split: splits class_col column with balanced labels and then replicates split to the data
def train_test_split_by_class(df, class_col, label_col, random_seed=0, train_size=0.75):
    # patients_train, patients_test = train_test_split(df[class_col].unique(), train_size=train_size, random_state=random_seed)

    df_patients = df[[class_col, label_col]].groupby(class_col).max().reset_index()
    patients_train, patients_test = train_test_split(df_patients, stratify=df_patients[label_col], train_size=train_size, random_state=random_seed)

    # print(patients_train)
    # print(patients_test)

    df_train = df[df[class_col].isin(patients_train[class_col])]
    df_train = shuffle(df_train, random_state=random_seed)

    df_test = df[df[class_col].isin(patients_test[class_col])]
    df_test = shuffle(df_test, random_state=random_seed)

    return df_train, df_test

# splits class_col column into 1 for testing and everytginf else for training
def train_test_split_LOO(df, class_col, ind=0, random_seed=0, train_size=0.75):
    class_key = df[class_col].unique()[ind]
    # patients_train = df[df[class_col]!=class_key][class_col]
    patients_train = [item for item in df[class_col].unique() if item != class_key]
    patients_test = [class_key]

    # print(patients_train)
    print(patients_test)

    df_train = df[df[strat_col].isin(patients_train)]
    df_train = shuffle(df_train, random_state=random_seed)

    df_test = df[df[strat_col].isin(patients_test)]
    df_test = shuffle(df_test, random_state=random_seed)

    return df_train, df_test


def select_most_relevant_features(df, feature_cols, label_col, strat_col, clf, N):
    selected_cols = {}

    for i in trange(N):
        df_train, df_test = train_test_split_by_class(df, class_col=strat_col, label_col=label_col, random_seed=seed+i, train_size=train_size)

        selector = RFECV(clf, min_features_to_select=1)
        selector = selector.fit(df_train[feature_cols], df_train[label_col])
        for i in range(len(feature_cols)):
        # for feature in feature_cols:
            if selector.support_[i]:
                feature = feature_cols[i]
                if feature in selected_cols:
                    selected_cols[feature] += 1
                else:
                    selected_cols[feature] = 1

    return selected_cols


def run_simple_classifier(df, feature_cols, label_col, strat_col):
    df_train, df_test = train_test_split_by_class(df, class_col=strat_col, label_col=label_col, random_seed=seed, train_size=train_size)

    df_res = df_test[['short_name', 'roi_number', label_col]]
    opt_clf_list = classifier_search(classifiers, df_train[feature_cols], df_train[label_col], df_test[feature_cols], df_test[label_col], df_res, cv_folds=cv_folds)

    return opt_clf_list


def run_classifier_with_nested_cv(df, N, feature_cols, label_col, strat_col):
    res_lists = []

    for i in trange(N):
        df_train, df_test = train_test_split_by_class(df, class_col=strat_col, label_col=label_col, random_seed=seed+i, train_size=train_size)

        df_pred        = df_test[['short_name', 'roi_number', label_col]]
        opt_clf_list = classifier_search(classifiers, df_train[feature_cols], df_train[label_col], df_test[feature_cols], df_test[label_col], df_pred, cv_folds=cv_folds)

        df_res = pd.DataFrame(opt_clf_list)
        df_res['seed'] = seed+i
        res_lists.append(df_res)

    df_res = pd.concat(res_lists)
    columns = ['name', 'accuracy', 'recall', 'precision', 'sensitivity', 'specificity']
    df_scores = df_res[columns].groupby(by='name', axis=0).agg(['mean','std'])
    df_scores[('total','mean')] = df_scores.xs('mean', axis=1, level=1, drop_level=False).mean(axis=1)
    df_scores[('total','std')] = df_scores.xs('std', axis=1, level=1, drop_level=False).mean(axis=1)

    return df_scores


# %%  read file with labeled data
folder_path = '/Users/mykhaylo.zayats/perfusion'

file_with_features = 'data/data_example.csv'

file_path = os.path.join(folder_path, file_with_features)
df = pd.read_csv(file_path, encoding='utf-8')

df.head(5)

# %% classifiers
seed = 1
np.random.seed(seed)

classifiers = list_explainable_classifiers()

use_cv = True
cv_folds = 10
train_size = 0.75
N = 20    # number of repeats (nested CV) 

label_map = {
    'Cancer' : 1,
    'Healthy': 0,
    'Benign' : 0
}
df['label'] = df['finding'].apply(lambda x: label_map[x])

df = df[df['finding']!='Healthy']         # classifying cancer vs benign series
# df = df[df['finding']!='Benign']        # classifying cancer vs healthy series

# filter out some patients 
df=df[df['short_name']!='IBM 36']

stratify_split = 'by_patient'
feature_cols = df.columns[2:-2].tolist()
label_col    = 'label'
strat_col = 'short_name'

# %% run simple classification
df_sub = df.dropna()
opt_clf_list = run_simple_classifier(df_sub, feature_cols, label_col, strat_col)
print_output(opt_clf_list, show_training=False)

# %% select most informative features
feature_cols = df.columns[2:-2].tolist()

# choose which classifier to use
# clf_ind = 1   # Linear SVM
clf_ind = 3   # decision tree
# clf_ind = 5   # logistic regression
clf = opt_clf_list[clf_ind]['clf']

init_len = len(feature_cols) + 1
cont = True
while cont:
    features_dict = select_most_relevant_features(df_sub, feature_cols, label_col, strat_col, clf, N)

    features_dict = {key:features_dict[key] for key in features_dict if features_dict[key] > N/2.0}
    features_dict = dict(sorted(features_dict.items(), key=lambda x:x[1], reverse=True))

    init_len = len(feature_cols)
    print(f'step: selected {len(features_dict)} out of {init_len} features')
    print(features_dict)

    feature_cols = list(features_dict.keys())
    # cont = init_len > len(feature_cols)
    cont = False

# %% randomised over multiple runs train/test splits, training and metrics
N = 20
print(feature_cols)
df_scores = run_classifier_with_nested_cv(df_sub, N, feature_cols, label_col, strat_col)
print(df_scores)

# %% running feature selection by selecting optimal pair using averaged across multiple runs metrics
feature_pairs = []
for i in range(len(feature_cols)):
    for j in range(i+1, len(feature_cols)):
        feature_pairs.append([feature_cols[i], feature_cols[j]])

df_scores_list = []
for ind in range(len(feature_pairs)):
    feature_cols = feature_pairs[ind]

    df_scores = run_classifier_with_nested_cv(df_sub, N, feature_pairs[ind], label_col, strat_col)
    df_scores_list.append(df_scores)

# %% computing performance metrics
df_metrics = pd.DataFrame(columns=['features', 'best_total', 'best_total_mean', 'best_total_std', 'best_avg', 'best_avg_mean', 'best_avg_std', 'best_accuracy', 'best_accuracy_mean', 'best_accuracy_std', 'best_recall', 'best_recall_mean', 'best_recall_std', 'best_precision', 'best_precision_mean', 'best_precision_std', 'best_specificity', 'best_specificity_mean', 'best_specificity_std', 'data'])

for ind in range(len(feature_pairs)):
    df_scores = df_scores_list[ind]

    # df_scores.drop(('total',''), axis=1, inplace=True)
    df_scores[('total','mean')] = df_scores.xs('mean', axis=1, level=1, drop_level=False).mean(axis=1)
    df_scores[('total','std')] = df_scores.xs('std', axis=1, level=1, drop_level=False).mean(axis=1)

    df_scores[('avg','mean')] = df_scores[[('accuracy','mean'),('recall','mean'),('specificity','mean')]].mean(axis=1)
    df_scores[('avg','std')] = df_scores[[('accuracy','mean'),('recall','mean'),('specificity','mean')]].std(axis=1)

    df_metrics.at[ind,'features'] = feature_pairs[ind]

    df_metrics.at[ind,'best_total'] = df_scores[('total','mean')].idxmax()
    df_metrics.at[ind,'best_total_mean'] = df_scores[('total','mean')].max()
    df_metrics.at[ind,'best_total_std'] = df_scores[('total','std')][df_scores[('total','mean')].idxmax()]

    df_metrics.at[ind,'best_avg'] = df_scores[('avg','mean')].idxmax()
    df_metrics.at[ind,'best_avg_mean'] = df_scores[('avg','mean')].max()
    df_metrics.at[ind,'best_avg_std'] = df_scores[('avg','std')][df_scores[('total','mean')].idxmax()]

    df_metrics.at[ind,'best_accuracy'] = df_scores[('accuracy','mean')].idxmax()
    df_metrics.at[ind,'best_accuracy_mean'] = df_scores[('accuracy','mean')].max()
    df_metrics.at[ind,'best_accuracy_std'] = df_scores[('accuracy','std')][df_scores[('accuracy','mean')].idxmax()]

    df_metrics.at[ind,'best_recall'] = df_scores[('recall','mean')].idxmax()
    df_metrics.at[ind,'best_recall_mean'] = df_scores[('recall','mean')].max()
    df_metrics.at[ind,'best_recall_std'] = df_scores[('recall','std')][df_scores[('recall','mean')].idxmax()]

    df_metrics.at[ind,'best_precision'] = df_scores[('precision','mean')].idxmax()
    df_metrics.at[ind,'best_precision_mean'] = df_scores[('precision','mean')].max()
    df_metrics.at[ind,'best_precision_std'] = df_scores[('precision','std')][df_scores[('precision','mean')].idxmax()]

    df_metrics.at[ind,'best_specificity'] = df_scores[('specificity','mean')].idxmax()
    df_metrics.at[ind,'best_specificity_mean'] = df_scores[('specificity','mean')].max()
    df_metrics.at[ind,'best_specificity_std'] = df_scores[('specificity','std')][df_scores[('specificity','mean')].idxmax()]

    df_metrics.at[ind,'data'] = df_scores
