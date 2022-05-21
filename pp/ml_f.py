from pp.log import logger
from pp.util import *
from pp.data import *
from pp.viz import *

#python standard libraries
#import pathlib
#from tempfile import mkdtemp
#from shutil import rmtree
#from joblib import dump

#non-standard libraries
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np

#from configparser import ConfigParser
#from PK import *
# import pickle
# from pandas.plotting import scatter_matrix as pdsm 
# from collections.abc import Iterable

# catch sklearn verbose warnings
import warnings
from sklearn.exceptions import ConvergenceWarning
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=ConvergenceWarning)
from sklearn import datasets, linear_model
from sklearn.utils import estimator_html_repr
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.compose import make_column_transformer
from sklearn.compose import ColumnTransformer
from sklearn.compose import make_column_selector as selector
from sklearn.impute import SimpleImputer
from sklearn.svm import LinearSVC
from sklearn.linear_model import RidgeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA, NMF
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import make_scorer
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_recall_curve, auc
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.feature_selection import VarianceThreshold
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
# vizualize pipeline
#from sklearn import set_config
#set_config(display='diagram')  
# for sklearn transformer caching

    
# MACHINE LEARNING 'FEATURE SELECTION' ACTIONS

@registerService()
def ML_SELECT_FEATURES_NONE_ZERO_VARIANCE(df):
    '''Select numerical features / columns with non-zero variance'''
    return DATA_COL_DELETE_EXCEPT(df, columns=_selectFeatures(df, method='VarianceThreshold'))

@registerService(
    target=OPTION_FIELD_SINGLE_COL_NUMBER,
    n=FIELD_INTEGER,
)
def ML_SELECT_FEATURES_N_BEST(df, target, n=10):
    '''Select best n numerical features / columns for classifying target column'''
    return DATA_COL_DELETE_EXCEPT(df, columns=_selectFeatures(df, method='SelectKBest', target=target, n=n))

def _selectFeatures(df, method=None, target=None, n=10):
    target = colHelper(df, target, max=1, forceReturnAsList=False)
    if method == 'VarianceThreshold':
        sel = VarianceThreshold() #remove '0 variance'
        x = df[colHelper(df, type='number')]
        sel.fit_transform(x)
        return sel.get_feature_names_out().tolist()
    elif method == 'SelectKBest':
        sel = SelectKBest(k=n)
        x = df[removeElementsFromList(colHelper(df, type='number'), [target])]
        y = df[target]
        sel.fit_transform(X=x, y=y)
        features = sel.get_feature_names_out().tolist()
        features.append(target)
        return features
'''
#@ignore_warnings
def ML_TRAIN_AND_SAVE_CLASSIFIER(self, target, path='classifier.joblib'):
    #Train a classification model for provided target, save model to specified location and display summary of model performance

    # BUILD MODEL

    # FEATURE TRANSFORMERS

    # temporary manual addition of method to SimpleImputer class
    SimpleImputer.get_feature_names_out = (lambda self, names=None:
                                   self.feature_names_in_)

    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())])

    categorical_transformer = OneHotEncoder(handle_unknown='ignore')

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, selector(dtype_include='number')),
            ('cat', categorical_transformer, selector(dtype_include=['object', 'category']))])

    # PIPELINE

    # prepare cache
    cachedir = mkdtemp()

    # Append classifier to preprocessing pipeline.
    # Now we have a full prediction pipeline.
    clf = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', LogisticRegression())
    ], memory=cachedir)

    param_grid = {
        'preprocessor__num__imputer__strategy': ['mean', 'median'],
        'classifier__C': [0.1, 1.0, 10, 100],
    }

    # SCORERS

    # The scorers can be either one of the predefined metric strings or a scorer
    # callable, like the one returned by make_scorer
    scoring = {
        'AUC': 'roc_auc', 
        'Accuracy': make_scorer(accuracy_score)
    }

    # BUILD GRID FOR PARAM SEARCH

    # Setting refit='AUC', refits an estimator on the whole dataset with the
    # parameter setting that has the best cross-validated AUC score.
    # That estimator is made available at ``gs.best_estimator_`` along with
    # parameters like ``gs.best_score_``, ``gs.best_params_`` and
    # ``gs.best_index_``

    grid = GridSearchCV(clf,
                        n_jobs=1, 
                        param_grid=param_grid, 
                        cv=10,
                        scoring=scoring, 
                        refit='AUC', 
                        return_train_score=True)

    # SPLIT & FIT!

    train_df, test_df = train_test_split(self._df, test_size=0.2, random_state=0)
    grid.fit(train_df.drop(target, axis=1), train_df[target])


    # after hard work of model fitting, we can clear pipeline/transformer cache
    rmtree(cachedir)

    # SAVE/'PICKLE' MODEL

    # generate path
    if path in self._config.sections() and 'model' in self._config[path]:
        path = self._config[path]['model']

    # save
    dump(grid.best_estimator_, path, compress = 1) 

    # force evaluation
    self._ML_EVAL_CLASSIFIER(path, target, test_df, train_df, pos_label='Yes')
    return self

def ML_EVAL_CLASSIFIER(self, target, path='classifier.joblib', pos_label='Yes'):
    #Load a save classifier model from specified location and evaluate with current dataframe data
    self._ML_EVAL_CLASSIFIER(path, target, test_df=self._df, trainf_df=None, pos_label=pos_label)
    return self

def _ML_EVAL_CLASSIFIER(self, path, target, test_df, train_df, pos_label, **kwargs):

    # generate path
    if path in self._config.sections() and 'model' in self._config[path]:
        path = self._config[path]['model']

    from joblib import load
    # load saved model again to be sure
    clf = load(path) 

    #PREPARE DATA

    # X, y columns
    X, y = self._removeElementsFromList(list(test_df.columns), target), target

    # test
    test_df['Split'] = 'test'
    test_df['Prediction'] = clf.predict(test_df[X])
    test_df['Score'] = clf.predict_proba(test_df[X])[:, 1]

    # train
    train_df['Split'] = 'train'
    train_df['Prediction'] = clf.predict(train_df[X])
    train_df['Score'] = clf.predict_proba(train_df[X])[:, 1]

    # combined test/train
    eval_df = test_df.append(train_df)

    # separately add count column, sort
    test_df.sort_values(by = 'Score', inplace=True)
    test_df.insert(0, 'Count', range(1, test_df.shape[0] + 1))
    train_df.sort_values(by = 'Score', inplace=True)
    train_df.insert(0, 'Count', range(1, train_df.shape[0] + 1))
    eval_df.sort_values(by = 'Score', inplace=True)
    eval_df.insert(0, 'Count', range(1, eval_df.shape[0] + 1))

    # CONFUSION MATRIX
    # use test
    np = confusion_matrix(test_df[y], test_df['Prediction'], labels=['No', 'Yes'], normalize='all')
    df = pd.DataFrame(data=np.ravel(), columns=['value']) 
    df['true'] = ['Negative', 'Negative', 'Positive', 'Positive']
    df['name'] = ['True negative', 'False Positive', 'False negative', 'True positive']

    # show confusion matrix as 'treemap'
    self._appendDF(df).VIZ_TREEMAP(path=['true', 'name'], 
                     values='value',
                     root='Top',
                     #width=600,
                     #height=450,
                     title='Classification Results (Confusion Matrix)')._popDF()

    # table of actual target, classifier scores and predictions based on those scores: use test
    self._appendDF(test_df).VIZ_TABLE(x=['Count', y, 'Score', 'Prediction'])
    self._figs[-1].update_layout(
        title="Classification Results (Details)",
        width=600, 
        height=450,
    ) 

    # histogram of scores compared to true labels: use test
    self.VIZ_HIST(title='Classifier score vs True labels',
                  x='Score', 
                  color=target,
                  height=400,
                  nbins=50, 
                  labels=dict(color='True Labels', x='Classifier Score')
                 )._popDF()

    # preliminary viz & roc
    fpr, tpr, thresholds = roc_curve(test_df[y], test_df['Score'], pos_label=pos_label)

    # tpr, fpr by threshold chart
    df = pd.DataFrame({
        'False Positive Rate': fpr,
        'True Positive Rate': tpr
    }, index=thresholds)
    df.index.name = "Thresholds"
    df.columns.name = "Rate"

    self._appendDF(df).VIZ_LINE(title='True Positive Rate and False Positive Rate at every threshold', 
                  width=600, 
                  height=450,
                  range_x=[0,1], 
                  range_y=[0,1],
                  markers=False
                 )._popDF()

    # roc chart
    self.VIZ_AREA(x=fpr, y=tpr,
                  #title=f'ROC Curve (AUC: %.2f)'% roc_auc_score(y_test, y_score),
                  width=600, 
                  height=450,
                  labels=dict(x='False Positive Rate', y='True Positive Rate'),
                  range_x=[0,1], 
                  range_y=[0,1],
                  markers=False
                 )

    self._figs[-1].add_shape(type='line', line=dict(dash='dash', color='firebrick'),x0=0, x1=1, y0=0, y1=1)

    precision, recall, thresholds = precision_recall_curve(test_df[target], test_df['Score'], pos_label=pos_label)

    # precision/recall chart
    self.VIZ_AREA(x=recall, y=precision,
                  title=f'Precision-Recall Curve (AUC={auc(fpr, tpr):.4f})',
                  width=600, 
                  height=450,
                  labels=dict(x='Recall', y='Precision'),
                  range_x=[0,1], 
                  range_y=[0,1],
                  markers=False)

    self._figs[-1].add_shape(type='line', line=dict(dash='dash', color='firebrick'),x0=0, x1=1, y0=0, y1=1)

    self._fig(preview = 6)
    return


# MACHINE LEARNING 'MODEL TRAINING' ACTIONS

#@ignore_warnings
def ML_TRAIN_AND_SAVE_REGRESSOR(self, target, path='classifier.joblib'):
    #Train a regression model for provided target, save model to specified location and display summary of model performance

    # BUILD MODEL

    # FEATURE TRANSFORMERS    
    # temporary manual addition of method to SimpleImputer class
    SimpleImputer.get_feature_names_out = (lambda self, names=None:
                                   self.feature_names_in_)

    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())])

    categorical_transformer = OneHotEncoder(handle_unknown='ignore')

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, selector(dtype_include='number')),
            ('cat', categorical_transformer, selector(dtype_include=['object', 'category']))])

    # PIPELINE

    # prepare cache
    cachedir = mkdtemp()

    # Append classifier to preprocessing pipeline.
    # Now we have a full prediction pipeline.
    clf = Pipeline(steps=[
        ('preprocessor', preprocessor),
        #('classifier', LogisticRegression())
        ('regressor',  LinearRegression())
    ], memory=cachedir)

    param_grid = {
        'preprocessor__num__imputer__strategy': ['mean', 'median'],
        #'classifier__C': [0.1, 1.0, 10, 100],
    }

    # SCORERS

    # The scorers can be either one of the predefined metric strings or a scorer
    # callable, like the one returned by make_scorer
    scoring = {
        'Mean_squared_error': 'neg_mean_squared_error',
        'r2': 'r2'
    }

    # BUILD GRID FOR PARAM SEARCH

    # Setting refit='AUC', refits an estimator on the whole dataset with the
    # parameter setting that has the best cross-validated AUC score.
    # That estimator is made available at ``gs.best_estimator_`` along with
    # parameters like ``gs.best_score_``, ``gs.best_params_`` and
    # ``gs.best_index_``

    grid = GridSearchCV(clf,
                        n_jobs=1, 
                        param_grid=param_grid, 
                        cv=10,
                        scoring=scoring, 
                        refit='r2', 
                        return_train_score=True)

    # PREPARE DATA & FIT!

    train_df, test_df = train_test_split(self._df, test_size=0.2, random_state=0)
    grid.fit(train_df.drop(target, axis=1), train_df[target])

    # after hard work of model fitting, we can clear pipeline/transformer cache
    rmtree(cachedir)

    # SAVE/'PICKLE' MODEL

    # generate path
    if path in self._config.sections() and 'model' in self._config[path]:
        path = self._config[path]['model']

    # save
    dump(grid.best_estimator_, path, compress = 1) 

    # force evaluation
    #return self._ML_EVAL_REGRESSOR(path, X_test, y_test, X_train, y_train)
    self._ML_EVAL_REGRESSOR(path, target=target, test_df=test_df, train_df=train_df)
    return self

def ML_EVAL_REGRESSOR(self, target, path='classifier.joblib'):
    #Evaluate a regressor with TEST data
    self._ML_EVAL_REGRESSOR(path, target, test_df=self._df, train_df=None)
    return self

def _ML_EVAL_REGRESSOR(self, path, target, test_df, train_df, **kwargs):
    #Evaluate a regressor

    # generate path
    if path in self._config.sections() and 'model' in self._config[path]:
        path = self._config[path]['model']

    from joblib import load
    # load saved model again to be sure
    clf = load(path) 

    # PREPARE DATA

    # X, y columns
    X, y = self._removeElementsFromList(list(test_df.columns), target), target

    # test
    test_df['Split'] = 'test'
    test_df['Prediction'] = clf.predict(test_df[X])
    test_df['Residual'] = test_df['Prediction'] - test_df[y]

    # train
    train_df['Split'] = 'train'
    train_df['Prediction'] = clf.predict(train_df[X])
    train_df['Residual'] = train_df['Prediction'] - train_df[y]

    # combined test/train
    eval_df = test_df.append(train_df)

    # separately add count column, sort
    test_df.sort_values(by = 'Prediction', inplace=True)
    test_df.insert(0, 'Count', range(1, test_df.shape[0] + 1))
    train_df.sort_values(by = 'Prediction', inplace=True)
    train_df.insert(0, 'Count', range(1, train_df.shape[0] + 1))
    eval_df.sort_values(by = 'Prediction', inplace=True)
    eval_df.insert(0, 'Count', range(1, eval_df.shape[0] + 1))

    #PREDICTIONS VS ACTUAL
    # scatter: use combined test/train
    self.VIZ_SCATTER(data_frame=eval_df,
                     x=y,
                     y='Prediction',
                     title='Predicted ' + y + ' vs actual ' + y,
                     width=800,
                     height=600,
                     labels={target: 'Actual '+y, 'Prediction': 'Predicted '+y},
                     marginal_x='histogram', marginal_y='histogram',
                     trendline='ols',
                     color='Split'
                     #opacity=0.65
                    )
    self._figs[-1].add_shape(
       type="line", line=dict(dash='dash'),
       x0=eval_df[y].min(), y0=eval_df[y].min(),
       x1=eval_df[y].max(), y1=eval_df[y].max()
    )
    self._figs[-1].update_yaxes(nticks=10).update_xaxes(nticks=10)

    # table of actual target, classifier scores and predictions based on those scores: use test
    self.VIZ_TABLE(data_frame=test_df,
                  x=['Count', y, 'Prediction', 'Residual'], 
                  )
    self._figs[-1].update_layout(
        title="Regression Results (Details)",
        width=600, 
        height=450,
    )

    #RESIDUALS
    # use combined train/test
    self.VIZ_SCATTER(data_frame=eval_df,
                     x='Prediction',
                     y='Residual',
                     title='Gap between predicted '+y +' and actual '+ y,
                     labels={'Prediction': 'Predicted '+y, 'Residual': 'Gap (predicted - actual)'},
                     width=800,
                     height=600,
                     marginal_y='violin',
                     trendline='ols',
                     color='Split'
                     #opacity=0.65
                    )
    self._figs[-1].update_yaxes(nticks=10).update_xaxes(nticks=10)

    # COEFFICIENT/S

    # use test

    if(len(test_df[X]) == 1):
        df = pd.DataFrame({
            'X': test_df[X].to_numpy(),
            'y': test_df[y].to_numpy()
            #'X': X_test.iloc[:, 0].to_numpy(),
            #'y': y_test.to_numpy()
        })
        self.VIZ_SCATTER(data_frame=df,
                         x='X',
                         y='y',
                      title='Regression plot (r2: TBD)',
                      width=600, 
                      height=450
                    )
        # add prediction line
        x_range = test_df[X].sort_values(by=X)
        y_range = clf.predict(x_range)
        self._figs[-1].add_traces(go.Scatter(x=x_range.iloc[:, 0].to_numpy(), y=y_range, name='Regression Fit'))

    else:
        df = pd.DataFrame({
            'X': clf.named_steps['preprocessor'].get_feature_names_out(),
            'y': clf.named_steps['regressor'].coef_
        })
        colors = ['Positive' if c > 0 else 'Negative' for c in clf.named_steps['regressor'].coef_]
        self.VIZ_BAR(
            x='X', 
            y='y',
            data_frame=df,
            color=colors,
            width=1200,
            height=600,
            #color_discrete_sequence=['red', 'blue'],
            labels=dict(x='Feature', y='Linear coefficient'),
            title='Weight of each feature when predicting '+target
        )

    self._fig(preview = 4)
    return
'''