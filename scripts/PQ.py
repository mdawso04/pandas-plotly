import pandas as pd
from pandas.plotting import scatter_matrix as pdsm 
#import numpy as np
import pathlib
#import matplotlib.pyplot as plt
from collections.abc import Iterable
import plotly.express as px
import plotly.graph_objects as go
from PK import *
from configparser import ConfigParser

import numpy as np
from sklearn import datasets, linear_model
# vizualize pipeline
from sklearn import set_config
set_config(display='diagram')  

# for transformer caching
from tempfile import mkdtemp
from shutil import rmtree
import pickle

#from sklearn.utils._testing import ignore_warnings
#from sklearn.exceptions import ConvergenceWarning
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

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

# ## PQ CLASS - QUERY INTERFACE ###

class SOURCE(object):
    
    def __init__(self, source):
        super(SOURCE, self).__init__()
        file_ext = pathlib.Path(source).suffix
        self._config = ConfigParser()
        self._config.read('config.ini', encoding='utf_8')
            
        # config.ini 'kintone app' section
        if source in self._config.sections() and 'kintone_domain' in self._config[source]:
            domain = self._config.get(source,'kintone_domain')
            app_id = self._config.getint(source,'app_id')
            api_token = self._config.get(source,'api_token')
            model_csv = self._config.get(source,'model_csv')
        
            m = jsModelFactory.get(model_csv)
            js = JinSapo(domain=domain, app_id=app_id, api_token=api_token, model=m)
            self._df = js.select_df()
        
        # config.ini 'csv' section
        elif source in self._config.sections() and 'csv' in self._config[source]:
            self._df = pd.read_csv(self._config[source]['csv'])
            
        # config.ini 'xlsx' section
        elif source in self._config.sections() and 'xlsx' in self._config[source]:
            self._df = pd.read_csv(self._config[source]['xlsx'])
        
        # csv
        elif file_ext == '.csv':
            self._df = pd.read_csv(source)
            
        # excel
        elif file_ext == '.xlsx':
            self._df = pd.read_excel(source)
        
        else:
            self._df = pd.DataFrame()
        self._showFig = False
        self._preview = 'no_chart' #'current_chart' 'all_charts' 'full' 'color_swatches'
        self._colorSwatch = px.colors.qualitative.Plotly
        
        self._figs = []
        
        self._fig_config =  {
            'displaylogo': False,
            'toImageButtonOptions': {
                'format': 'png', # one of png, svg, jpeg, webp
                'filename': 'custom_image',
                'height': None,
                'width': None,
                'scale': 5 # Multiply title/legend/axis/canvas sizes by this factor
            },
            'edits': {
                'axisTitleText': True,
                'legendPosition': True,
                'legendText': True,
                'titleText': True,
                'annotationPosition': True,
                'annotationText': True
            }
        }
        
    def _repr_pretty_(self, p, cycle): 
        if self._preview == 'current_chart':
            return self._figs[-1].show(config=self._fig_config), display(self._df)
        elif self._preview == 'all_charts':
            return tuple([f.show(config=self._fig_config) for f in self._figs]), display(self._df)
        elif self._preview == 'full':
            return tuple([f.show(config=self._fig_config) for f in self._figs]), display(self._df), display(self._df.info())
        elif self._preview == 'color_swatches':
            return px.colors.qualitative.swatches().show(), display(self._df)
        else:
            return display(self._df)
        
    def __repr__(self): 
        return self._df.__repr__()
    
    def __str__(self): 
        return self._df.__str__()
    
    def _fig(self, fig = None, preview = 'no_chart'):
        if fig == None:
            self._preview = preview
        else:
            self._figTidy(fig)
            self._figs.append(fig)
            self._preview = 'current_chart'
            
    def _figTidy(self, fig):
        #fig.update_traces()
        fig.update_layout(
            overwrite=True,
            #colorway=self._colorSwatch,
            dragmode='drawopenpath',
            #newshape_line_color='cyan',
            #title_text='Draw a path to separate versicolor and virginica',
            modebar_add=['drawline',
                'drawcircle',
                'drawrect',
                'eraseshape',
                'pan2d'
            ],
            modebar_remove=['resetScale', 'lasso2d'] #'select', 'zoom', 
        )
        #fig.update_annotations()
        #fig.update_xaxes()
            
    # DATAFRAME 'COLUMN' ACTIONS
    
    def DF_COL_ADD_FIXED(self, value, name = 'new_column'):
        '''Add a new column with a 'fixed' value as content'''
        name = self._toUniqueColName(name)
        self._df[name] = value
        self._fig()
        return self
    
    def DF_COL_ADD_INDEX(self, name = 'new_column', start = 1):
        '''Add a new column with a index/serial number as content'''
        name = self._toUniqueColName(name)
        self._df[name] = range(start, self._df.shape[0] + start)
        self._fig()
        return self
    
    def DF_COL_ADD_INDEX_FROM_0(self, name = 'new_column'):
        '''Convenience method for DF_COL_ADD_INDEX'''
        return self.DF_COL_ADD_INDEX(name, start = 0)
    
    def DF_COL_ADD_INDEX_FROM_1(self, name = 'new_column'):
        '''Convenience method for DF_COL_ADD_INDEX'''
        return self.DF_COL_ADD_INDEX(name, start = 1)
    
    def DF_COL_ADD_CUSTOM(self, column, lmda, name = 'new_column'):
        '''Add a new column with custom (lambda) content'''
        name = self._toUniqueColName(name)
        self._df[name] = self._df[column].apply(lmda)
        self._fig()
        return self
    
    def DF_COL_ADD_EXTRACT_POSITION_AFTER(self, column, pos, name = 'new_column'):
        '''Add a new column with content extracted from after char pos in existing column'''
        self._df = self.DF_COL_ADD_CUSTOM(self._df, column, lambda x: x[pos:], name = name)
        self._fig()
        return self
    
    def DF_COL_ADD_EXTRACT_POSITION_BEFORE(self, column, pos, name = 'new_column'):
        '''Add a new column with content extracted from before char pos in existing column'''
        self._df = self.DF_COL_ADD_CUSTOM(self._df, column, lambda x: x[:pos], name = name)
        self._fig()
        return self
    
    def DF_COL_ADD_EXTRACT_CHARS_FIRST(self, column, chars, name = 'new_column'):
        '''Add a new column with first N chars extracted from column'''
        self._df = self.DF_COL_ADD_CUSTOM(self._df, column, lambda x: x[:chars], name = name)
        self._fig()
        return self
    
    def DF_COL_ADD_EXTRACT_CHARS_LAST(self, column, chars, name = 'new_column'):
        '''Add a new column with last N chars extracted from column'''
        self._df = self.DF_COL_ADD_CUSTOM(self._df, column, lambda x: x[-chars:], name = name)
        self._fig()
        return self
    
    def DF_COL_ADD_DUPLICATE(self, column, name = 'new_column'):
        '''Add a new column by copying an existing column'''
        name = self._toUniqueColName(name)
        self._df[name] = self._df[column]
        self._fig()
        return self
    
    def DF_COL_DELETE(self, columns):
        '''Delete specified column/s'''
        columns = self._colHelper(columns)
        self._df = self._df.drop(columns, axis = 1)
        self._fig()
        return self
    
    def DF_COL_DELETE_EXCEPT(self, columns):
        '''Deleted all column/s except specified'''
        columns = self._colHelper(columns)
        cols = self._removeElementsFromList(self._df.columns.values.tolist(), columns)
        self._fig()
        return self.DF_COL_DELETE(cols).DF_COL_MOVE_TO_FRONT(columns)
    
    def DF_COL_MOVE_TO_FRONT(self, columns):
        '''Move specified column/s to new index'''
        colsToMove = self._colHelper(columns)
        otherCols = self._removeElementsFromList(self._df.columns.values.tolist(), colsToMove)
        self._df = self._df[colsToMove + otherCols]
        self._fig()
        return self
    
    def DF_COL_MOVE_TO_BACK(self, columns):
        '''Move specified column/s to new index'''
        colsToMove = self._colHelper(columns)
        otherCols = self._removeElementsFromList(self._df.columns.values.tolist(), colsToMove)
        self._df = self._df[otherCols + colsToMove]
        self._fig()
        return self
    
    def DF_COL_RENAME(self, columns):
        '''Rename specfied column/s'''
        # we handle dict for all or subset, OR list for all
        if isinstance(columns, dict):
            self._df.rename(columns = columns, inplace = True)
        else:
            self._df.columns = columns
        self._fig()
        return self
    
    #col_reorder list of indices, list of colnames
    
    def DF_COL_FORMAT_TO_UPPERCASE(self, columns = None):
        '''Format specified column/s values to uppercase'''
        if columns == None: columns = self._df.columns.values.tolist()
        self._df[columns] = self._df[columns].apply(lambda s: s.str.upper(), axis=0)
        self._fig()
        return self
    
    def DF_COL_FORMAT_TO_LOWERCASE(self, columns = None):
        '''Format specified column/s values to lowercase'''
        if columns == None: columns = self._df.columns.values
        self._df[columns] = self._df[columns].apply(lambda s: s.str.lower(), axis=0)
        self._fig()
        return self
    
    def DF_COL_FORMAT_TO_TITLECASE(self, columns = None):
        '''Format specified column/s values to titlecase'''
        if columns == None: columns = self._df.columns.values
        self._df[columns] = self._df[columns].apply(lambda s: s.str.title(), axis=0)
        self._fig()
        return self
    
    def DF_COL_FORMAT_STRIP(self, columns = None):
        '''Format specified column/s values by stripping invisible characters'''
        if columns == None: columns = self._df.columns.values
        self._df[columns] = self._df[columns].apply(lambda s: s.str.strip(), axis=0)
        self._fig()
        return self
    
    def DF_COL_FORMAT_STRIP_LEFT(self, columns = None):
        '''Convenience method for DF_COL_FORMAT_STRIP'''
        df = self._df
        if columns == None: columns = df.columns.values
        df[columns] = df[columns].apply(lambda s: s.str.lstrip(), axis=0)
        self._fig()
        return self
    
    def DF_COL_FORMAT_STRIP_RIGHT(self, columns = None):
        '''Convenience method for DF_COL_FORMAT_STRIP'''
        if columns == None: columns = self._df.columns.values
        self._df[columns] = self._df[columns].apply(lambda s: s.str.rstrip(), axis=0)
        self._fig()
        return self
    
    def DF_COL_FORMAT_ADD_PREFIX(self, prefix, column):
        '''Format specified single column values by adding prefix'''
        self._df[column] = str(prefix) + self._df[column].astype(str)
        self._fig()
        return self
    
    def DF_COL_FORMAT_ADD_SUFFIX(self, suffix, column):
        '''Format specified single column values by adding suffix'''
        self._df[column] = self._df[column].astype(str) + str(suffix)
        self._fig()
        return self
    
    def DF_COL_FORMAT_TYPE(self, columns, typ = 'str'):
        if columns == None: 
            self._df = self._df.astype(typ)
        else:
            convert_dict = {c:typ for c in columns}
            self._df = self._df.astype(convert_dict)
        self._fig()
        return self
    
    def DF_COL_FORMAT_ROUND(self, decimals):
        '''Round numerical column values to specified decimal'''
        self._df = self._df.round(decimals)
        self._fig()
        return self
    
    # DATAFRAME 'ROW' ACTIONS
    
    #def DF_ROW_ADD(self, row, index = 0):
    #    self._df.loc[index] = row
    #    self._showFig = False
    #    return self
    
    #def DF_ROW_DELETE(self, rowNums):
    #    pos = rowNums - 1
    #    self._df.drop(self._df.index[pos], inplace=True)
    #    self._showFig = False
    #    return self
    
    def DF_ROW_FILTER(self, criteria):
        '''Filter rows with specified filter criteria'''
        self._df.query(criteria, inplace = True)
        self._fig()
        return self
    
    def DF_ROW_KEEP_BOTTOM(self, numRows):
        '''Delete all rows except specified bottom N rows'''
        self._df = self._df.tail(numRows)
        self._fig()
        return self
    
    def DF_ROW_KEEP_TOP(self, numRows):
        '''Delete all rows except specified top N rows'''
        self._df = self._df.head(numRows)
        self._fig()
        return self
    
    def DF_ROW_REVERSE(self):
        '''Reorder all rows in reverse order'''
        self._df = self._df[::-1].reset_index(drop = True)
        self._fig()
        return self
    
    def DF_ROW_SORT(self, columns, descending = False):
        '''Reorder dataframe by specified columns in ascending/descending order'''
        ascending = 1
        if descending == True: ascending = 0
        self._df = self._df.sort_values(by = columns, axis = 0, ascending = ascending, na_position ='last')
        self._fig()
        return self
    
    # DATAFRAME ACTIONS
    
    def DF__APPEND(self, otherdf):
        '''Append a table to bottom of current table'''
        self._df = self._df.append(otherdf._df, ignore_index=True)
        self._fig()
        return self
    
    def DF__FILL_DOWN(self):
        '''Fill blank cells with values from last non-blank cell above'''
        self._df = self._df.fillna(method="ffill", axis = 'index', inplace = True)
        self._fig()
        return self
    
    def DF__FILL_UP(self):
        '''Fill blank cells with values from last non-blank cell below'''
        self._df = self._df.fillna(method="bfill", axis = 'index', inplace = True)
        self._fig()
        return self
    
    def DF__FILL_RIGHT(self):
        '''Fill blank cells with values from last non-blank cell from left'''
        self._df = self._df.fillna(method="ffill", axis = 'columns', inplace = True)
        self._fig()
        return self
    
    def DF__FILL_LEFT(self):
        '''Fill blank cells with values from last non-blank cell from right'''
        self._df = self._df.fillna(method="bfill", axis = 'columns', inplace = True)
        self._fig()
        return self
    
    def DF__GROUP(self, groupby, aggregates = None):
        '''Group table contents by specified columns with optional aggregation (sum/max/min etc)'''
        if aggregates == None:
            self._df = self._df.groupby(groupby, as_index=False).first()
        else:
            self._df = self._df.groupby(groupby, as_index=False).agg(aggregates)
            self._df.columns = ['_'.join(col).rstrip('_') for col in self._df.columns.values]
        self._fig()
        return self
    
    def DF__MERGE(self, otherdf, on, how = 'left'):
        self._df = pd.merge(self._df, otherdf._df, on=on, how=how)
        self._fig()
        return self
    
    def DF__REPLACE(self, before, after):
        self._df = self._df.apply(lambda s: s.str.replace(before, after, regex=False), axis=0)
        self._fig()
        return self
    
    '''
    def TAB_TRANSPOSE(self):
        self.df = self.df.transpose(copy = True)
        return self
    '''
    
    def DF__UNPIVOT(self, indexCols):
        self._df = pd.melt(self._df, id_vars = indexCols)
        self._fig()
        return self
    
    def DF__PIVOT(self, indexCols, cols, vals):
        #indexCols = list(set(df.columns) - set(cols) - set(vals))
        self._df = self._df.pivot(index = indexCols, columns = cols, values = vals).reset_index().rename_axis(mapper = None,axis = 1)
        self._fig()
        return self
    
    def DF_COLHEADER_PROMOTE(self, row = 1):
        '''Promote row at specified index to column headers'''
        # make new header, fill in blank values with ColN
        i = row - 1
        newHeader = self._df.iloc[i:row].squeeze()
        newHeader = newHeader.values.tolist()
        for i in newHeader:
            if i == None: i = 'Col'
        # set new col names
        self.DF_COL_RENAME(newHeader)
        
        # delete 'promoted' rows
        self.DF_ROW_DELETE(row)
        
        self._fig()
        return self
    
    def DF_COLHEADER_DEMOTE(self):
        '''Demote column headers to make 1st row of table'''
        # insert 'demoted' column headers
        self.DF_ROW_ADD(self._df.columns)
        # make new header as Col1, Col2, Coln
        newHeader = ['Col' + str(x) for x in range(len(self._df.columns))]
        # set new col names
        self.DF_COL_RENAME(newHeader)
        self._fig()
        return self
    
    def DF_COLHEADER_REORDER_ASC(self):
        '''Reorder column titles in ascending order'''
        self._df.columns = sorted(self._df.columns.values.tolist())
        self._fig()
        return self
    
    def DF_COLHEADER_REORDER_DESC(self):
        '''Reorder column titles in descending order'''
        self._df.columns = sorted(self._df.columns.values.tolist(), reverse = True)
        self._fig()
        return self
    
    def DF_COLHEADER_REORDER(self, columns):
        '''Reorder column titles in specified order'''
        # if not all columns are specified, we order to front and add others to end
        return self.DF_COL_MOVE_TO_FRONT(columns)
    
    def DF__STATS(self):
        '''Show basic summary statistics of table contents'''
        self._df = self._df.describe()
        self._fig()
        return self
    
    # VIZUALIZATION ACTIONS
    
    def VIZ_BOX(self, x=None, y=None, color=None, facet_col=None, facet_row=None, data_frame=None, **kwargs):
        '''Draw a box plot'''
        if data_frame is None: data_frame = self._df
        fig = px.box(data_frame=data_frame, x=x, y=y, color=color, facet_col=facet_col, facet_row=facet_row, 
                     color_discrete_sequence=self._colorSwatch, **kwargs)
        self._fig(fig)
        return self
        
    def VIZ_VIOLIN(self, x=None, y=None, color=None, facet_col=None, facet_row=None, data_frame=None, **kwargs):
        '''Draw a violin plot'''
        if data_frame is None: data_frame = self._df
        fig = px.violin(data_frame=data_frame, x=x, y=y, color=color, facet_col=facet_col, facet_row=facet_row, box=True, 
                     color_discrete_sequence=self._colorSwatch, **kwargs)
        self._fig(fig)
        return self
        
    def VIZ_HIST(self, x=None, color=None, facet_col=None, facet_row=None, data_frame=None, **kwargs):
        '''Draw a hisotgram'''
        if data_frame is None: data_frame = self._df
        fig = px.histogram(data_frame=data_frame, x=x, color=color, facet_col=facet_col, facet_row=facet_row, 
                     color_discrete_sequence=self._colorSwatch, **kwargs)
        self._fig(fig)
        return self
    
    def VIZ_HIST_LIST(self, color=None, data_frame=None, **kwargs):
        '''Draw a histogram for all fields in current dataframe'''
        if data_frame is None: data_frame = self._df
        for c in data_frame.columns:
            fig = px.histogram(data_frame=data_frame, x=c, color=color, color_discrete_sequence=self._colorSwatch, **kwargs)
            self._fig(fig)
        self._fig(preview = 'all_charts')
        return self
    
    def VIZ_SCATTER(self, x=None, y=None, color=None, size=None, symbol=None, facet_col=None, facet_row=None, data_frame=None, **kwargs):
        '''Draw a scatter plot'''
        if data_frame is None: data_frame = self._df
        fig = px.scatter(data_frame=data_frame, x=x, y=y, color=color, size=size, symbol=symbol, facet_col=facet_col, facet_row=facet_row, 
                     color_discrete_sequence=self._colorSwatch, **kwargs)
        self._fig(fig)
        return self
        
    def VIZ_BAR(self, x=None, y=None, color=None, facet_col=None, facet_row=None, data_frame=None, **kwargs):
        '''Draw a bar plot'''
        if data_frame is None: data_frame = self._df
        fig = px.bar(data_frame=data_frame, x=x, y=y, color=color, facet_col=facet_col, facet_row=facet_row, 
                     color_discrete_sequence=self._colorSwatch, **kwargs)
        self._fig(fig)
        return self
    
    def VIZ_LINE(self, x=None, y=None, color=None, facet_col=None, facet_row=None, markers=True, data_frame=None, **kwargs):
        '''Draw a line plot'''
        if data_frame is None: data_frame = self._df
        fig = px.line(data_frame=data_frame, x=x, y=y, color=color, facet_col=facet_col, facet_row=facet_row, markers=markers, 
                     color_discrete_sequence=self._colorSwatch, **kwargs)
        self._fig(fig)
        return self
    
    def VIZ_AREA(self, x=None, y=None, color=None, facet_col=None, facet_row=None, markers=True, data_frame=None, **kwargs):
        '''Draw a line plot'''
        if data_frame is None: data_frame = self._df
        fig = px.area(data_frame=data_frame, x=x, y=y, color=color, facet_col=facet_col, facet_row=facet_row, markers=markers, 
                     color_discrete_sequence=self._colorSwatch, **kwargs)
        self._fig(fig)
        return self
    
    def VIZ_TREEMAP(self, path, values, root='Top', data_frame=None, **kwargs):
        '''Draw a treemap plot'''
        path = [px.Constant("Top")] + path
        if data_frame is None: data_frame = self._df
        fig = px.treemap(data_frame=data_frame, path=path, values=values, color_discrete_sequence=self._colorSwatch, **kwargs)
        fig.update_traces(root_color="lightgrey")
        fig.update_layout(margin = dict(t=50, l=25, r=25, b=25))
        self._fig(fig)
        return self
    
    def VIZ_SCATTERMATRIX(self, dimensions=None, color=None, data_frame=None, **kwargs):
        '''Draw a scatter matrix plot'''
        if data_frame is None: data_frame = self._df
        fig = px.scatter_matrix(data_frame=data_frame, dimensions=dimensions, color_discrete_sequence=self._colorSwatch, color=color, **kwargs)
        self._fig(fig)
        return self
    
    # MACHINE LEARNING 'FEATURE SELECTION' ACTIONS
    
    def ML_SELECT_FEATURES_NONE_ZERO_VARIANCE(self):
        '''Select numerical features / columns with non-zero variance'''
        return self.DF_COL_DELETE_EXCEPT(self._selectFeatures(method='VarianceThreshold'))
    
    def ML_SELECT_FEATURES_N_BEST(self, target, n=10):
        '''Select best n numerical features / columns for classifying target column'''
        return self.DF_COL_DELETE_EXCEPT(self._selectFeatures(method='SelectKBest', target=target, n=n))
    
    #@ignore_warnings
    def ML_TRAIN_AND_SAVE_CLASSIFIER(self, target, path='classifier.joblib'):
        '''Train several classification models & select the best one'''
        
        # PREP TRAIN/TEST DATA
        
        # separate features, target
        X = self._df[self._removeElementsFromList(self._colHelper(colsOnNone=True), [target])]
        y = self._df[target]
        
        # train/test split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
        
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
        
        # FIT!
        
        # fit with 'train' only!
        grid.fit(X_train, y_train)
        
        # after hard work of model fitting, we can clear pipeline/transformer cache
        rmtree(cachedir)
        
        # SAVE/'PICKLE' MODEL
        
        # generate path
        if path in self._config.sections() and 'model' in self._config[path]:
            path = self._config[path]['model']
        
        # save
        from joblib import dump
        dump(grid.best_estimator_, path, compress = 1) 
        
        # load saved model again to be sure
        #new_clf = load(path) 
        
        # force evaluation
        return self._ML_EVAL_CLASSIFIER(path, X_test, y_test, pos_label='Yes')
    
    def ML_EVAL_CLASSIFIER(self, target, path='classifier.joblib', pos_label='Yes'):
        '''Evaluate a classfier with TEST data'''
        # separate features, target
        X = self._df[self._removeElementsFromList(self._colHelper(colsOnNone=True), [target])]
        y = self._df[target]
        
        return self._ML_EVAL_CLASSIFIER(path, X, y, pos_label)
        
    def _ML_EVAL_CLASSIFIER(self, path, X_test, y_test, pos_label, **kwargs):
        '''Draw a ROC plot'''
        
        # generate path
        if path in self._config.sections() and 'model' in self._config[path]:
            path = self._config[path]['model']
            
        from joblib import load
        # load saved model again to be sure
        clf = load(path) 
        
        # predict/score
        y_predict = clf.predict(X_test)
        y_score = clf.predict_proba(X_test)[:, 1]
        
        # classification report
        #print(classification_report(y_test, y_predict))

        # confusion matrix
        np = confusion_matrix(y_test, y_predict, labels=['No', 'Yes'], normalize='all')
        df = pd.DataFrame(data=np.ravel(), columns=['value']) 
        df['true'] = ['Negative', 'Negative', 'Positive', 'Positive']
        df['name'] = ['True negative', 'False Positive', 'False negative', 'True positive']
        
        # show confusion matrix as 'treemap'
        self.VIZ_TREEMAP(data_frame=df, 
                         path=['true', 'name'], 
                         values='value',
                         root='Top',
                         #width=600,
                         #height=450,
                         title='Classification Results (Confusion Matrix)')
        
        # histogram of scores compared to true labels
        self.VIZ_HIST(x=y_score, 
                      color=y_test, 
                      nbins=50, 
                      labels=dict(color='True Labels', x='Score'),
                      title='Classifier score vs True labels')
        
        # preliminary viz & roc
        fpr, tpr, thresholds = roc_curve(y_test, y_score, pos_label=pos_label)
        
        # tpr, fpr by threshold chart
        df = pd.DataFrame({
            'False Positive Rate': fpr,
            'True Positive Rate': tpr
        }, index=thresholds)
        df.index.name = "Thresholds"
        df.columns.name = "Rate"
        
        self.VIZ_LINE(data_frame=df, 
                      title='True Positive Rate and False Positive Rate at every threshold', 
                      width=600, 
                      height=450,
                      range_x=[0,1], 
                      range_y=[0,1],
                      markers=False)
        
        # roc chart
        self.VIZ_AREA(x=fpr, y=tpr,
                      title=f'ROC Curve (AUC: %.2f)'% roc_auc_score(y_test, y_score),
                      width=600, 
                      height=450,
                      labels=dict(x='False Positive Rate', y='True Positive Rate'),
                      range_x=[0,1], 
                      range_y=[0,1],
                      markers=False)
        
        self._figs[-1].add_shape(type='line', line=dict(dash='dash', color='firebrick'),x0=0, x1=1, y0=0, y1=1)

        precision, recall, thresholds = precision_recall_curve(y_test, y_score, pos_label=pos_label)

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
        
        return self
        
    
    def _selectFeatures(self, method=None, target=None, n=10):
        from sklearn.feature_selection import VarianceThreshold
        from sklearn.feature_selection import SelectKBest
        from sklearn.feature_selection import chi2
        
        if method == 'VarianceThreshold':
            sel = VarianceThreshold() #remove '0 variance'
            x = self._df[self._colHelper(type='number')]
            sel.fit_transform(x)
            return sel.get_feature_names_out().tolist()
        elif method == 'SelectKBest':
            sel = SelectKBest(k=n)
            x = self._df[self._colHelper(type='number')]
            y = self._df[target]
            sel.fit_transform(X=x, y=y)
            features = sel.get_feature_names_out().tolist()
            features.append(target)
            return features
    
    # MACHINE LEARNING 'MODEL TRAINING' ACTIONS
    
    #@ignore_warnings
    def ML_TRAIN_AND_SAVE_REGRESSOR(self, target, path='classifier.joblib'):
        '''Train several classification models & select the best one'''
        
        # PREP TRAIN/TEST DATA
        
        # separate features, target
        X = self._df[self._removeElementsFromList(self._colHelper(colsOnNone=True), [target])]
        y = self._df[target]
        
        # train/test split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
        
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
        
        # FIT!
        
        # fit with 'train' only!
        grid.fit(X_train, y_train)
        
        # after hard work of model fitting, we can clear pipeline/transformer cache
        rmtree(cachedir)
        
        # SAVE/'PICKLE' MODEL
        
        # generate path
        if path in self._config.sections() and 'model' in self._config[path]:
            path = self._config[path]['model']
        
        # save
        from joblib import dump
        dump(grid.best_estimator_, path, compress = 1) 
        
        # load saved model again to be sure
        #new_clf = load(path) 
        
        # force evaluation
        return self._ML_EVAL_REGRESSOR(path, X_test, y_test)
    
    def ML_EVAL_REGRESSOR(self, target, path='classifier.joblib'):
        '''Evaluate a regressor with TEST data'''
        # separate features, target
        X = self._df[self._removeElementsFromList(self._colHelper(colsOnNone=True), [target])]
        y = self._df[target]
        
        return self._ML_EVAL_REGRESSOR(path, X, y)
        
    def _ML_EVAL_REGRESSOR(self, path, X_test, y_test, **kwargs):
        '''Evaluate a regressor'''
        
        # generate path
        if path in self._config.sections() and 'model' in self._config[path]:
            path = self._config[path]['model']
            
        from joblib import load
        # load saved model again to be sure
        clf = load(path) 
        
        # predict/score
        #y_predict = clf.predict(X_test)
        #y_score = clf.predict_proba(X_test)[:, 1]
        
        #print(clf)
        #print(clf.named_steps['preprocessor'].get_feature_names_out())
        #print(clf.get_feature_names_out())
        
        if(len(X_test.columns) == 1):
            df = pd.DataFrame({
                'X': X_test.iloc[:, 0].to_numpy(),
                'y': y_test.to_numpy()
            })
            #df.index.name = "Thresholds"
            #df.columns.name = "Rate"

            # chart
            self.VIZ_SCATTER(data_frame=df,
                             x='X',
                             y='y',
                          title='Regression plot (r2: TBD)',
                          width=600, 
                          height=450,
                          #labels=dict(x='False Positive Rate', y='True Positive Rate'),
                          #range_x=[0,1], 
                          #range_y=[0,1],
                          #markers=False
                        )
            # add prediction line
            #x_range = np.linspace(X_test.min(), X_test.max(), 100)
            x_range = X_test.sort_values(by=list(X_test)[0])
            y_range = clf.predict(x_range)
            self._figs[-1].add_traces(go.Scatter(x=x_range.iloc[:, 0].to_numpy(), y=y_range, name='Regression Fit'))
        
        else:
            df = pd.DataFrame({
                'X': clf.named_steps['preprocessor'].get_feature_names_out(),  #X_test.columns,
                'y': clf.named_steps['regressor'].coef_
            })
            colors = ['Positive' if c > 0 else 'Negative' for c in clf.named_steps['regressor'].coef_]
            self.VIZ_BAR(
                x='X', 
                y='y',
                data_frame=df,
                color=colors,
                #color_discrete_sequence=['red', 'blue'],
                labels=dict(x='Feature', y='Linear coefficient'),
                title='Weight of each feature for predicting DailyRate'
            )
        
        return self
    
    @property
    def REPORT_SET_VIZ_COLORS_PLOTLY(self):
        '''Set plot/report colors to 'Plotly'''
        return self._REPORT_SET_VIZ_COLORS(px.colors.qualitative.Plotly)
    
    @property
    def REPORT_SET_VIZ_COLORS_D3(self):
        '''Set plot/report colors to 'D3'''
        return self._REPORT_SET_VIZ_COLORS(px.colors.qualitative.D3)
    
    @property
    def REPORT_SET_VIZ_COLORS_G10(self):
        '''Set plot/report colors to 'G10'''
        return self._REPORT_SET_VIZ_COLORS(px.colors.qualitative.G10)
    
    @property
    def REPORT_SET_VIZ_COLORS_T10(self):
        '''Set plot/report colors to 'T10'''
        return self._REPORT_SET_VIZ_COLORS(px.colors.qualitative.T10)
    
    @property
    def REPORT_SET_VIZ_COLORS_ALPHABET(self):
        '''Set plot/report colors to 'Alphabet'''
        return self._REPORT_SET_VIZ_COLORS(px.colors.qualitative.Alphabet)
    
    @property
    def REPORT_SET_VIZ_COLORS_DARK24(self):
        '''Set plot/report colors to 'Dark24'''
        return self._REPORT_SET_VIZ_COLORS(px.colors.qualitative.Dark24)
    
    @property
    def REPORT_SET_VIZ_COLORS_LIGHT24(self):
        '''Set plot/report colors to 'Light24'''
        return self._REPORT_SET_VIZ_COLORS(px.colors.qualitative.Light24)
    
    @property
    def REPORT_SET_VIZ_COLORS_SET1(self):
        '''Set plot/report colors to 'Set1'''
        return self._REPORT_SET_VIZ_COLORS(px.colors.qualitative.Set1)
    
    @property
    def REPORT_SET_VIZ_COLORS_PASTEL1(self):
        '''Set plot/report colors to 'Pastel1'''
        return self._REPORT_SET_VIZ_COLORS(px.colors.qualitative.Pastel1)
    
    @property
    def REPORT_SET_VIZ_COLORS_DARK2(self):
        '''Set plot/report colors to 'Dark2'''
        return self._REPORT_SET_VIZ_COLORS(px.colors.qualitative.Dark2)
    
    @property
    def REPORT_SET_VIZ_COLORS_SET2(self):
        '''Set plot/report colors to 'Set2'''
        return self._REPORT_SET_VIZ_COLORS(px.colors.qualitative.Set2)
    
    @property
    def REPORT_SET_VIZ_COLORS_PASTEL2(self):
        '''Set plot/report colors to 'Pastel2'''
        return self._REPORT_SET_VIZ_COLORS(px.colors.qualitative.Pastel2)
    
    @property
    def REPORT_SET_VIZ_COLORS_SET3(self):
        '''Set plot/report colors to 'Set3'''
        return self._REPORT_SET_VIZ_COLORS(px.colors.qualitative.Set3)
    
    @property
    def REPORT_SET_VIZ_COLORS_ANTIQUE(self):
        '''Set plot/report colors to 'Antique'''
        return self._REPORT_SET_VIZ_COLORS(px.colors.qualitative.Antique)
    
    @property
    def REPORT_SET_VIZ_COLORS_BOLD(self):
        '''Set plot/report colors to 'Bold'''
        return self._REPORT_SET_VIZ_COLORS(px.colors.qualitative.Bold)
    
    @property
    def REPORT_SET_VIZ_COLORS_PASTEL(self):
        '''Set plot/report colors to 'Pastel'''
        return self._REPORT_SET_VIZ_COLORS(px.colors.qualitative.Pastel)
    
    @property
    def REPORT_SET_VIZ_COLORS_PRISM(self):
        '''Set plot/report colors to 'Prism'''
        return self._REPORT_SET_VIZ_COLORS(px.colors.qualitative.Prism)
    
    @property
    def REPORT_SET_VIZ_COLORS_SAFE(self):
        '''Set plot/report colors to 'Safe'''
        return self._REPORT_SET_VIZ_COLORS(px.colors.qualitative.Safe)
    
    @property
    def REPORT_SET_VIZ_COLORS_VIVID(self):
        '''Set plot/report colors to 'Vivid'''
        return self._REPORT_SET_VIZ_COLORS(px.colors.qualitative.Vivid)
    
    def _REPORT_SET_VIZ_COLORS(self, swatch = px.colors.qualitative.Plotly):
        self._colorSwatch = swatch
        #self._fig(preview = 'color_swatches')
        return self
    
    @property
    def REPORT_PREVIEW(self):
        self._fig(preview = 'all_charts')
        return self
    
    @property
    def REPORT_PREVIEW_FULL(self):
        self._fig(preview = 'full')
        return self
    
    def REPORT_SAVE_ALL(self, path = None):
        self.REPORT_SAVE_DF(path = path)
        #self.REPORT_SAVE_VIZ_PNG(path = path)
        self.REPORT_SAVE_VIZ_HTML(path = path)
        return self
    
    #def REPORT_SAVE_VIZ_PNG(self, path = None):
    #    'Save all figures into separate png files'
    #    path = self._pathHelper(path, filename='figure')
    #    for i, fig in enumerate(self._figs):
    #        fig.write_image(path+'%d.png' % i, width=1040, height=360, scale=10) 
    #    return self
    
    def REPORT_SAVE_VIZ_HTML(self, path = None, write_type = 'w'):
        'Save all figures into a single html file'
        import datetime
        #path = path if path == Null else path.encode().decode('unicode-escape')
        
        if path in self._config.sections() and 'html' in self._config[path]:
            path = self._config[path]['html']
        else:
            path = self._pathHelper(path, filename='html_report.html')
        with open(path, write_type) as f:
            f.write("Report generated: " + str(datetime.datetime.today()))
            for i, fig in enumerate(self._figs):
                f.write(fig.to_html(full_html=False, include_plotlyjs='cdn', default_height=360, default_width='95%', config=self._fig_config))
            #f.write(self._df.describe(include='all').fillna(value='').T.to_html())
        return self
    
    #def REPORT_SAVE_VIZ_HTML_APPEND(self, path = None):
    #    'Save all figures into a single html file'
    #    return self.REPORT_SAVE_VIZ_HTML(path=path, write_type='a')
    
    def REPORT_SAVE_DF(self, path = None):
        if path in self._config.sections() and 'csv' in self._config[path]:
            path = self._config[path]['csv']
        else:
            #path = path if path == Null else path.encode().decode('unicode-escape')
            path = self._pathHelper(path, filename='dataframe.csv') #pandas needs file extension
        self._df.to_csv(path, index=False)
        return self
    
    def REPORT_SAVE_DF_KINTONE_SYNCH(self, config_section):
        if source in self._config.sections():
            domain = self._config.get(source,'kintone_domain')
            app_id = self._config.getint(source,'app_id')
            api_token = self._config.get(source,'api_token')
            model_csv = self._config.get(source,'model_csv')
            
            m = jsModelFactory.get(model_csv)
            js = JinSapo(domain=domain, app_id=app_id, api_token=api_token, model=m)
            data = jsDataLoader(m, self._df)
            js.synch(data.models())
        else:
            print('Config section not in config.ini: ' + config_section)
        return self
    
    def REPORT_SAVE_DF_KINTONE_ADD(self, config_section):
        if source in self._config.sections():
            domain = self._config.get(source,'kintone_domain')
            app_id = self._config.getint(source,'app_id')
            api_token = self._config.get(source,'api_token')
            model_csv = self._config.get(source,'model_csv')
            
            m = jsModelFactory.get(model_csv)
            js = JinSapo(domain=domain, app_id=app_id, api_token=api_token, model=m)
            data = jsDataLoader(m, self._df)
            js.create(data.models())
        else:
            print('Config section not in config.ini: ' + config_section)
        return self
    
    def REPORT_DASH(self):
        from jupyter_dash import JupyterDash
        import dash_core_components as dcc
        import dash_html_components as html
        from dash.dependencies import Input, Output

        # Load Data
        df = px.data.tips()
        # Build App
        app = JupyterDash(__name__)
        app.layout = html.Div([
            html.H1("JupyterDash Demo"),
            dcc.Graph(id='graph'),
            html.Label([
                "colorscale",
                dcc.Dropdown(
                    id='colorscale-dropdown', clearable=False,
                    value='plasma', options=[
                        {'label': c, 'value': c}
                        for c in px.colors.named_colorscales()
                    ])
            ]),
        ])
        # Define callback to update graph
        @app.callback(
            Output('graph', 'figure'),
            [Input("colorscale-dropdown", "value")]
        )
        def update_figure(colorscale):
            return px.scatter(
                df, x="total_bill", y="tip", color="size",
                color_continuous_scale=colorscale,
                render_mode="webgl", title="Tips"
            )
        # Run app and display result inline in the notebook
        app.run_server(mode='jupyterlab')

    

# ## UTILITIES ###

    def _removeElementsFromList(self, l1, l2):
        '''Remove from list1 any elements also in list2'''
        # if not list type ie string then covert
        if not isinstance(l1, list):
            list1 = []
            list1.append(l1)
            l1 = list1
        if not isinstance(l2, list):
            list2 = []
            list2.append(l2)
            l2 = list2
        #return list(set(l1) - set(l2)) + list(set(l2) - set(l1))
        return [i for i in l1 if i not in l2]
    
    def _commonElementsInList(self, l1, l2):
        if l1 is None or l2 is None: return None
        if not isinstance(l1, list): l1 = [l1]
        if not isinstance(l2, list): l2 = [l2]
        #a_set = set(l1)
        #b_set = set(l2)
        
        # check length
        #if len(a_set.intersection(b_set)) > 0:
        #    return list(a_set.intersection(b_set)) 
        #else:
        #    return None
        return [i for i in l1 if i in l2]
    
    def _colHelper(self, columns = None, max = None, type = None, colsOnNone = True):
        
        # pre-process: translate to column names
        if isinstance(columns, slice) or isinstance(columns, int):
            columns = self._df.columns.values.tolist()[columns]
        elif isinstance(columns, list) and all(isinstance(c, int) for c in columns):
            columns = self._df.columns[columns].values.tolist()
        
        # process: limit possible columns by type (number, object, datetime)
        df = self._df.select_dtypes(include=type) if type is not None else self._df
        
        #process: fit to limited column scope
        if colsOnNone == True and columns is None: columns = df.columns.values.tolist()
        elif columns is None: return None
        else: columns = self._commonElementsInList(columns, df.columns.values.tolist())           
        
        # apply 'max' check    
        if isinstance(columns, list) and max != None: 
            if max == 1: columns = columns[0]
            else: columns = columns[:max]
        
        return columns
    
    def _rowHelper(self, df, max = None, head = True):
        if max == None: return df
        else: 
            if head == True: return df.head(max)
            else: return df.tail(max)
    
    def _toUniqueColName(self, name):
        n = 1
        while name in self._df.columns.values.tolist():
            name = name + str(n)
        return name
    
    def _pathHelper(self, path, filename):
        import os
        if path == None:
            from pathlib import Path
            home = str(Path.home())
            path = os.path.join(home, 'report')
        else:
            path = os.path.join(path, 'report')
        os.makedirs(path, exist_ok = True)
        path = os.path.join(path, filename)
        return path

    
