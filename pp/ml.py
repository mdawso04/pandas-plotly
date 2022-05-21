import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np

from configparser import ConfigParser
import pathlib
from tempfile import mkdtemp
from shutil import rmtree
from joblib import dump

from PK import *

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

# ## QUERY INTERFACE ###

class GET(object):
    
    def __init__(self, source):
        super(GET, self).__init__()
        file_ext = pathlib.Path(source).suffix
        self._config = ConfigParser()
        self._config.read('config.ini', encoding='utf_8')
            
        # CONFIG.INI sources
        # kintone app
        if source in self._config.sections() and 'kintone_domain' in self._config[source]:
            domain = self._config.get(source,'kintone_domain')
            app_id = self._config.getint(source,'app_id')
            api_token = self._config.get(source,'api_token')
            model_csv = self._config.get(source,'model_csv')
            m = jsModelFactory.get(model_csv)
            js = JinSapo(domain=domain, app_id=app_id, api_token=api_token, model=m)
            self._df = js.select_df()
        # csv
        elif source in self._config.sections() and 'csv' in self._config[source]:
            self._df = pd.read_csv(self._config[source]['csv'])
        # xlsx
        elif source in self._config.sections() and 'xlsx' in self._config[source]:
            self._df = pd.read_excel(self._config[source]['xlsx'])
        # OTHER sources
        # csv
        elif file_ext == '.csv':
            self._df = pd.read_csv(source)
        # excel
        elif file_ext == '.xlsx':
            self._df = pd.read_excel(source)
        # blank
        else:
            self._df = pd.DataFrame()
        
        self._dfs = []
        self._showFig = False
        self._preview = 'no_chart' #'current_chart' 'all_charts'
        self.REPORT_SET_VIZ_COLORS_ANTIQUE
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
        self._fig_base_config =  {
            'dragmode': 'drawopenpath',
            'modebar_remove': ['resetScale', 'lasso2d'], #'select', 'zoom',
            'modebar_add': ['drawline', 'drawcircle',  'drawrect', 'eraseshape', 'pan2d'],
            'legend':{
                'traceorder':'reversed'
            },
            'title': {
                    'x': 0
            }
        }
        
        # format floats as percentages for readability
        #pd.options.display.float_format = '{:.2%}'.format
        #pd.options.display.max_rows = 500
        #pd.options.display.max_columns = 100
        
    def DF_COL_ADD_CONCATENATE(self, columns=None, separator='_', name='new_column'):
        '''Add a single new column with a 'fixed' value as content'''
        eval_string = '"{}".join(map(str, row.tolist()))'.format(separator)
        return self._DF_COL_ADD_CUSTOM(columns=columns, eval_string=eval_string, name=name)

    def _DF_COL_ADD_CUSTOM(self, columns=None, eval_string='""', name='new_column'):
        '''Add a single new column with custom (lambda) content'''
        columns = self._colHelper(columns)
        name = self._toUniqueColName(name)
        self._df[name] = self._df[columns].apply(lambda row: eval(eval_string), axis=1, result_type='expand')
        self._fig()
        return self

    #def DF_COL_ADD_CUSTOM(self, eval_string='row', name='new_column', data_frame=None):
    #    '''Add a new column with custom (lambda) content'''
    #    df = self._df if data_frame is None else data_frame[0] 
    #    name = self._toUniqueColName(name, data_frame=df)
    #    df[name] = df.apply(lambda row: eval(eval_string), axis=1, result_type='expand')
    #    if data_frame is None: self._df = df
    #    else: data_frame[0] = df
    #    self._fig()
    #    return self
    
    def DF_COL_ADD_DUPLICATE(self, column=None, name='new_column'):
        '''Add a single new column by copying an existing column'''
        column = self._colHelper(column, max=1, forceReturnAsList=False)
        eval_string = 'row.{}'.format(column)
        self._DF_COL_ADD_CUSTOM(eval_string=eval_string, name=name)
        self._fig()
        return self

    def DF_COL_ADD_EXTRACT_BEFORE(self, column=None, pos=None, name='new_column'):
        '''Add a single new column with text extracted from before char pos in existing column'''
        column = self._colHelper(column, max=1, forceReturnAsList=False)
        eval_string = 'str(row.{})[:{}]'.format(column, pos)
        return self._DF_COL_ADD_CUSTOM(eval_string=eval_string, name=name)

    def DF_COL_ADD_EXTRACT_FIRST(self, column=None, chars=None, name='new_column'):
        '''Add a single new column with first N chars extracted from column'''
        column = self._colHelper(column, max=1, forceReturnAsList=False)
        eval_string = 'str(row.{})[:{}]'.format(column, chars)
        return self._DF_COL_ADD_CUSTOM(eval_string=eval_string, name=name)

    def DF_COL_ADD_EXTRACT_FROM(self, column=None, pos=None, name='new_column'):
        '''Add a single new column of text extracted from after char pos in existing column'''
        column = self._colHelper(column, max=1, forceReturnAsList=False)
        eval_string = 'str(row.{})[{}:]'.format(column, pos)
        return self._DF_COL_ADD_CUSTOM(eval_string=eval_string, name=name)

    def DF_COL_ADD_EXTRACT_LAST(self, column=None, chars=None, name='new_column'):
        '''Add a single new column with last N chars extracted from column'''
        column = self._colHelper(column, max=1, forceReturnAsList=False)
        eval_string = 'str(row.{})[-{}:]'.format(column, chars)
        return self._DF_COL_ADD_CUSTOM(eval_string=eval_string, name=name)

    def DF_COL_ADD_FIXED(self, value=None, name='new_column'):
        '''Add a single new column with a 'fixed' value as content'''
        if isinstance(value, str): value = '"{}"'.format(value) # wrap string with extra commas!
        eval_string = '{}'.format(value)
        return self._DF_COL_ADD_CUSTOM(eval_string=eval_string, name=name)

    # COLUMN-WISE ADD
    def DF_COL_ADD_INDEX(self, start=1, name='new_column'):
        '''Add a single new column with a index/serial number as content'''
        name = self._toUniqueColName(name)
        self._df[name] = range(start, self._df.shape[0] + start)
        self._fig()
        return self

    def DF_COL_ADD_INDEX_FROM_0(self, name='new_column'):
        '''Convenience method for DF_COL_ADD_INDEX'''
        return self.DF_COL_ADD_INDEX(start=0, name=name)

    def DF_COL_ADD_INDEX_FROM_1(self, name='new_column'):
        '''Convenience method for DF_COL_ADD_INDEX'''
        return self.DF_COL_ADD_INDEX(start=1, name=name)

    def DF_COL_DELETE(self, columns=None):
        '''Delete specified column/s'''
        max = 1 if columns is None else None
        columns = self._colHelper(columns, max=max)
        self._df = self._df.drop(columns, axis = 1)
        self._fig()
        return self
    
    def DF_COL_DELETE_EXCEPT(self, columns=None):
        '''Deleted all column/s except specified'''
        max = 1 if columns is None else None
        columns = self._colHelper(columns, max=max)
        cols = self._removeElementsFromList(self._df.columns.values.tolist(), columns)
        self.DF_COL_DELETE(cols).DF_COL_REORDER_MOVE_TO_FRONT(columns)
        self._fig()
        return self
    
    def DF_COL_FILTER(self, columns=None, criteria=None):
        '''Filter rows with specified filter criteria'''
        self._df.query(criteria, inplace = True)
        self._df.reset_index(drop=True, inplace=True)
        self._fig()
        return self
    
    def DF_COL_FILTER_MISSING(self, columns=None):
        '''Filter rows with specified filter criteria'''
        columns = self._colHelper(columns, colsOnNone=True)
        self._df.dropna(inplace=True, subset=columns)
        self._fig()
        return self

    def DF_COL_FORMAT_ADD_PREFIX(self, columns=None, prefix='pre_'):
        '''Format specified column/s values by adding prefix'''
        eval_string = 'str("{}") + str(cell)'.format(prefix)
        return self._DF_COL_FORMAT_CUSTOM(columns=columns, eval_string=eval_string)

    def DF_COL_FORMAT_ADD_SUFFIX(self, columns=None, suffix='_suf'):
        '''Format specified single column values by adding suffix'''
        eval_string = 'str(cell) + str("{}")'.format(suffix)
        return self._DF_COL_FORMAT_CUSTOM(columns=columns, eval_string=eval_string)
    
    def DF_COL_FORMAT(self, columns=None, eval_string=None):
        '''Format specified column/s values to uppercase'''
        eval_string = 'cell{}'.format(eval_string)
        return self._DF_COL_FORMAT_CUSTOM(columns=columns, eval_string=eval_string)
    
    def _DF_COL_FORMAT_CUSTOM(self, columns=None, eval_string=None):
        '''Format specified column/s values to uppercase'''
        max = 1 if columns is None else None
        columns = self._colHelper(columns, max=max)
        self._df[columns] = pd.DataFrame(self._df[columns]).applymap(lambda cell: eval(eval_string))
        self._fig()
        return self
    
    def _DF_COL_FORMAT_CUSTOM_BATCH(self, columns=None, eval_string=None):
        '''Add a new column with custom (lambda) content'''
        max = 1 if columns is None else None
        columns = self._colHelper(columns, max=max)
        self._df[columns] = pd.DataFrame(self._df[columns]).apply(lambda row: eval(eval_string), axis=1)
        self._fig()
        return self
    
    # DF_FILL_DOWN
    def DF_COL_FORMAT_FILL_DOWN(self):
        '''Fill blank cells with values from last non-blank cell above'''
        eval_string = 'row.fillna(method="ffill")'.format(str(before), str(after))
        return self._DF_COL_FORMAT_CUSTOM_BATCH(columns=columns, eval_string=eval_string)
        
    #DF_FILL_UP
    def DF_COL_FORMAT_FILL_UP(self):
        '''Fill blank cells with values from last non-blank cell below'''
        eval_string = 'row.fillna(method="bfill")'.format(str(before), str(after))
        return self._DF_COL_FORMAT_CUSTOM_BATCH(columns=columns, eval_string=eval_string)
    
    # DF_REPLACE
    def DF_COL_FORMAT_REPLACE(self, columns=None, before='', after=''):
        '''Round numerical column values to specified decimal'''
        eval_string = 'cell.replace("{}","{}")'.format(str(before), str(after))
        return self._DF_COL_FORMAT_CUSTOM(columns=columns, eval_string=eval_string)
    
    # DF_REPLACE
    def DF_COL_FORMAT_REPLACE_MISSING(self, columns=None, after=''):
        '''Replace null (NaN) values to specified string'''
        eval_string = '"{}" if pd.isna(cell) else cell'.format(str(after))
        return self._DF_COL_FORMAT_CUSTOM(columns=columns, eval_string=eval_string)
    
    def DF_COL_FORMAT_ROUND(self, columns=None, decimals=0):
        '''Round numerical column values to specified decimal'''
        eval_string = 'round(cell,{})'.format(decimals)
        return self._DF_COL_FORMAT_CUSTOM(columns=columns, eval_string=eval_string)

    def DF_COL_FORMAT_STRIP(self, columns=None, chars=None):
        '''Format specified column/s values by stripping invisible characters'''
        eval_string = 'str(cell).strip()' if not chars else 'str(cell).strip("{}")'.format(chars)
        return self._DF_COL_FORMAT_CUSTOM(columns=columns, eval_string=eval_string)

    def DF_COL_FORMAT_STRIP_LEFT(self, columns=None, chars=None):
        '''Convenience method for DF_COL_FORMAT_STRIP'''
        eval_string = 'str(cell).lstrip()' if not chars else 'str(cell).lstrip("{}")'.format(chars)
        return self._DF_COL_FORMAT_CUSTOM(columns=columns, eval_string=eval_string)

    def DF_COL_FORMAT_STRIP_RIGHT(self, columns=None, chars=None):
        '''Convenience method for DF_COL_FORMAT_STRIP'''
        eval_string = 'str(cell).rstrip()' if not chars else 'str(cell).rstrip("{}")'.format(chars)
        return self._DF_COL_FORMAT_CUSTOM(columns=columns, eval_string=eval_string)

    def DF_COL_FORMAT_TO_LOWERCASE(self, columns=None):
        '''Format specified column/s values to lowercase'''
        eval_string = 'str(cell).lower()'
        return self._DF_COL_FORMAT_CUSTOM(columns=columns, eval_string=eval_string)

    def DF_COL_FORMAT_TO_TITLECASE(self, columns=None):
        '''Format specified column/s values to titlecase'''
        eval_string = 'str(cell).title()'
        return self._DF_COL_FORMAT_CUSTOM(columns=columns, eval_string=eval_string)

    def DF_COL_FORMAT_TO_UPPERCASE(self, columns=None):
        '''Format specified column/s values to uppercase'''
        eval_string = 'str(cell).upper()'
        return self._DF_COL_FORMAT_CUSTOM(columns=columns, eval_string=eval_string)

    def DF_COL_FORMAT_TYPE(self, columns=None, typ='str'):
        '''Format specified columns as specfied type'''
        max = 1 if columns is None else None
        columns = self._colHelper(columns, max=max)
        typ = [typ] if isinstance(typ, str) else typ
        convert_dict = {c:t for c,t in zip(columns, typ)}
        self._df = self._df.astype(convert_dict)
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
    
    def DF_COL_REORDER(self, columns):
        '''Reorder column titles in specified order. Convenience method for DF_COL_MOVE_TO_FRONT'''
        # if not all columns are specified, we order to front and add others to end
        return self.DF_COL_REORDER_MOVE_TO_FRONT(columns)

    def DF_COL_REORDER_ASCENDING(self):
        '''Reorder column titles in ascending order'''
        #df.columns = sorted(df.columns.values.tolist())
        self._df = self._df[sorted(self._df.columns.values.tolist())]
        self._fig()
        return self

    def DF_COL_REORDER_DESCENDING(self):
        '''Reorder column titles in descending order'''
        #df.columns = sorted(df.columns.values.tolist(), reverse = True)
        self._df = self._df[sorted(self._df.columns.values.tolist(), reverse=True)]
        self._fig()
        return self

    def DF_COL_REORDER_MOVE_TO_BACK(self, columns=None):
        '''Move specified column/s to back'''
        max = 1 if columns is None else None
        colsToMove = self._colHelper(columns, max=max)
        otherCols = self._removeElementsFromList(self._df.columns.values.tolist(), colsToMove)
        self._df = self._df[otherCols + colsToMove]
        self._fig()
        return self
    
    def DF_COL_REORDER_MOVE_TO_FRONT(self, columns=None):
        '''Move specified column/s to front'''
        max = 1 if columns is None else None
        colsToMove = self._colHelper(columns, max=max)
        otherCols = self._removeElementsFromList(self._df.columns.values.tolist(), colsToMove)
        self._df = self._df[colsToMove + otherCols]
        self._fig()
        return self
    
    def DF_COL_SORT(self, columns=None, ascending=True):
        '''Sort specified column/s in specified asc/desc order'''
        columns = self._colHelper(columns, colsOnNone=True)
        ascending = [ascending for _ in columns]
        self._df.sort_values(by=columns, ascending=ascending, inplace=True, na_position ='last')
        self._df.reset_index(inplace=True, drop=True)
        self._fig()
        return self
    
    def DF_COL_TRANSFORM_ADD(self, columns=None, num=0):
        '''Format specified column/s values to uppercase'''
        columns = self._colHelper(columns, type='number')
        eval_string = 'cell+{}'.format(num)
        return self._DF_COL_FORMAT_CUSTOM(columns=columns, eval_string=eval_string)
    
    def DF_COL_TRANSFORM_SUBTRACT(self, columns=None, num=0):
        '''Format specified column/s values to uppercase'''
        columns = self._colHelper(columns, type='number')
        eval_string = 'cell-{}'.format(num)
        return self._DF_COL_FORMAT_CUSTOM(columns=columns, eval_string=eval_string)
    
    def DF_COL_TRANSFORM_MULTIPLY(self, columns=None, num=0):
        '''Format specified column/s values to uppercase'''
        columns = self._colHelper(columns, type='number')
        eval_string = 'cell*{}'.format(num)
        return self._DF_COL_FORMAT_CUSTOM(columns=columns, eval_string=eval_string)
    
    def DF_COL_TRANSFORM_DIVIDE(self, columns=None, num=0):
        '''Format specified column/s values to uppercase'''
        columns = self._colHelper(columns, type='number')
        eval_string = 'cell/{}'.format(num)
        return self._DF_COL_FORMAT_CUSTOM(columns=columns, eval_string=eval_string)
    
    def DF_COL_TRANSFORM_EXPONENT(self, columns=None, num=0):
        '''Format specified column/s values to uppercase'''
        columns = self._colHelper(columns, type='number')
        eval_string = 'cell**{}'.format(num)
        return self._DF_COL_FORMAT_CUSTOM(columns=columns, eval_string=eval_string)
    
    def DF_COL_TRANSFORM_ROOT(self, columns=None, num=0):
        '''Format specified column/s values to uppercase'''
        columns = self._colHelper(columns, type='number')
        eval_string = 'cell**(1./{}.)'.format(num) if 0<=num else '-(-cell)**(1./{}.)'.format(num)
        return self._DF_COL_FORMAT_CUSTOM(columns=columns, eval_string=eval_string)
    
    def DF_COL_TRANSFORM_FLOORDIV(self, columns=None, num=0):
        '''Format specified column/s values to uppercase'''
        columns = self._colHelper(columns, type='number')
        eval_string = 'cell//{}'.format(num)
        return self._DF_COL_FORMAT_CUSTOM(columns=columns, eval_string=eval_string)
    
    def DF_COL_TRANSFORM_MODULUS(self, columns=None, num=0):
        '''Format specified column/s values to uppercase'''
        columns = self._colHelper(columns, type='number')
        eval_string = 'cell%{}'.format(num)
        return self._DF_COL_FORMAT_CUSTOM(columns=columns, eval_string=eval_string)


# DATAFRAME 'ROW' ACTIONS

    def DF_ROW_ADD(self, rows=None):
        '''Add row at specified index'''
        if rows is None: rows = self._rowHelper(max=1)
        if isinstance(rows, tuple):
            rows = list(rows)
        if isinstance(rows, list):
            self._df.loc[-1] = rows
            self._df.index = self._df.index + 1
            self._df.sort_index(inplace=True)
        #else:
        #    self._df = pd.concat([rows, self._df], ignore_index = True)
        self._fig()
        return self
    
    def DF_ROW_DELETE(self, rows=None):
        rows = list(rows) if isinstance(rows, tuple) else rows
        self._df.drop(self._df.index[rows], inplace=True)
        self._df.reset_index(drop=True, inplace=True)
        self._fig()
        return self
    
    def DF_ROW_KEEP_BOTTOM(self, numRows=1):
        '''Delete all rows except specified bottom N rows'''
        self._df = self._df.tail(numRows+1)
        self._df.reset_index(drop=True, inplace=True)
        self._fig()
        return self

    def DF_ROW_KEEP_TOP(self, numRows=1):
        '''Delete all rows except specified top N rows'''
        self._df = self._df.head(numRows+1)
        self._df.reset_index(drop=True, inplace=True)
        self._fig()
        return self

    def DF_ROW_REVERSE_ORDER(self):
        '''Reorder all rows in reverse order'''
        self._df = self._df[::-1].reset_index(drop = True)
        self._fig()
        return self

    def DF_ROW_TO_COLHEADER(self, row=0):
        '''Promote row at specified index to column headers'''
        # make new header, fill in blank values with ColN
        newHeader = self._df.iloc[row].squeeze()
        newHeader = newHeader.values.tolist()
        for i in newHeader:
            if i == None: i = 'Col'
        self.DF_COL_RENAME(newHeader)
        self.DF_ROW_DELETE([*range(row+1)])
        self._fig()
        return self

    def DF_ROW_FROM_COLHEADER(self):
        '''Demote column headers to make 1st row of table'''
        self.DF_ROW_ADD(list(self._df.columns.values))
        newHeader = ['Col' + str(x) for x in range(len(self._df.columns))]
        self.DF_COL_RENAME(newHeader)
        self._fig()
        return self
    
    # DATAFRAME ACTIONS
    def DF_APPEND(self, otherdf):
        '''Append a table to bottom of current table'''
        self._df = self._df.append(otherdf, ignore_index=True)
        self._fig()
        return self

    def DF_GROUP(self, groupby=None, aggregates=None):
        '''Group table contents by specified columns with optional aggregation (sum/max/min etc)'''
        max = 1 if groupby is None else None
        groupby = self._colHelper(groupby, max=max)
        if aggregates is None:
            self.DF_COL_ADD_FIXED(1, 'count')
            c = self._df.columns[-1]
            self._df = self._df.groupby(groupby, as_index=False, dropna=False).agg({c:'count'})
            #self._df = self._df.groupby(groupby, as_index=False, dropna=False).first()
        else:
            self._df = self._df.groupby(groupby, as_index=False, dropna=False).agg(aggregates)
            #self._df.columns = ['_'.join(col).rstrip('_') for col in self._df.columns.values]
        self._fig()
        return self

    def DF_MERGE(self, otherdf, on, how = 'left'):
        self._df = pd.merge(self._df, otherdf, on=on, how=how)
        self._fig()
        return self

    def DF_TRANSPOSE(self):
        self._df.transpose()
        self._fig()
        return self

    def DF_UNPIVOT(self, columns=None):
        columns = self._colHelper(columns)
        self._df = pd.melt(self._df, id_vars=columns)
        self._fig()
        return self

    def DF_PIVOT(self, indexCols, cols, vals):
        #indexCols = list(set(df.columns) - set(cols) - set(vals))
        self._df = self._df.pivot(index = indexCols, columns = cols, values = vals).reset_index().rename_axis(mapper = None,axis = 1)
        self._fig()
        return self

    # VIZUALIZATION ACTIONS
    
    def VIZ_AREA(self, x=None, y=None, color=None, facet_col=None, facet_row=None, markers=True, **kwargs):
        '''Draw a line plot with shaded area'''
        x, y, color, facet_col, facet_row = (self._colHelper(i, max=1, colsOnNone=False, forceReturnAsList=False) for i in [x, y, color, facet_col, facet_row])
        fig = px.area(data_frame=self._df, x=x, y=y, color=color, facet_col=facet_col, facet_row=facet_row, #markers=markers, 
                     color_discrete_sequence=self._colorSwatch, **kwargs)
        self._fig(fig)
        return self
    
    def VIZ_BAR(self, x=None, y=None, color=None, facet_col=None, facet_row=None, **kwargs):
        '''Draw a bar plot'''
        x, y, color, facet_col, facet_row = (self._colHelper(i, max=1, colsOnNone=False, forceReturnAsList=False) for i in [x, y, color, facet_col, facet_row])
        fig = px.histogram(data_frame=self._df, x=x, y=y, color=color, facet_col=facet_col, facet_row=facet_row, 
                     color_discrete_sequence=self._colorSwatch, **kwargs)
        self._fig(fig)
        return self
    
    def VIZ_BOX(self, x=None, y=None, color=None, facet_col=None, facet_row=None, **kwargs):
        '''Draw a box plot'''
        x, y, color, facet_col, facet_row = (self._colHelper(i, max=1, colsOnNone=False, forceReturnAsList=False) for i in [x, y, color, facet_col, facet_row])
        fig = px.box(data_frame=self._df, x=x, y=y, color=color, facet_col=facet_col, facet_row=facet_row, 
                     color_discrete_sequence=self._colorSwatch, **kwargs)
        self._fig(fig)
        return self
        
    def VIZ_DFSTATS(self):
        '''Show basic summary statistics of table contents'''
        stats = self._df.describe(include='all').T
        stats.insert(0, 'Feature', stats.index)
        self._appendDF(stats)
        self.DF_COL_ADD_INDEX_FROM_0(name='No').DF_COL_REORDER_MOVE_TO_FRONT(columns='No')
        self.VIZ_TABLE()
        self._popDF()
        return self
    
    def VIZ_HIST(self, x=None, color=None, facet_col=None, facet_row=None, **kwargs):
        '''Draw a hisotgram'''
        x, color, facet_col, facet_row = (self._colHelper(i, max=1, colsOnNone=False, forceReturnAsList=False) for i in [x, color, facet_col, facet_row])
        fig = px.histogram(data_frame=self._df, x=x, color=color, facet_col=facet_col, facet_row=facet_row, 
                     color_discrete_sequence=self._colorSwatch, **kwargs)
        self._fig(fig)
        return self
    
    def VIZ_HIST_LIST(self, color=None, **kwargs):
        '''Draw a histogram for all fields in current dataframe'''
        color = self._colHelper(color, max=1, colsOnNone=False, forceReturnAsList=False)
        for c in self._df.columns:
            fig = px.histogram(data_frame=self._df, x=c, color=color, color_discrete_sequence=self._colorSwatch, **kwargs)
            self._fig(fig)
        self._fig(preview = len(self._df.columns))
        return self
    
    def VIZ_ICICLE(self, path, values, root='All data', **kwargs):
        '''Draw a treemap plot'''
        path = [px.Constant("All data")] + self._colHelper(path)
        print(values)
        values = self._colHelper(values, max=1, type='number', forceReturnAsList=False)
        # make leaf dict (isna), update
        
        #p = self._colHelper(path)
        #d = self._df[p].groupby(p, as_index=False, dropna=False).first()
        #d = d[d.isna().any(axis=1)].to_dict(orient='records')
        #print(d)
        
        #d = df.groupby('Dia').apply(lambda a: dict(a.groupby('macap').apply(lambda x: dict(zip(x['transmission'], x['bytes'])))))
        #d = d.to_dict()
        
        # treemap, icicle, sunburst break on NaN. Replace with 'None' for this call
        df1 = self._df.where(pd.notnull, None)
        try:
            fig = px.icicle(data_frame=df1, path=path, values=values, color_discrete_sequence=self._colorSwatch, **kwargs)
            fig.update_traces(root_color="lightgrey")
            fig.update_layout(margin = dict(t=50, l=25, r=25, b=25))
            self._fig(fig)
        except ValueError:
            self.VIZ_ICICLE(path[1:-1], values, root=path[0])
        return self
    
    def VIZ_LINE(self, x=None, y=None, color=None, facet_col=None, facet_row=None, markers=True, **kwargs):
        '''Draw a line plot'''
        x, y, color, facet_col, facet_row = (self._colHelper(i, max=1, colsOnNone=False, forceReturnAsList=False) for i in [x, y, color, facet_col, facet_row])
        fig = px.line(data_frame=self._df, x=x, y=y, color=color, facet_col=facet_col, facet_row=facet_row, #markers=markers, 
                      color_discrete_sequence=self._colorSwatch, **kwargs)
        self._fig(fig)
        return self
    
    def VIZ_SCATTER(self, x=None, y=None, color=None, facet_col=None, facet_row=None, **kwargs): #size=None, symbol=None
        '''Draw a scatter plot'''
        x, y, color, facet_col, facet_row = (self._colHelper(i, max=1, colsOnNone=False, forceReturnAsList=False) for i in [x, y, color, facet_col, facet_row])
        fig = px.scatter(data_frame=self._df, x=x, y=y, color=color, facet_col=facet_col, facet_row=facet_row, 
                     color_discrete_sequence=self._colorSwatch, **kwargs)
        self._fig(fig)
        return self
        
    def VIZ_SCATTERMATRIX(self, dimensions=None, color=None, **kwargs):
        '''Draw a scatter matrix plot'''
        dimensions, color = (self._colHelper(i, max=j, colsOnNone=False, forceReturnAsList=False) for i, j in [(dimensions, None), (color, 1)])
        fig = px.scatter_matrix(data_frame=self._df, dimensions=dimensions, color_discrete_sequence=self._colorSwatch, color=color, **kwargs)
        self._fig(fig)
        return self
    
    def VIZ_SUNBURST(self, path, values, root='All data', **kwargs):
        '''Draw a treemap plot'''
        path = [px.Constant("All data")] + self._colHelper(path)
        values = self._colHelper(values, max=1, type='number', forceReturnAsList=False)
        # treemap, icicle, sunburst break on NaN. Replace with 'None' for this call
        df1 = self._df.where(pd.notnull, None)
        try:
            fig = px.sunburst(data_frame=df1, path=path, values=values, color_discrete_sequence=self._colorSwatch, **kwargs)
            fig.update_traces(root_color="lightgrey")
            fig.update_layout(margin = dict(t=50, l=25, r=25, b=25))
            self._fig(fig)
        except ValueError:
            self.VIZ_SUNBURST(path[1:-1], values, root=path[0])
        return self
    
    def VIZ_TABLE(self, columns=None, **kwargs):
        '''Draw a table'''
        columns = self._colHelper(columns)
        cell_values = self._df[columns].to_numpy().T
        fig = go.Figure(data=[go.Table(
            header=dict(values=columns,
                       align='left',
                       font_size=12,
                       height=30),
            cells=dict(values=cell_values,
                      align='left',
                       font_size=12,
                       height=30))
        ])
        self._fig(fig)
        return self
    
    def VIZ_TREEMAP(self, path, values, root='All data', **kwargs):
        '''Draw a treemap plot'''
        path = [px.Constant("All data")] + self._colHelper(path)
        values = self._colHelper(values, max=1, type='number', forceReturnAsList=False)
        # treemap, icicle, sunburst break on NaN. Replace with 'None' for this call
        df1 = self._df.where(pd.notnull, None)
        try:
            fig = px.treemap(data_frame=df1, path=path, values=values, color_discrete_sequence=self._colorSwatch, **kwargs)
            fig.update_traces(root_color="lightgrey")
            fig.update_layout(margin = dict(t=50, l=25, r=25, b=25))
            self._fig(fig)
        except ValueError:
            self.VIZ_TREEMAP(path[1:-1], values, root=path[0])
        return self
    
    def VIZ_VIOLIN(self, x=None, y=None, color=None, facet_col=None, facet_row=None, **kwargs):
        '''Draw a violin plot'''
        x, y, color, facet_col, facet_row = (self._colHelper(i, max=1, colsOnNone=False, forceReturnAsList=False) for i in [x, y, color, facet_col, facet_row])
        fig = px.violin(data_frame=self._df, x=x, y=y, color=color, facet_col=facet_col, facet_row=facet_row, box=True, 
                     color_discrete_sequence=self._colorSwatch, **kwargs)
        self._fig(fig)
        return self
    
    # MACHINE LEARNING 'FEATURE SELECTION' ACTIONS
    
    def ML_SELECT_FEATURES_NONE_ZERO_VARIANCE(self):
        '''Select numerical features / columns with non-zero variance'''
        return self.DF_COL_DELETE_EXCEPT(self._selectFeatures(method='VarianceThreshold'))
    
    def ML_SELECT_FEATURES_N_BEST(self, target, n=10):
        '''Select best n numerical features / columns for classifying target column'''
        return self.DF_COL_DELETE_EXCEPT(self._selectFeatures(method='SelectKBest', target=target, n=n))
    
    def _selectFeatures(self, method=None, target=None, n=10):
        target = self._colHelper(target, max=1, forceReturnAsList=False)
        if method == 'VarianceThreshold':
            sel = VarianceThreshold() #remove '0 variance'
            x = self._df[self._colHelper(type='number')]
            sel.fit_transform(x)
            return sel.get_feature_names_out().tolist()
        elif method == 'SelectKBest':
            sel = SelectKBest(k=n)
            x = self._df[self._removeElementsFromList(self._colHelper(type='number'), [target])]
            y = self._df[target]
            sel.fit_transform(X=x, y=y)
            features = sel.get_feature_names_out().tolist()
            features.append(target)
            return features
    
    #@ignore_warnings
    def ML_TRAIN_AND_SAVE_CLASSIFIER(self, target, path='classifier.joblib'):
        '''Train a classification model for provided target, save model to specified location and display summary of model performance'''
        
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
        '''Load a save classifier model from specified location and evaluate with current dataframe data'''
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
        '''Train a regression model for provided target, save model to specified location and display summary of model performance'''
        
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
        '''Evaluate a regressor with TEST data'''
        self._ML_EVAL_REGRESSOR(path, target, test_df=self._df, train_df=None)
        return self
        
    def _ML_EVAL_REGRESSOR(self, path, target, test_df, train_df, **kwargs):
        '''Evaluate a regressor'''
        
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
    
    #@property
    def REPORT_PREVIEW_CHARTS(self):
        self._fig(preview = 'all_charts')
        return self
    
    #@property
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
    
    def _REPORT_DASH(self):
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

    def _repr_pretty_(self, p, cycle): 
        '''Selects content for IPython display'''
        if self._preview == 'current_chart':
            return self._figs[-1].show(config=self._fig_config), display(self._dfForRepr())
        elif self._preview == 'all_charts':
            return tuple([f.show(config=self._fig_config) for f in self._figs]), display(self._dfForRepr())
        elif self._preview == 'full':
            return tuple([f.show(config=self._fig_config) for f in self._figs]), display(self._dfForRepr())
        elif self._preview == 'color_swatches':
            return px.colors.qualitative.swatches().show(), display(self._dfForRepr())
        elif isinstance(self._preview, int):
            return tuple([f.show(config=self._fig_config) for f in self._figs[-self._preview:]]), display(self._dfForRepr())
        else:
            #return display(pd.DataFrame(self._df.dtypes).T), display(self._df)
            #return display(self._df)
            return display(self._dfForRepr())
        
    def _dfForRepr(self):
        '''Prepares dataframe for IPython display'''
        # Update columns to include current datatypes as 2nd line
        dfr = self._df.copy()
        if isinstance(dfr.columns, pd.MultiIndex): 
            arrays = [range(0, len(dfr.columns)), dfr.columns.get_level_values(0), dfr.dtypes]
            mi = pd.MultiIndex.from_arrays(arrays, names=('Num', 'Name', 'Type'))
        else:
            arrays = [range(0, len(dfr.columns)), dfr.columns, dfr.dtypes]
            mi = pd.MultiIndex.from_arrays(arrays, names=('Num', 'Name', 'Type'))
        dfr.columns = mi
        return dfr
        
    def __repr__(self): 
        return self._df.__repr__()
    
    def __str__(self): 
        return self._df.__str__()
    
    @property
    def legendAsc(self):
        s = {
            'legend':{
                'traceorder':'reversed'
            }
        }
        self._fig(updateCurrentFigure=True, settings=s, overwrite=False)
        return self
    
    def _fig(self, fig=None, updateCurrentFigure=False, settings=None, overwrite=True, preview='no_chart'):
        '''Handles figure displaying for IPython'''
        if fig is not None:
            d = {**self._fig_base_config, **settings} if settings else self._fig_base_config
            fig.update_layout(dict1=d, overwrite=overwrite)
            self._figs.append(fig)
            self._preview = 'current_chart'
        elif updateCurrentFigure:
            fig = self._figs[-1]
            d = {**self._fig_base_config, **settings} if settings else self._fig_base_config
            fig.update_layout(dict1=d, overwrite=overwrite)
            self._figs.append(fig)
            self._preview = 'current_chart'
        else:
            self._preview = preview
            
    # DATAFRAME 'COLUMN' ACTIONS
    
    #utility to temporarily switch dataframe
    
    def _popDF(self):
        '''Remove current dataframe and replace with next on stack'''
        oldDF = self._df
        self._df = self._dfs.pop()
        return oldDF
    
    def _appendDF(self, df):
        '''Add dataframe to stack and make active dataframe'''
        self._dfs.append(self._df)
        self._df = df
        return self
    
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
        return [i for i in l1 if i in l2]
    
    def _colHelper(self, columns=None, max=None, type=None, colsOnNone=True, forceReturnAsList=True):
        
        if isinstance(columns, tuple):
            columns = list(columns)
        
        # pre-process: translate to column names
        if isinstance(columns, slice) or isinstance(columns, int):
            columns = self._df.columns.values.tolist()[columns]
        elif isinstance(columns, list) and all(isinstance(c, int) for c in columns):
            columns = self._df.columns[columns].values.tolist()
            
        # process: limit possible columns by type (number, object, datetime)
        df1 = self._df.select_dtypes(include=type) if type is not None else self._df
        
        #process: fit to limited column scope
        if colsOnNone == True and columns is None: columns = df1.columns.values.tolist()
        elif columns is None: return None
        else: columns = self._commonElementsInList(columns, df1.columns.values.tolist())           
        
        # apply 'max' check    
        if isinstance(columns, list) and max != None:
            if max == 1: columns = columns[0]
            else: columns = columns[:max]
            
        # if string format to list for return
        if forceReturnAsList and not isinstance(columns, list): 
            columns = [columns]
        
        return columns
    
    def _rowHelper(self, max = None, head = True):
        if max is None: return df
        else: 
            if head is True: return self._df.head(max)
            else: return self._df.tail(max)
    
    def _toUniqueColName(self, name):
        n = 1
        name = str(name)
        while name in self._df.columns.values.tolist():
            name = name + '_' + str(n)
        return name
    
    def _pathHelper(self, path, filename):
        import os
        if path == None:
            home = str(pathlib.Path.home())
            path = os.path.join(home, 'report')
        else:
            path = os.path.join(path, 'report')
        os.makedirs(path, exist_ok = True)
        path = os.path.join(path, filename)
        return path

    

figtest = {
    'data': [
        {
            'alignmentgroup': 'True', 
            'bingroup': 'x', 
            'hovertemplate': 
            'Attrition=Yes<br>Age=%{x}<br>count=%{y}<extra></extra>', 
            'legendgroup': 'Yes', 
            'marker': {
                'color': 'rgb(133, 92, 117)', 
                'pattern': {
                    'shape': ''
                }
            }, 
            'name': 'Yes', 
            'offsetgroup': 'Yes', 
            'orientation': 'v', 
            'showlegend': True, 
            'x': [], 
            'xaxis': 'x', 
            'yaxis': 'y', 
            'type': 'histogram'
        },
        {
            'alignmentgroup': 'True', 
            'bingroup': 'x', 
            'hovertemplate': 'Attrition=No<br>Age=%{x}<br>count=%{y}<extra></extra>', 
            'legendgroup': 'No', 
            'marker': {
                'color': 'rgb(217, 175, 107)', 
                'pattern': {
                    'shape': ''
                }
            }, 
            'name': 'No', 
            'offsetgroup': 'No', 
            'orientation': 'v', 
            'showlegend': True, 
            'x': [], 
            'xaxis': 'x', 
            'yaxis': 'y', 
            'type': 'histogram'
        }
    ], 
    'layout': {
        'template': {
            'data': {
                'bar': [{
                    'error_x': {
                        'color': '#2a3f5f'
                    }, 
                    'error_y': {
                        'color': '#2a3f5f'
                    }, 'marker': {
                        'line': {
                            'color': '#E5ECF6', 'width': 0.5
                        }, 
                        'pattern': {
                            'fillmode': 'overlay', 
                            'size': 10, 
                            'solidity': 0.2
                        }
                    }, 
                    'type': 'bar'
                }], 
                'barpolar': [{
                    'marker': {
                        'line': {
                            'color': '#E5ECF6', 'width': 0.5
                        }, 
                        'pattern': {
                            'fillmode': 'overlay', 
                            'size': 10, 
                            'solidity': 0.2
                        }
                    }, 
                    'type': 'barpolar'
                }], 
                'carpet': [{
                    'aaxis': {
                        'endlinecolor': '#2a3f5f', 
                        'gridcolor': 'white', 
                        'linecolor': 'white', 
                        'minorgridcolor': 'white', 
                        'startlinecolor': '#2a3f5f'
                    }, 
                    'baxis': {
                        'endlinecolor': '#2a3f5f', 
                        'gridcolor': 'white', 
                        'linecolor': 'white', 
                        'minorgridcolor': 'white', 
                        'startlinecolor': '#2a3f5f'
                    }, 
                    'type': 'carpet'
                }], 
                'choropleth': [{
                    'colorbar': {
                        'outlinewidth': 0, 
                        'ticks': ''
                    }, 
                    'type': 'choropleth'
                }], 
                'contour': [{
                    'colorbar': {
                        'outlinewidth': 0, 
                        'ticks': ''
                    }, 
                    'colorscale': [], 
                    'type': 'histogram2dcontour'
                }], 
                'mesh3d': [{
                    'colorbar': {
                        'outlinewidth': 0, 
                        'ticks': ''
                    }, 
                    'type': 'mesh3d'
                }], 
                'parcoords': [{
                    'line': {
                        'colorbar': {
                            'outlinewidth': 0, 
                            'ticks': ''
                        }}, 
                    'type': 'parcoords'
                }], 
                'pie': [{
                    'automargin': True, 'type': 'pie'
                }], 
                'scatter': [{
                    'marker': {
                        'colorbar': {
                            'outlinewidth': 0, 
                            'ticks': ''
                        }}, 
                    'type': 'scatter'
                }], 
                'scatter3d': [{
                    'line': {
                        'colorbar': {
                            'outlinewidth': 0, 
                            'ticks': ''
                        }}, 
                    'marker': {
                        'colorbar': {
                            'outlinewidth': 0, 
                            'ticks': ''
                        }}, 
                    'type': 'scatter3d'
                }], 
                'scattercarpet': [{
                    'marker': {
                        'colorbar': {
                            'outlinewidth': 0, 
                            'ticks': ''
                        }}, 
                    'type': 'scattercarpet'
                }], 
                'scattergeo': [{
                    'marker': {
                        'colorbar': {
                            'outlinewidth': 0, 
                            'ticks': ''
                        }}, 
                    'type': 'scattergeo'
                }], 
                'scattergl': [{
                    'marker': {
                        'colorbar': {
                            'outlinewidth': 0, 
                            'ticks': ''
                        }}, 
                    'type': 'scattergl'
                }], 
                'scattermapbox': [{
                    'marker': {
                        'colorbar': {
                            'outlinewidth': 0, 
                            'ticks': ''
                        }}, 
                    'type': 'scattermapbox'
                }], 
                'scatterpolar': [{
                    'marker': {
                        'colorbar': {
                            'outlinewidth': 0, 
                            'ticks': ''
                        }}, 
                    'type': 'scatterpolar'
                }], 
                'scatterpolargl': [{
                    'marker': {
                        'colorbar': {
                            'outlinewidth': 0, 
                            'ticks': ''
                        }}, 
                    'type': 'scatterpolargl'
                }], 
                'scatterternary': [{
                    'marker': {
                        'colorbar': {
                            'outlinewidth': 0, 
                            'ticks': ''
                        }}, 
                    'type': 'scatterternary'
                }], 
                'surface': [{
                    'colorbar': {
                        'outlinewidth': 0, 
                        'ticks': ''
                    }, 
                    'colorscale': [], 
                    'type': 'surface'
                }], 
                'table': [{
                    'cells': {
                        'fill': {
                            'color': '#EBF0F8'
                        }, 
                        'line': {
                            'color': 'white'
                        }}, 
                    'header': {
                        'fill': {
                            'color': '#C8D4E3'
                        }, 
                        'line': {
                            'color': 'white'
                        }}, 
                    'type': 'table'
                }]
            }, 
            'layout': {
                'annotationdefaults': {
                    'arrowcolor': '#2a3f5f', 
                    'arrowhead': 0, 
                    'arrowwidth': 1
                }, 
                'autotypenumbers': 'strict', 
                'coloraxis': {
                    'colorbar': {
                        'outlinewidth': 0, 
                        'ticks': ''
                    }
                }, 
                'colorscale': {
                    'diverging': []
                }, 
                'colorway': [], 
                'font': {
                    'color': '#2a3f5f'
                }, 
                'geo': {
                    'bgcolor': 'white', 
                    'lakecolor': 'white', 
                    'landcolor': '#E5ECF6', 
                    'showlakes': True, 
                    'showland': True, 
                    'subunitcolor': 'white'
                }, 
                'hoverlabel': {
                    'align': 'left'
                }, 
                'hovermode': 'closest', 
                'mapbox': {
                    'style': 'light'
                }, 
                'paper_bgcolor': 'white', 
                'plot_bgcolor': '#E5ECF6', 
                'polar': {
                    'angularaxis': {
                        'gridcolor': 'white', 
                        'linecolor': 'white', 
                        'ticks': ''
                    }, 
                    'bgcolor': '#E5ECF6', 
                    'radialaxis': {
                        'gridcolor': 'white', 
                        'linecolor': 'white', 
                        'ticks': ''
                    }
                }, 
                'scene': {
                    'xaxis': {
                        'backgroundcolor': '#E5ECF6', 
                        'gridcolor': 'white', 
                        'gridwidth': 2, 
                        'linecolor': 'white', 
                        'showbackground': True, 
                        'ticks': '', 
                        'zerolinecolor': 'white'
                    }, 
                    'yaxis': {
                        'backgroundcolor': '#E5ECF6', 
                        'gridcolor': 'white', 
                        'gridwidth': 2, 
                        'linecolor': 'white', 
                        'showbackground': True, 
                        'ticks': '', 
                        'zerolinecolor': 'white'
                    }, 
                    'zaxis': {
                        'backgroundcolor': '#E5ECF6', 
                        'gridcolor': 'white', 
                        'gridwidth': 2, 
                        'linecolor': 'white', 
                        'showbackground': True, 
                        'ticks': '', 
                        'zerolinecolor': 'white'
                    }
                }, 
                'shapedefaults': {
                    'line': {
                        'color': '#2a3f5f'
                    }
                }, 
                'ternary': {
                    'aaxis': {
                        'gridcolor': 'white', 
                        'linecolor': 'white', 
                        'ticks': ''
                    }, 
                    'baxis': {
                        'gridcolor': 'white', 
                        'linecolor': 'white', 
                        'ticks': ''
                    }, 
                    'bgcolor': '#E5ECF6', 
                    'caxis': {
                        'gridcolor': 'white', 
                        'linecolor': 'white', 
                        'ticks': ''
                    }
                }, 
                'title': {
                    'x': 0.05
                }, 
                'xaxis': {
                    'automargin': True, 
                    'gridcolor': 'white', 
                    'linecolor': 'white', 
                    'ticks': '', 
                    'title': {
                        'standoff': 15
                    }, 
                    'zerolinecolor': 'white', 
                    'zerolinewidth': 2
                }, 
                'yaxis': {
                    'automargin': True, 
                    'gridcolor': 'white', 
                    'linecolor': 'white', 
                    'ticks': '', 
                    'title': {
                        'standoff': 15
                    }, 
                    'zerolinecolor': 'white', 
                    'zerolinewidth': 2
                }
            }
        },
        'xaxis': {
            'anchor': 'y', 
            'domain': [0.0, 1.0], 
            'title': {
                'text': 'Age'
            }
        }, 
        'yaxis': {
            'anchor': 'x', 
            'domain': [0.0, 1.0], 
            'title': {
                'text': 'count'
            }
        }, 
        'legend': {
            'traceorder': 'reversed'
        }, 
        'margin': {
            't': 60
        }, 
        'barmode': 'relative', 
        'modebar': {
            'remove': ['resetScale', 'lasso2d'], 
            'add': ['drawline', 'drawcircle', 'drawrect', 'eraseshape', 'pan2d']
        }, 
        'dragmode': 'drawopenpath'
    }
}
