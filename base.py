import pandas as pd
from pandas.plotting import scatter_matrix as pdsm 
import numpy as np
import pathlib
import matplotlib.pyplot as plt
#import seaborn as sns

class pq(object):
    
    #column analyze from json
        
    def __init__(self, source):
        super(pq, self).__init__()
        file_ext = pathlib.Path(source).suffix
        if file_ext == '.csv':
            self._df = pd.read_csv(source)
        elif file_ext == '.xlsx':
            self._df = pd.read_excel(source)
        else:
            self._df = pd.DataFrame()
        self._showFig = False
        plt.ioff()
        
    def _repr_pretty_(self, p, cycle): 
        if self._showFig:
            return display(plt.gcf()), display(self._df)
        else:
            return display(self._df)
        
    def __repr__(self): 
        return self._df.__repr__()
    
    def __str__(self): 
        return self._df.__str__()
    
    # COLUMNS
    #add/copy/combine, delete, rename, reorder, change
    
    def DF_COL_ADD_FIXED(self, value, name = 'new_column'):
        n = 1
        while name in self._df.columns.values.tolist():
            name = name + str(n)
        self._df[name] = value
        
        self._showFig = False
        return self
    
    def DF_COL_ADD_INDEX(self, name = 'new_column'):
        n = 1
        while name in self._df.columns.values.tolist():
            name = name + str(n)
        self._df[name] = range(self._df.shape[0])
        
        self._showFig = False
        return self
    
    def DF_COL_ADD_CUSTOM(self, column, lmda, name = 'new_column'):
        n = 1
        while name in self._df.columns.values.tolist():
            name = name + str(n)
        self._df[name] = self._df[column].apply(lmda)
        
        self._showFig = False
        return self
    
    def DF_COL_ADD_EXTRACT_POSITION_AFTER(self, column, pos, name = 'new_column'):
        self._df = self.DF_COL_ADD_CUSTOM(self._df, column, lambda x: x[pos:], name = name)
        
        self._showFig = False
        return self
    
    def DF_COL_ADD_EXTRACT_POSITION_BEFORE(self, column, pos, name = 'new_column'):
        self._df = self.DF_COL_ADD_CUSTOM(self._df, column, lambda x: x[:pos], name = name)
        
        self._showFig = False
        return self
    
    def DF_COL_ADD_EXTRACT_CHARS_FIRST(self, column, chars, name = 'new_column'):
        self._df = self.DF_COL_ADD_CUSTOM(self._df, column, lambda x: x[:chars], name = name)
        
        self._showFig = False
        return self
    
    def DF_COL_ADD_EXTRACT_CHARS_LAST(self, column, chars, name = 'new_column'):
        self._df = self.DF_COL_ADD_CUSTOM(self._df, column, lambda x: x[-chars:], name = name)
        
        self._showFig = False
        return self
    
    def DF_COL_ADD_DUPLICATE(self, column, name = 'new_column'):
        n = 1
        while name in self._df.columns.values.tolist():
            name = name + str(n)
        self._df[name] = self._df[column]
        
        self._showFig = False
        return self
    
    def DF_COL_DELETE(self, columns):
        # if slice or int or int list then convert to col names
        if isinstance(columns, slice) or isinstance(columns, int):
            columns = self._df.columns.values.tolist()[columns]
        elif isinstance(columns, list) and all(isinstance(c, int) for c in columns):
            columns = self._df.columns[columns].values.tolist()
        self._df = self._df.drop(columns, axis = 1)
        
        self._showFig = False
        return self
    
    def DF_COL_DELETE_EXCEPT(self, columns):
        # if slice, int or int list then convert to col names
        if isinstance(columns, slice) or isinstance(columns, int):
            columns = self._df.columns.values.tolist()[columns]
        elif isinstance(columns, list) and all(isinstance(c, int) for c in columns):
            columns = self._df.columns[columns].values.tolist()
        cols = pq._diff(self._df.columns.values.tolist(), columns)
        
        self._showFig = False
        return self.DF_COL_DELETE(cols)
    
    def DF_COL_RENAME(self, columns):
        # we handle dict OR list
        if isinstance(columns, dict):
            self._df.rename(columns = columns, inplace = True)
        else:
            self._df.columns = columns
        
        self._showFig = False
        return self
    
    #col_reorder list of indices, list of colnames
    
    def DF_COL_REORDER_ASC(self):
        self._df.columns = sorted(self._df.columns.values.tolist())
        
        self._showFig = False
        return self
    
    def DF_COL_REORDER_DESC(self):
        self._df.columns = sorted(self._df.columns.values.tolist(), reverse = True)
        
        self._showFig = False
        return self
    
    def DF_COL_FORMAT_TO_UPPERCASE(self, columns = None):
        if columns == None: columns = self._df.columns.values.tolist()
        self._df[columns] = self._df[columns].apply(lambda s: s.str.upper(), axis=0)
        
        self._showFig = False
        return self
    
    def DF_COL_FORMAT_TO_LOWERCASE(self, columns = None):
        if columns == None: columns = self._df.columns.values
        self._df[columns] = self._df[columns].apply(lambda s: s.str.lower(), axis=0)
        
        self._showFig = False
        return self
    
    def DF_COL_FORMAT_TO_TITLECASE(self, columns = None):
        if columns == None: columns = self._df.columns.values
        self._df[columns] = self._df[columns].apply(lambda s: s.str.title(), axis=0)
        
        self._showFig = False
        return self
    
    def DF_COL_FORMAT_STRIP(self, columns = None):
        if columns == None: columns = self._df.columns.values
        self._df[columns] = self._df[columns].apply(lambda s: s.str.strip(), axis=0)
        
        self._showFig = False
        return self
    
    def DF_COL_FORMAT_STRIP_LEFT(self, columns = None):
        df = self._df
        if columns == None: columns = df.columns.values
        df[columns] = df[columns].apply(lambda s: s.str.lstrip(), axis=0)
        
        self._showFig = False
        return self
    
    def DF_COL_FORMAT_STRIP_RIGHT(self, columns = None):
        if columns == None: columns = self._df.columns.values
        self._df[columns] = self._df[columns].apply(lambda s: s.str.rstrip(), axis=0)
        
        self._showFig = False
        return self
    
    def DF_COL_FORMAT_ADD_PREFIX(self, prefix, column):
        self._df[column] = str(prefix) + self._df[column].astype(str)
        
        self._showFig = False
        return self
    
    def DF_COL_FORMAT_ADD_SUFFIX(self, suffix, column):
        self._df[column] = self._df[column].astype(str) + str(suffix)
        
        self._showFig = False
        return self
    
    def DF_COL_FORMAT_TYPE(self, columns, typ = 'str'):
        if columns == None: 
            self._df = self._df.astype(typ)
        else:
            convert_dict = {c:typ for c in columns}
            self._df = self._df.astype(convert_dict)
        
        self._showFig = False
        return self
    
    def DF_COL_FORMAT_ROUND(self, decimals):
        self._df = self._df.round(decimals)
        
        self._showFig = False
        return self
    
    #ROW
    
    def DF_ROW_ADD(self, row, index = 0):
        self._df.loc[index] = row
        
        self._showFig = False
        return self
    
    def DF_ROW_DELETE(self, rowNums):
        pos = rowNums - 1
        self._df.drop(self._df.index[pos], inplace=True)
        
        self._showFig = False
        return self
    
    def DF_ROW_FILTER(self, criteria):
        self._df.query(criteria, inplace = True)
        
        self._showFig = False
        return self
    
    def DF_ROW_KEEP_BOTTOM(self, numRows):
        self._df = self._df.tail(numRows)
        
        self._showFig = False
        return self
    
    def DF_ROW_KEEP_TOP(self, numRows):
        self._df = self._df.head(numRows)
        
        self._showFig = False
        return self
    
    def DF_ROW_REVERSE(self):
        self._df = self._df[::-1].reset_index(drop = True)
        
        self._showFig = False
        return self
    
    def DF_ROW_SORT(self, columns, descending = False):
        ascending = 1
        if descending == True: ascending = 0
        self._df = self._df.sort_values(by = columns, axis = 0, ascending = ascending, na_position ='last')
        
        self._showFig = False
        return self
    
    
    #TABLE
    
    def DF__APPEND(self, otherdf):
        self._df = self._df.append(otherdf._df, ignore_index=True)
        
        self._showFig = False
        return self
    
    def DF__FILL_DOWN(self):
        self._df = self._df.fillna(method="ffill", axis = 'index', inplace = True)
        
        self._showFig = False
        return self
    
    def DF__FILL_UP(self):
        self._df = self._df.fillna(method="bfill", axis = 'index', inplace = True)
        
        self._showFig = False
        return self
    
    def DF__FILL_RIGHT(self):
        self._df = self._df.fillna(method="ffill", axis = 'columns', inplace = True)
        
        self._showFig = False
        return self
    
    def DF__FILL_LEFT(self):
        self._df = self._df.fillna(method="bfill", axis = 'columns', inplace = True)
        
        self._showFig = False
        return self
    
    def DF__GROUP(self, groupby, aggregates = None):
        if aggregates == None:
            self._df = self._df.groupby(groupby).first()
        else:
            self._df = self._df.groupby(groupby).agg(aggregates).reset_index()
        # flatten multi-level columns created by aggregation
        self._df.columns = ['_'.join(col).rstrip('_') for col in self._df.columns.values]
        
        self._showFig = False
        return self
    
    def DF__MERGE(self, otherdf, on, how = 'left'):
        self._df = pd.merge(self._df, otherdf._df, on=on, how=how)
        
        self._showFig = False
        return self
    
    def DF__REPLACE(self, before, after):
        self._df = self._df.apply(lambda s: s.str.replace(before, after, regex=False), axis=0)
        
        self._showFig = False
        return self
    
    '''
    def TAB_TRANSPOSE(self):
        self.df = self.df.transpose(copy = True)
        return self
    '''
    
    def DF__UNPIVOT(self, indexCols):
        self._df = pd.melt(self._df, id_vars = indexCols)
        
        self._showFig = False
        return self
    
    def DF__PIVOT(self, indexCols, cols, vals):
        #indexCols = list(set(df.columns) - set(cols) - set(vals))
        self._df = self._df.pivot(index = indexCols, columns = cols, values = vals).reset_index().rename_axis(mapper = None,axis = 1)
        
        self._showFig = False
        return self
    
    def DF__PROMOTE_TO_HEADER(self, row = 1):
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
        
        self._showFig = False
        return self
    
    def DF__DEMOTE_HEADER(self):
        # insert 'demoted' column headers
        self.DF_ROW_ADD(self._df.columns)
        # make new header as Col1, Col2, Coln
        newHeader = ['Col' + str(x) for x in range(len(self._df.columns))]
        # set new col names
        self.DF_COL_RENAME(newHeader)
        
        self._showFig = False
        return self
    
    def DF__STATS(self):
        self._df = self._df.describe()
        
        self._showFig = False
        return self
    
    def VIZ_BOX(self, cols=None, by=None, **kwargs):
        self._df.plot.box(by, y=cols, **kwargs)
        self._showFig = True
        return self
    
    def VIZ_BOX_SUBS(self, cols=None, by=None, **kwargs):
        self._df.plot.box(by, y=cols, subplots=True, **kwargs)
        self._showFig = True
        return self
    
    def VIZ_HIST(self, cols=None, by=None, bins=10, **kwargs):
        kwargs['by'], kwargs['y'], kwargs['bins'] = by, cols, bins
        self._df.plot.hist(**kwargs)
        self._showFig = True
        return self
    
    def VIZ_SCATTER(self, x, y, s=None, c=None, **kwargs):
        self._df.plot.scatter(x=x, y=y, s=s, c=c, **kwargs)
        self._showFig = True
        return self
    
    def VIZ_SCATTER_MATRIX(self, cols=None, **kwargs):
        if cols == None:
            pdsm(self._df, **kwargs)
        else:
            pdsm(self._df[cols], **kwargs)
        self._showFig = True
        return self
    
    def VIZ_BAR(self, x, y, **kwargs):
        self._df.plot.bar(x=y, y=x, **kwargs)
        self._showFig = True
        return self
    
    def VIZ_LINE(self, x, y, **kwargs):
        self._df.plot.line(x=x, y=y, **kwargs)
        self._showFig = True
        return self
    
    def ABOUT_DF(self):
        print(self._df.info())
        
        self._showFig = False
        return self
    
    def SAVE_VIZ(self):
        for i in plt.get_fignums():
            plt.figure(i)
            plt.savefig('figure%d.png' % i)
        return self
    
    def SAVE_DF(self, path = 'query_write.csv'):
        self._df.to_csv(path, index=False)
        
        self._showFig = False
        return self
    
    @staticmethod
    def _diff(l1, l2):
        # if not list type ie string then covert
        if not isinstance(l1, list):
            list1 = []
            list1.append(l1)
            l1 = list1
        if not isinstance(l2, list):
            list2 = []
            list2.append(l2)
            l2 = list2
        return list(set(l1) - set(l2)) + list(set(l2) - set(l1))
    
    
    
