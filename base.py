import pandas as pd
import numpy as np
import pathlib

class pq(object):
    
    #column analyze from json
        
    def __init__(self, source):
        super(pq, self).__init__()
        file_ext = pathlib.Path(source).suffix
        if file_ext == '.csv':
            self.df = pd.read_csv(source)
        elif file_ext == '.xlsx':
            self.df = pd.read_excel(source)
        else:
            df = pd.DataFrame()
        
    def _repr_pretty_(self, p, cycle): 
        return display(self.df)
        
    def __repr__(self): 
        return self.df.__repr__()
    
    def __str__(self): 
        return self.df.__str__()
    
    #def READ(self, csv):
    #    s = pd.read_csv(csv)
    #    return s
    
    # COLUMNS
    #add/copy/combine, delete, rename, reorder, change
    
    def COL_ADD_FIXED(self, value, name = 'new_column'):
        n = 1
        while name in self.df.columns.values.tolist():
            name = name + str(n)
        self.df[name] = value
        return self
    
    def COL_ADD_INDEX(self, name = 'new_column'):
        n = 1
        while name in self.df.columns.values.tolist():
            name = name + str(n)
        self.df[name] = range(self.df.shape[0])
        return self
    
    def COL_ADD_CUSTOM(self, column, lmda, name = 'new_column'):
        n = 1
        while name in self.df.columns.values.tolist():
            name = name + str(n)
        self.df[name] = self.df[column].apply(lmda)
        return self
    
    def COL_ADD_EXTRACT_POSITION_AFTER(self, column, pos, name = 'new_column'):
        self.df = self.COL_ADD_CUSTOM(self.df, column, lambda x: x[pos:], name = name)
        return self
    
    def COL_ADD_EXTRACT_POSITION_BEFORE(self, column, pos, name = 'new_column'):
        self.df = self.COL_ADD_CUSTOM(self.df, column, lambda x: x[:pos], name = name)
        return self
    
    def COL_ADD_EXTRACT_CHARS_FIRST(self, column, chars, name = 'new_column'):
        self.df = self.COL_ADD_CUSTOM(self.df, column, lambda x: x[:chars], name = name)
        return self
    
    def COL_ADD_EXTRACT_CHARS_LAST(self, column, chars, name = 'new_column'):
        self.df = self.COL_ADD_CUSTOM(self.df, column, lambda x: x[-chars:], name = name)
        return self
    
    def COL_ADD_DUPLICATE(self, column, name = 'new_column'):
        n = 1
        while name in self.df.columns.values.tolist():
            name = name + str(n)
        self.df[name] = self.df[column]
        return self
    
    def COL_DELETE(self, columns):
        # if slice or int or int list then convert to col names
        if isinstance(columns, slice) or isinstance(columns, int):
            columns = self.df.columns.values.tolist()[columns]
        elif isinstance(columns, list) and all(isinstance(c, int) for c in columns):
            columns = self.df.columns[columns].values.tolist()
        self.df = self.df.drop(columns, axis = 1)
        return self
    
    def COL_DELETE_EXCEPT(self, columns):
        # if slice, int or int list then convert to col names
        if isinstance(columns, slice) or isinstance(columns, int):
            columns = self.df.columns.values.tolist()[columns]
        elif isinstance(columns, list) and all(isinstance(c, int) for c in columns):
            columns = self.df.columns[columns].values.tolist()
        cols = pq._diff(self.df.columns.values.tolist(), columns)
        return self.COL_DELETE(cols)
    
    def COL_RENAME(self, columns):
        # we handle dict OR list
        if isinstance(columns, dict):
            self.df.rename(columns = columns, inplace = True)
        else:
            self.df.columns = columns
        return self
    
    def COL_REORDER_ASC(self):
        self.df.columns = sorted(self.df.columns.values.tolist())
        return self
    
    def COL_REORDER_DESC(self):
        self.df.columns = sorted(self.df.columns.values.tolist(), reverse = True)
        return self
    
    def COL_FORMAT_TO_UPPERCASE(self, columns = None):
        if columns == None: columns = self.df.columns.values.tolist()
        self.df[columns] = self.df[columns].apply(lambda s: s.str.upper(), axis=0)
        return self
    
    def COL_FORMAT_TO_LOWERCASE(self, columns = None):
        if columns == None: columns = self.df.columns.values
        self.df[columns] = self.df[columns].apply(lambda s: s.str.lower(), axis=0)
        return self
    
    def COL_FORMAT_TO_TITLECASE(self, columns = None):
        if columns == None: columns = self.df.columns.values
        self.df[columns] = self.df[columns].apply(lambda s: s.str.title(), axis=0)
        return self
    
    def COL_FORMAT_STRIP(self, columns = None):
        if columns == None: columns = self.df.columns.values
        self.df[columns] = self.df[columns].apply(lambda s: s.str.strip(), axis=0)
        return self
    
    def COL_FORMAT_STRIP_LEFT(self, columns = None):
        df = self.df
        if columns == None: columns = df.columns.values
        df[columns] = df[columns].apply(lambda s: s.str.lstrip(), axis=0)
        return self
    
    def COL_FORMAT_STRIP_RIGHT(self, columns = None):
        if columns == None: columns = self.df.columns.values
        self.df[columns] = self.df[columns].apply(lambda s: s.str.rstrip(), axis=0)
        return self
    
    def COL_FORMAT_ADD_PREFIX(self, prefix, column):
        self.df[column] = str(prefix) + self.df[column].astype(str)
        return self
    
    def COL_FORMAT_ADD_SUFFIX(self, suffix, column):
        self.df[column] = self.df[column].astype(str) + str(suffix)
        return self
    
    def COL_FORMAT_TYPE(self, columns, typ = 'str'):
        if columns == None: 
            self.df = self.df.astype(typ)
        else:
            convert_dict = {c:typ for c in columns}
            self.df = self.df.astype(convert_dict)
        return self
    
    #ROW
    
    def ROW_ADD(self, row, index = 0):
        self.df.loc[index] = row
        return self
    
    def ROW_DELETE(self, rowNums):
        pos = rowNums - 1
        self.df.drop(self.df.index[pos], inplace=True)
        return self
    
    def ROW_FILTER(self, criteria):
        df.query(criteria, inplace = True)
        return self
    
    def ROW_KEEP_BOTTOM(self, numRows):
        self.df = self.df.tail(numRows)
        return self
    
    def ROW_KEEP_TOP(self, numRows):
        self.df = self.df.head(numRows)
        return self
    
    def ROW_REVERSE(self):
        self.df = self.df[::-1].reset_index(drop = True)
        return self
    
    def ROW_SORT(self, columns, descending = False):
        ascending = 1
        if descending == True: ascending = 0
        self.df = self.df.sort_values(by = columns, axis = 0, ascending = ascending, na_position ='last')
        return self
    
    
    #TABLE
    
    def TAB_APPEND(self, otherdf):
        self.df = self.df.append(otherdf.df, ignore_index=True)
        return self
    
    def TAB_FILL_DOWN(self):
        self.df = self.df.fillna(method="ffill", axis = 'index', inplace = True)
        return self
    
    def TAB_FILL_UP(self):
        self.df = self.df.fillna(method="bfill", axis = 'index', inplace = True)
        return self
    
    def TAB_FILL_RIGHT(self):
        self.df = self.df.fillna(method="ffill", axis = 'columns', inplace = True)
        return self
    
    def TAB_FILL_LEFT(self):
        self.df = self.df.fillna(method="bfill", axis = 'columns', inplace = True)
        return self
    
    def TAB_GROUP(self, groupby, aggregates = None):
        if aggregates == None:
            self.df = self.df.groupby(groupby).first()
        else:
            self.df = self.df.groupby(groupby).agg(aggregates).reset_index() #.rename_axis(mapper = None,axis = 1)
        return self
    
    def TAB_MERGE(self, otherdf, on, how = 'left'):
        self.df = pd.merge(self.df, otherdf.df, on=on, how=how)
        return self
    
    def TAB_REPLACE(self, before, after):
        self.df = self.df.apply(lambda s: s.str.replace(before, after, regex=False), axis=0)
        return self
    
    '''
    def TAB_TRANSPOSE(self):
        self.df = self.df.transpose(copy = True)
        return self
    '''
    
    def TAB_UNPIVOT(self, indexCols):
        self.df = pd.melt(self.df, id_vars = indexCols)
        return self
    
    def TAB_PIVOT(self, indexCols, cols, vals):
        #indexCols = list(set(df.columns) - set(cols) - set(vals))
        self.df = self.df.pivot(index = indexCols, columns = cols, values = vals).reset_index().rename_axis(mapper = None,axis = 1)
        return self
    
    def TAB_PROMOTE_TO_HEADER(self, row = 1):
        # make new header, fill in blank values with ColN
        i = row - 1
        newHeader = self.df.iloc[i:row].squeeze()
        newHeader = newHeader.values.tolist()
        for i in newHeader:
            if i == None: i = 'Col'
        
        # set new col names
        self.COL_RENAME(newHeader)
        
        # delete 'promoted' rows
        self.ROW_DELETE(row)
        return self
    
    def TAB_DEMOTE_HEADER(self):
        # insert 'demoted' column headers
        self.ROW_ADD(self.df.columns)
        # make new header as Col1, Col2, Coln
        newHeader = ['Col' + str(x) for x in range(len(self.df.columns))]
        # set new col names
        self.COL_RENAME(newHeader)
        return self
    
    def TAB_INFO(self):
        print(self.df.info())
        return self
    
    def TAB_SUMMARY_QUANT_STATS(self):
        self.df = self.df.describe()
        return self
    
    def WRITE(self, path = 'query_write.csv'):
        self.df.to_csv(path, index=False)
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
    
    
    
