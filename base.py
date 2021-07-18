import pandas as pd
import numpy as np

class pq:
    
    #source new xlsx
    #column analyze from json, xml
    #table group
    #table header promote 1st row, demote to first row
    #table merge, append
    #table row count
    #table pivot, unpivot

    
    @staticmethod
    def SOURCE(csv):
        s = pd.read_csv(csv)
        return s
    
    # COLUMNS
    #add/copy/combine, delete, rename, reorder, change
    
    @staticmethod
    def COLUMN_ADD_FIXED(df, value, name = 'new_column'):
        n = 1
        while name in df.columns.values:
            name = name + str(n)
        df[name] = value
        return df
    
    @staticmethod
    def COLUMN_ADD_INDEX(df, name = 'new_column'):
        n = 1
        while name in df.columns.values:
            name = name + str(n)
        df[name] = range(df.shape[0])
        return df
    
    @staticmethod
    def COLUMN_ADD_CUSTOM(df, column, lmda, name = 'new_column'):
        n = 1
        while name in df.columns.values:
            name = name + str(n)
        df[name] = df[column].apply(lmda)
        return df
    
    @staticmethod
    def COLUMN_ADD_EXTRACT_POSITION_AFTER(df, column, pos, name = 'new_column'):
        return pq.COLUMN_ADD_CUSTOM(df, column, lambda x: x[pos:], name = name)
    
    @staticmethod
    def COLUMN_ADD_EXTRACT_POSITION_BEFORE(df, column, pos, name = 'new_column'):
        return pq.COLUMN_ADD_CUSTOM(df, column, lambda x: x[:pos], name = name)
    
    @staticmethod
    def COLUMN_ADD_EXTRACT_CHARS_FIRST(df, column, chars, name = 'new_column'):
        return pq.COLUMN_ADD_CUSTOM(df, column, lambda x: x[:chars], name = name)
    
    @staticmethod
    def COLUMN_ADD_EXTRACT_CHARS_LAST(df, column, chars, name = 'new_column'):
        return pq.COLUMN_ADD_CUSTOM(df, column, lambda x: x[-chars:], name = name)
    
    @staticmethod
    def COLUMN_ADD_DUPLICATE(df, column, name = 'new_column'):
        n = 1
        while name in df.columns.values:
            name = name + str(n)
        df[name] = df[column]
        return df
    
    @staticmethod
    def COLUMN_ADD_COMBINE(df, columns, name = 'new_column', separator = ''):
        n = 1
        while name in df.columns.values:
            name = name + str(n)
        df[name] = df[columns].apply(lambda x: separator.join(x.tolist()), axis=1)
        return df
    
    @staticmethod
    def COLUMN_ADD_SPLIT_LEFT(df, column, separator = ',', splits = None):
        splitted = df[column].str.split(pat = separator, n = splits, expand = True)
        i = range(len(splitted.columns))
        for c in i:
            df[column + '_' + str(c)] = splitted.iloc[:, c]
        return df
    
    @staticmethod
    def COLUMN_ADD_SPLIT_RIGHT(df, column, separator = ',', splits = None):
        splitted = df[column].str.rsplit(pat = separator, n = splits, expand = True)
        i = range(len(splitted.columns))
        for c in i:
            df[column + '_' + str(c)] = splitted.iloc[:, c]
        return df
        
    @staticmethod
    def COLUMN_DELETE(df, columns):
        return df.drop(columns, axis = 1)
    
    @staticmethod
    def COLUMN_DELETE_EXCEPT(df, not_columns):
        columns = pq._diff(list(df.columns.values), not_columns)
        return pq.COLUMN_DELETE(df, columns)
    
    @staticmethod
    def COLUMN_RENAME(df, columns):
        df.rename(columns = columns, inplace = True)
        return df
    
    @staticmethod
    def COLUMN_REORDER_ASC(df):
        df.columns = sorted(df.columns.values.tolist())
        return df
    
    @staticmethod
    def COLUMN_REORDER_DESC(df):
        df.columns = sorted(df.columns.values.tolist(), reverse = True)
        return df
    
    @staticmethod
    def COLUMN_FORMAT_TO_UPPERCASE(df, columns = None):
        if columns == None: columns = df.columns.values
        df[columns] = df[columns].apply(lambda s: s.str.upper(), axis=0)
        return df
    
    @staticmethod
    def COLUMN_FORMAT_TO_LOWERCASE(df, columns = None):
        if columns == None: columns = df.columns.values
        df[columns] = df[columns].apply(lambda s: s.str.lower(), axis=0)
        return df
    
    @staticmethod
    def COLUMN_FORMAT_TO_TITLECASE(df, columns = None):
        if columns == None: columns = df.columns.values
        df[columns] = df[columns].apply(lambda s: s.str.title(), axis=0)
        return df
    
    @staticmethod
    def COLUMN_FORMAT_STRIP(df, columns = None):
        if columns == None: columns = df.columns.values
        df[columns] = df[columns].apply(lambda s: s.str.strip(), axis=0)
        return df
    
    @staticmethod
    def COLUMN_FORMAT_STRIP_LEFT(df, columns = None):
        if columns == None: columns = df.columns.values
        df[columns] = df[columns].apply(lambda s: s.str.lstrip(), axis=0)
        return df
    
    @staticmethod
    def COLUMN_FORMAT_STRIP_RIGHT(df, columns = None):
        if columns == None: columns = df.columns.values
        df[columns] = df[columns].apply(lambda s: s.str.rstrip(), axis=0)
        return df
    
    @staticmethod
    def COLUMN_FORMAT_ADD_PREFIX(df, prefix, column):
        df[column] = str(prefix) + df[column].astype(str)
        return df
    
    @staticmethod
    def COLUMN_FORMAT_ADD_SUFFIX(df, suffix, column):
        df[column] = df[column].astype(str) + str(suffix)
        return df
    
    @staticmethod
    def COLUMN_FORMAT_TYPE(df, columns, typ = 'str'):
        if columns == None: 
            df = df.astype(typ)
        else:
            convert_dict = {c:typ for c in columns}
            df = df.astype(convert_dict)
        return df
    
    #ROW
    
    @staticmethod
    def ROW_DELETE(df, rowNums):
        df.drop(df.index[pos], inplace=True)
        return df
    
    @staticmethod
    def ROW_FILTER(df, criteria):
        df.query(criteria, inplace = True)
        return df
    
    @staticmethod
    def ROW_KEEP_BOTTOM(df, numRows):
        df = df.tail(numRows)
        return df
    
    @staticmethod
    def ROW_KEEP_TOP(df, numRows):
        df = df.head(numRows)
        return df
    
    @staticmethod
    def ROW_REVERSE(df):
        df = df[::-1].reset_index(drop = True)
        return df
    
    @staticmethod
    def ROW_SORT(df, columns, descending = False):
        ascending = 1
        if descending == True: ascending = 0
        df = df.sort_values(columns, ascending)
        return df
    
    #TABLE
    
    @staticmethod
    def TABLE_FILL_DOWN(df):
        df = df.fillna(method="ffill", axis = 'index', inplace = True)
        return df
    
    @staticmethod
    def TABLE_FILL_UP(df):
        df = df.fillna(method="bfill", axis = 'index', inplace = True)
        return df
    
    @staticmethod
    def TABLE_FILL_RIGHT(df):
        df = df.fillna(method="ffill", axis = 'columns', inplace = True)
        return df
    
    @staticmethod
    def TABLE_FILL_LEFT(df):
        df = df.fillna(method="bfill", axis = 'columns', inplace = True)
        return df
    
    @staticmethod
    def TABLE_REPLACE(df, before, after):
        df = df.apply(lambda s: s.str.replace(before, after, regex=False), axis=0)
        return df
    
    @staticmethod
    def TABLE_TRANSPOSE(df):
        df = df.T
        return df
    
    @staticmethod
    def TABLE_INFO(df):
        print(df.info())
        return df
    
    @staticmethod
    def _diff(l1, l2):
        return list(set(l1) - set(l2)) + list(set(l2) - set(l1))
    