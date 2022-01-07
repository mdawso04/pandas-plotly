from .base import *
from .util import *

#non-standard libraries
import pandas as pd

class Data(Base):
    
    def __init__(self, source):
        super().__init__(source)
    
    def DATA_COL_ADD_CONCATENATE(self, columns=None, separator='_', name='new_column'):
        '''Add a single new column with a 'fixed' value as content'''
        eval_string = '"{}".join(map(str, row.tolist()))'.format(separator)
        return self._DATA_COL_ADD_CUSTOM(columns=columns, eval_string=eval_string, name=name)

    def _DATA_COL_ADD_CUSTOM(self, columns=None, eval_string='""', name='new_column'):
        '''Add a single new column with custom (lambda) content'''
        columns = colHelper(self.df, columns)
        name = toUniqueColName(self.df, name)
        self.df[name] = self.df[columns].apply(lambda row: eval(eval_string), axis=1, result_type='expand')
        self._preview()
        return self

    def DATA_COL_ADD_DUPLICATE(self, column=None, name='new_column'):
        '''Add a single new column by copying an existing column'''
        column = colHelper(self.df, column, max=1, forceReturnAsList=False)
        eval_string = 'row.{}'.format(column)
        self._DATA_COL_ADD_CUSTOM(eval_string=eval_string, name=name)
        self._preview()
        return self

    def DATA_COL_ADD_EXTRACT_BEFORE(self, column=None, pos=None, name='new_column'):
        '''Add a single new column with text extracted from before char pos in existing column'''
        column = colHelper(self.df, column, max=1, forceReturnAsList=False)
        eval_string = 'str(row.{})[:{}]'.format(column, pos)
        return self._DATA_COL_ADD_CUSTOM(eval_string=eval_string, name=name)

    def DATA_COL_ADD_EXTRACT_FIRST(self, column=None, chars=None, name='new_column'):
        '''Add a single new column with first N chars extracted from column'''
        column = colHelper(self.df, column, max=1, forceReturnAsList=False)
        eval_string = 'str(row.{})[:{}]'.format(column, chars)
        return self._DATA_COL_ADD_CUSTOM(eval_string=eval_string, name=name)

    def DATA_COL_ADD_EXTRACT_FROM(self, column=None, pos=None, name='new_column'):
        '''Add a single new column of text extracted from after char pos in existing column'''
        column = colHelper(self.df, column, max=1, forceReturnAsList=False)
        eval_string = 'str(row.{})[{}:]'.format(column, pos)
        return self._DATA_COL_ADD_CUSTOM(eval_string=eval_string, name=name)

    def DATA_COL_ADD_EXTRACT_LAST(self, column=None, chars=None, name='new_column'):
        '''Add a single new column with last N chars extracted from column'''
        column = colHelper(self.df, column, max=1, forceReturnAsList=False)
        eval_string = 'str(row.{})[-{}:]'.format(column, chars)
        return self._DATA_COL_ADD_CUSTOM(eval_string=eval_string, name=name)

    def DATA_COL_ADD_FIXED(self, value=None, name='new_column'):
        '''Add a single new column with a 'fixed' value as content'''
        if isinstance(value, str): value = '"{}"'.format(value) # wrap string with extra commas!
        eval_string = '{}'.format(value)
        return self._DATA_COL_ADD_CUSTOM(eval_string=eval_string, name=name)

    def DATA_COL_ADD_INDEX(self, start=1, name='new_column'):
        '''Add a single new column with a index/serial number as content'''
        name = toUniqueColName(self.df, name)
        self.df[name] = range(start, self.df.shape[0] + start)
        self._preview()
        return self

    def DATA_COL_ADD_INDEX_FROM_0(self, name='new_column'):
        '''Convenience method for DATA_COL_ADD_INDEX'''
        return self.DATA_COL_ADD_INDEX(start=0, name=name)

    def DATA_COL_ADD_INDEX_FROM_1(self, name='new_column'):
        '''Convenience method for DATA_COL_ADD_INDEX'''
        return self.DATA_COL_ADD_INDEX(start=1, name=name)

    def DATA_COL_DELETE(self, columns=None):
        '''Delete specified column/s'''
        max = 1 if columns is None else None
        columns = colHelper(self.df, columns, max=max)
        self.df = self.df.drop(columns, axis = 1)
        self._preview()
        return self
    
    def DATA_COL_DELETE_EXCEPT(self, columns=None):
        '''Deleted all column/s except specified'''
        max = 1 if columns is None else None
        columns = colHelper(self.df, columns, max=max)
        cols = removeElementsFromList(self.df.columns.values.tolist(), columns)
        self.DATA_COL_DELETE(cols).DATA_COL_REORDER_MOVE_TO_FRONT(columns)
        self._preview()
        return self
    
    def DATA_COL_FILTER(self, columns=None, criteria=None):
        '''Filter rows with specified filter criteria'''
        self.df.query(criteria, inplace = True)
        self.df.reset_index(drop=True, inplace=True)
        self._preview()
        return self
    
    def DATA_COL_FILTER_MISSING(self, columns=None):
        '''Filter rows with specified filter criteria'''
        columns = colHelper(self.df, columns, colsOnNone=True)
        self.df.dropna(inplace=True, subset=columns)
        self._preview()
        return self

    def DATA_COL_FORMAT_ADD_PREFIX(self, columns=None, prefix='pre_'):
        '''Format specified column/s values by adding prefix'''
        eval_string = 'str("{}") + str(cell)'.format(prefix)
        return self._DATA_COL_FORMAT_CUSTOM(columns=columns, eval_string=eval_string)

    def DATA_COL_FORMAT_ADD_SUFFIX(self, columns=None, suffix='_suf'):
        '''Format specified single column values by adding suffix'''
        eval_string = 'str(cell) + str("{}")'.format(suffix)
        return self._DATA_COL_FORMAT_CUSTOM(columns=columns, eval_string=eval_string)
    
    def DATA_COL_FORMAT(self, columns=None, eval_string=None):
        '''Format specified column/s values to uppercase'''
        eval_string = 'cell{}'.format(eval_string)
        return self._DATA_COL_FORMAT_CUSTOM(columns=columns, eval_string=eval_string)
    
    def _DATA_COL_FORMAT_CUSTOM(self, columns=None, eval_string=None):
        '''Format specified column/s values to uppercase'''
        max = 1 if columns is None else None
        columns = colHelper(self.df, columns, max=max)
        self.df[columns] = pd.DataFrame(self.df[columns]).applymap(lambda cell: eval(eval_string))
        self._preview()
        return self
    
    def _DATA_COL_FORMAT_CUSTOM_BATCH(self, columns=None, eval_string=None):
        '''Add a new column with custom (lambda) content'''
        max = 1 if columns is None else None
        columns = colHelper(self.df, columns, max=max)
        self.df[columns] = pd.DataFrame(self.df[columns]).apply(lambda row: eval(eval_string), axis=1)
        self._preview()
        return self
    
    # DATA_FILL_DOWN
    def DATA_COL_FORMAT_FILL_DOWN(self):
        '''Fill blank cells with values from last non-blank cell above'''
        eval_string = 'row.fillna(method="ffill")'.format(str(before), str(after))
        return self._DATA_COL_FORMAT_CUSTOM_BATCH(columns=columns, eval_string=eval_string)
        
    #DATA_FILL_UP
    def DATA_COL_FORMAT_FILL_UP(self):
        '''Fill blank cells with values from last non-blank cell below'''
        eval_string = 'row.fillna(method="bfill")'.format(str(before), str(after))
        return self._DATA_COL_FORMAT_CUSTOM_BATCH(columns=columns, eval_string=eval_string)
    
    # DATA_REPLACE
    def DATA_COL_FORMAT_REPLACE(self, columns=None, before='', after=''):
        '''Round numerical column values to specified decimal'''
        eval_string = 'cell.replace("{}","{}")'.format(str(before), str(after))
        return self._DATA_COL_FORMAT_CUSTOM(columns=columns, eval_string=eval_string)
    
    # DATA_REPLACE
    def DATA_COL_FORMAT_REPLACE_MISSING(self, columns=None, after=''):
        '''Replace null (NaN) values to specified string'''
        eval_string = '"{}" if pd.isna(cell) else cell'.format(str(after))
        return self._DATA_COL_FORMAT_CUSTOM(columns=columns, eval_string=eval_string)
    
    def DATA_COL_FORMAT_ROUND(self, columns=None, decimals=0):
        '''Round numerical column values to specified decimal'''
        eval_string = 'round(cell,{})'.format(decimals)
        return self._DATA_COL_FORMAT_CUSTOM(columns=columns, eval_string=eval_string)

    def DATA_COL_FORMAT_STRIP(self, columns=None, chars=None):
        '''Format specified column/s values by stripping invisible characters'''
        eval_string = 'str(cell).strip()' if not chars else 'str(cell).strip("{}")'.format(chars)
        return self._DATA_COL_FORMAT_CUSTOM(columns=columns, eval_string=eval_string)

    def DATA_COL_FORMAT_STRIP_LEFT(self, columns=None, chars=None):
        '''Convenience method for DATA_COL_FORMAT_STRIP'''
        eval_string = 'str(cell).lstrip()' if not chars else 'str(cell).lstrip("{}")'.format(chars)
        return self._DATA_COL_FORMAT_CUSTOM(columns=columns, eval_string=eval_string)

    def DATA_COL_FORMAT_STRIP_RIGHT(self, columns=None, chars=None):
        '''Convenience method for DATA_COL_FORMAT_STRIP'''
        eval_string = 'str(cell).rstrip()' if not chars else 'str(cell).rstrip("{}")'.format(chars)
        return self._DATA_COL_FORMAT_CUSTOM(columns=columns, eval_string=eval_string)

    def DATA_COL_FORMAT_TO_LOWERCASE(self, columns=None):
        '''Format specified column/s values to lowercase'''
        eval_string = 'str(cell).lower()'
        return self._DATA_COL_FORMAT_CUSTOM(columns=columns, eval_string=eval_string)

    def DATA_COL_FORMAT_TO_TITLECASE(self, columns=None):
        '''Format specified column/s values to titlecase'''
        eval_string = 'str(cell).title()'
        return self._DATA_COL_FORMAT_CUSTOM(columns=columns, eval_string=eval_string)

    def DATA_COL_FORMAT_TO_UPPERCASE(self, columns=None):
        '''Format specified column/s values to uppercase'''
        eval_string = 'str(cell).upper()'
        return self._DATA_COL_FORMAT_CUSTOM(columns=columns, eval_string=eval_string)

    def DATA_COL_FORMAT_TYPE(self, columns=None, typ='str'):
        '''Format specified columns as specfied type'''
        max = 1 if columns is None else None
        columns = colHelper(self.df, columns, max=max)
        typ = [typ] if isinstance(typ, str) else typ
        convert_dict = {c:t for c,t in zip(columns, typ)}
        self.df = self.df.astype(convert_dict)
        self._preview()
        return self
    
    def DATA_COL_RENAME(self, columns):
        '''Rename specfied column/s'''
        # we handle dict for all or subset, OR list for all
        if isinstance(columns, dict):
            self.df.rename(columns = columns, inplace = True)
        else:
            self.df.columns = columns
        self._preview()
        return self
    
    def DATA_COL_REORDER(self, columns):
        '''Reorder column titles in specified order. Convenience method for DATA_COL_MOVE_TO_FRONT'''
        # if not all columns are specified, we order to front and add others to end
        return self.DATA_COL_REORDER_MOVE_TO_FRONT(columns)

    def DATA_COL_REORDER_ASCENDING(self):
        '''Reorder column titles in ascending order'''
        #df.columns = sorted(df.columns.values.tolist())
        self.df = self.df[sorted(self.df.columns.values.tolist())]
        self._preview()
        return self

    def DATA_COL_REORDER_DESCENDING(self):
        '''Reorder column titles in descending order'''
        #df.columns = sorted(df.columns.values.tolist(), reverse = True)
        self.df = self.df[sorted(self.df.columns.values.tolist(), reverse=True)]
        self._preview()
        return self

    def DATA_COL_REORDER_MOVE_TO_BACK(self, columns=None):
        '''Move specified column/s to back'''
        max = 1 if columns is None else None
        colsToMove = colHelper(self.df, columns, max=max)
        otherCols = removeElementsFromList(self.df.columns.values.tolist(), colsToMove)
        self.df = self.df[otherCols + colsToMove]
        self._preview()
        return self
    
    def DATA_COL_REORDER_MOVE_TO_FRONT(self, columns=None):
        '''Move specified column/s to front'''
        max = 1 if columns is None else None
        colsToMove = colHelper(self.df, columns, max=max)
        otherCols = removeElementsFromList(self.df.columns.values.tolist(), colsToMove)
        self.df = self.df[colsToMove + otherCols]
        self._preview()
        return self
    
    def DATA_COL_SORT(self, columns=None, ascending=True):
        '''Sort specified column/s in specified asc/desc order'''
        columns = colHelper(self.df, columns, colsOnNone=True)
        ascending = [ascending for _ in columns]
        self.df.sort_values(by=columns, ascending=ascending, inplace=True, na_position ='last')
        self.df.reset_index(inplace=True, drop=True)
        self._preview()
        return self
    
    def DATA_COL_TRANSFORM_ADD(self, columns=None, num=0):
        '''Format specified column/s values to uppercase'''
        columns = colHelper(self.df, columns, type='number')
        eval_string = 'cell+{}'.format(num)
        return self._DATA_COL_FORMAT_CUSTOM(columns=columns, eval_string=eval_string)
    
    def DATA_COL_TRANSFORM_SUBTRACT(self, columns=None, num=0):
        '''Format specified column/s values to uppercase'''
        columns = colHelper(self.df, columns, type='number')
        eval_string = 'cell-{}'.format(num)
        return self._DATA_COL_FORMAT_CUSTOM(columns=columns, eval_string=eval_string)
    
    def DATA_COL_TRANSFORM_MULTIPLY(self, columns=None, num=0):
        '''Format specified column/s values to uppercase'''
        columns = colHelper(self.df, columns, type='number')
        eval_string = 'cell*{}'.format(num)
        return self._DATA_COL_FORMAT_CUSTOM(columns=columns, eval_string=eval_string)
    
    def DATA_COL_TRANSFORM_DIVIDE(self, columns=None, num=0):
        '''Format specified column/s values to uppercase'''
        columns = colHelper(self.df, columns, type='number')
        eval_string = 'cell/{}'.format(num)
        return self._DATA_COL_FORMAT_CUSTOM(columns=columns, eval_string=eval_string)
    
    def DATA_COL_TRANSFORM_EXPONENT(self, columns=None, num=0):
        '''Format specified column/s values to uppercase'''
        columns = colHelper(self.df, columns, type='number')
        eval_string = 'cell**{}'.format(num)
        return self._DATA_COL_FORMAT_CUSTOM(columns=columns, eval_string=eval_string)
    
    def DATA_COL_TRANSFORM_ROOT(self, columns=None, num=0):
        '''Format specified column/s values to uppercase'''
        columns = colHelper(self.df, columns, type='number')
        eval_string = 'cell**(1./{}.)'.format(num) if 0<=num else '-(-cell)**(1./{}.)'.format(num)
        return self._DATA_COL_FORMAT_CUSTOM(columns=columns, eval_string=eval_string)
    
    def DATA_COL_TRANSFORM_FLOORDIV(self, columns=None, num=0):
        '''Format specified column/s values to uppercase'''
        columns = colHelper(self.df, columns, type='number')
        eval_string = 'cell//{}'.format(num)
        return self._DATA_COL_FORMAT_CUSTOM(columns=columns, eval_string=eval_string)
    
    def DATA_COL_TRANSFORM_MODULUS(self, columns=None, num=0):
        '''Format specified column/s values to uppercase'''
        columns = colHelper(self.df, columns, type='number')
        eval_string = 'cell%{}'.format(num)
        return self._DATA_COL_FORMAT_CUSTOM(columns=columns, eval_string=eval_string)

# DATAFRAME 'ROW' ACTIONS

    def DATA_ROW_ADD(self, rows=None):
        '''Add row at specified index'''
        if rows is None: rows = rowHelper(self.df, max=1)
        if isinstance(rows, tuple):
            rows = list(rows)
        if isinstance(rows, list):
            self.df.loc[-1] = rows
            self.df.index = self.df.index + 1
            self.df.sort_index(inplace=True)
        #else:
        #    self._df = pd.concat([rows, self._df], ignore_index = True)
        self._preview()
        return self
    
    def DATA_ROW_DELETE(self, rows=None):
        rows = list(rows) if isinstance(rows, tuple) else rows
        self.df.drop(self.df.index[rows], inplace=True)
        self.df.reset_index(drop=True, inplace=True)
        self._preview()
        return self
    
    def DATA_ROW_KEEP_BOTTOM(self, numRows=1):
        '''Delete all rows except specified bottom N rows'''
        self.df = self.df.tail(numRows+1)
        self.df.reset_index(drop=True, inplace=True)
        self._preview()
        return self

    def DATA_ROW_KEEP_TOP(self, numRows=1):
        '''Delete all rows except specified top N rows'''
        self.df = self.df.head(numRows+1)
        self.df.reset_index(drop=True, inplace=True)
        self._preview()
        return self

    def DATA_ROW_REVERSE_ORDER(self):
        '''Reorder all rows in reverse order'''
        self.df = self.df[::-1].reset_index(drop = True)
        self._preview()
        return self

    def DATA_ROW_TO_COLHEADER(self, row=0):
        '''Promote row at specified index to column headers'''
        # make new header, fill in blank values with ColN
        newHeader = self.df.iloc[row].squeeze()
        newHeader = newHeader.values.tolist()
        for i in newHeader:
            if i == None: i = 'Col'
        self.DATA_COL_RENAME(newHeader)
        self.DATA_ROW_DELETE([*range(row+1)])
        self._preview()
        return self

    def DATA_ROW_FROM_COLHEADER(self):
        '''Demote column headers to make 1st row of table'''
        self.DATA_ROW_ADD(list(self.df.columns.values))
        newHeader = ['Col' + str(x) for x in range(len(self.df.columns))]
        self.DATA_COL_RENAME(newHeader)
        self._preview()
        return self
    
    # DATAFRAME ACTIONS
    def DATA_APPEND(self, otherdf):
        '''Append a table to bottom of current table'''
        self.df = self.df.append(otherdf, ignore_index=True)
        self._preview()
        return self

    def DATA_GROUP(self, groupby=None, aggregates=None):
        '''Group table contents by specified columns with optional aggregation (sum/max/min etc)'''
        max = 1 if groupby is None else None
        groupby = colHelper(self.df, groupby, max=max)
        if aggregates is None:
            self.DATA_COL_ADD_FIXED(1, 'count')
            c = self.df.columns[-1]
            self.df = self.df.groupby(groupby, as_index=False, dropna=False).agg({c:'count'})
            #self._df = self._df.groupby(groupby, as_index=False, dropna=False).first()
        else:
            self.df = self.df.groupby(groupby, as_index=False, dropna=False).agg(aggregates)
            #self._df.columns = ['_'.join(col).rstrip('_') for col in self._df.columns.values]
        self._preview()
        return self

    def DATA_MERGE(self, otherdf, on, how = 'left'):
        self.df = pd.merge(self.df, otherdf, on=on, how=how)
        self._preview()
        return self

    def DATA_TRANSPOSE(self):
        self.df.transpose()
        self._preview()
        return self

    def DATA_UNPIVOT(self, columns=None):
        columns = colHelper(self.df, columns)
        self.df = pd.melt(self.df, id_vars=columns)
        self._preview()
        return self

    def DATA_PIVOT(self, indexCols, cols, vals):
        #indexCols = list(set(df.columns) - set(cols) - set(vals))
        self.df = self.df.pivot(index = indexCols, columns = cols, values = vals).reset_index().rename_axis(mapper = None,axis = 1)
        self._preview()
        return self
    