import pandas as pd
from pandas.plotting import scatter_matrix as pdsm 
import numpy as np
import pathlib
import matplotlib.pyplot as plt
from collections.abc import Iterable
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
        plt.close("all") # in case of memory needs clearing from previous run
        
    def _repr_pretty_(self, p, cycle): 
        if self._showFig:
            return display(plt.gcf()), display(self._df)
        else:
            return display(self._df)
        
    def __repr__(self): 
        return self._df.__repr__()
    
    def __str__(self): 
        return self._df.__str__()
    
    # DATAFRAME 'COLUMN' ACTIONS
    
    def DF_COL_ADD_FIXED(self, value, name = 'new_column'):
        name = self._toUniqueColName(name)
        self._df[name] = value
        self._showFig = False
        return self
    
    def DF_COL_ADD_INDEX(self, name = 'new_column'):
        name = self._toUniqueColName(name)
        self._df[name] = range(self._df.shape[0])
        self._showFig = False
        return self
    
    def DF_COL_ADD_CUSTOM(self, column, lmda, name = 'new_column'):
        name = self._toUniqueColName(name)
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
        name = self._toUniqueColName(name)
        self._df[name] = self._df[column]
        self._showFig = False
        return self
    
    def DF_COL_DELETE(self, columns):
        columns = self._colNames(columns)
        self._df = self._df.drop(columns, axis = 1)
        self._showFig = False
        return self
    
    def DF_COL_DELETE_EXCEPT(self, columns):
        columns = self._colNames(columns)
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
    
    def DF_COLHEADER_PROMOTE(self, row = 1):
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
    
    def DF_COLHEADER_DEMOTE(self):
        # insert 'demoted' column headers
        self.DF_ROW_ADD(self._df.columns)
        # make new header as Col1, Col2, Coln
        newHeader = ['Col' + str(x) for x in range(len(self._df.columns))]
        # set new col names
        self.DF_COL_RENAME(newHeader)
        self._showFig = False
        return self
    
    def DF_COLHEADER_REORDER_ASC(self):
        self._df.columns = sorted(self._df.columns.values.tolist())
        self._showFig = False
        return self
    
    def DF_COLHEADER_REORDER_DESC(self):
        self._df.columns = sorted(self._df.columns.values.tolist(), reverse = True)
        self._showFig = False
        return self
    
    def DF__STATS(self):
        self._df = self._df.describe()
        self._showFig = False
        return self
    
    # VIZUALIZATION ACTIONS
    
    def VIZ_BOX(self, col=None, by=None, **kwargs):
        col = self._colNames(col, max=1)
        by = self._colNames(by, max=1)
        self._plotHelper(col=col, by=by, type='box', subplots=False, **kwargs)        
        self._showFig = True
        return self
        
    def VIZ_HIST(self, col=None, by=None, bins=10, **kwargs):
        col = self._colNames(col, max=1)
        by = self._colNames(by, max=1)
        self._plotHelper(col=col, by=by, type='hist', subplots=True, bins=bins, **kwargs)        
        self._showFig = True
        return self
        
    def VIZ_HIST_STACKED(self, col=None, by=None, bins=10, **kwargs):
        col = self._colNames(col, max=1)
        by = self._colNames(by, max=1)
        self._plotHelper(col=col, by=by, type='hist_stacked', subplots=False, bins=bins, **kwargs)        
        self._showFig = True
        return self
    
    def _plotHelper(self, col, by, type, **kwargs):
        grp = self._groupingHelper(col, by)
        fig, axs = self._subplotHelper(subplots=kwargs['subplots'], numGrps=grp['num_grp'])
        
        # 1 - n plots
        if type=='hist':
            for a, g in zip(axs, grp['grp_keys']):
                a.hist(grp['by_grp'][g], bins=kwargs['bins'])
                a.set_title(g)
        # 1 plot only
        elif type=='hist_stacked':
                a = axs[0]
                a.hist(grp['all_grp'], bins=kwargs['bins'], density=True, histtype='bar', stacked=True, label=grp['grp_keys'])
                a.legend()
                fig.suptitle('Histogram: ' + col + ' by ' + by)
        # 1 - n plots
        elif type=='box':
                a = axs[0]
                a.boxplot(grp['all_grp'], labels=grp['grp_keys'])
                #a.legend()
                fig.suptitle('Box plot: ' + col + ' by ' + by)
        else:
            return    
            
    def _groupingHelper(self, col, by):
        #before_grouping: self._df
        #num_grp: 0-n
        #by_group: g1: x, gn: xx
        #all_groups: [x, xx]
        #gr_keys: [g1, gn]
        gr_dic = {'before_grouping': self._df}
        # no grouping
        if by == None: 
            gr_dic['num_grp': 0]
        # grouping
        else: 
            gb = self._df.groupby(by)
            grps = {g: gb.get_group(g)[col] for g in list(gb.groups.keys())}
            gr_dic['num_grp'] = len(grps)
            gr_dic['by_grp'] = grps
            gr_dic['all_grp'] = list(grps.values())
            gr_dic['grp_keys'] = sorted(list(grps.keys())) 
        return gr_dic    
    
    def _subplotHelper(self, subplots, numGrps):
        # if no grouping OR 1 group OR groups without subplots, just make 1 x 1
        if numGrps == 0 or numGrps == 1 or subplots == False: 
            fig, axs = plt.subplots(1, 1)
        # otherwise, generate
        else:
            r = 3 if numGrps >= 3 else numGrps % 3
            c = 3 if numGrps >= 7 else ((numGrps - 1) // 3) + 1
            fig, axs = plt.subplots(r, c, sharey=True, tight_layout=True)
            if isinstance(axs, Iterable): axs = axs.flatten().tolist()
        # if single axis, add to list for simplified handling later
        if not isinstance(axs, list): axs = [axs]
        return fig, axs
    
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
    
    def VIZ_BAR(self, x=None, y=None, **kwargs):
        # catch data over 20 indices/groups
        self._df.plot.bar(x=x, y=y, **kwargs)
        self._showFig = True
        return self
    
    def VIZ_BAR_STACKED(self, x=None, y=None, **kwargs):
        self._df.plot.bar(x=x, y=y, stacked=True, **kwargs)
        self._showFig = True
        return self
    
    def VIZ_BARH(self, x=None, y=None, **kwargs):
        self._df.plot.barh(x=x, y=y, **kwargs)
        self._showFig = True
        return self
    
    def VIZ_BARH_STACKED(self, x=None, y=None, **kwargs):
        self._df.plot.barh(x=x, y=y, stacked=True, **kwargs)
        self._showFig = True
        return self
    
    def VIZ_LINE(self, x=None, y=None, **kwargs):
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
    
    def _colNames(self, columns, max = None):
        # convert to list of col names
        # if slice or int 
        if isinstance(columns, slice) or isinstance(columns, int):
            columns = self._df.columns.values.tolist()[columns]
        # or list of int 
        elif isinstance(columns, list) and all(isinstance(c, int) for c in columns):
            columns = self._df.columns[columns].values.tolist()
        # apply 'max' check    
        if isinstance(columns, list) and max != None: 
            if max == 1: columns = columns[0]
            else: columns = columns[:max]
        return columns
    
    def _toUniqueColName(self, name):
        n = 1
        while name in self._df.columns.values.tolist():
            name = name + str(n)
        return name
        
    
    
    
    
