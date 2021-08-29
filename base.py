import pandas as pd
from pandas.plotting import scatter_matrix as pdsm 
import numpy as np
import pathlib
import matplotlib.pyplot as plt
from collections.abc import Iterable
import plotly.express as px


### PLOTHELPER CLASS & SUBCLASSES ###
        
class PlotHelper(object):
    
    def __init__(self, df, col, by, **kwargs):
        super(PlotHelper, self).__init__()
        self._df = df
        self.col = col
        self.by = by
        self.kwargs = kwargs
        self.grp = self._groupingHelper()
        self.fig, self.axs = self._subplotHelper()
        
    def _groupingHelper(self):
        '''Prepare data into groups in preparation for plotting'''
        
        gr_dic = {'_no_grp': self._df}
        if self.by == None: 
            gr_dic['num_grp'] = 0
            if self.col is not None: gr_dic['by_grp'] = {'_no_grp': self._df[self.col]}
            else: gr_dic['by_grp'] = {'_no_grp': self._df}
            gr_dic['all_grp'] = list(gr_dic['by_grp'].values())
            gr_dic['grp_keys'] = sorted(list(gr_dic['by_grp'].keys()))
        else: 
            gb = self._df.groupby(self.by)
            if self.col is not None: grps = {g: gb.get_group(g)[self.col] for g in list(gb.groups.keys())}
            else: grps = {g: gb.get_group(g) for g in list(gb.groups.keys())}
            gr_dic['num_grp'] = len(grps)
            gr_dic['by_grp'] = grps
            gr_dic['all_grp'] = list(grps.values())
            gr_dic['grp_keys'] = sorted(list(grps.keys())) 
        return gr_dic    
    
    def _subplotHelper(self):
        '''Prepare base figure/plot(s)'''
        subplots=self.kwargs['subplots']
        numGrps=self.grp['num_grp']
        
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
    
    def plot(self):
        return
    
    def _colorHelper(self, num):
        num = 9 if num > 9 else num
        col_pal = {
            0: 'tab:blue',
            1: 'tab:orange',
            2: 'tab:green',
            3: 'tab:red',
            4: 'tab:purple',
            5: 'tab:brown',
            6: 'tab:pink',
            7: 'tab:gray',
            8: 'tab:olive',
            9: 'tab:cyan'}
        return col_pal[num]
    
    def _formatHelper(self, **kwargs):
        #default
        axesProp = {
            #'adjustable': None,
            #'agg_filter': None,
            #'alpha': None,
            #'anchor': None,
            #'animated': None,
            #'aspect': None,
            #'autoscale_on': None,
            #'autoscalex_on': None,
            #'autoscaley_on':None,
            #'axes_locator': None,
            #'axisbelow': None,
            #'box_aspect': None,
            #'clip_box': None,
            #'clip_on': None,
            #'clip_path': None,
            #'contains': None,
            #'facecolor': None,
            #'figure': None,
            #'frame_on': None,
            #'gid': None,
            #'in_layout': None,
            #'label': None,
            #'navigate': None,
            #'navigate_mode': None,
            #'path_effects': None,
            #'picker': None,
            #'position': None,
            #'prop_cycle': None,
            #'rasterization_zorder': None,
            #'rasterized': None,
            #'sketch_params': None,
            #'snap': None,
            #'title': None,
            #'transform': None,
            #'url': None,
            #'visible': None
            #'xbound': None,
            #'xlabel': None,
            #'xlim': None,
            #'xmargin': None,
            #'xscale': None,
            #'xticklabels': None,
            #'xticks': None,
            #'ybound': None,
            #'ylabel': None,
            #'ylim': None,
            #'ymargin': None,
            #'yscale': None,
            #'yticklabel': None,
            #'yticks': None,
            #'zorder': None
        }
        
        xticklabelProp = {
            #'fontsize': rcParams['axes.titlesize'],
            #'fontweight': rcParams['axes.titleweight'],
            #'verticalalignment': 'baseline',
            #'horizontalalignment': loc
        }
        
        #given
        #use kwargs if given
        ##plt.setp(axs, xticks=[y + 1 for y in range(len(all_data))], xticklabels=['x1', 'x2', 'x3', 'x4'])
        plt.setp(plt.gca(), **axesProp)

class BoxPlotHelper(PlotHelper):
    
    def __init__(self, df, col, by, **kwargs):
        PlotHelper.__init__(self, df, col, by, **kwargs) 
        
    def plot(self):
        grp = self.grp
        axs = self.axs
        kwargs = self.kwargs
        fig = self.fig
        for a, g in zip(axs, grp['grp_keys']):
            a.boxplot(grp['by_grp'][g], labels=self.col)
            a.set_title(g)
        fig.suptitle('Box plot')
        #self._formatHelper()
        #a.legend()
            
class HistSubplotsPlotHelper(PlotHelper):
    
    def __init__(self, df, col, by, **kwargs):
        PlotHelper.__init__(self, df, col, by, **kwargs) 
        
    def plot(self):
        grp = self.grp
        axs = self.axs
        kwargs = self.kwargs
        for a, g in zip(axs, grp['grp_keys']):
            a.hist(grp['by_grp'][g], bins=kwargs['bins'])
            a.set_title(g)       
    
class HistPlotHelper(PlotHelper):
    
    def __init__(self, df, col, by, **kwargs):
        PlotHelper.__init__(self, df, col, by, **kwargs) 
        
    def plot(self):
        grp = self.grp
        axs = self.axs
        kwargs = self.kwargs
        fig = self.fig
        a = axs[0]
        a.hist(grp['all_grp'], bins=kwargs['bins'], density=True, histtype='bar', stacked=True, label=grp['grp_keys'])
        a.legend()
        fig.suptitle('Histogram')

class BarStackedPlotHelper(PlotHelper):
    
    def __init__(self, df, col, by, **kwargs):
        PlotHelper.__init__(self, df, col, by, **kwargs) 
        
    def plot(self):
        grp = self.grp
        axs = self.axs
        kwargs = self.kwargs
        fig = self.fig
        a = axs[0]
        bottom = None
        x_axis = grp['_no_grp'][self.by]
        for c in self.col:
            data = grp['_no_grp'][c].to_numpy()
            a.bar(x_axis, data, bottom=bottom, label=c)
            if bottom is None: bottom = data
            else: bottom = np.add(data, bottom)
        a.legend()
        a.set_title('Bar plot')
        
class LinePlotHelper(PlotHelper):
    
    def __init__(self, df, col, by, **kwargs):
        PlotHelper.__init__(self, df, col, by, **kwargs) 
        
    def plot(self):
        grp = self.grp
        axs = self.axs
        kwargs = self.kwargs
        fig = self.fig
        a = axs[0]
        x_axis = self.col
        for g, c in zip(grp['grp_keys'], range(len(grp['grp_keys']))):
            data = grp['by_grp'][g]
            color = self._colorHelper(c)
            for index, rows in data.iterrows():
                # Create list for the current row
                y_axis =list(rows)
                #a.plot(x_axis, y_axis, color=color, linewidth=0.5)
            # plot mean
            a.plot(x_axis, data.mean(), color=color, linewidth=3, label=g)
        a.legend()
        a.set_title('Line plot')

class ScatterPlotHelper(PlotHelper):
    
    def __init__(self, df, col, by, **kwargs):
        PlotHelper.__init__(self, df, col, by, **kwargs) 
        
    def plot(self):
        grp = self.grp
        axs = self.axs
        kwargs = self.kwargs
        for a, g in zip(axs, grp['grp_keys']):
            a.scatter()
            a.set_title(g)       
    
    

### PLOTLYHELPER CLASS & SUBCLASSES ###
        
class PlotlyHelper(object):
    
    def __init__(self, df, col, by, **kwargs):
        super(PlotHelper, self).__init__()
        self._df = df
        self.col = col
        self.by = by
        self.kwargs = kwargs
        self.grp = self._groupingHelper()
        self.fig, self.axs = self._subplotHelper()
        
    def _groupingHelper(self):
        '''Prepare data into groups in preparation for plotting'''
        
        gr_dic = {'_no_grp': self._df}
        if self.by == None: 
            gr_dic['num_grp'] = 0
            if self.col is not None: gr_dic['by_grp'] = {'_no_grp': self._df[self.col]}
            else: gr_dic['by_grp'] = {'_no_grp': self._df}
            gr_dic['all_grp'] = list(gr_dic['by_grp'].values())
            gr_dic['grp_keys'] = sorted(list(gr_dic['by_grp'].keys()))
        else: 
            gb = self._df.groupby(self.by)
            if self.col is not None: grps = {g: gb.get_group(g)[self.col] for g in list(gb.groups.keys())}
            else: grps = {g: gb.get_group(g) for g in list(gb.groups.keys())}
            gr_dic['num_grp'] = len(grps)
            gr_dic['by_grp'] = grps
            gr_dic['all_grp'] = list(grps.values())
            gr_dic['grp_keys'] = sorted(list(grps.keys())) 
        return gr_dic    
    
    def _subplotHelper(self):
        '''Prepare base figure/plot(s)'''
        subplots=self.kwargs['subplots']
        numGrps=self.grp['num_grp']
        
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
    
    def plot(self):
        return
    
    def _colorHelper(self, num):
        num = 9 if num > 9 else num
        col_pal = {
            0: 'tab:blue',
            1: 'tab:orange',
            2: 'tab:green',
            3: 'tab:red',
            4: 'tab:purple',
            5: 'tab:brown',
            6: 'tab:pink',
            7: 'tab:gray',
            8: 'tab:olive',
            9: 'tab:cyan'}
        return col_pal[num]
    
    def _formatHelper(self, **kwargs):
        #default
        axesProp = {
            #'adjustable': None,
            #'agg_filter': None,
            #'alpha': None,
            #'anchor': None,
            #'animated': None,
            #'aspect': None,
            #'autoscale_on': None,
            #'autoscalex_on': None,
            #'autoscaley_on':None,
            #'axes_locator': None,
            #'axisbelow': None,
            #'box_aspect': None,
            #'clip_box': None,
            #'clip_on': None,
            #'clip_path': None,
            #'contains': None,
            #'facecolor': None,
            #'figure': None,
            #'frame_on': None,
            #'gid': None,
            #'in_layout': None,
            #'label': None,
            #'navigate': None,
            #'navigate_mode': None,
            #'path_effects': None,
            #'picker': None,
            #'position': None,
            #'prop_cycle': None,
            #'rasterization_zorder': None,
            #'rasterized': None,
            #'sketch_params': None,
            #'snap': None,
            #'title': None,
            #'transform': None,
            #'url': None,
            #'visible': None
            #'xbound': None,
            #'xlabel': None,
            #'xlim': None,
            #'xmargin': None,
            #'xscale': None,
            #'xticklabels': None,
            #'xticks': None,
            #'ybound': None,
            #'ylabel': None,
            #'ylim': None,
            #'ymargin': None,
            #'yscale': None,
            #'yticklabel': None,
            #'yticks': None,
            #'zorder': None
        }
        
        xticklabelProp = {
            #'fontsize': rcParams['axes.titlesize'],
            #'fontweight': rcParams['axes.titleweight'],
            #'verticalalignment': 'baseline',
            #'horizontalalignment': loc
        }
        
        #given
        #use kwargs if given
        ##plt.setp(axs, xticks=[y + 1 for y in range(len(all_data))], xticklabels=['x1', 'x2', 'x3', 'x4'])
        plt.setp(plt.gca(), **axesProp)

class HistPlotlyHelper(PlotlyHelper):
    
    def __init__(self, df, col, by, **kwargs):
        PlotlyHelper.__init__(self, df, col, by, **kwargs) 
        
    def plot(self):
        grp = self.grp
        axs = self.axs
        kwargs = self.kwargs
        
        fig = px.histogram(df, x="total_bill")
        fig.show()

        #fig = self.fig
        #a = axs[0]
        #a.hist(grp['all_grp'], bins=kwargs['bins'], density=True, histtype='bar', stacked=True, label=grp['grp_keys'])
        #a.legend()
        #fig.suptitle('Histogram')

### PQ CLASS - QUERY INTERFACE ###

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
        
        self._figs = []
        
    def _repr_pretty_(self, p, cycle): 
        if self._showFig:
            #return display(plt.gcf()), display(self._df)
            return display(self._figs[-1]), display(self._df)
        else:
            return display(self._df)
        
    def __repr__(self): 
        return self._df.__repr__()
    
    def __str__(self): 
        return self._df.__str__()
    
    def _fig(self, fig = None):
        if fig == None:
            self._showFig = False
        else:
            self._showFig = True
            fig.update_traces(dict(marker_line_width=0))
            self._figs.append(fig)
            
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
            self._df = self._df.groupby(groupby, as_index=False).first()
        else:
            self._df = self._df.groupby(groupby, as_index=False).agg(aggregates)
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
    
    def VIZ_BOX(self, x=None, y=None, color=None, facet_col=None, facet_row=None):
        fig = px.box(self._df, x=x, y=y, color=color, facet_col=facet_col, facet_row=facet_row)
        self._fig(fig)
        #fig.show()
        return self
        
    def VIZ_VIOLIN(self, x=None, y=None, color=None, facet_col=None, facet_row=None):
        fig = px.violin(self._df, x=x, y=y, color=color, facet_col=facet_col, facet_row=facet_row, box=True)
        self._fig(fig)
        #fig.show()
        return self
        
    def VIZ_HIST(self, x=None, color=None, facet_col=None, facet_row=None, bins=20):
        fig = px.histogram(self._df, x=x, color=color, facet_col=facet_col, facet_row=facet_row, nbins=bins)
        self._fig(fig)
        #fig.show()
        return self
        
    def VIZ_SCATTER(self, x=None, y=None, color=None, size=None, symbol=None, facet_col=None, facet_row=None):
        fig = px.scatter(self._df, x=x, y=y, color=color, size=size, symbol=symbol, facet_col=facet_col, facet_row=facet_row)
        self._fig(fig)
        #fig.show()
        return self
        
    def VIZ_BAR(self, x=None, y=None, color=None, facet_col=None, facet_row=None):
        fig = px.bar(self._df, x=x, y=y, color=color, facet_col=facet_col, facet_row=facet_row)
        self._fig(fig)
        #fig.show()
        return self
    
    def VIZ_LINE(self, x=None, y=None, color=None, facet_col=None, facet_row=None, markers=True):
        fig = px.line(self._df, x=x, y=y, color=color, facet_col=facet_col, facet_row=facet_row, markers=markers)
        self._fig(fig)
        #fig.show()
        return self
    
    def VIZ_TREEMAP(self, path, values, color=None):
        fig = px.treemap(self._df, path=path, values=values, color=color)
        self._fig(fig)
        #fig.show()
        return self
    
    def ABOUT_DF(self):
        print(self._df.info())
        self._showFig = False
        return self
    
    def SAVE_VIZ_PNG(self):
        #for i in plt.get_fignums():
        #    plt.figure(i)
        #    plt.savefig('figure%d.png' % i)
        for i, fig in enumerate(self._figs):
            fig.write_image('figure%d.png' % i) 
        return self
    
    def SAVE_VIZ_HTML(self):
        for i, fig in enumerate(self._figs):
            fig.write_html('figure%d.html' % i) 
        return self
    
    def SAVE_DF(self, path = 'query_write.csv'):
        self._df.to_csv(path, index=False)
        self._showFig = False
        return self
    
### UTILITIES ###
    
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
    
    def _common(self, l1, l2):
        if l1 is None or l2 is None: return None
        if not isinstance(l1, list): l1 = [l1]
        if not isinstance(l2, list): l2 = [l2]
        a_set = set(l1)
        b_set = set(l2)
        
        # check length
        if len(a_set.intersection(b_set)) > 0:
            return list(a_set.intersection(b_set)) 
        else:
            return None
    
    def _colHelper(self, columns, max = None, type = None, colsOnNone = True):
        
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
        else: columns = self._common(columns, df.columns.values.tolist())           
        
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
    
    def DF_ROW_KEEP_TOP(self, numRows):
        self._df = self._df.head(numRows)
        self._showFig = False
    
    def _toUniqueColName(self, name):
        n = 1
        while name in self._df.columns.values.tolist():
            name = name + str(n)
        return name
    
    '''    
    def VIZ_BOX_OLD(self, col=None, by=None, **kwargs):
        col = self._colHelper(col, max=5, type='number', colsOnNone=True)
        by = self._colHelper(by, max=1, colsOnNone=False)
        BoxPlotHelper(df=self._df, col=col, by=by, subplots=True, **kwargs).plot()
        self._showFig = True
        return self
        
    def VIZ_HIST_OLD(self, col=None, by=None, bins=10, **kwargs):
        col = self._colHelper(col, max=1, colsOnNone=True)
        by = self._colHelper(by, max=1, colsOnNone=False)
        #HistPlotHelper(df=self._df, col=col, by=by, subplots=False, bins=bins, **kwargs).plot()
        HistPlotlyHelper(df=self._df, col=col, by=by, subplots=False, bins=bins, **kwargs).plot()
        self._showFig = True
        return self

    def VIZ_SCATTER_OLD(self, col=None, by=None, **kwargs):
        col = self._colHelper(col, max=2, colsOnNone=True)
        by = self._colHelper(by, max=1, colsOnNone=False)
        ScatterPlotHelper(df=self._df, col=col, by=by, subplots=False, **kwargs).plot()
        self._showFig = True
        return self

    def VIZ_BAR_OLD(self, col=None, by=None, **kwargs):
        col = self._colHelper(col, max=1, type='number', colsOnNone=True)
        by = self._colHelper(by, max=1, colsOnNone=False)
        df = self._rowHelper(self._df, max = 10, head = True)
        BarStackedPlotHelper(df=df, col=col, by=by, subplots=False, **kwargs).plot()
        self._showFig = True
        return self

    def VIZ_LINE_OLD(self, col=None, by=None, **kwargs):
        col = self._colHelper(col, type='number')
        by = self._colHelper(by, max=1, colsOnNone=False)
        LinePlotHelper(df=self._df, col=col, by=by, subplots=False, **kwargs).plot()
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
        # 1 plot only
        elif type=='bar_stacked':
                a = axs[0]
                bottom = None
                x_axis = grp['no_grp'][by]
                for c in col:
                    print(bottom)
                    data = grp['no_grp'][c].to_numpy()
                    a.bar(x_axis, data, bottom=bottom, label=c)
                    if bottom is None: bottom = data
                    else: bottom = np.add(data, bottom)
                a.legend()
                a.set_title('Bar plot')
        # 1 plot, 
        elif type=='line':
                a = axs[0]
                x_axis = col
                for g, c in zip(grp['grp_keys'], range(len(grp['grp_keys']))):
                    data = grp['by_grp'][g]
                    color = self._colorHelper(c)
                    for index, rows in data.iterrows():
                        # Create list for the current row
                        y_axis =list(rows)
                        #a.plot(x_axis, y_axis, color=color, linewidth=0.5)
                    # plot mean
                    a.plot(x_axis, data.mean(), color=color, linewidth=3, label=g)
                a.legend()
                a.set_title('Line plot')
        else:
            return
    
    def _colorHelper(self, num):
        num = 9 if num > 9 else num
        col_pal = {
            0: 'tab:blue',
            1: 'tab:orange',
            2: 'tab:green',
            3: 'tab:red',
            4: 'tab:purple',
            5: 'tab:brown',
            6: 'tab:pink',
            7: 'tab:gray',
            8: 'tab:olive',
            9: 'tab:cyan'}
        return col_pal[num]
            
    def _groupingHelper(self, col, by):
        gr_dic = {'no_grp': self._df}
        if by == None: 
            gr_dic['num_grp'] = 0
            gr_dic['by_grp'] = {'no_grp': self._df[col]}
            gr_dic['all_grp'] = list(gr_dic['by_grp'].values())
            gr_dic['grp_keys'] = sorted(list(gr_dic['by_grp'].keys()))
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
'''
    

    
