import pandas as pd
from pandas.plotting import scatter_matrix as pdsm 
#import numpy as np
import pathlib
#import matplotlib.pyplot as plt
from collections.abc import Iterable
import plotly.express as px
from PK import *
from configparser import ConfigParser

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
        name = self._toUniqueColName(name)
        self._df[name] = value
        self._fig()
        return self
    
    def DF_COL_ADD_INDEX(self, name = 'new_column', start = 0):
        name = self._toUniqueColName(name)
        self._df[name] = range(start, self._df.shape[0] + start)
        self._fig()
        return self
    
    def DF_COL_ADD_INDEX_FROM_0(self, name = 'new_column'):
        return self.DF_COL_ADD_INDEX(name, start = 0)
    
    def DF_COL_ADD_INDEX_FROM_1(self, name = 'new_column'):
        return self.DF_COL_ADD_INDEX(name, start = 1)
    
    def DF_COL_ADD_CUSTOM(self, column, lmda, name = 'new_column'):
        name = self._toUniqueColName(name)
        self._df[name] = self._df[column].apply(lmda)
        self._fig()
        return self
    
    def DF_COL_ADD_EXTRACT_POSITION_AFTER(self, column, pos, name = 'new_column'):
        self._df = self.DF_COL_ADD_CUSTOM(self._df, column, lambda x: x[pos:], name = name)
        self._fig()
        return self
    
    def DF_COL_ADD_EXTRACT_POSITION_BEFORE(self, column, pos, name = 'new_column'):
        self._df = self.DF_COL_ADD_CUSTOM(self._df, column, lambda x: x[:pos], name = name)
        self._fig()
        return self
    
    def DF_COL_ADD_EXTRACT_CHARS_FIRST(self, column, chars, name = 'new_column'):
        self._df = self.DF_COL_ADD_CUSTOM(self._df, column, lambda x: x[:chars], name = name)
        self._fig()
        return self
    
    def DF_COL_ADD_EXTRACT_CHARS_LAST(self, column, chars, name = 'new_column'):
        self._df = self.DF_COL_ADD_CUSTOM(self._df, column, lambda x: x[-chars:], name = name)
        self._fig()
        return self
    
    def DF_COL_ADD_DUPLICATE(self, column, name = 'new_column'):
        name = self._toUniqueColName(name)
        self._df[name] = self._df[column]
        self._fig()
        return self
    
    def DF_COL_DELETE(self, columns):
        columns = self._colHelper(columns)
        self._df = self._df.drop(columns, axis = 1)
        self._fig()
        return self
    
    def DF_COL_DELETE_EXCEPT(self, columns):
        columns = self._colHelper(columns)
        cols = pq._diff(self._df.columns.values.tolist(), columns)
        self._fig()
        return self.DF_COL_DELETE(cols)
    
    def DF_COL_RENAME(self, columns):
        # we handle dict for all or subset, OR list for all
        if isinstance(columns, dict):
            self._df.rename(columns = columns, inplace = True)
        else:
            self._df.columns = columns
        self._fig()
        return self
    
    #col_reorder list of indices, list of colnames
    
    def DF_COL_FORMAT_TO_UPPERCASE(self, columns = None):
        if columns == None: columns = self._df.columns.values.tolist()
        self._df[columns] = self._df[columns].apply(lambda s: s.str.upper(), axis=0)
        self._fig()
        return self
    
    def DF_COL_FORMAT_TO_LOWERCASE(self, columns = None):
        if columns == None: columns = self._df.columns.values
        self._df[columns] = self._df[columns].apply(lambda s: s.str.lower(), axis=0)
        self._fig()
        return self
    
    def DF_COL_FORMAT_TO_TITLECASE(self, columns = None):
        if columns == None: columns = self._df.columns.values
        self._df[columns] = self._df[columns].apply(lambda s: s.str.title(), axis=0)
        self._fig()
        return self
    
    def DF_COL_FORMAT_STRIP(self, columns = None):
        if columns == None: columns = self._df.columns.values
        self._df[columns] = self._df[columns].apply(lambda s: s.str.strip(), axis=0)
        self._fig()
        return self
    
    def DF_COL_FORMAT_STRIP_LEFT(self, columns = None):
        df = self._df
        if columns == None: columns = df.columns.values
        df[columns] = df[columns].apply(lambda s: s.str.lstrip(), axis=0)
        self._fig()
        return self
    
    def DF_COL_FORMAT_STRIP_RIGHT(self, columns = None):
        if columns == None: columns = self._df.columns.values
        self._df[columns] = self._df[columns].apply(lambda s: s.str.rstrip(), axis=0)
        self._fig()
        return self
    
    def DF_COL_FORMAT_ADD_PREFIX(self, prefix, column):
        self._df[column] = str(prefix) + self._df[column].astype(str)
        self._fig()
        return self
    
    def DF_COL_FORMAT_ADD_SUFFIX(self, suffix, column):
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
        self._df = self._df.round(decimals)
        self._fig()
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
        self._fig()
        return self
    
    def DF_ROW_KEEP_BOTTOM(self, numRows):
        self._df = self._df.tail(numRows)
        self._fig()
        return self
    
    def DF_ROW_KEEP_TOP(self, numRows):
        self._df = self._df.head(numRows)
        self._fig()
        return self
    
    def DF_ROW_REVERSE(self):
        self._df = self._df[::-1].reset_index(drop = True)
        self._fig()
        return self
    
    def DF_ROW_SORT(self, columns, descending = False):
        ascending = 1
        if descending == True: ascending = 0
        self._df = self._df.sort_values(by = columns, axis = 0, ascending = ascending, na_position ='last')
        self._fig()
        return self
    
    #TABLE
    
    def DF__APPEND(self, otherdf):
        self._df = self._df.append(otherdf._df, ignore_index=True)
        self._fig()
        return self
    
    def DF__FILL_DOWN(self):
        self._df = self._df.fillna(method="ffill", axis = 'index', inplace = True)
        self._fig()
        return self
    
    def DF__FILL_UP(self):
        self._df = self._df.fillna(method="bfill", axis = 'index', inplace = True)
        self._fig()
        return self
    
    def DF__FILL_RIGHT(self):
        self._df = self._df.fillna(method="ffill", axis = 'columns', inplace = True)
        self._fig()
        return self
    
    def DF__FILL_LEFT(self):
        self._df = self._df.fillna(method="bfill", axis = 'columns', inplace = True)
        self._fig()
        return self
    
    def DF__GROUP(self, groupby, aggregates = None):
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
        # insert 'demoted' column headers
        self.DF_ROW_ADD(self._df.columns)
        # make new header as Col1, Col2, Coln
        newHeader = ['Col' + str(x) for x in range(len(self._df.columns))]
        # set new col names
        self.DF_COL_RENAME(newHeader)
        self._fig()
        return self
    
    def DF_COLHEADER_REORDER_ASC(self):
        self._df.columns = sorted(self._df.columns.values.tolist())
        self._fig()
        return self
    
    def DF_COLHEADER_REORDER_DESC(self):
        self._df.columns = sorted(self._df.columns.values.tolist(), reverse = True)
        self._fig()
        return self
    
    def DF__STATS(self):
        self._df = self._df.describe()
        self._fig()
        return self
    
    # VIZUALIZATION ACTIONS
    
    def VIZ_BOX(self, x=None, y=None, color=None, facet_col=None, facet_row=None, **kwargs):
        fig = px.box(self._df, x=x, y=y, color=color, facet_col=facet_col, facet_row=facet_row, 
                     color_discrete_sequence=self._colorSwatch, **kwargs)
        self._fig(fig)
        return self
        
    def VIZ_VIOLIN(self, x=None, y=None, color=None, facet_col=None, facet_row=None, **kwargs):
        fig = px.violin(self._df, x=x, y=y, color=color, facet_col=facet_col, facet_row=facet_row, box=True, 
                     color_discrete_sequence=self._colorSwatch, **kwargs)
        self._fig(fig)
        return self
        
    def VIZ_HIST(self, x=None, color=None, facet_col=None, facet_row=None, **kwargs):
        fig = px.histogram(self._df, x=x, color=color, facet_col=facet_col, facet_row=facet_row, 
                     color_discrete_sequence=self._colorSwatch, **kwargs)
        self._fig(fig)
        return self
    
    def VIZ_HIST_LIST(self, color=None, **kwargs):
        for c in self._df.columns:
            fig = px.histogram(self._df, x=c, color=color, color_discrete_sequence=self._colorSwatch, **kwargs)
            self._fig(fig)
        self._fig(preview = 'all_charts')
        return self
    
    def VIZ_SCATTER(self, x=None, y=None, color=None, size=None, symbol=None, facet_col=None, facet_row=None, **kwargs):
        fig = px.scatter(self._df, x=x, y=y, color=color, size=size, symbol=symbol, facet_col=facet_col, facet_row=facet_row, 
                     color_discrete_sequence=self._colorSwatch, **kwargs)
        self._fig(fig)
        return self
        
    def VIZ_BAR(self, x=None, y=None, color=None, facet_col=None, facet_row=None, **kwargs):
        fig = px.bar(self._df, x=x, y=y, color=color, facet_col=facet_col, facet_row=facet_row, 
                     color_discrete_sequence=self._colorSwatch, **kwargs)
        self._fig(fig)
        return self
    
    def VIZ_LINE(self, x=None, y=None, color=None, facet_col=None, facet_row=None, markers=True, **kwargs):
        fig = px.line(self._df, x=x, y=y, color=color, facet_col=facet_col, facet_row=facet_row, markers=markers, 
                     color_discrete_sequence=self._colorSwatch, **kwargs)
        self._fig(fig)
        return self
    
    def VIZ_TREEMAP(self, path, values, color=None, **kwargs):
        fig = px.treemap(self._df, path=path, values=values, color=color, color_discrete_sequence=self._colorSwatch, **kwargs)
        self._fig(fig)
        return self
    
    def VIZ_SCATTERMATRIX(self, dimensions=None, color=None, **kwargs):
        fig = px.scatter_matrix(self._df, dimensions=dimensions, color_discrete_sequence=self._colorSwatch, color=color, **kwargs)
        self._fig(fig)
        return self
    
    @property
    def REPORT_SET_VIZ_COLORS_PLOTLY(self):
        return self._REPORT_SET_VIZ_COLORS(px.colors.qualitative.Plotly)
    
    @property
    def REPORT_SET_VIZ_COLORS_D3(self):
        return self._REPORT_SET_VIZ_COLORS(px.colors.qualitative.D3)
    
    @property
    def REPORT_SET_VIZ_COLORS_G10(self):
        return self._REPORT_SET_VIZ_COLORS(px.colors.qualitative.G10)
    
    @property
    def REPORT_SET_VIZ_COLORS_T10(self):
        return self._REPORT_SET_VIZ_COLORS(px.colors.qualitative.T10)
    
    @property
    def REPORT_SET_VIZ_COLORS_ALPHABET(self):
        return self._REPORT_SET_VIZ_COLORS(px.colors.qualitative.Alphabet)
    
    @property
    def REPORT_SET_VIZ_COLORS_DARK24(self):
        return self._REPORT_SET_VIZ_COLORS(px.colors.qualitative.Dark24)
    
    @property
    def REPORT_SET_VIZ_COLORS_LIGHT24(self):
        return self._REPORT_SET_VIZ_COLORS(px.colors.qualitative.Light24)
    
    @property
    def REPORT_SET_VIZ_COLORS_SET1(self):
        return self._REPORT_SET_VIZ_COLORS(px.colors.qualitative.Set1)
    
    @property
    def REPORT_SET_VIZ_COLORS_PASTEL1(self):
        return self._REPORT_SET_VIZ_COLORS(px.colors.qualitative.Pastel1)
    
    @property
    def REPORT_SET_VIZ_COLORS_DARK2(self):
        return self._REPORT_SET_VIZ_COLORS(px.colors.qualitative.Dark2)
    
    @property
    def REPORT_SET_VIZ_COLORS_SET2(self):
        return self._REPORT_SET_VIZ_COLORS(px.colors.qualitative.Set2)
    
    @property
    def REPORT_SET_VIZ_COLORS_PASTEL2(self):
        return self._REPORT_SET_VIZ_COLORS(px.colors.qualitative.Pastel2)
    
    @property
    def REPORT_SET_VIZ_COLORS_SET3(self):
        return self._REPORT_SET_VIZ_COLORS(px.colors.qualitative.Set3)
    
    @property
    def REPORT_SET_VIZ_COLORS_ANTIQUE(self):
        return self._REPORT_SET_VIZ_COLORS(px.colors.qualitative.Antique)
    
    @property
    def REPORT_SET_VIZ_COLORS_BOLD(self):
        return self._REPORT_SET_VIZ_COLORS(px.colors.qualitative.Bold)
    
    @property
    def REPORT_SET_VIZ_COLORS_PASTEL(self):
        return self._REPORT_SET_VIZ_COLORS(px.colors.qualitative.Pastel)
    
    @property
    def REPORT_SET_VIZ_COLORS_PRISM(self):
        return self._REPORT_SET_VIZ_COLORS(px.colors.qualitative.Prism)
    
    @property
    def REPORT_SET_VIZ_COLORS_SAFE(self):
        return self._REPORT_SET_VIZ_COLORS(px.colors.qualitative.Safe)
    
    @property
    def REPORT_SET_VIZ_COLORS_VIVID(self):
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
    
    def MAKE_WINDOWS_SCRIPT_FILES(self):
        return
        #get this filename (except extension)
        #make .bat file with same name
        # write:
        # call C:\Users\mdaws\anaconda3\Scripts\activate.bat
        # call FILENAME.py
        # do same for .py file (don't use extensions)
    
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

    
