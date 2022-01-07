from .base import *
from .data import *
from .util import *

#non-standard libraries
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

DATATYPE_VIZ = 'viz'
DATATYPES.extend([
    DATATYPE_VIZ
])

PREVIEWER_CHART_CURRENT = 'previewer_chart_current'
PREVIEWTYPES.extend([
    PREVIEWER_CHART_CURRENT
])

WRITER_SIMPLE_VIZ = 'writer_simple_viz'
WRITERTYPES.extend([
    WRITER_SIMPLE_VIZ
])

FIGURE_CONFIG_SHOW = {
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

FIGURE_CONFIG_BASE =  {
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

class Viz(Data):   
    def __init__(self, source):
        super().__init__(source)
        
        #extend base data structure
        self._data[DATATYPE_VIZ] = {
                'active':None,
                'stack':[]
        }
        
        #set plotly color palette to our preferred one
        self.REPORT_SET_VIZ_COLORS_ANTIQUE
    
    @property
    def viz(self):
        return self._data[DATATYPE_VIZ]['active']
    
    @viz.setter
    def viz(self, v):
        self._data[DATATYPE_VIZ]['active'] = v
    
    def _fig(self, fig=None, updateCurrentFigure=False, settings=None, overwrite=True, preview=PREVIEWER_CHART_CURRENT):
        '''Handles figure displaying for IPython'''
        if fig is not None and not isinstance(fig, list):
            d = {**FIGURE_CONFIG_BASE, **settings} if settings else FIGURE_CONFIG_BASE
            fig.update_layout(dict1=d, overwrite=overwrite)
            self._append(DATATYPE_VIZ, fig)
            self._preview(preview=PREVIEWER_CHART_CURRENT)
        elif fig is not None and isinstance(fig, list):
            d = {**FIGURE_CONFIG_BASE, **settings} if settings else FIGURE_CONFIG_BASE
            for f in fig:
                f.update_layout(dict1=d, overwrite=overwrite)
            self._append(DATATYPE_VIZ, fig)
            self._preview(preview=PREVIEWER_CHART_CURRENT)
        elif updateCurrentFigure:
            fig = self._figs[-1]
            d = {**FIGURE_CONFIG_BASE, **settings} if settings else FIGURE_CONFIG_BASE
            fig.update_layout(dict1=d, overwrite=overwrite)
            self._append(DATATYPE_VIZ, fig)
            self._preview(preview=PREVIEWER_CHART_CURRENT)
        else:
            self._preview(preview) 
    
# VIZUALIZATION ACTIONS
    
    def REPORT_SAVE_VIZ_AS_HTML(self, tar):
        self._write(tar)
        return self
        
    def VIZ_AREA(self, x=None, y=None, color=None, facet_col=None, facet_row=None, markers=True, **kwargs):
        '''Draw a line plot with shaded area'''
        x, y, color, facet_col, facet_row = (colHelper(df=self.df, columns=i, max=1, colsOnNone=False, forceReturnAsList=False) for i in [x, y, color, facet_col, facet_row])
        fig = px.area(data_frame=self.df, x=x, y=y, color=color, facet_col=facet_col, facet_row=facet_row, #markers=markers, 
                     color_discrete_sequence=self._colorSwatch, **kwargs)
        self._fig(fig)
        return self
    
    def VIZ_BAR(self, x=None, y=None, color=None, facet_col=None, facet_row=None, **kwargs):
        '''Draw a bar plot'''
        x, y, color, facet_col, facet_row = (colHelper(df=self.df, columns=i, max=1, colsOnNone=False, forceReturnAsList=False) for i in [x, y, color, facet_col, facet_row])
        fig = px.histogram(data_frame=self.df, x=x, y=y, color=color, facet_col=facet_col, facet_row=facet_row, 
                     color_discrete_sequence=self._colorSwatch, **kwargs)
        self._fig(fig)
        return self
    
    def VIZ_BOX(self, x=None, y=None, color=None, facet_col=None, facet_row=None, **kwargs):
        '''Draw a box plot'''
        x, y, color, facet_col, facet_row = (colHelper(df=self.df, columns=i, max=1, colsOnNone=False, forceReturnAsList=False) for i in [x, y, color, facet_col, facet_row])
        fig = px.box(data_frame=self.df, x=x, y=y, color=color, facet_col=facet_col, facet_row=facet_row, 
                     color_discrete_sequence=self._colorSwatch, **kwargs)
        self._fig(fig)
        return self
        
    def VIZ_DATASTATS(self):
        '''Show basic summary statistics of table contents'''
        stats = self.df.describe(include='all').T
        stats.insert(0, 'Feature', stats.index)
        self._append(DATATYPE_DATAFRAME, stats)
        self.DATA_COL_ADD_INDEX_FROM_0(name='No').DATA_COL_REORDER_MOVE_TO_FRONT(columns='No')
        self.VIZ_TABLE()
        self._pop(DATATYPE_DATAFRAME)
        return self
    
    def VIZ_HIST(self, x=None, color=None, facet_col=None, facet_row=None, **kwargs):
        '''Draw a hisotgram'''
        x, color, facet_col, facet_row = (colHelper(self.df, columns=i, max=1, colsOnNone=False, forceReturnAsList=False) for i in [x, color, facet_col, facet_row])
        fig = px.histogram(data_frame=self.df, x=x, color=color, facet_col=facet_col, facet_row=facet_row, 
                     color_discrete_sequence=self._colorSwatch, **kwargs)
        self._fig(fig)
        return self
    
    def VIZ_HIST_LIST(self, color=None, **kwargs):
        '''Draw a histogram for all fields in current dataframe'''
        color = colHelper(df=self.df, columns=color, max=1, colsOnNone=False, forceReturnAsList=False)
        v = [px.histogram(data_frame=self.df, x=c, color=color, color_discrete_sequence=self._colorSwatch, **kwargs) for c in self.df.columns]
        self._fig(v)
        return self
    
    def VIZ_ICICLE(self, path, values, root='All data', **kwargs):
        '''Draw a treemap plot'''
        path = [px.Constant("All data")] + colHelper(df=self.df, columns=path)
        values = colHelper(df=self.df, columns=values, max=1, type='number', forceReturnAsList=False)
        # make leaf dict (isna), update
        
        #p = self._colHelper(path)
        #d = self._df[p].groupby(p, as_index=False, dropna=False).first()
        #d = d[d.isna().any(axis=1)].to_dict(orient='records')
        #print(d)
        
        #d = df.groupby('Dia').apply(lambda a: dict(a.groupby('macap').apply(lambda x: dict(zip(x['transmission'], x['bytes'])))))
        #d = d.to_dict()
        
        # treemap, icicle, sunburst break on NaN. Replace with 'None' for this call
        df1 = self.df.where(pd.notnull, None)
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
        x, y, color, facet_col, facet_row = (colHelper(df=self.df, columns=i, max=1, colsOnNone=False, forceReturnAsList=False) for i in [x, y, color, facet_col, facet_row])
        fig = px.line(data_frame=self.df, x=x, y=y, color=color, facet_col=facet_col, facet_row=facet_row, #markers=markers, 
                      color_discrete_sequence=self._colorSwatch, **kwargs)
        self._fig(fig)
        return self
    
    def VIZ_SCATTER(self, x=None, y=None, color=None, facet_col=None, facet_row=None, **kwargs): #size=None, symbol=None
        '''Draw a scatter plot'''
        x, y, color, facet_col, facet_row = (colHelper(df=self.df, columns=i, max=1, colsOnNone=False, forceReturnAsList=False) for i in [x, y, color, facet_col, facet_row])
        fig = px.scatter(data_frame=self.df, x=x, y=y, color=color, facet_col=facet_col, facet_row=facet_row, 
                     color_discrete_sequence=self._colorSwatch, **kwargs)
        self._fig(fig)
        return self
        
    def VIZ_SCATTERMATRIX(self, dimensions=None, color=None, **kwargs):
        '''Draw a scatter matrix plot'''
        dimensions, color = (colHelper(df=self.df, columns=i, max=j, colsOnNone=False, forceReturnAsList=False) for i, j in [(dimensions, None), (color, 1)])
        fig = px.scatter_matrix(data_frame=self.df, dimensions=dimensions, color_discrete_sequence=self._colorSwatch, color=color, **kwargs)
        self._fig(fig)
        return self
    
    def VIZ_SUNBURST(self, path, values, root='All data', **kwargs):
        '''Draw a treemap plot'''
        path = [px.Constant("All data")] + colHelper(df=self.df, columns=path)
        values = colHelper(df=self.df, columns=values, max=1, type='number', forceReturnAsList=False)
        # treemap, icicle, sunburst break on NaN. Replace with 'None' for this call
        df1 = self.df.where(pd.notnull, None)
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
        columns = colHelper(df=self.df, columns=columns)
        cell_values = self.df[columns].to_numpy().T
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
        path = [px.Constant("All data")] + colHelper(df=self.df, columns=path)
        values = colHelper(df=self.df, columns=values, max=1, type='number', forceReturnAsList=False)
        # treemap, icicle, sunburst break on NaN. Replace with 'None' for this call
        df1 = self.df.where(pd.notnull, None)
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
        x, y, color, facet_col, facet_row = (colHelper(df=self.df, columns=i, max=1, colsOnNone=False, forceReturnAsList=False) for i in [x, y, color, facet_col, facet_row])
        fig = px.violin(data_frame=self.df, x=x, y=y, color=color, facet_col=facet_col, facet_row=facet_row, box=True, 
                     color_discrete_sequence=self._colorSwatch, **kwargs)
        self._fig(fig)
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

"""    
    #@property
    def REPORT_PREVIEW_CHARTS(self):
        self._fig(preview = 'all_charts')
        return self
    
    #@property
    def REPORT_PREVIEW_FULL(self):
        self._fig(preview = 'full')
        return self
    
    def REPORT_SAVE_ALL(self, path = None):
        self.REPORT_SAVE_DATA(path = path)
        #self.REPORT_SAVE_VIZ_PNG(path = path)
        self.REPORT_SAVE_VIZ_HTML(path = path)
        return self
"""  

@registerPreviewer 
class PreviewerChartCurrent(BasePreviewer):
    @classmethod
    def type(cls):
        '''Returns key used to regsiter type'''
        return PREVIEWER_CHART_CURRENT
    
    @classmethod
    def preview(self, data):
        '''Returns dataframe based on config'''
        viz = data[DATATYPE_VIZ]['active']
        
        #if viz contains multiple plots eg. HIST_LIST 
        if isinstance(viz, list):
            return tuple([v.show(config=FIGURE_CONFIG_SHOW) for v in viz]), PREVIEWERS[PREVIEWER_SIMPLEDATA].preview(data)
            
        return viz.show(config=FIGURE_CONFIG_SHOW), PREVIEWERS[PREVIEWER_SIMPLEDATA].preview(data)

@registerWriter 
class SimpleVizWriter(BaseWriter):
    def __init__(self, cfg=None, tar=None):
        super().__init__(cfg=cfg, tar=tar)
        
    @classmethod
    def type(cls):
        '''Returns key used to register type'''
        return WRITER_SIMPLE_VIZ
        
    @classmethod
    def ok(cls, tar):
        '''Returns key used to register type'''
        if isinstance(tar, str) and tar[-5:]=='.html':
            return True
        return False
        
    def write(self, data):
        '''Writes viz based on config'''
        import datetime
        vizs = data[DATATYPE_VIZ]['stack']
        write_type = 'w'
        def wr(path, vizs):
            #handle mixed lists (individual viz & list of viz)
            vizs1 = []
            for v in vizs:
                vizs1.extend(v) if isinstance(v, list) else vizs1.extend([v]) 
            with open(path, write_type) as f:
                f.write("Report generated: " + str(datetime.datetime.today()))
                for v in vizs1:
                    f.write(v.to_html(full_html=False, include_plotlyjs='cdn', default_height=360, default_width='95%', config=FIGURE_CONFIG_SHOW))
        if self._cfg:
            c = self._cfg
            if 'html' in c.keys():
                wr(c['html'], vizs)
                return
            else:
                pass
        t = self._tar
        if isinstance(t, str) and tar[-5:]=='.html':
            wr(tar, vizs)
            return 
        else:
            raise TypeError("Invalid writer target")
                
fig_defaults = {
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
