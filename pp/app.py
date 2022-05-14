from pp.constants import *
import pp.config as config
from pp.log import logger
from pp.util import *
from pp.ml_f import *

#python standard libraries
import functools, inspect

#non-standard libraries
import pandas as pd

class App(object):
    def __init__(self, current=None):
        #TODO load from string param
        if current is None or not isinstance(current, dict): 
            self.src, self.pre, self.viz = None, [], []
        else:
            self.src = current['src'] if 'src' in current.keys() else None
            self.pre = current['pre'] if 'pre' in current.keys() else []
            self.viz = current['viz'] if 'viz' in current.keys() else []
        self.services = SERVICES.keys()

    def current(self):
        return {'src': self.src,  'pre': self.pre, 'viz': self.viz}
    
    def tostring():
        pass
    
    def addSrc(self, service, options):
        # only allow a single source
        self.src = {'service': service, 'options': options}
    
    def addPre(self, service, options, index=None):
        self.pre.insert(len(self.pre) if index is None else index, {'service': service, 'options': options})
    
    def addViz(self, service, options, index=None):
        self.viz.insert(len(self.viz) if index is None else index, {'service': service, 'options': options})
        
    def options(self, service, df=None):
        if df is not None:
            return SERVICES[service].options(df)
        elif self.src is not None:
            return SERVICES[service].options(self.src)
        return "Dataframe not found"
    
    def isvalid():
        #TODO
        # MUST/WANT param check
        # param type check
        return True
    
    def call(self, df):
        #if not self.isvalid():
            #exception
        if len(self.pre) > 0:
            for p in self.pre:
                fn = SERVICES[p['service']].fn
                df = fn(df, **p['options'])
        
        if len(self.viz) > 0:
            fn = SERVICES[self.viz[-1]['service']].fn
            o = self.viz[-1]['options']
            return fn(df, **o)
        
        return df

class Base(object):
    
    def __init__(self, source):
        super(Base, self).__init__()
        
        #Build base data structure
        self._data = {
            DATATYPE_DATAFRAME:{
                'active':None,
                'stack':[]
            },
        }
        logger.debug('Data structure built')
        
        #read user supplied source
        self._read(source)
        logger.debug('Source read: {}'.format(source))
        
        #call default preview
        self._preview()
               
    def _pop(self, key):
        '''Return current data item and replace with next from stack'''
        #TODO if empty
        s = self._data[key]['stack']
        old = s.pop()
        self._data[key]['active'] = s[-1] if len(s) > 0 else None
        return old
    
    def _append(self, key, data):
        '''Add data item to stack and make active'''
        #TODO if empty
        self._data[key]['stack'].append(data); self._data[key]['active'] = data
        return self
    
    def _active(self, key, data):
        '''Replace active data'''
        self._pop(key); self._append(key, data)
        return self
    
    def REPORT_SAVE_DATA_AS_CSV_EXCEL(self, tar):
        self._write(tar)
        return self
        
    @property
    def df(self):
        return self._data[DATATYPE_DATAFRAME]['active']
    
    @df.setter
    def df(self, df1):
        '''Replace active df without pushing current to stack'''
        self._active(DATATYPE_DATAFRAME, df1)
    
    def _repr_pretty_(self, p, cycle): 
        '''Selects content for IPython display'''
        selected = self._previewMode
        d = self._data
        return PREVIEWERS[selected].preview(data=self._data)
        
    def __repr__(self): 
        return self._df.__repr__()
    
    def __str__(self): 
        return self._df.__str__()

