from pp.constants import *
import pp.config as config
from pp.log import logger

#python standard libraries
import functools, inspect

#non-standard libraries
import pandas as pd

#class store
READERS = {}

#class store
WRITERS = {}

#class store
PREVIEWERS = {}

class Base(object):
    
    def __init__(self, source):
        super(Base, self).__init__()
        
        #Build base data structure
        self._data = {
            DATATYPE_DATAFRAME:{
                'active':None,
                'stack':[]
            },
            DATATYPE_READER:{
                'active':None,
                'stack':[]
            },
            DATATYPE_WRITER:{
                'active':None,
                'stack':[]
            }
        }
        logger.debug('Data structure built')
        
        #read user supplied source
        self._read(source)
        logger.debug('Source read: {}'.format(source))
        
        #call default preview
        self._preview()
    
    def _read(self, src=None):
        #check config for *valid* section matching our src
        #cfg = self._cfgHelper(src)
        c = config.section(src)
        if c and (DATATYPE_READER in c.keys()) and any(c[DATATYPE_READER] == r for r in READERTYPES):
            r = READERS[c[DATATYPE_READER]](cfg=c)
        
        #else, fallback to 1-by-1 check of readers supporting our src - use first 'OK' reader
        else:
            for r in READERS.values():
                if r.ok(src):
                    r = r(src=src)
                    break
            else:
                print('Reader not found')
                return
        
        #If success, instantiate Reader, read df, append to our data
        df = r.read()
        self._append(DATATYPE_DATAFRAME, df)
        logger.info('Read from: {}'.format(src))
        return self
    
    def _write(self, tar=None):
        #check config for *valid* section matching our src
        c = config.section(src)
        if c and (DATATYPE_WRITER in c.keys()) and any(c[DATATYPE_WRITER] == w for w in WRITERTYPES):
            w = WRITERS[c[DATATYPE_WRITER]](cfg=c)
        
        #else, fallback to 1-by-1 check of readers supporting our src - use first 'OK' reader
        else:
            for w in WRITERS.values():
                if w.ok(tar):
                    w = w(tar=tar)
                    break
            else:
                print('Writer not found')
                return
        
        w.write(self._data)
        logger.info('Wrote to: {}'.format(tar))
        return self
    
    def _preview(self, preview=PREVIEWER_SIMPLEDATA): 
        '''Handles figure displaying for IPython'''
        p = preview
        if (not p) or not any(p == pt for pt in PREVIEWTYPES):
            print('Previewer not found')
            return
        self._previewMode = p
           
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
    
    def _call(self, params):
        '''Call this object with dictionary provided'''
        '''
        test_params = [
            {
                'method': 'DATA_COL_DELETE',
                'params': {
                    'columns': '年齢',
                }
            },
        ]
        '''
        
        try:
            for p in params:
                func = getattr(self, p['method'])
                if p['params'] is None:
                    print(inspect.signature(func))
                    func()
                else:
                    print(inspect.signature(func))
                    func(**p['params'])
        except AttributeError:
            print("dostuff not found")
        
        return self
    
    def REPORT_SAVE_DATA_AS_CSV_EXCEL(self, tar):
        self._write(tar)
        return self
    
    def CALL(self, params):
        self._call(params)
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

#READERS, WRITERS & PREVIEWERS
    
def registerReader(cls):
    '''Register Reader objects'''
    t = cls.type()
    if t is None or t not in READERTYPES:
        raise ValueError('Not valid Reader')
    READERS[t] = cls
    logger.debug('Registered Reader: {}'.format(cls))
    return cls
    
class BaseReader():
    def __init__(self, cfg=None, src=None):
        self._cfg = cfg
        self._src = src
        
    @classmethod
    def type(cls):
        '''Returns key used to regsiter Reader type'''
        return None #don't register BaseReader
    
    @classmethod
    def ok(cls, src):
        '''Returns key used to regsiter Reader type'''
        return False #don't register BaseReader
        
    def read(self):
        '''Returns dataframe based on config'''
        #check cfg, read, return df
        return

@registerReader 
class SimpleCsvExcelReader(BaseReader):
    def __init__(self, cfg=None, src=None):
        super().__init__(cfg=cfg, src=src)
        
    @classmethod
    def type(cls):
        '''Returns key used to regsiter Reader type'''
        return READER_SIMPLE_CSV_EXCEL
        
    @classmethod
    def ok(cls, src):
        '''Returns key used to regsiter Reader type'''
        if isinstance(src, str) and (src[-4:]=='.csv' or src[-5:]=='.xlsx'):
            return True
        return False #don't register BaseReader
        
    def read(self):
        '''Returns dataframe based on config'''
        if self._cfg:
            c = self._cfg
            if 'csv' in c.keys():
                return pd.read_csv(c['csv'])
            elif 'excel' in c.keys():
                return pd.read_excel(c['excel'])
            else:
                pass
        s = self._src
        if isinstance(s, str) and (s[-4:]=='.csv'):
            return pd.read_csv(s)
        elif isinstance(s, str) and (s[-5:]=='.xlsx'):
            return pd.read_excel(s)
        else:
            raise TypeError("Invalid reader source")
        
def registerWriter(cls):
    '''Register Writer objects'''
    t = cls.type()
    if t is None or t not in WRITERTYPES:
        raise ValueError('Not valid Writer')
    WRITERS[t] = cls
    logger.debug('Registered Writer: {}'.format(cls))
    return cls
    
class BaseWriter():
    def __init__(self, cfg=None, tar=None):
        self._cfg = cfg
        self._tar = tar
        
    @classmethod
    def type(cls):
        '''Returns key used to regsiter type'''
        return None #don't register Base
        
    @classmethod
    def ok(cls, tar):
        '''Returns key used to register type'''
        return False #don't register Base
        
    def write(self, data):
        '''Writes based on config'''
        #check cfg, write, return
        return

@registerWriter 
class SimpleCsvExcelWriter(BaseWriter):
    def __init__(self, cfg=None, tar=None):
        super().__init__(cfg=cfg, tar=tar)
        
    @classmethod
    def type(cls):
        '''Returns key used to regsiter Reader type'''
        return WRITER_SIMPLE_CSV_EXCEL
        
    @classmethod
    def ok(cls, tar):
        '''Returns key used to register type'''
        if isinstance(tar, str) and (tar[-4:]=='.csv' or tar[-5:]=='.xlsx'):
            return True
        return False #don't register BaseReader
        
    def write(self, data):
        '''Writes dataframe based on config'''
        if self._cfg:
            c = self._cfg
            df = data[DATATYPE_DATAFRAME]['active']
            if 'csv' in c.keys():
                return df.to_csv(c['csv'], index=False)
            elif 'excel' in c.keys():
                return df.to_excel(c['excel'], index=False)
            else:
                pass
        t = self._tar
        if isinstance(t, str) and (t[-4:]=='.csv'):
            return df.to_csv(t, index=False)
        elif isinstance(t, str) and (t[-5:]=='.xlsx'):
            return df.to_excel(t, index=False)
        else:
            raise TypeError("Invalid writer target")

def registerPreviewer(cls):
    '''Register objects'''
    t = cls.type()
    if t is None or t not in PREVIEWTYPES:
        raise ValueError('Not valid Previewer')
    PREVIEWERS[t] = cls
    logger.debug('Registered Previewer: {}'.format(cls))
    return cls
        
class BasePreviewer():
    @classmethod
    def type(cls):
        '''Returns key used to regsiter Reader type'''
        return None #don't register Base
    
    @classmethod    
    def preview(self, data):
        '''Returns dataframe based on config'''
        return

@registerPreviewer 
class SimpleDATAPreviewer(BasePreviewer):
    @classmethod
    def type(cls):
        '''Returns key used to regsiter type'''
        return PREVIEWER_SIMPLEDATA
    
    @classmethod
    def preview(self, data):
        '''Returns dataframe based on config'''
        df = data[DATATYPE_DATAFRAME]['active']
        if isinstance(df.columns, pd.MultiIndex): 
            arrays = [range(0, len(df.columns)), df.columns.get_level_values(0), df.dtypes]
            mi = pd.MultiIndex.from_arrays(arrays, names=('Num', 'Name', 'Type'))
        else:
            arrays = [range(0, len(df.columns)), df.columns, df.dtypes]
            mi = pd.MultiIndex.from_arrays(arrays, names=('Num', 'Name', 'Type'))
        df.columns = mi
        return display(df)