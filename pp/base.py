#python standard libraries
from configparser import ConfigParser
import functools

#non-standard libraries
import pandas as pd

#DATA TYPES 
#directory
DATATYPES = []
DATATYPE_DATAFRAME = 'df'
DATATYPE_READER = 'reader'
DATATYPE_WRITER = 'writer'
DATATYPE_PREVIEWER = 'previewer'
DATATYPES.extend([
    DATATYPE_DATAFRAME,
    DATATYPE_READER,
    DATATYPE_WRITER,
    DATATYPE_PREVIEWER
])

#READER TYPES 
#directory
READERTYPES = []
READER_SIMPLE_CSV_EXCEL = 'reader_simple_csv_excel'
READERTYPES.extend([
    READER_SIMPLE_CSV_EXCEL
])

#class store
READERS = {}

#WRITER TYPES
#directory
WRITERTYPES = []
WRITER_SIMPLE_CSV_EXCEL = 'writer_simple_csv_excel'
WRITERTYPES.extend([
    WRITER_SIMPLE_CSV_EXCEL
])

#class store
WRITERS = {}

#PREVIEWER TYPES
#directory
PREVIEWTYPES = []
PREVIEWER_SIMPLEDATA = 'previewer_simpledata'
PREVIEWTYPES.extend([
    PREVIEWER_SIMPLEDATA
])

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
        
        #read user supplied config
        self._cfg = ConfigParser()
        self._cfg.read('config.ini', encoding='utf_8')
        
        #read user supplied source
        self._read(source)
        
        #call default preview
        self._preview()
    
    def _cfgHelper(self, section):
            return self._cfg[section] if section in self._cfg.sections() else None 
            
    def _read(self, src=None):
        #check config for *valid* section matching our src
        cfg = self._cfgHelper(src)
        if cfg and (DATATYPE_READER in cfg.keys()) and any(cfg[DATATYPE_READER] == r for r in READERTYPES):
            r = READERS[cfg[DATATYPE_READER]](cfg=cfg)
        
        #else, fallback to 1-by-1 check of readers supporting our src - use first 'OK' reader
        else:
            for r in READERTYPES:
                if r.ok(src):
                    r = r(src=src)
                    break
            else:
                print('Reader not found')
                return
        
        #If success, instantiate Reader, read df, append to our data
        df = r.read()
        self._append(DATATYPE_DATAFRAME, df)
        return self
    
    def _write(self, tar=None):
        #check config for *valid* section matching our src
        cfg = self._cfgHelper(tar)
        if cfg and (DATATYPE_WRITER in cfg.keys()) and any(cfg[DATATYPE_WRITER] == w for w in WRITERTYPES):
            w = WRITERS[cfg[DATATYPE_WRITER]](cfg=cfg)
        
        #else, fallback to 1-by-1 check of readers supporting our src - use first 'OK' reader
        else:
            for w in WRITERTYPES:
                if w.ok(tar):
                    w = w(tar=tar)
                    break
            else:
                print('Writer not found')
                return
        
        w.write(self._data)
        return self
        
    def REPORT_SAVE_DATA_AS_CSV_EXCEL(self, tar):
        self._write(tar)
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
    
    @property
    def df(self):
        return self._data[DATATYPE_DATAFRAME]['active']
    
    @df.setter
    def df(self, df1):
        '''Replace active df without saving to stack'''
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
        return
    READERS[t] = cls
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
                return pd.read_csv(filepath_or_buffer=c['csv'])
            elif 'excel' in c.keys():
                return pd.read_excel(filepath_or_buffer=c['excel'])
            else:
                pass
        s = self._src
        if isinstance(src, str) and (src[-4:]=='.csv'):
            return pd.read_csv(filepath_or_buffer=s)
        elif isinstance(src, str) and (src[-5:]=='.xlsx'):
            return pd.read_excel(filepath_or_buffer=s)
        else:
            raise TypeError("Invalid reader source")
        
def registerWriter(cls):
    '''Register Writer objects'''
    t = cls.type()
    if t is None or t not in WRITERTYPES:
        return
    WRITERS[t] = cls
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
            return pd.read_csv(filepath_or_buffer=s)
        elif isinstance(t, str) and (t[-5:]=='.xlsx'):
            return pd.read_excel(filepath_or_buffer=s)
        else:
            raise TypeError("Invalid writer target")

def registerPreviewer(cls):
    '''Register objects'''
    t = cls.type()
    if t is None or t not in PREVIEWTYPES:
        return
    PREVIEWERS[t] = cls
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