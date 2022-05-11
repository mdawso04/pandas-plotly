import pandas as pd
from pathlib import Path
from pp.log import logger

from inspect import signature
#from types import MappingProxyType
from collections import OrderedDict

#SERVICES DIRECTORY 
SERVICES = {}

#SERVICE KEYS
# type, number of selections possible
OPTION_FIELD_SINGLE_COL_ANY = (None, 1)
OPTION_FIELD_MULTI_COL_ANY = (None, None)
OPTION_FIELD_SINGLE_COL_NUMBER = ('number', 1)
OPTION_FIELD_MULTI_COL_NUMBER = ('number', None)
OPTION_FIELD_SINGLE_COL_STRING = ('object', 1)
OPTION_FIELD_MULTI_COL_STRING = ('object', None)
OPTION_FIELD_SINGLE_BOOLEAN = ('boolean', 1)
OPTION_FIELD_SINGLE_COLORSWATCH = ('colorswatch', 1)
OPTION_FIELDS = []
OPTION_FIELDS.extend([
    OPTION_FIELD_SINGLE_COL_ANY,
    OPTION_FIELD_MULTI_COL_ANY,
    OPTION_FIELD_SINGLE_COL_NUMBER,
    OPTION_FIELD_MULTI_COL_NUMBER,
    OPTION_FIELD_SINGLE_COL_STRING,
    OPTION_FIELD_MULTI_COL_STRING,
    OPTION_FIELD_SINGLE_BOOLEAN,
    OPTION_FIELD_SINGLE_COLORSWATCH,
])
FIELD_STRING = 'string'
FIELD_INTEGER = 'int'
FIELD_NUMBER = 'number'
FIELD_FLOAT = 'float'

class ServiceFactory(object):
    class Service(object):
        def __init__(self, fn, d):
            self.name = fn.__name__
            self.fn = fn
            self._d = d
            
        def options(self, df):
            #TODO: orderedDict 
            return {k: (colHelper(df, type=v[0], colsOnNone=True) if v in OPTION_FIELDS else None) for k, v in self._d.items()}
    
    def __init__(self, fn, d):
        self._fn = fn
        self._d = d
        
    def get(self):
        return self.Service(self._fn, self._d)
    
class ServiceManager(object):
    def __init__(self, pre=None, viz=None):
        #TODO load from string param
        self.pre = [] if pre is None else pre
        self.viz = [] if viz is None else viz

    def current(self):
        return {'pre': self.pre, 'viz': self.viz}
    
    def tostring():
        pass
    
    @property
    def services(self):
        return SERVICES
    
    def addPre(self, service, options):
        self.pre.append({'service': service.name, 'function': service.fn, 'options': options})
    
    def addViz(self, service, options):
        self.viz.append({'service': service.name, 'function': service.fn, 'options': options})
    
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
                df = p['function'](df, **p['options'])
        
        if len(self.viz) > 0:
            fn = self.viz[-1]['function'] 
            o = self.viz[-1]['options']
            return fn(df, **o)
        
        return df

def registerService(**d):
    def inner(fn):
        #sig = signature(fn)
        SERVICES[fn.__name__] = ServiceFactory(fn, d).get()
        logger.debug('Registered Service: {}'.format(fn.__name__))
        return fn
    return inner

'''
Viz

{
    pre:[{
    }]
    viz:[]
}

'''


'''
def registerServiceOLD(**d):
    def inner(fn):
        #sig = signature(fn)
        SERVICES[fn.__name__] = {
            'fn': fn,
            'options': {k: lambda df, t=v: colHelper(df, type=t, colsOnNone=True) for k, v in d.items()}
        }
        logger.debug('Registered Service: {}'.format(fn.__name__))
        return fn
    return inner
'''

'''
#ColumnName = str | int
#ColumnNames = str | int | list[str] | list[int] | tuple[str] | tuple[int]

class Column(Object):
    @classmethod
    def typeMatch(cls, t):
        pass    
        
class AnyColumn(Column):
    @classmethod
    def typeMatch(cls, t):
        if t is None or t not in ('number', 'object'):
            return False
        return True

class NumberColumn(Column):
    @classmethod
    def typeMatch(cls, t):
        if t is None or t not in ('number'):
            return False
        return True

class TextColumn(Column):
    @classmethod
    def typeMatch(cls, t):
        if t is None or t not in ('object'):
            return False
        return True
'''


# ## UTILITIES ###
def removeElementsFromList(l1, l2):
    '''Remove from list1 any elements also in list2'''
    # if not list type ie string then covert
    if not isinstance(l1, list):
        list1 = []
        list1.append(l1)
        l1 = list1
    if not isinstance(l2, list):
        list2 = []
        list2.append(l2)
        l2 = list2
    return [i for i in l1 if i not in l2]

def commonElementsInList(l1, l2):
    if l1 is None or l2 is None: return None
    if not isinstance(l1, list): l1 = [l1]
    if not isinstance(l2, list): l2 = [l2]
    return [i for i in l1 if i in l2]

def colHelper(df, columns=None, max=None, type=None, colsOnNone=True, forceReturnAsList=True):

    if isinstance(columns, tuple):
        columns = list(columns)

    # pre-process: translate to column names
    if isinstance(columns, slice) or isinstance(columns, int):
        columns = df.columns.values.tolist()[columns]
    elif isinstance(columns, list) and all(isinstance(c, int) for c in columns):
        columns = df.columns[columns].values.tolist()

    # process: limit possible columns by type (number, object, datetime)
    df1 = df.select_dtypes(include=type) if type is not None else df

    #process: fit to limited column scope
    if colsOnNone == True and columns is None: columns = df1.columns.values.tolist()
    elif columns is None: return None
    else: columns = commonElementsInList(columns, df1.columns.values.tolist())           

    # apply 'max' check    
    if isinstance(columns, list) and max != None:
        if max == 1: columns = columns[0]
        else: columns = columns[:max]

    # if string format to list for return
    if forceReturnAsList and not isinstance(columns, list): 
        columns = [columns]

    return columns

def colValues(df, col):
    cv = df[col].unique()
    return cv

def toMultiIndex(df):
    if isinstance(df.columns, pd.MultiIndex): 
        arrays = [range(0, len(df.columns)), df.columns.get_level_values(0), df.dtypes]
        mi = pd.MultiIndex.from_arrays(arrays, names=('Num', 'Name', 'Type'))
    else:
        arrays = [range(0, len(df.columns)), df.columns, df.dtypes]
        mi = pd.MultiIndex.from_arrays(arrays, names=('Num', 'Name', 'Type'))
    df.columns = mi
    return df

def toSingleIndex(df):
    if isinstance(df.columns, pd.MultiIndex): 
        df.columns = df.columns.get_level_values(1)
    return df

def rowHelper(df, max = None, head = True):
    if max is None: return df
    else: 
        if head is True: return df.head(max)
        else: return df.tail(max)

def toUniqueColName(df, name):
    n = 1
    name = str(name)
    while name in df.columns.values.tolist():
        name = name + '_' + str(n)
    return name

def pathHelper(path, filename):
    import os
    if path == None:
        home = str(pathlib.Path.home())
        path = os.path.join(home, 'report')
    else:
        path = os.path.join(path, 'report')
    os.makedirs(path, exist_ok = True)
    path = os.path.join(path, filename)
    return path
