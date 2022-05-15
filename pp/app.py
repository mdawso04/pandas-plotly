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
    def __init__(self, todos=None):
        #TODO load from string param
        if todos is None or not isinstance(todos, dict): 
            self.todos = {k: [] for k in ('read', 'data', 'viz', 'write')} 
        else:
            self.todos = todos
    
    def _service_helper(self, index=None, group=None, return_type='group_service'):
        #if group_service: asis
        #if group_service_names: grp/ser names
        #if service: ser
        
        if return_type=='group_service':
            return SERVICES if group is None else SERVICES[group]
        elif return_type=='group_service_names':
            return ({k: list(v.keys()) for k, v in SERVICES.items()} if group is None 
                    else [k for k, v in SERVICES[group].items()])
        elif return_type=='service':
            return {k: v for dic in SERVICES.values() for k, v in dic.items()}
        return "SERVICE NOT FOUND"
    
    def services(self, group=None):
        return self._service_helper(return_type='group_service_names', group=group)
    
    def options(self, service, df=None):
        services_dict = self._service_helper(return_type='service')
        if df is not None:
            return services_dict[service].options(df)
        else:
            return services_dict[service].options(self.call(group='read')) 
        return "SERVICE NOT FOUND"
    
    def add(self, service, options=None, index=None):
        group = service.split('_', 1)[0].lower()
        l = self.todos[group]
        l.insert(len(l) if index is None else index, {'service': service, 'options': options})
    
    def isvalid(self):
        #TODO
        # MUST/WANT param check
        # param type check
        return True
    
    def call(self, df=None, index=None, group=None):
        if not self.isvalid():
            #exception
            return "ERROR"
        #groups = (group) if group is not None else ('io', 'data', 'viz')
        #df = df if df is not None else...
        if group is not None:
            todos = self.todos[group]
        else:
            todos = self.todos['read'] + self.todos['data'] + self.todos['viz'] + self.todos['write']
        
        service_list = self._service_helper(return_type='service')
            
        result, results = None, []
        for item in todos:
            fn = service_list[item['service']].fn
            if item['options'] is not None:
                result = fn(df=df, **item['options'])
            else:
                result = fn(df=df)
            if isinstance(result, pd.DataFrame):
                df = result
            else:
                results.append(result)
        if isinstance(result, pd.DataFrame):
            results.append(result)
        if len(results) == 1: return results[0]
        else: return results

    def tostring():
        pass
    
    
    
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

