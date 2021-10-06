# -*- coding: utf-8 -*-
# %%
import pykintone
from pykintone import model, kintoneService as ks
from pykintone.structure import kintoneStructure as kst
import pandas as pd
import timeit


class jsKintoneModel(model.kintoneModel):
    
    #UNIQUE_KEY='record_id'
    
    def __init__(self):
        super(jsKintoneModel, self).__init__()
    
class jsModelFactory(object):
    
    @staticmethod
    def get(mdl='model.csv'):
        df = pd.read_csv(mdl)
        fieldNames = df.columns
        
        # fields for dynamic class as dict. Field starting with '*' is UNIQUE_KEY
        fields = {i if i[0] != '*' else i[1:]:None for i in fieldNames} 
        
        #add fields from super class (ie kintonemodel)
        fields['record_id'] = None
        fields['revision'] = None
        
        # handmade field check method to add to our dyn class template. Point: override '__dir__()'
        def attGetter(self): return fields
        
        fields1 = fields.copy()
        fields1['__dir__'] =  attGetter
        
        #build dynamic class
        mod = type(mdl, (jsKintoneModel,), fields1)
        globals()[mdl] = mod
        for i in df.columns:
            if i[0] == '*':
                mod.UNIQUE_KEY = i[1:] 
        if hasattr(mod, 'UNIQUE_KEY') == False: mod.UNIQUE_KEY = 'NO_KEY_SET'
        return mod

class jsDataLoader(object):
    
    def __init__(self, model, dataframe):
        super(jsDataLoader, self).__init__()
        self.dataframe = dataframe
        self.model = model
        
    def models(self):
        df = self.dataframe
        
        if self.model.UNIQUE_KEY != 'NO_KEY_SET':
            #check, return None for duplicated UNIQUE_KEY in our data
            if df[df.duplicated(subset=self.model.UNIQUE_KEY)].size > 0:
                print('Error: source data includes duplicated UNIQUE_KEY')
                return None
        
        #build list of models from our csv/df
        models = []
        for index, row in df.iterrows():
            m = self.model()
            d = dir(m) #call our custom field check function
            for i in d:
                if i == 'record_id' or i == 'revision': continue #dont attempt load from csv of kintone system-only fields
                setattr(m, i, row[i])
            models.append(m) 
        return models
    
    
class JinSapo(object):
    
    def __init__(self, domain, app_id, api_token, model):
        # patched account.py line 48
        #pk = pykintone.load(account)
        #self.app = pk.app(app)
        self.app = pykintone.app(domain, app_id, api_token)
        self.model = model
    
    @staticmethod
    def clean (json):
        cdf = pd.json_normalize(json)
        cdf = cdf.loc[:,['.value' in i for i in cdf.columns]]
        cdf.columns = [c.replace('.value', '') for c in cdf.columns]
        cdf = cdf[cdf.columns[~cdf.columns.isin(['$id','$revision', '更新日時', '作成日時', '更新者.code', '更新者.name', '作成者.code', '作成者.name'])]]
        return cdf
    
    def select(self, query='', fields=()):
        allResults = []
        query = query+' ' if query!='' else ''
        totalCount = self.app.select(query + 'limit 1').total_count
        downloadCount = 0
        while downloadCount < totalCount:
            s = query + 'limit '+ str(ks.SELECT_LIMIT) + ' offset ' + str(downloadCount)
            results = self.app.select(s)
            models = results.models(self.model)
            allResults.extend(models)
            downloadCount += len(models)
        return allResults
    
    def select_df(self, query='', fields=()):
        df = pd.DataFrame()
        allResults = self.select(query=query, fields=fields)
        for r in allResults:
            dic = {key:val for key, val in r.__dict__.items() if key[0:1] != '_'}
            df = df.append(dic, ignore_index = True) 
        return df

    def create_dummy(self, count):
        if count is None: 
            print('Warning: cannot execute with empty count')
            return None
        lst = [self.model() for i in range(count)]
        self.create(lst)
            
    def create(self, lst = []):
        if lst is None: 
            print('Warning: cannot execute with empty list')
            return None
        splitList = self._split(lst, ks.UPDATE_LIMIT) 
        for i in splitList:
            self.app.batch_create(i)
            
    def update(self, lst = []):
        if lst is None: 
            print('Warning: cannot execute with empty list')
            return None
        splitList = self._split(lst, ks.UPDATE_LIMIT) 
        for i in splitList:
            self.app.batch_update(i)
        
    def delete(self, lst = []):
        if lst is None: 
            print('Warning: cannot execute with empty list')
            return None
        splitList = self._split(lst, ks.UPDATE_LIMIT) 
        for i in splitList:
            self.app.delete(i)
    
    def deleteAll(self):
        self.delete(lst = self.select())
            
    def _split(self, lst, limit=ks.UPDATE_LIMIT):
        n = limit
        l = lst
        return [l[i:i + n] for i in range(0, len(l), n)] 
    
    #replace all kintone data with provided new data, using UNIQUE_KEY
    #to decide whether to update/create/delete
    
    def synch(self, new_models):
        
        if new_models is None: 
            print('Warning: cannot execute with empty list')
            return None
        
        if self.model.UNIQUE_KEY == 'NO_KEY_SET':
            print('Warning: cannot synch without UNIQUE_KEY')
            return None
        
        #get all kintone data
        current = self.select()
        
        new = new_models
        
        #build lists of <uniqueKey : model> dicts for comparison
        cur = {getattr(i, self.model.UNIQUE_KEY):i for i in current}
        new = {getattr(i, self.model.UNIQUE_KEY):i for i in new}
        
        #build delete list of models (uniqueKey in cur, but not in new)
        delKeys = [key for key in cur.keys() if key not in new.keys()]
        delLst = [value for key, value in cur.items() if key in delKeys]
        
        #build create list of models (uniqueKey not in cur, but in new)
        creKeys = [key for key in new.keys() if key not in cur.keys()]
        creLst = [value for key, value in new.items() if key in creKeys]
        
        #build update list of models (uniqueKey in cur and new)
        updKeys = [key for key in new.keys() if key in cur.keys()]
        updLst = [value for key, value in new.items() if key in updKeys]
        
        #delete, add, update
        self.delete(delLst)
        self.create(creLst)
        self.update(updLst)
        

#1) load kintone app/s, model, jsApp
#pk = pykintone.load('account.yml')
#model = jsModelFactory.get('model3.csv')
#js = JinSapo(account = 'account.yml', model = model)

#select
#q = ''
#data = js.select(q)

#create: dummy
#js.create_dummy(200)

#create: from csv
#data = jsDataLoader(model, 'data3.csv')
#js.create(data.models())

#delete: from kintone
#q = '社員番号 = ' + str(5)
#deleteData = js.select(q)
#js.delete(deleteData)

#delete: all from kintone
#js.deleteAll()

#update: from kintone
#updateData = js.select()
#do domething
#js.update(updateData)

#synch: auto
#data = jsDataLoader(model, 'data3.csv')
#js.synch(data.models())

#synch: manual
#for i in range(30000, 30602, 20):
#    q = 'レコード番号 > '+str(i)
#    print(q)
#    deleteData = js.select(query=q)
#    js.delete(deleteData)


# %%




