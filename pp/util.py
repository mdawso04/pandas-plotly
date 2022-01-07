import pandas as pd
from pathlib import Path

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
