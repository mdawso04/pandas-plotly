# test_module.py
import unittest
import sys
import os.path
#sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir)))
import pp
import pandas as pd

class TestData(unittest.TestCase):
    def setUp(self):
        self.df = pp.IO_READ_CSV('https://raw.githubusercontent.com/IBM/employee-attrition-aif360/master/data/emp_attrition.csv')

    def tearDown(self):
        pass
    
    def test_DATA_COL_ADD_CONCATENATE(self):
        #Equal
        df2 = pp.DATA_COL_ADD_CONCATENATE(self.df, columns=None, separator='_', name='new_column')
        self.assertEqual(len(self.df.columns)+1, len(df2.columns), 'DATA_COL_ADD_CONCATENATE failed')
    
    '''
    def test_init(self):
        self.assertIsInstance(self.app, pp.base.Base, 'app instantiation failed')
        self.assertIsInstance(self.app.df, pd.DataFrame, 'dataframe not found')
        
    def test_SimpleCsvExcelReader_ok(self):
        #True
        self.assertTrue(SimpleCsvExcelReader.ok('some.csv'), 'some.csv failed')
        self.assertTrue(SimpleCsvExcelReader.ok('some.xlsx'), 'some.csv failed')
        self.assertTrue(SimpleCsvExcelReader.ok('parent/some.csv'), 'parent/some.csv failed')
        self.assertTrue(SimpleCsvExcelReader.ok('parent/some.xlsx'), 'parent/some.csv failed')
        #False
        self.assertFalse(SimpleCsvExcelReader.ok('some.excel'), 'some.excel failed')
        self.assertFalse(SimpleCsvExcelReader.ok('some.xls'), 'some.xls failed')
        self.assertFalse(SimpleCsvExcelReader.ok('parent/some.excel'), 'some.excel failed')
        self.assertFalse(SimpleCsvExcelReader.ok('parent/some.xls'), 'parent/some.excel failed')
        
    def test_SimpleCsvExcelWriter_ok(self):
        #True
        self.assertTrue(SimpleCsvExcelWriter.ok('some.csv'), 'some.csv failed')
        self.assertTrue(SimpleCsvExcelWriter.ok('some.xlsx'), 'some.csv failed')
        self.assertTrue(SimpleCsvExcelWriter.ok('parent/some.csv'), 'parent/some.csv failed')
        self.assertTrue(SimpleCsvExcelWriter.ok('parent/some.xlsx'), 'parent/some.csv failed')
        #False
        self.assertFalse(SimpleCsvExcelWriter.ok('some.excel'), 'some.excel failed')
        self.assertFalse(SimpleCsvExcelWriter.ok('some.xls'), 'some.xls failed')
        self.assertFalse(SimpleCsvExcelWriter.ok('parent/some.excel'), 'some.excel failed')
        self.assertFalse(SimpleCsvExcelWriter.ok('parent/some.xls'), 'parent/some.excel failed')
'''
"""
from pandas.testing import assert_frame_equal
df1 = pd.DataFrame({'a': [1, 2], 'b': [3, 4]})
df2 = pd.DataFrame({'a': [1, 2], 'b': [3.0, 4.0]})
"""
        
"""
    def test_upper(self):
        self.assertEqual('foo'.upper(), 'FOO')

    def test_isupper(self):
        self.assertTrue('FOO'.isupper())
        self.assertFalse('Foo'.isupper())

    def test_split(self):
        s = 'hello world'
        self.assertEqual(s.split(), ['hello', 'world'])
        # check that s.split fails when the separator is not a string
        with self.assertRaises(TypeError):
            s.split(2)
"""

if __name__ == '__main__':
    unittest.main()