# pandas-plotly

A python wrapper library for pandas & plotly that has a similar feel to Microsoft Power Query. 
By using this library, python novices familiar with Power Query or similar tools can 
quickly 1) get & transform data, 2) vizualize the data and 3) save the data as a report.

Methods are grouped by function: DF_ = dataframe operations, VIZ_ = vizualizations, REPORT_ = save data/vizualizations are report

## Example

```python
from scripts import SOURCE
(
    SOURCE('kintone_app1')
    .DF_COL_DELETE(['revision', 'record_id'])
    .REPORT_SET_VIZ_COLORS_ANTIQUE
    .VIZ_HIST_LIST('Attrition')
    .DF_ROW_FILTER('Age < 29')
    .DF__GROUP('Department', {'Age': ['mean', 'count']})
    .DF_COL_RENAME({'Age_mean': 'AvgAge', 'Age_count': 'NoEmployees'})
    .DF_COL_ADD_INDEX_FROM_1('DeptID')
    .VIZ_BAR('Department', 'NoEmployees')
    .REPORT_SAVE_VIZ_HTML('html_report')
)
```

![image](https://user-images.githubusercontent.com/87593190/133514752-5eb39b13-ca8d-4cd9-a058-2c8a411db05f.png)


## Requirements

pandas, plotly, optional: pycurl

## Methods

| Method |  Description & example
| --- | --- |
| ```DF_COL_ADD_FIXED``` | Add a new column with a 'fixed' value as content  ```.DF_COL_ADD_FIXED('Tokyo')```<br /> ```.DF_COL_ADD_FIXED('Tokyo', 'City')``` 
| ```DF_COL_ADD_INDEX``` |  Add a new column with a index/serial number as content  ```.DF_COL_ADD_INDEX()```__ ```.DF_COL_ADD_INDEX('No.')```__ ```.DF_COL_ADD_INDEX('No.', 0)```
| ```DF_COL_ADD_INDEX_FROM_0``` |  Convenience method for DF_COL_ADD_INDEX  ```.DF_COL_ADD_INDEX_FROM_0()```  ```.DF_COL_ADD_INDEX_FROM_0('No.')```
| ```DF_COL_ADD_INDEX_FROM_1``` |  Convenience method for DF_COL_ADD_INDEX  ```.DF_COL_ADD_INDEX_FROM_1()```  ```.DF_COL_ADD_INDEX_FROM_1('No.')```
| ```DF_COL_ADD_CUSTOM``` |  Add a new column with custom (lambda) content  
| ```DF_COL_ADD_EXTRACT_POSITION_AFTER``` |  Add a new column with content extracted from after char pos in existing column  ```.DF_COL_ADD_EXTRACT_POSITION_AFTER('OldCol', 5)```  ```.DF_COL_ADD_EXTRACT_POSITION_AFTER('OldCol', 5, 'NewCol')```
| ```DF_COL_ADD_EXTRACT_POSITION_BEFORE``` |  Add a new column with content extracted from before char pos in existing column  ```.DF_COL_ADD_EXTRACT_POSITION_BEFORE('OldCol', 5)```  ```.DF_COL_ADD_EXTRACT_POSITION_BEFORE('OldCol', 5, 'NewCol')```
| ```DF_COL_ADD_EXTRACT_CHARS_FIRST``` |  Add a new column with first N chars extracted from column  ```.DF_COL_ADD_EXTRACT_CHARS_FIRST('OldCol', 5)```  ```.DF_COL_ADD_EXTRACT_CHARS_FIRST('OldCol', 5, 'NewCol')```
| ```DF_COL_ADD_EXTRACT_CHARS_LAST``` |  Add a new column with last N chars extracted from column  ```.DF_COL_ADD_EXTRACT_CHARS_FIRST('OldCol', 5)```  ```.DF_COL_ADD_EXTRACT_CHARS_FIRST('OldCol', 5, 'NewCol')```
| ```DF_COL_ADD_DUPLICATE``` | Add a new column by copying an existing column
| DF_COL_DELETE |  Delete specified column/s 
| DF_COL_DELETE_EXCEPT |  Deleted all column/s except specified  ```.COL_DELETE_EXCEPT('Age')```  ```.COL_DELETE_EXCEPT([0,3,6])```  ```.COL_DELETE_EXCEPT(slice(0,3))```
| DF_COL_RENAME |  Rename specfied column/s
| DF_COL_FORMAT_TO_UPPERCASE |  Format specified column/s values to uppercase
| DF_COL_FORMAT_TO_LOWERCASE |  Format specified column/s values to lowercase
| DF_COL_FORMAT_TO_TITLECASE |  Format specified column/s values to titlecase
| DF_COL_FORMAT_STRIP |  Format specified column/s values by stripping invisible characters
| DF_COL_FORMAT_STRIP_LEFT |  Format specified column/s values by stripping invisible characters from left
| DF_COL_FORMAT_STRIP_RIGHT |  Format specified column/s values by stripping invisible characters from right
| DF_COL_FORMAT_ADD_PREFIX | Format specified single column values by adding prefix
| DF_COL_FORMAT_ADD_SUFFIX |  Format specified single column values by adding suffix
| DF_COL_FORMAT_TYPE | 
| DF_COL_FORMAT_ROUND | Round numerical column values to specified decimal  | ```.COL_FORMAT_ROUND(2)``` ```.COL_FORMAT_ROUND({'c1':2, 'c2':0})``` 
| DF_ROW_FILTER |  Filter rows with specified filter criteria
| DF_ROW_KEEP_BOTTOM |  Delete all rows except specified bottom N rows
| DF_ROW_KEEP_TOP |  Delete all rows except specified top N rows
| DF_ROW_REVERSE |  Reorder all rows in reverse order
| DF_ROW_SORT |  Reorder specified column contents in ascending/descending order
| DF__APPEND |  Append a table to bottom of current table
| DF__FILL_DOWN |  Fill blank cells with values from last non-blank cell above
| DF__FILL_UP |  Fill blank cells with values from last non-blank cell below
| DF__FILL_RIGHT |  Fill blank cells with values from last non-blank cell from left
| DF__FILL_LEFT |  Fill blank cells with values from last non-blank cell from right
| DF__GROUP |  Group table contents by specified columns with optional aggregation (sum/max/min etc)
| DF__MERGE |  Merge a table with current table with specified type (left/right/inner/outer) 
| DF__REPLACE |  Replace string values in table
| DF__UNPIVOT |  Unpivot table on specified columns
| DF__PIVOT |  Pivot table on specified columns
| DF_COLHEADER_PROMOTE |  Promote row at specified index to column headers
| DF_COLHEADER_DEMOTE |  Demote column headers to make 1st row of table
| DF_COLHEADER_REORDER_ASC |  Reorder column titles in ascending order
| DF_COLHEADER_REORDER_DESC |  Reorder column titles in descending order
| DF__STATS |  Show basic summary statistics of table contents
| VIZ_BOX |  Draw a box plot
| VIZ_VIOLIN |  Draw a violin plot
| VIZ_HIST |  Draw a hisotgram
| VIZ_HIST_LIST |  Draw a histogram for all fields in current dataframe
| VIZ_SCATTER |  Draw a scatter plot
| VIZ_BAR |  Draw a bar plot
| VIZ_LINE |  Draw a line plot
| VIZ_TREEMAP |  Draw a treemap plot
| VIZ_SCATTERMATRIX |  Draw a scatter matrix plot
| REPORT_SET_VIZ_COLORS_PLOTLY |  Set plot/report colors to 'Plotly'
| REPORT_SET_VIZ_COLORS_D3 |  Set plot/report colors to 'D3'
| REPORT_SET_VIZ_COLORS_G10 |  Set plot/report colors to 'G10'
| REPORT_SET_VIZ_COLORS_T10 |  Set plot/report colors to 'T10'
| REPORT_SET_VIZ_COLORS_DARK24 |  Set plot/report colors to 'Dark24'
| REPORT_SET_VIZ_COLORS_LIGHT24 |  Set plot/report colors to 'Light24'
| REPORT_SET_VIZ_COLORS_SET1 |  Set plot/report colors to 'Set1'
| REPORT_SET_VIZ_COLORS_PASTEL1 |  Set plot/report colors to 'Pastel1'
| REPORT_SET_VIZ_COLORS_DARK2 |  Set plot/report colors to 'Dark2'
| REPORT_SET_VIZ_COLORS_SET2 |  Set plot/report colors to 'Set2'
| REPORT_SET_VIZ_COLORS_PASTEL2 |  Set plot/report colors to 'Pastel2'
| REPORT_SET_VIZ_COLORS_SET3 |  Set plot/report colors to 'Set3'
| REPORT_SET_VIZ_COLORS_ANTIQUE |  Set plot/report colors to 'Antique'
| REPORT_SET_VIZ_COLORS_BOLD |  Set plot/report colors to 'Bold'
| REPORT_SET_VIZ_COLORS_PASTEL |  Set plot/report colors to 'Pastel'
| REPORT_SET_VIZ_COLORS_PRISM |  Set plot/report colors to 'Prism'
| REPORT_SET_VIZ_COLORS_SAFE |  Set plot/report colors to 'Safe'
| REPORT_SET_VIZ_COLORS_VIVID |  Set plot/report colors to 'Vivid'
| REPORT_PREVIEW |  Preview all plots on screen (for use in JupyterLab)
| REPORT_PREVIEW_FULL |  Preview all plots, dataframe and column summary on screen (for use in JupyterLab)
| REPORT_SAVE_ALL |  Save html format report/plots and dataframe to specified location
| REPORT_SAVE_VIZ_HTML |  Save html format report/plots to specified location
| REPORT_SAVE_DF |  Save dataframe to specified location
