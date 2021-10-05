# pyPowerQ

A python library with a similar feel to Microsoft Power Query. 
By using this library, python novices familiar with Power Query or similary tools can quickly do most of the same things in python.

TAB_ = TABLE transformations, COL_ = COLUMN transformations, ROW_ = ROW transformations

## Example

```python
from pyPquery.base import pq as READ
(
    READ('pyQuery_data.csv')
    .COL_ADD_FIXED('test')
    .COL_DELETE(['workclass', 'race', 'sex', 'y', 'relationship'])
    .COL_DELETE('education')
    .COL_DELETE('age')
    .COL_ADD_FIXED('more', '固定')
    .COL_DELETE(['capital-loss', 'occupation', 'capital-gain', 'marital-status'])
    .COL_DELETE('固定')
    .COL_ADD_FIXED(12345, 'Num')
    .COL_ADD_INDEX('no')
    .COL_REORDER_ASC()
    .TAB_GROUP('hours-per-week', {'education-num': ['mean', 'max']})
    .WRITE('query_write.csv')
)
```

![image](https://user-images.githubusercontent.com/87593190/133514752-5eb39b13-ca8d-4cd9-a058-2c8a411db05f.png)


## Requirements

pandas, numpy, plotly, pathlib

## Methods

| Group | Method |  Description | Example
| --- | --- | --- | ---
| DF | DF_COL_ADD_FIXED | Add a new column with a 'fixed' value as content
| DF | DF_COL_ADD_INDEX |  Add a new column with a index/serial number as content
| DF | DF_COL_ADD_CUSTOM |  Add a new column with custom (lambda) content
| DF | DF_COL_ADD_EXTRACT_POSITION_AFTER |  Add a new column with content extracted from after char pos in existing column
| DF | DF_COL_ADD_EXTRACT_POSITION_BEFORE |  Add a new column with content extracted from before char pos in existing column
| DF | DF_COL_ADD_EXTRACT_CHARS_FIRST |  Add a new column with first N chars extracted from column
| DF | DF_COL_ADD_EXTRACT_CHARS_LAST |  Add a new column with last N chars extracted from column
| DF | DF_COL_ADD_DUPLICATE | Add a new column by copying an existing column
| DF | DF_COL_DELETE |  Delete specified column/s 
| DF | DF_COL_DELETE_EXCEPT |  Deleted all column/s except specified  | ```.COL_DELETE_EXCEPT('Age')``` ```.COL_DELETE_EXCEPT([0,3,6])``` ```.COL_DELETE_EXCEPT(slice(0,3))```
| DF | DF_COL_RENAME |  Rename specfied column/s
| DF | DF_COL_FORMAT_TO_UPPERCASE |  Format specified column/s values to uppercase
| DF | DF_COL_FORMAT_TO_LOWERCASE |  Format specified column/s values to lowercase
| DF | DF_COL_FORMAT_TO_TITLECASE |  Format specified column/s values to titlecase
| DF | DF_COL_FORMAT_STRIP |  Format specified column/s values by stripping invisible characters
| DF | DF_COL_FORMAT_STRIP_LEFT |  Format specified column/s values by stripping invisible characters from left
| DF | DF_COL_FORMAT_STRIP_RIGHT |  Format specified column/s values by stripping invisible characters from right
| DF | DF_COL_FORMAT_ADD_PREFIX | Format specified single column values by adding prefix
| DF | DF_COL_FORMAT_ADD_SUFFIX |  Format specified single column values by adding suffix
| DF | DF_COL_FORMAT_TYPE | 
| DF | DF_COL_FORMAT_ROUND | Round numerical column values to specified decimal  | ```.COL_FORMAT_ROUND(2)``` ```.COL_FORMAT_ROUND({'c1':2, 'c2':0})``` 
| DF | DF_ROW_FILTER |  Filter rows with specified filter criteria
| DF | DF_ROW_KEEP_BOTTOM |  Delete all rows except specified bottom N rows
| DF | DF_ROW_KEEP_TOP |  Delete all rows except specified top N rows
| DF | DF_ROW_REVERSE |  Reorder all rows in reverse order
| DF | DF_ROW_SORT |  Reorder specified column contents in ascending/descending order
| DF | DF__APPEND |  Append a table to bottom of current table
| DF | DF__FILL_DOWN |  Fill blank cells with values from last non-blank cell above
| DF | DF__FILL_UP |  Fill blank cells with values from last non-blank cell below
| DF | DF__FILL_RIGHT |  Fill blank cells with values from last non-blank cell from left
| DF | DF__FILL_LEFT |  Fill blank cells with values from last non-blank cell from right
| DF | DF__GROUP |  Group table contents by specified columns with optional aggregation (sum/max/min etc)
| DF | DF__MERGE |  Merge a table with current table with specified type (left/right/inner/outer) 
| DF | DF__REPLACE |  Replace string values in table
| DF | DF__UNPIVOT |  Unpivot table on specified columns
| DF | DF__PIVOT |  Pivot table on specified columns
| DF | DF_COLHEADER_PROMOTE |  Promote row at specified index to column headers
| DF | DF_COLHEADER_DEMOTE |  Demote column headers to make 1st row of table
| DF | DF_COLHEADER_REORDER_ASC |  Reorder column titles in ascending order
| DF | DF_COLHEADER_REORDER_DESC |  Reorder column titles in descending order
| DF | DF__STATS |  Show basic summary statistics of table contents
| VIZ | VIZ_BOX |  Draw a box plot
| VIZ | VIZ_VIOLIN |  Draw a violin plot
| VIZ | VIZ_HIST |  Draw a hisotgram
| VIZ | VIZ_HIST_LIST |  Draw a histogram for all fields in current dataframe
| VIZ | VIZ_SCATTER |  Draw a scatter plot
| VIZ | VIZ_BAR |  Draw a bar plot
| VIZ | VIZ_LINE |  Draw a line plot
| VIZ | VIZ_TREEMAP |  Draw a treemap plot
| VIZ | VIZ_SCATTERMATRIX |  Draw a scatter matrix plot
| REPORT | REPORT_SET_VIZ_COLORS_PLOTLY |  Set plot/report colors to 'Plotly'
| REPORT | REPORT_SET_VIZ_COLORS_D3 |  Set plot/report colors to 'D3'
| REPORT | REPORT_SET_VIZ_COLORS_G10 |  Set plot/report colors to 'G10'
| REPORT | REPORT_SET_VIZ_COLORS_T10 |  Set plot/report colors to 'T10'
| REPORT | REPORT_SET_VIZ_COLORS_DARK24 |  Set plot/report colors to 'Dark24'
| REPORT | REPORT_SET_VIZ_COLORS_LIGHT24 |  Set plot/report colors to 'Light24'
| REPORT | REPORT_SET_VIZ_COLORS_SET1 |  Set plot/report colors to 'Set1'
| REPORT | REPORT_SET_VIZ_COLORS_PASTEL1 |  Set plot/report colors to 'Pastel1'
| REPORT | REPORT_SET_VIZ_COLORS_DARK2 |  Set plot/report colors to 'Dark2'
| REPORT | REPORT_SET_VIZ_COLORS_SET2 |  Set plot/report colors to 'Set2'
| REPORT | REPORT_SET_VIZ_COLORS_PASTEL2 |  Set plot/report colors to 'Pastel2'
| REPORT | REPORT_SET_VIZ_COLORS_SET3 |  Set plot/report colors to 'Set3'
| REPORT | REPORT_SET_VIZ_COLORS_ANTIQUE |  Set plot/report colors to 'Antique'
| REPORT | REPORT_SET_VIZ_COLORS_BOLD |  Set plot/report colors to 'Bold'
| REPORT | REPORT_SET_VIZ_COLORS_PASTEL |  Set plot/report colors to 'Pastel'
| REPORT | REPORT_SET_VIZ_COLORS_PRISM |  Set plot/report colors to 'Prism'
| REPORT | REPORT_SET_VIZ_COLORS_SAFE |  Set plot/report colors to 'Safe'
| REPORT | REPORT_SET_VIZ_COLORS_VIVID |  Set plot/report colors to 'Vivid'
| REPORT | REPORT_PREVIEW |  Preview all plots on screen (for use in JupyterLab)
| REPORT | REPORT_PREVIEW_FULL |  Preview all plots, dataframe and column summary on screen (for use in JupyterLab)
| REPORT | REPORT_SAVE_ALL |  Save html format report/plots and dataframe to specified location
| REPORT | REPORT_SAVE_VIZ_HTML |  Save html format report/plots to specified location
| REPORT | REPORT_SAVE_DF |  Save dataframe to specified location


