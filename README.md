# pandas-plotly

A simple, unified interface for pandas & plotly for data wrangling, vizualization & report generating. 

## Example

```python
from scripts import SOURCE
(
    SOURCE('attrition_csv')
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

![image](https://user-images.githubusercontent.com/87593190/136625699-3d0d11a0-3268-44fb-9c33-eb9f41f94fa3.png)

## Use cases

1) Basic data wrangling / vizualization /report generation
Use in JupyterLab to see data wrangling & vizualization results in real time. When you
find an interesting vizualization or data, save on the spot with no need to keep your script.

2) Automated data wrangling / vizualization / report generation
As above play in JupyterLab to make the results you want. Once your script is ready,
save to use whenever needed.

For Windows users: Copy the supplied Windows .bat template to quickly call your script manually, 
from Windows Task Scheduler, or if in corporate environment, user start up tasks.

## Requirements

pandas, plotly, optional: pycurl

## Methods

| Method |  Description & example
| --- | --- |
| ```DF_COL_ADD_FIXED``` | Add a new column with a 'fixed' value as content<br />```.DF_COL_ADD_FIXED('Tokyo')```<br />```.DF_COL_ADD_FIXED('Tokyo', 'Location')``` 
| ```DF_COL_ADD_INDEX``` |  Add a new column with a index/serial number as content<br /> ```.DF_COL_ADD_INDEX()```<br /> ```.DF_COL_ADD_INDEX('Index_number')```<br />```.DF_COL_ADD_INDEX('No.', 0)```
| ```DF_COL_ADD_INDEX_FROM_0``` |  Convenience method for DF_COL_ADD_INDEX<br /> ```.DF_COL_ADD_INDEX_FROM_0()```<br /> ```.DF_COL_ADD_INDEX_FROM_0('Index_number')```
| ```DF_COL_ADD_INDEX_FROM_1``` |  Convenience method for DF_COL_ADD_INDEX<br /> ```.DF_COL_ADD_INDEX_FROM_1()```<br /> ```.DF_COL_ADD_INDEX_FROM_1('Index_number')```
| ```DF_COL_ADD_CUSTOM``` |  Add a new column with custom (lambda) content  
| ```DF_COL_ADD_EXTRACT_POSITION_AFTER``` |  Add a new column with content extracted from after char pos in existing column<br />```.DF_COL_ADD_EXTRACT_POSITION_AFTER('Old_column', 5)```<br /> ```.DF_COL_ADD_EXTRACT_POSITION_AFTER('Old_column', 5, 'New_column')```
| ```DF_COL_ADD_EXTRACT_POSITION_BEFORE``` |  Add a new column with content extracted from before char pos in existing column<br /> ```.DF_COL_ADD_EXTRACT_POSITION_BEFORE('Old_column', 5)```<br /> ```.DF_COL_ADD_EXTRACT_POSITION_BEFORE('Old_column', 5, 'New_column')```
| ```DF_COL_ADD_EXTRACT_CHARS_FIRST``` |  Add a new column with first N chars extracted from column<br /> ```.DF_COL_ADD_EXTRACT_CHARS_FIRST('Old_column', 5)```<br />```.DF_COL_ADD_EXTRACT_CHARS_FIRST('Old_column', 5, 'New_column')```
| ```DF_COL_ADD_EXTRACT_CHARS_LAST``` |  Add a new column with last N chars extracted from column<br /> ```.DF_COL_ADD_EXTRACT_CHARS_FIRST('Old_column', 5)```<br /> ```.DF_COL_ADD_EXTRACT_CHARS_FIRST('Old_column', 5, 'New_column')```
| ```DF_COL_ADD_DUPLICATE``` | Add a new column by copying an existing column<br /> ```.DF_COL_ADD_DUPLICATE('Old_column')```<br /> ```.DF_COL_ADD_DUPLICATE('New_column')```
| ```DF_COL_DELETE``` |  Delete specified column/s<br /> ```.DF_COL_DELETE('Del_column')```<br />```.DF_COL_DELETE(['Del_column1', 'Del_column2', 'Del_columnN')```<br />```.DF_COL_DELETE([0, 3, 6])```<br />```.DF_COL_DELETE(Slice(0,3))```
| ```DF_COL_DELETE_EXCEPT``` |  Deleted all column/s except specified<br /> ```.DF_COL_DELETE_EXCEPT('Keep_column')```<br />```.DF_COL_DELETE_EXCEPT(['Keep_column1', 'Keep_column2', 'Keep_columnN')```<br />```.DF_COL_DELETE_EXCEPT([0, 3, 6])```<br />```.DF_COL_DELETE_EXCEPT(Slice(0,3))```
| ```DF_COL_RENAME``` |  Rename specfied column/s```.DF_COL_RENAME({'OldCol1':'NewCol1', 'OldColN:'NewColN})```<br />```.DF_COL_RENAME(['NewCol1','NewCol2', 'NewColN'])```<br />
| ```DF_COL_FORMAT_TO_UPPERCASE``` |  Format specified column/s values to uppercase<br /> ```.DF_COL_FORMAT_TO_UPPERCASE(['Column1', 'ColumnN'])```
| ```DF_COL_FORMAT_TO_LOWERCASE``` |  Format specified column/s values to lowercase<br /> ```.DF_COL_FORMAT_TO_LOWERCASE(['Column1', 'ColumnN'])```
| ```DF_COL_FORMAT_TO_TITLECASE``` |  Format specified column/s values to titlecase<br /> ```.DF_COL_FORMAT_TO_TITLECASE(['Column1', 'ColumnN'])```
| ```DF_COL_FORMAT_STRIP``` |  Format specified column/s values by stripping invisible characters<br /> ```.DF_COL_FORMAT_STRIP(['Column1', 'ColumnN'])```
| ```DF_COL_FORMAT_STRIP_LEFT``` |  Convenience method for DF_COL_FORMAT_STRIP<br /> ```.DF_COL_FORMAT_STRIP_LEFT(['Column1', 'ColumnN'])```
| ```DF_COL_FORMAT_STRIP_RIGHT``` |  Convenience method for DF_COL_FORMAT_STRIP<br /> ```.DF_COL_FORMAT_STRIP_RIGHT(['Column1', 'ColumnN'])```
| ```DF_COL_FORMAT_ADD_PREFIX``` | Format specified single column values by adding prefix<br /> ```.DF_COL_FORMAT_ADD_PREFIX('Prefix_', 'Column')```
| ```DF_COL_FORMAT_ADD_SUFFIX``` |  Format specified single column values by adding suffix<br /> ```.DF_COL_FORMAT_ADD_SUFFIX(['_suffix', 'Column')```
| ```DF_COL_FORMAT_TYPE``` | 
| ```DF_COL_FORMAT_ROUND``` | Round numerical column values to specified decimal<br /> ```.DF_COL_FORMAT_ROUND(2)```<br /> ```.DF_COL_FORMAT_ROUND({'Column1':2, 'Column2':1, 'ColumnN':1})```
| ```DF_ROW_FILTER``` |  Filter rows with specified filter criteria<br /> ```.DF_ROW_FILTER('Age < 29')```
| ```DF_ROW_KEEP_BOTTOM``` |  Delete all rows except specified bottom N rows<br /> ```.DF_ROW_KEEP_BOTTOM(5)```
| ```DF_ROW_KEEP_TOP``` |  Delete all rows except specified top N rows<br /> ```.DF_ROW_KEEP_TOP(5)```
| ```DF_ROW_REVERSE``` |  Reorder all rows in reverse order<br /> ```.DF_ROW_REVERSE()```
| ```DF_ROW_SORT``` |  Reorder dataframe by specified columns in ascending/descending order<br /> ```.DF_ROW_SORT('Sort_column1')```<br /> ```.DF_ROW_SORT(['Sort_column1', 'Sort_Column2'])```<br /> ```.DF_ROW_SORT(['Sort_column1', 'Sort_Column2'], True)```
| ```DF__APPEND``` |  Append a table to bottom of current table<br /> ```.DF__APPEND(dataframe_to_append)```
| ```DF__FILL_DOWN``` |  Fill blank cells with values from last non-blank cell above<br /> ```.DF__FILL_DOWN()```
| ```DF__FILL_UP``` |  Fill blank cells with values from last non-blank cell below<br /> ```.DF__FILL_UP()```
| ```DF__FILL_RIGHT``` |  Fill blank cells with values from last non-blank cell from left<br /> ```.DF__FILL_RIGHT()```
| ```DF__FILL_LEFT``` |  Fill blank cells with values from last non-blank cell from right<br /> ```.DF__FILL_LEFT()```
| ```DF__GROUP``` |  Group table contents by specified columns with optional aggregation (sum/max/min etc)<br /> ```.DF__GROUP('Department', {'Age': ['mean', 'count']})```
| ```DF__MERGE``` |  Merge a table with current table with specified type (left/right/inner/outer) 
| ```DF__REPLACE``` |  Replace string values in table
| ```DF__UNPIVOT``` |  Unpivot table on specified columns
| ```DF__PIVOT``` |  Pivot table on specified columns
| ```DF_COLHEADER_PROMOTE``` |  Promote row at specified index to column headers<br /> ```.DF_COLHEADER_PROMOTE(1)```
| ```DF_COLHEADER_DEMOTE``` |  Demote column headers to make 1st row of table<br /> ```.DF_COLHEADER_DEMOTE()```
| ```DF_COLHEADER_REORDER_ASC``` |  Reorder column titles in ascending order<br /> ```.DF_COLHEADER_REORDER_ASC()```
| ```DF_COLHEADER_REORDER_DESC``` |  Reorder column titles in descending order<br /> ```.DF_COLHEADER_REORDER_DESC()```
| ```DF__STATS``` |  Show basic summary statistics of table contents<br /> ```.DF__STATS()```
| ```VIZ_BOX``` |  Draw a box plot<br /> ```.VIZ_BOX('Department', 'NoEmployees')```
| ```VIZ_VIOLIN``` |  Draw a violin plot<br /> ```.VIZ_VIOLIN('Department', 'NoEmployees')```
| ```VIZ_HIST``` |  Draw a hisotgram<br /> ```.VIZ_HIST('Department', 'NoEmployees')```
| ```VIZ_HIST_LIST``` |  Draw a histogram for all fields in current dataframe<br /> ```.VIZ_HIST_LIST()```
| ```VIZ_SCATTER``` |  Draw a scatter plot
| ```VIZ_BAR``` |  Draw a bar plot<br /> ```.VIZ_BAR('Department', 'NoEmployees')```
| ```VIZ_LINE``` |  Draw a line plot<br /> ```.VIZ_LINE('Department', 'NoEmployees')```
| ```VIZ_TREEMAP``` |  Draw a treemap plot
| ```VIZ_SCATTERMATRIX``` |  Draw a scatter matrix plot
| ```REPORT_SET_VIZ_COLORS_PLOTLY``` |  Set plot/report colors to 'Plotly'<br /> ```.REPORT_SET_VIZ_COLORS_PLOTLY```
| ```REPORT_SET_VIZ_COLORS_D3``` |  Set plot/report colors to 'D3'<br /> ```.REPORT_SET_VIZ_COLORS_D3```
| ```REPORT_SET_VIZ_COLORS_G10``` |  Set plot/report colors to 'G10'<br /> ```.REPORT_SET_VIZ_COLORS_G10```
| ```REPORT_SET_VIZ_COLORS_T10``` |  Set plot/report colors to 'T10'<br /> ```.REPORT_SET_VIZ_COLORS_T10```
| ```REPORT_SET_VIZ_COLORS_ALPHABET``` |  Set plot/report colors to 'Alphabet'<br /> ```.REPORT_SET_VIZ_COLORS_ALPHABET```
| ```REPORT_SET_VIZ_COLORS_DARK24``` |  Set plot/report colors to 'Dark24'<br /> ```.REPORT_SET_VIZ_COLORS_DARK24```
| ```REPORT_SET_VIZ_COLORS_LIGHT24``` |  Set plot/report colors to 'Light24'<br /> ```.REPORT_SET_VIZ_COLORS_LIGHT24```
| ```REPORT_SET_VIZ_COLORS_SET1``` |  Set plot/report colors to 'Set1'<br /> ```.REPORT_SET_VIZ_COLORS_SET1```
| ```REPORT_SET_VIZ_COLORS_PASTEL1``` |  Set plot/report colors to 'Pastel1'<br /> ```.REPORT_SET_VIZ_COLORS_PASTEL1```
| ```REPORT_SET_VIZ_COLORS_DARK2``` |  Set plot/report colors to 'Dark2'<br /> ```.REPORT_SET_VIZ_COLORS_DARK2```
| ```REPORT_SET_VIZ_COLORS_SET2``` |  Set plot/report colors to 'Set2'<br /> ```.REPORT_SET_VIZ_COLORS_SET2```
| ```REPORT_SET_VIZ_COLORS_PASTEL2``` |  Set plot/report colors to 'Pastel2'<br /> ```.REPORT_SET_VIZ_COLORS_PASTEL2```
| ```REPORT_SET_VIZ_COLORS_SET3``` |  Set plot/report colors to 'Set3'<br /> ```.REPORT_SET_VIZ_COLORS_SET3```
| ```REPORT_SET_VIZ_COLORS_ANTIQUE``` |  Set plot/report colors to 'Antique'<br /> ```.REPORT_SET_VIZ_COLORS_ANTIQUE```
| ```REPORT_SET_VIZ_COLORS_BOLD``` |  Set plot/report colors to 'Bold'<br /> ```.REPORT_SET_VIZ_COLORS_BOLD```
| ```REPORT_SET_VIZ_COLORS_PASTEL``` |  Set plot/report colors to 'Pastel'<br /> ```.REPORT_SET_VIZ_COLORS_PASTEL```
| ```REPORT_SET_VIZ_COLORS_PRISM``` |  Set plot/report colors to 'Prism'<br /> ```.REPORT_SET_VIZ_COLORS_PRISM```
| ```REPORT_SET_VIZ_COLORS_SAFE``` |  Set plot/report colors to 'Safe'<br /> ```.REPORT_SET_VIZ_COLORS_SAFE```
| ```REPORT_SET_VIZ_COLORS_VIVID``` |  Set plot/report colors to 'Vivid'<br /> ```.REPORT_SET_VIZ_COLORS_VIVID```
| ```REPORT_PREVIEW``` |  Preview all plots on screen (for use in JupyterLab)<br /> ```.REPORT_PREVIEW()```
| ```REPORT_PREVIEW_FULL``` |  Preview all plots, dataframe and column summary on screen (for use in JupyterLab)<br /> ```.REPORT_PREVIEW_FULL()```
| ```REPORT_SAVE_ALL``` |  Save html format report/plots and dataframe to specified location<br /> ```.REPORT_SAVE_ALL()```
| ```REPORT_SAVE_VIZ_HTML``` |  Save html format report/plots to specified location<br /> ```.REPORT_SAVE_VIZ_HTML()```
| ```REPORT_SAVE_DF``` |  Save dataframe to specified location<br /> ```.REPORT_SAVE_DF()```
