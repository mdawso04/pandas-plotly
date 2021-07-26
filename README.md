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

## Requirements

pandas, numpy, pathlib

## Methods

| Group | Method |  Description
| --- | --- | ---
| COLUMN | COL_ADD_FIXED | Add a new column with a 'fixed' value as content
| COLUMN | COL_ADD_INDEX |  Add a new column with a index/serial number as content
| COLUMN | COL_ADD_CUSTOM |  Add a new column with custom (lambda) content
| COLUMN | COL_ADD_EXTRACT_POSITION_AFTER |  Add a new column with content extracted from after char pos in existing column
| COLUMN | COL_ADD_EXTRACT_POSITION_BEFORE |  Add a new column with content extracted from before char pos in existing column
| COLUMN | COL_ADD_EXTRACT_CHARS_FIRST |  Add a new column with first N chars extracted from column
| COLUMN | COL_ADD_EXTRACT_CHARS_LAST |  Add a new column with last N chars extracted from column
| COLUMN | COL_ADD_DUPLICATE |  Add a new column with content duplicated from existing column
