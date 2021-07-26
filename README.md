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
