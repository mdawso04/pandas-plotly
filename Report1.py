# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.12.0
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

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


