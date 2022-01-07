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
    SOURCE('attrition_csv')
    .ML_TRAIN_AND_SAVE_CLASSIFIER('Attrition')
    .REPORT_SAVE_VIZ_HTML('html_report')
)


