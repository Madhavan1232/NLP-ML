# General Rules for All Files:

write codes in file that add with prompt and write code in opened file.
Apply the following rules to all code generation tasks.
Use add file blocks for all code snippets.
Do not include comment lines within the code.

# Rules for Notebook Files (**/*.ipynb):

Use matplotlib for plotting (use seaborn only if specifically required).
If the task is a plotting task, generate the plot only.
Do not print unnecessary information or dataframes.
Plot only the provided reference data or image descriptions.

# Rules for Python Files (**/*.py):

Use the following standard for reading CSV files: df = pd.read_csv(os.path.join(sys.path[0], input()))
Ensure import os, import sys, and import pandas as pd are included.


