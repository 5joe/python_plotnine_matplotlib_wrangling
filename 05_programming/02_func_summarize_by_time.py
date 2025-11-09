# DS4B 101-P: PYTHON FOR DATA SCIENCE AUTOMATION ----
# Module 5 (Programming): Functions ----

# Imports

import pandas as pd
import numpy as np

from my_pandas_extensions.database import collect_data

df = collect_data()

# WHAT WE WANT TO STREAMLINE

freq = "Q"

df[['category_2', 'order_date', 'total_price']] \
    .set_index('order_date') \
    .groupby('category_2') \
    .resample(rule, kind = 'period') \
    .agg(np.sum) \
    .unstack("category_2") \
    .reset_index() \
    .set_index("order_date")

# BUILDING SUMMARIZE BY TIME

data = df

def summarize_by_time(
    data, date_column, value_column,
    groups = None, 
    rule = "D",
    agg_func = np.sum,
    kind = "timestamp",
    **kwargs
    ):
    
    #CHECKS
    
    #BODY
    
    # handle date column
    data = data.set_index(date_column)
    
    # Handle groupby
    if groups is not None:
        data = data.groupby(groups)
    
    # Handle resample
    data = data.resample(
        rule = rule,
        kind = kind,
        **kwargs,
    )
    

    return data


summarize_by_time(
    data,
    date_column = 'order_date',
    value_column= 'total_price',
    #groups= ["category_1", "category_2"],
    rule="M",
    kind="period"
    ).sum()

# ADDING TO OUR TIME SERIES MODULE

