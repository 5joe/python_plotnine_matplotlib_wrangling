# DS4B 101-P: PYTHON FOR DATA SCIENCE AUTOMATION ----
# Module 5 (Programming): Functions ----

# Imports

import pandas as pd
import numpy as np
from pandas.core import groupby
from pandas.core.series import Series

from my_pandas_extensions.database import collect_data


df = collect_data()

# 1.0 EXAMINING FUNCTIONS ----

# Pandas Series Function
# ?pd.Series.max
# ?np.max

?pd.Series.max

type(df.total_price)

df.total_price.max()
df.total_price.min()


my_max = pd.Series.max

my_max(df.total_price)

type(pd.Series.max)

# Pandas Data Frame Function
# ?pd.DataFrame.aggregate

pd.DataFrame.aggregate(
    self = df,
    func=np.sum
)

df.aggregate(func=np.sum)

pd.DataFrame.aggregate(
    self = df[['total_price']],
    func=np.quantile,
    q= 0.5
)


# 2.0 OUTLIER DETECTION FUNCTION ----
# - Works with a Pandas Series

x = df['total_price']

iqr_multiplier = "abc"

def detect_outliers(x, iqr_multiplier = 1.5, how = "both"):
    """Used to detect outliers using the  1.5 IQR (Inner Quartile Range) Method

    Args:
        x (Pandas Series): A numeric pandas series
        iqr_multiplier (Int, float, optional):
        A multiplier used to modify IQR sensitivity.
        Must be positive, lower values will add more outliers and 
        larger values add fewer outliers.
        Defaults to 1.5.
        
        how (str, optional): 
        One of "both", "upper" or "lower". Defaults to "both".
        "both" returns both upper and lower outliers.
        "lower" flags lower outliers only
        "upper" flags upper outliers only

    Raises:
        Exception: _description_
        Exception: _description_
        Exception: _description_
        Exception: _description_

    Returns:
        [Pandas Series]: A boolean series that flags outliers as true/ false 
    """
    
    
    #CHECKS
    if type(x) is not pd.Series:
        raise Exception("'x' must be a pandas series")
    
    if not isinstance(iqr_multiplier, (float, int)):
        raise Exception("'iqr_multiplier' must be an int or float")
    if iqr_multiplier<=0:
        raise Exception("'iqr_multiplier' must be a positive value")
    
    how_options = ['both','upper','lower']
    
    if how not in how_options:
        raise Exception(
            f"Invalid 'how', Expected one of {how_options}"
        )
    
    # IQR LOGIC
    
    q75 = np.quantile(x, 0.75)
    q25 = np.quantile(x, 0.25)
    iqr = q75 - q25
    
    lower_limit = q25 - iqr_multiplier*iqr
    upper_limit = q75 + iqr_multiplier*iqr
    
    outliers_upper = x >= upper_limit
    outliers_lower = x <= lower_limit
    
    
    if how == "both":
        outliers = outliers_upper | outliers_lower
    elif how == "lower":
        outliers = outliers_lower
    else:
        outliers = outliers_upper
    return outliers

# ?detect_outliers

detect_outliers(df['total_price'], iqr_multiplier=0.5)


# This then gives all the rows where we have outliers
df[
    detect_outliers(df['total_price'], 
    iqr_multiplier=0.1,
    how = "lower")
    ]

# doing the first check
detect_outliers(1)

# doing a second test
detect_outliers(df['total_price'], iqr_multiplier="abc")
detect_outliers(df['total_price'], iqr_multiplier=-5)
detect_outliers(df['total_price'], how= "abc")

# Groupby Example
df\
    .groupby("category_2")\
    .apply(
        lambda x: x[
            detect_outliers(
                x = x['total_price'],
                iqr_multiplier=1.5,
                how="upper"
            )
        ]
    )

# 3.0 EXTENDING A CLASS ----

pd.Series.detect_outliers = detect_outliers

df['total_price'].detect_outliers()

# ?pd.Series.detect_outliers

