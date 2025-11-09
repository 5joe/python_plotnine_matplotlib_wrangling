# DS4B 101-P: PYTHON FOR BUSINESS ANALYSIS ----
# Module 4 (Time Series): Working with Time Series Data ----

# IMPORTS

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from my_pandas_extensions.database import collect_data

# DATA

df = collect_data()

# 1.0 DATE BASICS

df['order_date']

# Conversion

pd.to_datetime("2011-01-07").to_period(freq = "W")
pd.to_datetime("2011-01-07")\
    .to_period(freq = "W")\
    .to_timestamp()

# Accessing elements

df.order_date

# Months

df.order_date.dt.month

df.order_date.dt.month_name()

# Days

df.order_date.dt.day
df.order_date.dt.day_name()

#Year
df.order_date.dt.year

# DATE MATH

import datetime

today = datetime.date.today()

pd.to_datetime(today + pd.Timedelta("1 day"))

# I need to look into this and handle
df.order_date + pd.Timedelta("1 year")


df.order_date + pd.Timedelta("30 min")

# Duration(Time between 2 time stamps)

### this part is incomplete, there comeback and finish it later

today = datetime.date.today()

today + pd.Timedelta('1Y')


# DATE SEQUENCES

pd.date_range(
    start   =   pd.to_datetime("2011-01"),
    periods =   10,
    freq    =   "2D"
    )

pd.date_range(
    start   =   pd.to_datetime("2011-01"),
    end     =   pd.to_datetime("2011-12-31"),
    freq    =   "1W"
    )


# PERIODS
# - Periods represent timestamps that fall within an interval using a frequency.
# - IMPORTANT: {sktime} requires periods to model univariate time series


# Convert to Time Stamp

# Get the Frequency



# TIME-BASED GROUPING (RESAMPLING)
# - The beginning of our Summarize by Time Function

# Using kind = "timestamp"
##Single time series
bike_sales_m_df = df[['order_date', 'total_price']]\
    .set_index('order_date')\
    .resample("M", kind="period")\
    .sum()

bike_sales_m_df

df[['order_date', 'total_price']]\
    .set_index('order_date')\
    .resample("M", kind="period")\
    .sum()\
    .reset_index()\
    .assign(order_date = lambda x: x.order_date.dt.to_timestamp())

# Using kind = "period"
###Grouped Time series
## there is another error here that I need to correct here
bike_sale_cat2_m_wide_df = df[['category_2', 'order_date', 'total_price']]\
    .set_index('order_date')\
    .groupby('category_2')\
    .resample('M', kind='period')\
    .agg(np.sum)\
    .drop('category_2', axis=1)\
    .unstack('category_2')\
    .reset_index()\
    .assign(order_date = lambda x: x['order_date'].dt.to_period())\
    .set_index('order_date')

bike_sale_cat2_m_wide_df

# MEASURING CHANGE

# Difference from Previous Timestamp

#  - Single (No Groups)

bike_sales_m_df\
    .assign(total_price_lag1 = lambda x: x['total_price'].shift(1))\
    .assign(diff = lambda x: x.total_price - x.total_price_lag1)\
    .plot(y='diff')
    
bike_sales_m_df\
    .apply(lambda x: (x - x.shift(1))/x.shift(1))\
    .plot()
    

#  - Multiple Groups: Key is to use wide format with apply

bike_sale_cat2_m_wide_df\
    .apply(lambda x: (x - x.shift(1)) / x.shift(1))\
    .plot()

#  - Difference from First Timestamp

#this is for the singles

bike_sales_m_df\
    .apply(lambda x: (x -x[0])/ x[0])\
    .plot()


#This is for the group
bike_sale_cat2_m_wide_df\
    .apply(lambda x: (x -x[0])/ x[0])\
    .plot()



# CUMULATIVE CALCULATIONS

bike_sales_m_df\
    .resample("Y")\
    .sum()\
    .cumsum()\
    .plot(kind = "bar")
    
###This is the old method that isnt used anymore 

# bike_sales_m_df\
#     .resample("Y")\
#     .sum()\
#     .cumsum()\
#     .reset_index()\
#     .assign(order_date = lambda x: x.order_date.dt.to_period())\
#     .set_index('order_date')\
#     .plot(kind = "bar")

# we do the samething for the group

bike_sale_cat2_m_wide_df\
    .resample("Y")\
    .sum()\
    .cumsum()\
    .plot(kind="bar", stacked=True)

# ROLLING CALCULATIONS

# Single

bike_sales_m_df.plot()

bike_sales_m_df['total_price']\
    .rolling(
        window=12
    )\
    .mean()

# doing with window=3
bike_sales_m_df\
    .assign(
        total_price_roll12 = lambda x: x['total_price']\
            .rolling(
                window=3,
                center=True,
                min_periods=1
            )\
            .mean()
    )\
    .plot()
    
# doing with window=12
bike_sales_m_df\
    .assign(
        total_price_roll12 = lambda x: x['total_price']\
            .rolling(
                window=12,
                center=True,
                min_periods=1
            )\
            .mean()
    )\
    .plot()

# Groups - Can't use assign(), we'll use merging

bike_sale_cat2_m_wide_df\
    .apply(
        lambda x: x.rolling(
            window =24,
            center = True,
            min_periods =1
        )\
        .mean()
    )\
    .rename(lambda x: x+ "roll 24", axis = 1)\
    .merge(
        bike_sale_cat2_m_wide_df,
        how="right",
        left_index=True,
        right_index=True
    )\
    .plot()


