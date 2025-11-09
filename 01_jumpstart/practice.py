
# import libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# plotting

from plotnine import (
    ggplot, aes, 
    geom_col, geom_line, geom_smooth,
    facet_wrap,
    scale_y_continuous, scale_x_datetime,
    labs,
    theme, theme_minimal, theme_matplotlib, theme_light, theme_classic,
    expand_limits,
    element_text
    
)

from mizani.breaks import date_breaks
from mizani.formatters import date_format, currency_format

# checking what my date format is going to be.
date_format()

#Misc --- importing libraries that can help in the creation of directories.

from os import mkdir, getcwd

from rich import pretty
pretty.install()



# Importing Data Files ---

bikes_df        =   pd.read_excel("./00_data_raw/bikes.xlsx")
bikeshops_df    =   pd.read_excel("./00_data_raw/bikeshops.xlsx")
orderlines_df   =   pd.read_excel(
    io="./00_data_raw/orderlines.xlsx",
    converters={'order.date':str})

s = bikes_df['description']
freq_count_series = s.value_counts()# count out how many times a certain description is appearing
freq_count_series.nlargest(5)#selects out the top 5 most frequent descritions.

top5_bikes_series = bikes_df['description'].value_counts().nlargest(5)
fig =  top5_bikes_series.plot(kind= "barh")
fig.invert_yaxis()

fig
plt.show()


## Here here, lets get into some data joining
orderlines_df = pd.DataFrame(orderlines_df)

orderlines_df.drop(columns='Unnamed: 0', axis=1)

bike_orderlines_joined_df = orderlines_df\
    .drop(columns='Unnamed: 0', axis=1)\
    .merge(
        right=bikes_df,
        how='left',
        left_on='product.id',
        right_on='bike.id'
    )\
    .merge(
        right=bikeshops_df, 
        how='left',
        left_on ='customer.id',
        right_on='bikeshop.id'
    )


bike_orderlines_joined_df

# doing some further data wrangling(just getting that data ready for getting cooked)
df = bike_orderlines_joined_df

df2 = bike_orderlines_joined_df.copy()

# handle the dates
df.info()
df['order.date'] = pd.to_datetime(df['order.date'])

df.description
df.T

# splitting Description into category_1, category_2 and frame_material

temp_df = df['description'].str.split(pat=" - ", expand= True)

df['category.1']        =   temp_df[0]
df['category.2']        =   temp_df[1]
df['frame.material']    =   temp_df[2]

df.info()

# Splitting Location into city and state
temp_df = df['location'].str.split(pat=", ", expand = True)
df['city']      =   temp_df[0]
df['state']     =   temp_df[1]

df

# Price Extended

df['total.price'] = df['quantity'] * df['price']

df.sort_values('total.price', ascending= False)

# Reorganizing

df.columns

cols_to_keep_list = [
    'order.id', 
    'order.line',
    'order.date',
    #'customer.id',
    #'product.id',
    'quantity', 
    #'bike.id', 
    'model', 
    #'description', 
    'price', 
    #'bikeshop.id',
    'bikeshop.name', 
    'location', 
    'category.1', 
    'category.2',
    'frame.material', 
    'city', 
    'state', 
    'total.price']

df = df[cols_to_keep_list]

df.columns = df.columns.str.replace(".", "_")

df.order_id

bike_orderlines_joined_df = df

# following the course, visualising a Time Series

#mkdir("01_data_information")

bike_orderlines_joined_df.to_pickle("./01_data_information/bike_orderlines_wrangled_df.pkl")
df = pd.read_pickle("./01_data_information/bike_orderlines_wrangled_df.pkl")
df
# calculate the total sales by month

df = pd.DataFrame(df)
df['order_date']

order_date_series = df['order_date']
order_date_series.dt.year # this is returning the year

# returns a single column dataframe
sales_by_month_df = df[['order_date', 'total_price']]\
    .set_index('order_date')\
    .resample(rule='MS')\
    .aggregate(np.sum)\
    .reset_index()

sales_by_month_df

# Quick plot ----

sales_by_month_df.plot(x='order_date', y ='total_price')
plt.show()


# report plot ---

usd = currency_format(prefix="$", accuracy= 1, big_mark=",")
usd([1000])

ggplot(aes(x='order_date', y='total_price'), sales_by_month_df)+\
    geom_line()+\
    geom_smooth(
        method = 'loess',
        se = False,
        color = "blue",
        span = 0.3
    )+\
    scale_y_continuous(labels=usd)+\
    labs(
        
        title = "Revenue by Month",
        x="",
        y="Revenue"
        
    )+\
    theme_matplotlib()+\
    expand_limits(y=0)

# Sales by Year and category 2 ----
# Step 1 - Manipulate ---

sales_by_month_cat_2 = df[['category_2','order_date','total_price']] \
    .set_index('order_date')\
    .groupby('category_2')\
    .resample('W')\
    .agg(func={'total_price':np.sum})\
    .reset_index()

#step 2 - Visualize
# simple plot
sales_by_month_cat_2\
    .pivot(
        index       =       'order_date',
        columns     =       'category_2',
        values      =       'total_price'
    )\
    .fillna(0)\
    .plot(kind="line", subplots=True, layout = (3,3))

plt.show()

# reporting plot

ggplot(
    mapping = aes(x='order_date', y = 'total_price'),
    data = sales_by_month_cat_2
)+\
    geom_line(color = "#2C3E50")+\
    geom_smooth(method ="lm", se = False, color = "blue")+\
    facet_wrap(
        facets="category_2",
        ncol=3,
        scales="free_y"
    )+\
    scale_y_continuous(labels=usd)+\
    scale_x_datetime(
        breaks=date_breaks("2 years"),
        labels=date_format(fmt="%y-%m")
    )+\
    labs(
        title="Revenue By Week",
        x="",
        y="Revenue"
    )+\
    theme_matplotlib()+\
    theme(
        subplots_adjust={'wspace':0.25},
        axis_text_y=element_text(size=6),
        axis_text_x=element_text(size=6)
    )

plt.show()

# writing files in python ---

#pickle files---

mkdir("02_prac_storage")

df.to_pickle("./02_prac_storage/bike_orderlines_wrangled_df.pkl")

#csv file ---
df.to_csv("./02_prac_storage/bike_orderlines_wrangles_df.csv")

# Excel
df.to_excel("./02_prac_storage/bike_orderlines_wrangled_df.xlsx")













