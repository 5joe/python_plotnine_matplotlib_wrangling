import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from os import mkdir

from plotnine import (
ggplot, aes, 
geom_col, geom_line, geom_smooth,
facet_wrap,
scale_y_continuous, scale_x_datetime,
labs,
theme, theme_minimal, theme_matplotlib,
expand_limits,
element_text,
theme_light,
theme_classic
)

from mizani.formatters import date_format, currency_format


bikes_df = pd.read_excel("00_data_raw/bikes.xlsx")

bikes_df

bikeshops_df = pd.read_excel("00_data_raw/bikeshops.xlsx")
bikeshops_df



orderlines_df = pd.read_excel(
    io = "00_data_raw/orderlines.xlsx",
    converters= {'order.date':str}
    )
orderlines_df.info()

# Lets us do the joining of the data
orderlines_df = pd.DataFrame(orderlines_df)

# here we are now doing that important merging. Using what you would refer to as the primary keys
bike_orderlines_joined_df = orderlines_df \
    .drop(columns='Unnamed: 0', axis=1) \
    .merge(
       right=bikes_df,
        how='left',
        left_on='product.id',
        right_on='bike.id'
    )\
    .merge(
        
        right=bikeshops_df,
        how = 'left',
        left_on='customer.id',
        right_on='bikeshop.id'
        
    )
    
bike_orderlines_joined_df

# 5.0 Wrangling Data ---

# No Copy
df = bike_orderlines_joined_df
df

# here there is a copy
df2 = bike_orderlines_joined_df.copy()   
df2 

# handling dates
df['order.date']

df['order.date'] = pd.to_datetime(df['order.date'])

df.info()

# what is the addecr of coping vs not copying
bike_orderlines_joined_df.info()# since df and this share the same information u find that any update to bike_orderlines_joined_df() affects df and viceversa, unless there isnt any it was a copy

df.T# Transpose

# splitting description in category_1, category_2 and frame_material

"Mountain - Over Mountain - Carbon".split(" - ")

temp_df = df['description'].str.split(pat=' - ', expand= True)

df['category.1'] = temp_df[0]
df['category.2'] = temp_df[1]
df['frame.material'] = temp_df[2]
df

# we want to split the location column for this as well 
temp_df = df['location'].str.split(pat=', ', n=1, expand=True)

df['City'] = temp_df[0]
df['State'] = temp_df[1]

# Price Extended

df['Total.Price'] =  df['quantity']*df['price']

df.sort_values('Total.Price', ascending=False)

# we can reorganise the columns
df.columns# since there arent any parenthesis needed, this is an attribute
cols_to_keep_list = ['order.line', 
    'order.date', 
    #    'customer.id',
    #    'product.id', 
    #    'quantity', 
    #    'bike.id',
    'model', 
    #    'description', 
    'quantity',
    'price', 
    'Total.Price',
    'bikeshop.id',
    #    'bikeshop.name',
    'location',
    'category.1',
    'category.2', 
    'frame.material',
    'City', 
    'State', 
     ]

df[cols_to_keep_list]

df['order.date']

'order.date'.replace(".", "_")

# here we rename the df columns and change the "." to "_"
df.columns = df.columns.str.replace(".", "_")

df.order_id
df['order_id']

bike_orderlines_joined_df
bike_orderlines_wrangle_df = df
bike_orderlines_wrangle_df

# we can do some visualizing a Time series
mkdir("00_data_wrangles")

bike_orderlines_wrangle_df.to_pickle("00_data_wrangles/bike_orderlines_wrangled_df.pkl")


df = pd.read_pickle("00_data_wrangles/bike_orderlines_wrangled_df.pkl")


# Total sales by month

df = pd.DataFrame(df)
df['order_date']# this returns a series


order_date_series = df['order_date']
order_date_series.dt.year

#this is at the end of the year
df[['order_date', 'Total_Price']]\
    .set_index('order_date')\
    .resample(rule='Y')\
    .sum()

#Year Strta
df.info()

df['order_date'] = pd.to_datetime(df['order_date'])

df[['order_date', 'Total_Price']]\
    .set_index('order_date')\
    .resample(rule='YS')\
    .sum()

#by Month(this is at the end of the Month)
df[['order_date', 'Total_Price']]\
    .set_index('order_date')\
    .resample(rule='M')\
    .aggregate(np.sum)


#by Month(this is at the start of the month
sales_by_month_df = df[['order_date', 'Total_Price']]\
    .set_index('order_date')\
    .resample(rule='MS')\
    .aggregate(np.sum) \
    .reset_index()

sales_by_month_df

# Quick plot ----
sales_by_month_df.plot(x = 'order_date', y = "Total_Price")
plt.show()



##Report plots
# here these are the plots that I will place in a report plot

#create the currency label
usd = currency_format(prefix="$", accuracy= 1, big_mark=",")
usd([1000])

ggplot(sales_by_month_df,aes(x = 'order_date', y = 'Total_Price'))+\
    geom_line()+\
    geom_smooth(method = 'loess')

ggplot(sales_by_month_df,aes(x = 'order_date', y = 'Total_Price'))+\
    geom_line()+\
    geom_smooth(method = 'loess',
                se = False, 
                color= "blue", 
                span = 0.3
    )+\
    scale_y_continuous(labels=usd)+\
    labs(
            title="Revenue by Month",
            x = "",
            y="Revenue"
            
        )+\
    theme_minimal()+\
    expand_limits(y= 0)
    
    
#sales by year and category 2
## Manipulate
df.info()

#be meticulous about learning this part
df[['category_2','order_date','Total_Price']]\
    .set_index('order_date')\
    .groupby('category_2')\
    .resample('W')\
    .agg(func = np.sum)

#be meticulous about learning this part
##Here we are doing it specifically for the Total Price column
sales_by_month_cat2 = df[['category_2','order_date','Total_Price']]\
    .set_index('order_date')\
    .groupby('category_2')\
    .resample('W')\
    .agg(func = {'Total_Price':np.sum})\
    .reset_index()
    
sales_by_month_cat2

## lets do some visulizations with the Matplotlib
#this plot is a little messy
sales_by_month_cat2\
    .pivot(
        index = 'order_date',
        columns='category_2',
        values = 'Total_Price'
    )\
    .fillna(0)\
    .plot()


sales_by_month_cat2\
    .pivot(
        index = 'order_date',
        columns='category_2',
        values = 'Total_Price'
    )\
    .fillna(0)\
    .plot(kind = 'line', subplots = True)
#Lets make subplots so that its much more clear
sales_by_month_cat2\
    .pivot(
        index = 'order_date',
        columns='category_2',
        values = 'Total_Price'
    )\
    .fillna(0)\
    .plot(kind = 'line', subplots = True, layout=(3,3))
    
plt.show()
        
# Reporting

ggplot(
    mapping = aes(x = 'order_date', y= 'Total_Price', color = 'category_2'),
    data = sales_by_month_cat2
    ) +\
    geom_line()+\
    facet_wrap(
        facets = "category_2",
        ncol=3
    )        









