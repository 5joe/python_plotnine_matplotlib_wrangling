import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#Plotting
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

from mizani.breaks import date_breaks
from mizani.formatters import date_format, currency_format

# Misc
from os import mkdir, getcwd

from rich import pretty
pretty.install()

# Lets us do some summation for this(numpy useage)
np.sum([1,2,3])
np.sum([1,4,6])

# Lets do some subtraction
np.subtract(5,1)
np.subtract([6,7],[1,1])

#Lets do some division
np.divide(6,3)

# then some multiplication
np.multiply([1,2,3],[4,5,6])

help(pd.read_excel)

# 2.0 Importing Data Files -----
getcwd()
bikes_df = pd.read_excel("00_data_raw/bikes.xlsx")
bikes_df

bikeshops_df = pd.read_excel("00_data_raw/bikeshops.xlsx")
bikeshops_df

orderlines_df = pd.read_excel(
    io = "00_data_raw/orderlines.xlsx",
    converters= {'order.date':str}
    )

orderlines_df

# 3.0 Examining Data ---

#Data counting and determing which have the highest frequency and the least

s = bikes_df['description']
freq_count_series = s.value_counts()

# for the five most frequent in description
top5_bikes = freq_count_series.nlargest(5)

fig = top5_bikes.plot(kind="barh")
fig.invert_yaxis()

# for the 5 least frequent
last5_bikes = freq_count_series.nsmallest(5)
fig1 = last5_bikes.plot(kind="barh")
fig1.invert_yaxis()

pd.Series.plot(top5_bikes)

plt.show()

# 4.0 Joining Data ----
orderlines_df.info()
type(orderlines_df)

orderlines_df = pd.DataFrame(orderlines_df)

##help(pd.DataFrame.drop)
orderlines_df.drop(columns='Unnamed: 0', axis =1)

bike_orderlines_joined_df = orderlines_df\
    .drop(columns='Unnamed: 0', axis=1)\
    .merge(
        right = bikes_df,
        how='left',
        left_on='product.id',
        right_on='bike.id'
    )\
    .merge(
        right=bikeshops_df,
        how='left',
        left_on='customer.id',
        right_on='bikeshop.id'
    )

bike_orderlines_joined_df

df = bike_orderlines_joined_df

df2 = bike_orderlines_joined_df.copy()

df.info()
df['order.date']
df['order.date'] = pd.to_datetime(df['order.date'])

# Splitting Description into Category_1, Category_2 and frame_material
"Mountain - Over Mountain - Carbon".split(" - ")

temp_df = df['description'].str.split(pat=" - ", expand=True)
df['Category.1'] = temp_df[0]
df['Category.2'] = temp_df[1]
df['frame.material'] = temp_df[2]

df

# Splitting Location into City and State
df.location
temp_df1 = df['location'].str.split(pat=', ', expand=True)

temp_df1
df['city'] = temp_df1[0]
df['state'] = temp_df1[1]

df


# Price Extended
df['total.price'] = df['quantity'] * df['price']
df.sort_values('total.price',ascending=False)

# Reorganizing
df.columns
cols_to_keep = ['order.id',
                 'order.line', 
                 'order.date', 
                # 'customer.id', 
                # 'product.id',
                 'quantity',   
                # 'bike.id', 
                 'model', 
                # 'description', 
                 'price', 
                # 'bikeshop.id',
                 'bikeshop.name', 
                 'location', 
                 'Category.1', 
                 'Category.2',
                 'frame.material', 
                 'city', 
                 'state', 
                 'total.price'
                 ]

df = df[cols_to_keep]

# Remaining Columns

df['order.date']
'order.date'.replace(".","_")

df.columns = df.columns.str.replace(".","_")

df.order_date

df['order_date']

bike_orderlines_joined_df

bike_orderlines_wrangle_df = df

# 6.0 Visualizing a Time Series ---

mkdir("00_data_wrangled")

bike_orderlines_wrangle_df.to_pickle("00_data_wrangled/bike_orderlines_wrangled_df.pkl")

df = pd.read_pickle("00_data_wrangled/bike_orderlines_wrangled_df.pkl")

# 6.1 Total Sales by Month ---

df = pd.DataFrame(df)

df['order_date']

order_date_series = df['order_date']
order_date_series.dt.year

# returns a single column dataframe
sales_by_month_df = df[['order_date', 'total_price']]\
            .set_index('order_date')\
            .resample(rule='MS')\
            .aggregate(np.sum)\
            .reset_index()

sales_by_month_df


# Quick Plot ---
sales_by_month_df.plot(x = 'order_date', y = 'total_price')

# Report Plot ----
usd = currency_format(prefix="$",accuracy=1, big_mark=",")

usd([10000])

ggplot(aes(x='order_date', y='total_price'), sales_by_month_df) +\
    geom_line()+\
    geom_smooth(method = 'loess',
                se = False,
                color = "blue",
                span = 0.3
                )+\
    scale_y_continuous(labels=usd)+\
    labs(
        title="Revenue by Month",
        x="",
        y="Revenue"
    )+\
    theme_minimal()+\
    expand_limits(y=0)

# 6.2 Sales by Year and Category 2 --
    
# Step 1 - Manipulate ---

sales_by_month_cat_2 = df[['Category_2','order_date','total_price']]\
    .set_index('order_date')\
    .groupby('Category_2')\
    .resample('W')\
    .agg(func={'total_price':np.sum})\
    .reset_index()

#step 2 - Visualize ----

# simple Plot

sales_by_month_cat_2\
    .pivot(
        index = 'order_date',
        columns = 'Category_2',
        values = 'total_price'
    )\
    .fillna(0)\
    .plot(kind="line", subplots=True, layout = (3,3))

plt.show()


# Reporting Plot

type(sales_by_month_cat_2.order_date)
sales_by_month_cat_2.dtypes

ggplot(
    mapping = aes(x = 'order_date', y ='total_price'),
    data = sales_by_month_cat_2
    )+ \
        geom_line(color = "#2C3E50") +\
        geom_smooth(method = "lm", se = False, color = "blue")+ \
        facet_wrap(
            facets = "Category_2",
            ncol = 3,
            scales="free_y"
        ) +\
        scale_y_continuous(labels= usd)+\
        scale_x_datetime( 
            breaks=date_breaks("2 years"),
            labels=date_format(fmt = "%Y-%m")
            )+\
        labs(
            title= "Revenue By Week",
            x = "",
            y = "Revenue"
        )+\
        theme_matplotlib()+\
        theme(
            subplots_adjust={'wspace':0.25},
            axis_text_y=element_text(size = 6),
            axis_text_x=element_text(size=6)
        )
       








