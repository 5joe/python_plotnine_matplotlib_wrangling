#imports 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from my_pandas_extensions.database import collect_data

df = collect_data()

df


# 1.0 select the columns
df[['order_date', 'order_id', 'order_line']]

df.info() # here we can look at all the columns that are in the data frame

df_selected = df[['order_id', 'order_date', 'model', 'category_1', 'category_2', 'price', 'total_price']]
df_selected.info()
df_selected

type(df_selected) # here shows that this a dataframe

df_selected.dtypes # here this shows the different types for the columns that are in the data frame


type(df[['order_date', 'order_id', 'order_line']]) # here this shows that it is a dataframe

#Below now I am going to work with iloc[]
##iloc[]
## Select by position
df.iloc[:,0:3]
df.iloc[:,-3:]
df.iloc[:,3:]
df.iloc[:,5:]
df.iloc[:,5]


# select by position
##lets use the iloc[]
df.iloc[:,0:3]

df.iloc[:, -3:]
 

##select by text matching
df.filter(regex="(^model)|(^cat)", axis=1)

df.filter(regex="(^model)|(^cat)", axis=1)

#here lets do it using the Rowise version
#this hasnt done what I expected it to do
df.filter(regex="(^Jekyll)|(^CAAD)", axis=0)

###VERY IMPORTANT DETAIL RIGHT HERE
###axis=1 → filter columns
###axis=0 → filter rows

# Rearranging columns
df.columns
#----single approach-----
l = df.columns.to_list()

#---single approach---
l = df.columns.tolist()

l.remove('model')

l

#therefore this can be used to just reorder the columns that are in the data Frame
['model', *l]
df[['model', *l]]

#----Here let us redo the multiple method-----#
l = df.columns.tolist()

#here below we remove both the category_1 and category_2 columns
l.remove('category_1')
l.remove('category_2')

#---here lets use the names and the list to reorder---
df_ordered_by_removal = df[['model','category_1','category_2',*l]]
df_ordered_by_removal

cols_to_the_front = ['model','category_1','category_2']

#here we are checking throught the list l column by column
#the list will be reproduced
[col for col in l]

#-------further more ----------#
# here we are checking to ensure that there isn't a column that is being repeated
l3 = [col for col in l if col not in cols_to_the_front]

df[[*cols_to_the_front, *l3]]

df_ordered_by_removal.columns # if you run this you can see that it takes the new arrangement that we just created

#---the multiple approach -----------
l = df.columns.tolist()
l.remove('category_1')
l.remove('category_2')

df[['model','category_1','category_2', *l]]

# --List Comprehensive
l = df.columns.tolist()
l

cols_to_front = ['model', 'category_1', 'category_2']

#here we are interating through the list in l Column for column
##therefore the list is reproduced here
[col for col in l]



#--further more----
l2 = [col for col in l if col not in cols_to_front]

##note that the approach below is actually prett great
df[[*cols_to_front, *l2]]

#selecting by type
df.info()

#--here we can use include or exclude, so it will include that datatype----
df.select_dtypes(include=object)
test = df.select_dtypes(include=object)

test.info() # you observer that it grabs all the columns that are the object dtype()

df.info()

#here lets select the columns that are of the dtype object only
df1 = df.select_dtypes(include = object)
df1
df1.info()

#here let us do the same things without the objects dtypes
df2 = df.select_dtypes(exclude = object)
df2
df2.info()

#here we can do for only columns with the integer dtype though I see that we shall have to include ''
df8 = df.select_dtypes(include = 'Int64')
df8
df8.info()

#lets us do the same thing for the datetime dtype
df9 = df.select_dtypes(include = 'datetime64[ns]')
df9
df9.info()

#combines multiple dataframes that are contained in a list
pd.concat([df1, df2], axis = 1)

df3 = df[['model','category_1','category_2']]

df4 = df.drop(['model','category_1','category_2'], axis = 1)

pd.concat([df3, df4], axis = 1)

#dropping columns (deselecting)
df10 = df.copy()
df.drop(['model','category_1','category_2'], axis = 1)
df10.drop(['model', 'category_1','category_2'], axis = 1)




## Here we are handling the arranging of ROWS
# 2.0 ARRANGING ROWS
df.sort_values('total_price', ascending=False)

df.sort_values('order_date', ascending=False)

# a more direct approach
df['price'].sort_values(ascending=False)

df['price'].sort_values()

#3.0 FILTERING ------
# simple filters(rowwise filtering in this case)

df.order_date>= pd.to_datetime("2015-01-01")

#the above can be used to subset
#therefore here we have used the filter to subset the dataframe
df[df.order_date>= pd.to_datetime("2015-01-01")]



df.model == "Trigger Carbon 1"
df[df.model == "Trigger Carbon 1"]

#filtering for anything that starts with trigger
df.model.str.startswith('Trigger')

#then use that to subset
df[df.model.str.startswith('Trigger')]

#we can also use contains
df.model.str.contains('Carbon')

#then we can use that to subset
df[df.model.str.contains('Carbon')]

#QUERY METHOD
price_threshold = 5000

df.query("price >= @price_threshold")

#we can combine queries
price_threshold_1 = 5000
price_threshold_2 = 1000

df.query("(price >= @price_threshold_1) | (price >= @price_threshold_2)")

df.query(f"price >= {price_threshold_1}")

# filtering in a list
df['category_2'].unique()

df['category_2'].value_counts()

df['category_2'].isin(['Triathalon','Over Mountain'])
df[df['category_2'].isin(['Triathalon','Over Mountain'])]

#I can do this with the negation(~) to get the opposite
df[~df['category_2'].isin(['Triathalon','Over Mountain'])]

#Slicing

#remember when doing the iloc[] the first part is for
##the rows and the second for the columns
###they are divided by a comma(,)

df[:5] # the first 5 rows
# above is the same as
df.head(5)

df.tail(5)

#index slicing
df.iloc[0:5, [1,3,5]]

df.iloc[0:5, :]
df.iloc[:,[1,3,5]]


#unique/ Distinct values
##we use .drop_duplicates()
df[['model','category_1','category_2','frame_material']]\
    .drop_duplicates()

df['model'].unique()

# Top / Bottom
# getting the nlargest from a dataframe
df.nlargest(n = 20, columns = 'total_price')

df['total_price'].nlargest(n=20)

# for the smallest orders
df.nsmallest(n=15, columns='total_price')

df.total_price.nsmallest(n=15)

# Sampling rows()
##if the random_state is set to 123.
###it will always return the same 10 rows
df.sample(n = 10, random_state=123)

df.sample(frac = 0.10, random_state=123)

#WE ARE ADDING CALCULATED COLUMNS(MUTATING)

df2 = df.copy()

df2['new_col'] = df2['price'] * df2['quantity']
df2
df2['new_col_2'] = df2['model'].str.lower()
df2

#METHOD 2- ASSIGN
##the result here is the frame_material column but the values are in Lower case
df['frame_material'].str.lower()

#---in the case below x is the dataframe---

###here the result is the whole dataframe with all the columns and the frame_material column having lower case
df.assign(frame_material = lambda x: x['frame_material'].str.lower())

####in this case it does the same thing as above but then the inintial column isnt overwritten
df.assign(frame_material_lower = lambda x: x['frame_material'].str.lower())

#small illustration
##here we have an interesting use of the assign() function.
###here we are normalizing the illustration
df[['model','price']]\
    .drop_duplicates()\
    .assign(price = lambda x: np.log(x['price']))\
    .set_index('model')\
    .plot(kind = 'hist')

# Adding flags/ we can make Booleans using the assign function
##False/ True

"Supersix Evo Hi-Mod Team".lower().find("supersix") >=0

"Supersix Evo Black Inc.".lower().find("supersix") >=0

"Jekyll Carbon 4".lower().find("supersix") >= 0

#the method below doesnt work coz its a string
##it works better for SeriesS
###"Jekyll Carbon 4".lower().contains("supersix")

type("Jekyll Carbon 4")

df.model.str.lower().str.contains('Carbon')

##in some cases we are adding the str twice there be very linient 
df.assign(flag_supersix= lambda x: x['model'].str.lower().str.contains('supersix'))

#Binning

pd.cut(df.price, bins = 3, labels = ['low','medium','high'])

type(pd.cut(df.price, bins = 3, labels = ['low','medium','high']))

#here converted to a str
pd.cut(df.price, bins = 3, labels = ['low','medium','high']).astype("str")
## here I have used the assign() function 
df.assign(price_bins = lambda x: pd.cut(x['price'], bins = 3, labels = ['low','medium','high']))

df[['model','price']]\
    .drop_duplicates()\
    .assign(price_group = lambda x: pd.cut(x['price'], bins = 3))\
    .pivot(
        index = 'model',
        columns = 'price_group',
        values = 'price'
    )\
    .style.background_gradient(cmap='Blues')

# qcut() - another cutting function
##this cuts numeric columns into quantiles
###this can help with the much more precise quantiles
pd.qcut(df.price, q=[0, 0.33, 0.66, 1], labels = ['low','medium','high'])

df[['model','price']]\
    .drop_duplicates()\
    .assign(price_group = lambda x: pd.qcut(x['price'], q = 3))\
    .pivot(
        index = 'model',
        columns = 'price_group',
        values = 'price'
    )\
    .style.background_gradient(cmap='Blues')


#5.0 GROUPING
##5.1 Aggreagations (No Grouping)
df.sum()
#this initailly returns a panda core Series
type(df[['total_price']].sum())

#here we are converting this to a dataframe
df[['total_price']].sum().to_frame()

df\
    .select_dtypes(exclude=['object'])\
    .drop('order_date', axis = 1)\
    .sum()

df\
    .select_dtypes(exclude=['object'])\
    .drop('order_date', axis = 1)\
    .agg(np.sum)


#here make sure that you have checked the data type that is going to be summed
df\
    .select_dtypes(np.number)\
    .agg([np.sum,np.mean,np.std])

df.agg(
    {
        'quantity':np.sum,
        'total_price':[np.sum, np.mean]
    }
)

# Comman summaries
df['model'].value_counts()
df[['model','category_1']].value_counts()

# the number of unique for the columns
df.nunique()

#this checks if there are any missing data per column
## its then returns the total number of missing data
###if none then 0 is returned
df.isna().sum()

df\
    .select_dtypes(np.number)\
    .std()


#I hope that you have realised that whenever we are 
##passing more that 1 function we are placing the functions
###into []
df\
    .select_dtypes(np.number)\
    .aggregate([np.mean,np.std])

#5.2 Groupby  and Aggregate
###a very important part is numeric_only = True in the sum() function
df.info()

df.groupby(['city']).sum(numeric_only=True)

df.groupby(['city','state']).sum(numeric_only=True)

df\
    .groupby(['city','state'])\
    .agg(
        dict(
            total_price = np.sum,
            quantity = np.sum,
            price = [np.sum, np.mean, np.std]
            )
        )


# IMPORTANT NOTE THAT THESE TWO ARE THE SAME
##dict(total_price = np.sum) and {'total_price': np.sum}

# Get the sum and the median by groups

summary_df_1 = df[['category_1','category_2','total_price']]\
    .groupby(['category_1','category_2'])\
    .agg([np.sum, np.median])\
    .reset_index()

summary_df_1

# Apply Summary Functions to specific columns

summary_df_2 = df[['category_1','category_2','total_price','quantity']]\
    .groupby(['category_1','category_2'])\
    .agg(
        {
<<<<<<< HEAD
            'quantity': np.sum,
=======
            'quantity':np.sum,
>>>>>>> 54a509f (python data wrangling)
            'total_price':np.sum
        }
    )\
    .reset_index()

summary_df_2

<<<<<<< HEAD
#Detecting NAs 
=======
# Here let us detect some nas
>>>>>>> 54a509f (python data wrangling)
summary_df_1.columns

summary_df_1.isna().sum()


<<<<<<< HEAD
#Groupby + Trasform(APply)
df[['category_2','order_date','total_price','quantity']]\
    .set_index('order_date')\
    .groupby('category_2', as_index = False)\
    .resample("W")\
    .agg(np.sum)\
    .reset_index()

#lets proceed with this
summary_df_3 = df[['category_2','order_date','total_price','quantity']]\
    .set_index('order_date')\
    .groupby('category_2')\
    .resample("W")\
    .agg(np.sum)\
    .reset_index(level = 1)

# there are some issues with the plot below
#### I need to handle here
summary_df_3\
    .drop_duplicates()\
    .set_index('order_date')\
    .groupby('category_2')\
    .apply(lambda x : (x.total_price - x.total_price.mean())/x.total_price.std())\
    .reset_index()\
    .pivot(
        index = 'order_date',
        columns = 'category_2',
        values = 'total_price'
    )\
    .plot()

#5.4 Groupby + Filter (Apply)
#need to get these too
df.tail(3)
summary_df_3\
    .groupby('category_2')\
    .tail(5)

summary_df_3\
    .groupby('category_2')\
    .apply(lambda x: x.iloc[10,20])


# Renaming 
# single index
summary_df_2\
    .rename(columns = dict(category_1 = "Category 1"))

summary_df_2.columns.str.replace("_", " ").str.title()

summary_df_2\
    .rename(columns = lambda x: x.replace("_", " ").title())

#Target specific columns
summary_df_2\
    .rename(columns = {'total_price' : 'Revenue'})

# Multi index
summary_df_1.columns

#List Comprehensions
##A preferred way to make lists using singe-line iteration
["_".join(col).rstrip("_") for col in summary_df_1.columns.tolist()]

"_".join(('category_1', 'median'))

summary_df_1\
    .set_axis(
        ["_".join(col).rstrip("_") for col in summary_df_1.columns.tolist()], 
        axis=1
     )

# RESHAPING (MELT & PIVOT_TABLE)

#Aggregate Revenue by Bikeshop by Category_1

bikeshop_revenue_df = df[['bikeshop_name','category_1','total_price']]\
    .groupby(['bikeshop_name','category_1'])\
    .sum()\
    .reset_index()\
    .sort_values('total_price', ascending = False)\
    .rename(columns = lambda x: x.replace("_", " ").title())

bikeshop_revenue_df
# the wide format
##is very important for:
### 1. plotting with matplotlib backend
#### 2. Illustrating in Tables

# 7.1 Pivot & Melt
##Pivot - note that pivot turns a column values into a row headers
### melt - turns particular row headers in Columns
#Pivot (pivot wider)


#IMPORTANT: MEMORISE HOW TO DO THIS 
bikeshop_revenue_wide_df = bikeshop_revenue_df\
.pivot(
    index =['Bikeshop Name'],
    columns = ['Category 1'],
    values = ['Total Price']
)\
.reset_index()\
.set_axis(
    ['Bikeshop Name', 'Mountain','Road'],
    axis =1
)

#Try to be very analytical while doing this

bikeshop_revenue_wide_df\
    .sort_values("Mountain")\
    .plot(
        x = "Bikeshop Name", 
        y = ["Mountain", "Road"],
        kind =  'barh'
)
# Wide format:
#is great for reports

# not the changes
from mizani.labels import dollar
usd = dollar

usd([1000])[0]

bikeshop_revenue_wide_df\
    .sort_values("Mountain", ascending = False)\
    .style\
    .highlight_max()\
    .format(
        {
            "Mountain": lambda x: usd([x])[0],
            "Road" : lambda x: usd([x])[0]
        }
    )\
    .to_excel("03_pandas_core/bikeshop_revenue_wide.xlsx")


# Melt (Pivoting long)
## melt() this does the oppposite of pivot

bikeshop_revenue_long_df = pd.read_excel("03_pandas_core/bikeshop_revenue_wide.xlsx")\
    .iloc[:, 1:]\
    .melt(
        value_vars=["Mountain","Road"],
        var_name = "Category 1",
        value_name = "Revenue",
        id_vars = "Bikeshop Name"
    )

bikeshop_order = bikeshop_revenue_long_df\
    .groupby("Bikeshop Name")\
    .sum()\
    .sort_values("Revenue")\
    .index\
    .tolist()

from plotnine import (
    ggplot, aes, geom_col, facet_wrap,
    coord_flip,
    theme_minimal
)

# Categorical Data Type
##we use this when we want to sort text data.
##Categorical data Combines a label (text)
###and a numeric value (numeric order)


bikeshop_revenue_long_df["Bikeshop Name"] = pd.Categorical(
    bikeshop_revenue_long_df['Bikeshop Name'],
    categories=bikeshop_order
    )


bikeshop_revenue_long_df.info()



ggplot(
    mapping = aes(x = "Bikeshop Name", y = "Revenue", fill = "Category 1"),
    data = bikeshop_revenue_long_df
    ) +\
    geom_col()+\
    coord_flip()+\
    facet_wrap("Category 1")+\
    theme_minimal()

#7.2 Pivot Table (Pivot + Summarization, Excel Pivot Table)

df\
    .pivot_table(
        columns = None,
        values = "total_price",
        index = "category_1",
        aggfunc= np.sum
    )

df\
    .pivot_table(
        columns = "frame_material",
        values = "total_price",
        index = "category_1",
        aggfunc= np.sum
    )

df\
    .pivot_table(
        columns = None,
        values = "total_price",
        index = ["category_1", "frame_material"],
        aggfunc= np.sum
    )

df.info()


# Note that this worked after I set the values
##otherwise it will try to sum columns that arent numeric
###which inturn case and error
df\
    .assign(year = lambda x: x.order_date.dt.year)\
    .pivot_table(
        index = "year",
        aggfunc= np.sum,
        columns = ["category_1","category_2"],
        values = ['total_price']
    )


#To invert the above
sales_by_cat1_cat2_year = df\
    .assign(year = lambda x: x.order_date.dt.year)\
    .pivot_table(
        columns = "year",
        aggfunc= np.sum,
        index = ["category_1","category_2"],
        values = ['total_price']
    )



# 7.3 stack and unstack
# unstack - Pivots Wider 1 level (Pivot)

# note the unstack without any values will pivot the inner
## most level
sales_by_cat1_cat2_year\
    .unstack()

#level determines which level to pivot

sales_by_cat1_cat2_year\
    .unstack(
        level= "category_2",
        fill_value = 0
    )

sales_by_cat1_cat2_year\
    .stack()

sales_by_cat1_cat2_year\
    .stack(
        level = "year"
    )


#this the equivalent of transposing the dataframe
sales_by_cat1_cat2_year\
    .stack(
        level = "year"
    )\
    .unstack(
        level = ["category_1","category_2"]
    )

# 8.0 JOINING DATA ----

orderlines_df = pd.read_excel("00_data_raw/orderlines.xlsx")
bikes_df = pd.read_excel("00_data_raw/bikes.xlsx")

#Merge (joining)
pd.merge(
    left = orderlines_df,
    right = bikes_df,
    left_on = "product.id",
    right_on="bike.id"
)

# Concatenate (Binding)
# Rows
df_1 = df.head(5)
df_2 = df.tail(5)

pd.concat([df_1, df_2], axis = 0)

# Columns 
df_1 = df.iloc[:,5:]
df_2 = df.iloc[:,-5:]

pd.concat([df_1, df_2], axis = 1)

# 9.0 SPLITTING 

#separate

df_2 = df['order_date'].astype('str').str.split("-", expand = True)\
    .set_axis(["year", "month", "day"], axis = 1)

df_2

pd.concat([df, df_2], axis =1)

# Combine
df_2

df_2['year'] + "-" + df_2['month'] + "-" + df_2['day']

# 10.0 APPLY
# - Apply functions across rows

sales_cat2_daily_df = df[['category_2','order_date','total_price']]\
    .set_index('order_date')\
    .groupby('category_2')\
    .resample('D')\
    .sum()

sales_cat2_daily_df

np.mean([1,2,3]) # Aggregation

np.sqrt([1,2,3]) # Transformation

# Here we had to be specific about the numeric columns being used
sales_cat2_daily_df.select_dtypes(include='number').apply(np.mean)

sales_cat2_daily_df.select_dtypes(include='number').apply(np.sqrt)

sales_cat2_daily_df.select_dtypes(include = 'number').apply(np.mean, result_type= "broadcast")
sales_cat2_daily_df.select_dtypes(include = 'number').apply(lambda x: np.repeat(np.mean(x), len(x)))

sales_cat2_daily_df\
    .groupby('category_2')\
    .apply(np.mean)

# Alternate to apply (Transform)
sales_cat2_daily_df\
    .groupby('category_2')\
    .transform(np.mean)


# 11.0 PIPE
# - Functional programming helper for "data" functions

data = df

def add_column(data, **kwargs):

    data_copy = data.copy()

    # print(kwargs)

    data_copy[list(kwargs.keys())] = pd.DataFrame(kwargs)
    
    return data_copy

add_column(df, total_price_2 = df.total_price * 2)

df\
    .pipe(
        add_column, category_2_lower = df.category_2.str.lower(),
        category_2_upper = df.category_2.str.upper()
        ) 

