# DS4B 101-P: PYTHON FOR DATA SCIENCE AUTOMATION ----
# Module 3 (Pandas Core): Data Structures ----

# IMPORTS ----

import pandas as pd
import numpy as np

from my_pandas_extensions.database import collect_data

df= collect_data()

df



df['frame_material'].str.lower()# make the frame_material items lower case

df['frame_material'].str.upper()# make the frame_material items upper case

# 1.0 HOW PYTHON WORKS - OBJECTS

# Objects

type(df)

# Objects have classes

type(df).mro()

type("string").mro()

# Objects have attributes

df.shape # can return some information on the object

df.columns

# Objects have methods

df.query("model == 'Jekyll Carbon 2'")

df.query("model == 'Jekyll Carbon 2'")

df.query("frame_material == 'Carbon'")

df.query("frame_material == 'Aluminum'")

df['order_date'].values
df['frame_material'].value_counts()

# 2.0 KEY DATA STRUCTURES FOR ANALYSIS (The three below are the Data Structure Hierarchy)

# - PANDAS DATA FRAME

type(df) # pandas.core.frame.DataFrame


# - PANDAS SERIES

type(df['order_date']) #pandas.core.series.Series

df['order_date'].dt.year

df.dt

# - NUMPY

type(df['order_date'].values).mro()

df['order_date'].values.dtype
df['order_date'].value_counts()
df['order_date'].values.dtype

df['order_date'].dtype

# Data Types

df['price'].values.dtype 


# 3.0 DATA STRUCTURES - PYTHON

# Dictionaries

d = {'a':1}

type(d)

d.keys()

d.values()

d['a']

# Lists

l = [1, "A", [2, "B"]]

l[0]
l[1]
l[2]

m = [1, 4, 6, [3, "really"], 9]

m[0]
m[4]
m[3]


list(d.values())[0] # need a deeper understanding of lists

# Tuples

type(df.shape).mro()

type(df.shape).mro()

t = (10, 20)

t[0]

# over writing this usually does work

t[0] = 30 # this is instead causes and error to emerge 


# Base Data Types

1.5
type(1.5).mro()

1
type(1)

"here"
type("here")


df.total_price.dtype
df.total_price.values

type(df['model'].values[0])


#Let me also try casting

mode ="Jekyll Carbon 2"
price =  10101

f"The model is {mode} and the price {price}"


f"The price for the Jekyll Carbon 2 is {price}"

type(price)

# Casting

model = " Jekyll Carbon 2"
price = 6070

f"The first model is {model} " # allows us to place a string of text in

f"The price of the first model {price}" # an automatic datatype conversion is done and then the 6070 is added

type(price)

price + "Some Text"# this requires that the price is first converted to a string then its combined to the text.

str(price) + "Some text" # in this concept this is called casting

type(range(1, 50)).mro()


# casting to a list(from Low level objects to high level objects)
list(range(1, 51))


r =list(range(1, 51))

np.array(r)

pd.Series(r).to_frame()


# converting column data types

df['order_date'].astype('str').str.replace("-", '/')


