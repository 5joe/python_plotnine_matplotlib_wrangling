import pandas as pd
import sqlalchemy as sql

import os

engine = sql.create_engine("sqlite:///00_database/bike_order_database.sqlite")
conn = engine.connect()

# Read in the excel files and populate the database
bikes_df = pd.read_excel("./00_data_raw/bikes.xlsx")

bikeshops_df = pd.read_excel("./00_data_raw/bikeshops.xlsx")

orderlines_df = pd.read_excel("./00_data_raw/orderlines.xlsx")

# Create Tables
##here we are adding the excel info to the database as tables
bikes_df.to_sql("bikes", con=conn)
pd.read_sql("SELECT * FROM bikes", con = conn)

bikeshops_df.to_sql("bikeshops", con =conn)
pd.read_sql("SELECT * FROM bikeshops", con = conn)

# here i drop the unnamed: 0 column before I read the dataset in the database
orderlines_df\
    .iloc[:, 1:]\
    .to_sql("orderlines", con = conn, if_exists="replace")

pd.read_sql("SELECT * FROM orderlines", con =conn)
pd.read_sql("SELECT name FROM sqlite_master WHERE type='table';", con=conn)
#close the connection
##always close the database--good practice
conn.close()

#Reconnection to the databased
##connecting is the same as creating
engine = sql.create_engine("sqlite:///00_database/bike_order_database.sqlite")
conn = engine.connect()

# get the table names
inspector = sql.inspect(conn)

inspector.get_schema_names()

inspector.get_table_names()


#here let us read the data from the tables
table = inspector.get_table_names()
pd.read_sql(f"SELECT * FROM {table[0]}", con =conn)

#close the connection
conn.close()



