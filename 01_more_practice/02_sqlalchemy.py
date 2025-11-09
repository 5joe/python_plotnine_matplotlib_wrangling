# imports ---
import pandas as pd
import sqlalchemy as sql


import os

os.mkdir("./00_prac_database")

# CREATINF DATABASE FROM EXISTING FILES
# instatiate a database

engine = sql.create_engine("sqlite:///00_prac_database/this_database.sqlite")

conn = engine.connect()

# Read Excel Files

bikes_df        =   pd.read_excel("./00_data_raw/bikes.xlsx")
bikeshops_df    =   pd.read_excel("./00_data_raw/bikeshops.xlsx")
orderlines_df   =   pd.read_excel("./00_data_raw/orderlines.xlsx")

# create tables --- that is inside the database
bikes_df.to_sql("bikes", con=conn)
pd.read_sql("SELECT * FROM bikes", con=conn)

bikeshops_df.to_sql("bikeshops", con=conn)
pd.read_sql("SELECT * FROM bikeshops", con=conn)

orderlines_df\
    .iloc[: , 1:]\
    .to_sql("orderlines", con=conn, if_exists="replace")

pd.read_sql("SELECT * FROM orderlines", con=conn)

conn.close()


# Let us try to reconnect to the database that we have just closed.

engine = sql.create_engine("sqlite:///00_prac_database/this_database.sqlite")

conn = engine.connect()

# Getting Data from the Database
## Get the table names

inspector = sql.inspect(conn)

inspector.get_schema_names()

table = inspector.get_table_names()
pd.read_sql(f"SELECT * FROM {table[0]}", con=conn)
pd.read_sql(f"SELECT * FROM {table[1]}", con=conn)
pd.read_sql(f"SELECT * FROM {table[2]}", con=conn)

conn.close()
















