# DS4B 101-P: PYTHON FOR BUSINESS ANALYSIS ----
# Module 2 (Pandas Import): Importing Files ----

# IMPORTS ----
# %%
import pandas as pd


# %%

# 1.0 FILES ----
##my_practice part for read_pickle
pkl_df = pd.read_pickle("./00_data_wrangled/bike_orderlines_wrangled_df.pkl")
pkl_df.info()
# - Pickle ---- pretty firs therefore look to use the .pkl

# The original one
pickle_df = pd.read_pickle("./00_data_wrangled/bike_orderlines_wrangled_df.pkl")
pickle_df.info()

# - CSV ----
##my_practice part
jude_df = pd.read_csv("./00_data_wrangled/bike_orderlines_wrangled_df.csv")
jude_df.info()

# The original one
csv_df = pd.read_csv("./00_data_wrangled/bike_orderlines_wrangled_df.csv", parse_dates=['order_date'])

csv_df.info()
# - Excel ----
##my_practice part
edr_df = pd.read_excel("./00_data_wrangled/bike_orderlines_wrangled_df.xlsx")
edr_df.info()

# The original one
excel_df = pd.read_excel("./00_data_wrangled/bike_orderlines_wrangled_df.xlsx")

excel_df.info()



