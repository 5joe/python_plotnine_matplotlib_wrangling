
import pandas as pd

#1.0 files importation ---
pickle_df = pd.read_pickle("./00_data_wrangled/bike_orderlines_wrangled_df.pkl")
pickle_df.info()
pickle_df

# Pickle files are pretty fast at loading data therefore look to use more of them than csv or excel

csv_df = pd.read_csv("./00_data_wrangled/bike_orderlines_wrangled_df.csv")

csv_df


excel_df = pd.read_excel("./00_data_wrangled/bike_orderlines_wrangled_df.xlsx")







