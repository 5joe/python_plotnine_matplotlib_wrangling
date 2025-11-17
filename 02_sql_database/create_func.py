# Here I am going to create some functions that I can use for data collection

import sqlalchemy as sql, inspect
import pandas as pd



def my_function(a = 1):
    b = 2
    return(a + b)

my_function()


# the main event for functions
def collect_data(
    conn_string = "sqlite:///00_database/bike_orders_database.sqlite"
):
    """
    Collects and combines the bike orders data.

    Args:
        conn_string (str, optional): A sqlqclachemy connection string to find the database. 
        Defaults to "sqlite:///00_database/bike_orders_database.sqlite".

    Returns:
        Dataframe: A pandas dataframe the combines data from tables:
            -orderlines: Transactions data
            -bikes: Products data
            -bikeshops: Customers data
    """
    #Body
    ##1.0 connect the database
    engine = sql.create_engine(conn_string)
    
    conn = engine.connect()
    
  
    table_names = ['bikes', 'orderlines', 'bikeshops']
    data_dict = {}
    for table in table_names:
        data_dict[table] = pd.read_sql(f"SELECT * FROM {table}", con  = conn)\
            .drop("index", axis=1)
    #data_dict.values[0]
    
    conn.close()
    
    ##2.0 Combine and clean the data
    joined_df = pd.DataFrame(data_dict['orderlines'])\
        .merge(
            right=data_dict['bikes'],
            how='left',
            left_on='product.id',
            right_on='bike.id'
        )\
        .merge(
            right=data_dict['bikeshops'],
            how='left',
            left_on='customer.id',
            right_on='bikeshop.id'
        )
        
    #3.0 cleaning the data(wrangled)
    
    df = joined_df
    df['order.date'] = pd.to_datetime(df['order.date'])
    
    temp_df = df['description'].str.split(" - ", expand=True)
    df['category.1'] = temp_df[0]
    df['category.2'] = temp_df[1]
    df['frame.material'] = temp_df[2]
    
    temp_df = df['location'].str.split(", ", expand = True)
    df['city'] = temp_df[0]
    df['state'] = temp_df[1]
    
    df['total.price'] = df['quantity'] * df['price']
    
    df.columns
    
    cols_to_keep_list = [
        'order.id', 'order.line', 'order.date', 
        'quantity', 'price','total.price', 'bikeshop.name',
        'model','category.1', 'category.2', 'frame.material', 
        'city', 'state'
    ]
    
    df = df[cols_to_keep_list]
    
    df.columns = df.columns.str.replace(".", "_")
    
    df.info()
    
    return df

collect_data()





