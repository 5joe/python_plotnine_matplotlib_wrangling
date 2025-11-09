# DS4B 101-P: PYTHON FOR DATA SCIENCE AUTOMATION ----
# Module 4 (Time Series): Profiling Data ----


# IMPORTS

import pandas as pd
import numpy as np

from ydata_profiling import ProfileReport

from my_pandas_extensions.database import collect_data

df = collect_data()
df

# PANDAS PROFILING

# Get a Profile

profile = ProfileReport(
    df = df
)
profile

# Sampling - Big Datasets

df.profile_report()

## we can do the same thing without having to print the whole report
df.sample(frac=0.5).profile_report()


# Pandas Helper
# ?pd.DataFrame.profile_report

df.profile_report().to_file("04_time_series/profile_report.html")

# Saving Output


# VSCode Extension - Browser Preview



