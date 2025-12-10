#  IMPORTING LIBRARIES

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from datetime import timedelta
from prophet import Prophet  # For forecasting
from sklearn.metrics import mean_squared_error, mean_absolute_error

plt.style.use("seaborn-v0_8-darkgrid")   # to make the graph to look cleaner and readable
plt.rcParams["figure.figsize"] = (12,6)  # plots the clearer visualization of the plots

# COVERTING THE EXCEL TO CSV FILE AND LOADING THE DATASET

excel_file = "/content/Daily_Public_Transport_Passenger_Journeys_by_Service_Type_20250603.xlsx"
csv_file = "/content/transport_dataset.csv"

excel_df = pd.read_excel(excel_file)
excel_df.to_csv(csv_file, index=False)

df = pd.read_csv(csv_file)

print("Excel file converted to CSV and loaded successfully.\n")
display(df.head())

# DETECTING DATE AND COLUMN

date_col = None
for col in df.columns:
    if "date" in col.lower():
        date_col = col
        break
if not date_col:
    raise ValueError("No date column found! Please check dataset.")

df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
df = df.dropna(subset=[date_col]).sort_values(date_col)

# DETECTING NUMERIC VALUES AND CLEAN THEM

num_cols = df.select_dtypes(include="number").columns

df[num_cols] = df[num_cols].fillna(method="ffill").fillna(method="bfill")  # CLEANING THE MISSING VALUES

# FEATURE ENGINEERING

for col in num_cols:
    df[col + "_DailyGrowth"] = df[col].pct_change().fillna(0) # ADD DAILY GROWTH FOR EACH NUMERIC FEATURE

df["Traffic_Index"] = df[num_cols].sum(axis=1) # COMBINED TRAFFIC INDEX

# MULTI-LINE TREND OF ALL NUMERIC FEATURES

plt.figure(figsize=(14,7))
for col in num_cols:
    plt.plot(df[date_col], df[col], linewidth=2, label=col)

plt.title("Passenger Trends Across Transport Services", fontsize=16, weight="bold")
plt.xlabel("Date")
plt.ylabel("Passenger Count")
plt.legend()
plt.show()

# HEATMAP OF THE DAILY GROWTH

plt.figure(figsize=(10,6))
growth_cols = [c for c in df.columns if "DailyGrowth" in c]
sns.heatmap(df[growth_cols].corr(), annot=True, cmap="crest")

plt.title("Correlation Between Daily Growth Rates", fontsize=14, weight="bold")
plt.show()

# TRAFFIC INDEX OVER TIME

plt.figure(figsize=(13,6))
plt.plot(df[date_col], df["Traffic_Index"], linewidth=3)

plt.title("Overall Public Transport Traffic Index Over Time", fontsize=16, weight="bold")
plt.xlabel("Date")
plt.ylabel("Traffic Index")
plt.show()

# DISTRIBUTION OF EACH SERVICE TYPE

plt.figure(figsize=(14,7))
for col in num_cols:
    sns.kdeplot(df[col], linewidth=2, label=col)

plt.title("Distribution of Passenger Counts Across Services", fontsize=16, weight="bold")
plt.xlabel("Passenger Count")
plt.legend()
plt.show()

# FORECAST FOR NEXT 7 DAYS

os.makedirs("../images", exist_ok=True)
forecast_results = pd.DataFrame()

services_to_forecast = ["Local Route", "Light Rail", "Peak Service", "Rapid Route", "School"]

for service in services_to_forecast:
    if service not in df.columns:
        continue  # Skip if column not in dataset

    # Prepare data for Prophet
    prophet_df = df[[date_col, service]].rename(columns={date_col: 'ds', service: 'y'})
    
    model = Prophet(daily_seasonality=True)
    model.fit(prophet_df)
    
    future = model.make_future_dataframe(periods=7)  # next 7 days
    forecast = model.predict(future)
    
    # Save forecast values
    forecast_7days = forecast[['ds', 'yhat']].tail(7).rename(columns={'yhat': service})
    if forecast_results.empty:
        forecast_results = forecast_7days
    else:
        forecast_results = forecast_results.merge(forecast_7days, on='ds')
    
    # Plot forecast
    fig = model.plot(forecast)
    plt.title(f"7-Day Forecast: {service}", fontsize=14, weight="bold")
    plt.savefig(f"../images/forecast_{service.replace(' ','_')}.png")
    plt.show()

# Save all forecasts to CSV
forecast_results.to_csv("../forecasts_next7.csv", index=False)
print("7-day forecasts saved to 'forecasts_next7.csv'.")
display(forecast_results)
