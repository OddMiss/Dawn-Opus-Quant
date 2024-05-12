import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.dates as mdates
from math import sqrt
import numpy as np
import pandas as pd
import warnings

# Suppress the warning
warnings.filterwarnings("ignore", category=pd.errors.SettingWithCopyWarning)

path = "D:\\AI_data_analysis\\CY\\"

def Draw_Industry_Daily(Long_position_df):
    # Long_position_df = pd.read_csv(
    #     path + f"{long_position_df_name}.csv",
    #     parse_dates=["trade_date"],
    #     index_col="trade_date",
    # )
    SWClass_all = pd.read_csv(path + 'SWClass_all.csv')
    Class = SWClass_all[['股票代码', '一级行业名称', '二级行业名称', '三级行业名称']]

    # Create a dictionary mapping stock codes to industries
    industry_first_dict = dict(zip(Class['股票代码'], Class['一级行业名称']))
    industry_second_dict = dict(zip(Class['股票代码'], Class['二级行业名称']))
    industry_third_dict = dict(zip(Class['股票代码'], Class['三级行业名称']))

    # Drop Monday column
    # var_names = Long_position_df_Monday.columns.drop("Monday")

    # Reshape the DataFrame
    var_names = Long_position_df.columns
    position_df_long = Long_position_df.reset_index().melt(
        id_vars="trade_date", value_vars=var_names, var_name='ts_code'
    )
    position_df_long = position_df_long.sort_values(by="trade_date").reset_index(drop=True)

    # Drop rows where 'value' column is 0
    position_df_long = position_df_long[position_df_long['value'] != 0]
    position_df_long.reset_index(drop=True, inplace=True)
    position_df_long = position_df_long.drop(columns=['value'])
    # Add industry columns
    position_df_long['industry_sw_first'] = position_df_long['ts_code'].map(industry_first_dict)
    position_df_long['industry_sw_second'] = position_df_long['ts_code'].map(industry_second_dict)
    position_df_long['industry_sw_third'] = position_df_long['ts_code'].map(industry_third_dict)

    # Count the number of occurrences of each industry each day
    industry_counts = (
        position_df_long.groupby(["trade_date", "industry_sw_first"])
        .size()
        .unstack(fill_value=0)
    )

    # Set the font properties to include Chinese characters
    plt.rcParams["font.sans-serif"] = [
        "SimHei"
    ]  # Specify the font family to use (SimHei is a common Chinese font)
    # Ensure that minus sign is displayed correctly for Chinese characters
    plt.rcParams["axes.unicode_minus"] = False

    # Format the x-axis tick labels to show only year-month-day
    industry_counts.index = industry_counts.index.strftime("%Y-%m-%d")

    # Plot the counts as a bar graph
    industry_counts.plot(kind="bar", stacked=True, figsize=(10, 5))

    # Set the frequency of x-axis tick labels to display every nth label
    n = 8  # Display every n nd label
    # plt.xticks(range(0, len(industry_counts.index), n), industry_counts.index[::n])
    plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=n))

    # Rotate the x-axis tick labels for better visibility
    plt.xticks(rotation=40, ha='right')

    plt.xlabel("Trade Time")
    plt.ylabel("Count")
    plt.title("Number of Industries Each Day")
    plt.legend(title="Industry", bbox_to_anchor=(1, 0.5), loc="center left")
    plt.show()

    return industry_counts