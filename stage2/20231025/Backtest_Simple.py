import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objs as go
from plotly.subplots import make_subplots
# import matplotlib.cm as cm
# from datetime import date, timedelta

path = 'D:\\AI_data_analysis\\CY\\'
price_return_backtest = pd.read_csv(path + 'price_return_backtest.csv',
                                parse_dates=['trade_date'],
                                index_col='trade_date')
HS_300_backtest = pd.read_csv(path + 'HS_300_backtest.csv',
                          parse_dates=["trade_date"],
                          index_col="trade_date")
HS_300_backtest = HS_300_backtest['000300.SH']

CS_500_backtest = pd.read_csv(path + 'CS_500_backtest.csv',
                          parse_dates=["trade_date"],
                          index_col="trade_date")
CS_500_backtest = CS_500_backtest['000905.SH']

CS_1000_backtest = pd.read_csv(path + 'CS_1000_backtest.csv',
                           parse_dates=["trade_date"],
                           index_col="trade_date")
CS_1000_backtest = CS_1000_backtest['000852.SH']

def Ensure_position(DF):
    # We add a column named 'Monday' and change positions every Monday.
    DF = DF.shift(1)  # Move one step forward to ensure position
    DF["Monday"] = (DF.index.dayofweek == 0).astype(int)  # Shift except Monday

    # Get the columns to shift (all columns except 'Monday')
    cols_to_shift = DF.columns[DF.columns != "Monday"]

    DF.loc[DF["Monday"] == 0, cols_to_shift] = np.nan
    # DF.fillna(
    #     method="ffill", inplace=True
    # )  # Forward fill, holing positions for a week.
    DF.ffill(inplace=True)
    DF.fillna(value=0, inplace=True)  # Fill remaining NaN with 0
    return DF

def Simple_Backtest(factor_df, stock_num, Ascending):
    # Calculte the ranks of factors daily.
    factor_ranks = factor_df.rank(axis=1, ascending=Ascending)

    # Create position_df based on top 3 ranks
    position_df = factor_ranks.apply(lambda x: x <= stock_num).astype(int)
    position_df = Ensure_position(position_df)

    # Delete 'Monday' to fit into yield dataframe.
    del position_df["Monday"]

    # Calculate the sum of each line in turn.
    stock_amount_sum = position_df.sum(axis=1)

    # Calculate the weight of each stock. (Average distribution at the same level)
    weight_allocation = position_df.apply(lambda x: x / stock_amount_sum, axis=0).fillna(0)

    # Calculate the daily profit rate. And prepare to calculate cumprod.
    profit = (weight_allocation * price_return_backtest).sum(axis=1)

    plt.rcParams["axes.unicode_minus"] = False  # 正常显示负号
    plt.figure(figsize=(10, 5))
    plt.axhline(y=1, color="grey", linestyle="--")
    # Calculate the daily equity and draw.
    profit = profit.apply(lambda x: x + 1 if x == 0 else x + (1 - 0.0003))
    profit.cumprod().plot(label="Stocks", legend=True, color="#800080")

    profit_HS300 = HS_300_backtest.apply(lambda x: x + 1 if x == 0 else x + (1 - 0.0003))
    profit_HS300.cumprod().plot(label="HS 300 index", legend=True, color="r")
    profit_CS500 = CS_500_backtest.apply(lambda x: x + 1 if x == 0 else x + (1 - 0.0003))
    profit_CS500.cumprod().plot(label="CS 500 index", legend=True, color="g")
    profit_CS1000 = CS_1000_backtest.apply(lambda x: x + 1 if x == 0 else x + (1 - 0.0003))
    profit_CS1000.cumprod().plot(label="CS 1000 index", legend=True, color="b")

    plt.title(f"Equity of {stock_num} stocks")
    plt.legend(title="Index", bbox_to_anchor=(1, 0.5), loc="center left")
    plt.show()

def Dynamic_Drawback(factor_df, stock_num, Ascending):
    # Calculate the ranks of factors daily.
    factor_ranks = factor_df.rank(axis=1, ascending=Ascending)

    # Create position_df based on top 3 ranks
    position_df = factor_ranks.apply(lambda x: x <= stock_num).astype(int)
    position_df = Ensure_position(position_df)

    # Delete 'Monday' to fit into yield dataframe.
    del position_df["Monday"]

    # Calculate the sum of each line in turn.
    stock_amount_sum = position_df.sum(axis=1)

    # Calculate the weight of each stock. (Average distribution at the same level)
    weight_allocation = position_df.apply(lambda x: x / stock_amount_sum, axis=0).fillna(0)

    plt.rcParams["axes.unicode_minus"] = False  # Display negative sign correctly
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 5), sharex=True)

    # Calculate the daily profit rate. And prepare to calculate cumprod.
    profit = (weight_allocation * price_return_backtest).sum(axis=1)
    profit = profit.apply(lambda x: x + 1 if x == 0 else x + (1 - 0.0003))
    equity = profit.cumprod()
    max_equity = max(equity)

    ax1.axhline(y=1, color="grey", linestyle="--")
    ax1.plot(equity, label="Stocks equity", color="#800080")
    ax1.set_title(f"Equity of {stock_num} stocks")
    ax1.legend(bbox_to_anchor=(1, 0.5), loc="center left")
    # Annotate max value of equity
    ax1.annotate(f'Max: {max_equity:.2f}', xy=(equity.idxmax(), max_equity),
                xytext=(equity.idxmax(), max_equity + max_equity / 5),
                arrowprops=dict(facecolor='black', arrowstyle='->'))

    # Calculate drawdown
    rolling_max = equity.cummax()
    drawdown = (equity - rolling_max) / rolling_max
    max_drawdown = min(drawdown)
    
    ax2.plot(drawdown, label="Drawdown", color="maroon")
    # Shade the area where drawdown is negative
    ax2.fill_between(drawdown.index, 
                     drawdown, 
                     0, 
                     where=drawdown < 0, 
                     color='maroon', 
                     alpha=0.3)
    ax2.set_title("Drawdown Curve")
    ax2.legend(bbox_to_anchor=(1, 0.5), loc="center left")
    # Annotate min value of ax2
    ax2.annotate(f'Min: {max_drawdown:.2f}', xy=(drawdown.idxmin(), max_drawdown),
                xytext=(drawdown.idxmin(), max_drawdown + max_drawdown / 3),
                arrowprops=dict(facecolor='black', arrowstyle='->'))

    ax1.grid(True)
    ax2.grid(True)
    plt.tight_layout()
    plt.show()

def Ensure_Future_position_small(factor_df, stock_num, Ascending):
    last_day = factor_df.iloc[-1, :]

    # Calculte the ranks of factors daily.
    factor_ranks = last_day.rank(ascending=Ascending)

    # Create a boolean mask to identify columns where values are between 0 and stock_num - 1
    mask = (factor_ranks > 0) & (factor_ranks <= stock_num)
    print(factor_ranks[mask].sort_values())
    selected_stocks = list(factor_ranks[mask].sort_values().index)
    return selected_stocks

if __name__ == "__main__":
    Ensemble_factor_df = pd.read_csv(path + 'Ensemble_factor_df.csv',
                                    parse_dates=['trade_date'],
                                    index_col='trade_date')
    Dynamic_Drawback(Ensemble_factor_df, 6, True)