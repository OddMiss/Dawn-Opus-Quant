import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from datetime import date, timedelta

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
    DF = DF.shift(1) # Move one step forward to ensure position
    DF["Monday"] = (DF.index.dayofweek == 0).astype(int) # Shift except Monday

    # Get the columns to shift (all columns except 'Monday')
    cols_to_shift = DF.columns[DF.columns != 'Monday']

    DF.loc[DF['Monday'] == 0, cols_to_shift] = np.nan
    # DF.fillna(method='ffill', inplace=True) # Forward fill, holing positions for a week.
    DF.ffill(inplace=True)
    DF.fillna(value=0, inplace=True) # Fill remaining NaN with 0
    return DF

def Layer_Backtest(factor_df, type):
    # layer number
    num_layers = 20

    # Calculte the ranks of factors daily.
    factor_ranks = factor_df.rank(axis=1, ascending=False)

    # The factor ordering is divided into num_layers, each of which allocates funds equally.
    layer_allocation = (factor_ranks // (len(factor_df.columns) / num_layers)).fillna(0)
    layer_allocation

    # import matplotlib.cm as cm

    plt.rcParams['axes.unicode_minus'] = False # 正常显示负号
    plt.figure(figsize=(10, 5))
    plt.axhline(y=1, color='grey', linestyle='--')

    # Define a color map to use for changing colors progressively
    # colors = plt.cm.jet(np.linspace(0, 1, num_layers))

    global profit_long, profit_short
    profit_long = profit_short = None
    def Long_Short(Num_layers, Layer, Profit):
        global profit_long, profit_short
        long_layer = Num_layers - 1
        short_layer = 0
        if Layer == short_layer:
            profit_short = Profit
            # The short profit comes from the decline of the stock.
            profit_short = profit_short.apply(lambda x: x + 1 if x == 0 else -x + (1 - 0.0003))
            profit_short = profit_short.cumprod()
            profit_short *= 0.5
        elif Layer == long_layer:
            profit_long = Profit
            profit_long = profit_long.apply(lambda x: x + 1 if x == 0 else x + (1 - 0.0003))
            profit_long = profit_long.cumprod()
            profit_long *= 0.5

    Long_position_df = None
    for layer in range(0, num_layers):
        # Ensure holding stocks
        hold_flag_matrix = layer_allocation.mask(layer_allocation != layer, 0).mask(layer_allocation == layer, 1)
        hold_flag_matrix = Ensure_position(hold_flag_matrix)
        if layer == num_layers - 1:
            Long_position_df = hold_flag_matrix.copy()

        # Delete 'Monday' to fit into yield dataframe.
        del hold_flag_matrix["Monday"]

        # Calculate the sum of each line in turn.
        stock_amount_sum = hold_flag_matrix.sum(axis=1)

        # Calculate the weight of each stock. (Average distribution at the same level)
        weight_allocation = hold_flag_matrix.apply(lambda x: x / stock_amount_sum, axis=0).fillna(0)

        # Calculate the daily profit rate. And prepare to calculate cumprod.
        profit = (weight_allocation * price_return_backtest).sum(axis=1)

        # Create Long and Short position
        Long_Short(num_layers, layer, profit)

        # Calculate the daily equity and draw.
        # Using the 'viridis' colormap with a gradient based on layer number
        colors = cm.viridis(layer / num_layers)
        profit = profit.apply(lambda x: x + 1 if x == 0 else x + (1 - 0.0003))
        profit.cumprod().plot(label=layer, legend=True, color=colors)

    profit_HS300 = HS_300_backtest.apply(lambda x: x + 1 if x == 0 else x + (1 - 0.0003))
    profit_HS300.cumprod().plot(label="HS 300 index", legend=True, color='r')
    profit_CS500 = CS_500_backtest.apply(lambda x: x + 1 if x == 0 else x + (1 - 0.0003))
    profit_CS500.cumprod().plot(label="CS 500 index", legend=True, color='g')
    profit_CS1000 = CS_1000_backtest.apply(lambda x: x + 1 if x == 0 else x + (1 - 0.0003))
    profit_CS1000.cumprod().plot(label="CS 1000 index", legend=True, color='b')

    (profit_long + profit_short).plot(color='orange', label='long_short', legend=True)
    plt.title(f"20-Layered Portfolio Equity ({type})")
    plt.legend(title='Layer', bbox_to_anchor=(1, 0.5), loc='center left')
    plt.show()

    return profit_long, Long_position_df

if __name__ == "__main__":
    print(price_return_backtest.index)