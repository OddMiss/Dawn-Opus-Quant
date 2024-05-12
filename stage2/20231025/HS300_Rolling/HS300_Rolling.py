# Modeling
import tensorflow as tf
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import xgboost as xgb

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Attention, GRU, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

import numpy as np
import pandas as pd
import warnings
import matplotlib.pyplot as plt
# from math import sqrt
# import matplotlib.cm as cm
# import seaborn as sns
from datetime import date, timedelta
import psutil
import os

# Get the current process ID of the IPython kernel
pid = os.getpid()
# Get the process associated with the IPython kernel
process = psutil.Process(pid)

from tqdm import tqdm  # For status process bar
# from IPython.display import clear_output
# import baostock as bs
import pickle

# Suppress the warning
# warnings.filterwarnings(
#     "ignore", 
#     category=pd.core.common.SettingWithCopyWarning)

zscore = StandardScaler()

# Suppress the warning
warnings.filterwarnings('ignore', category=pd.errors.SettingWithCopyWarning)
Main_bactest = False
Online = False

# Get the directory of the Python script
path = os.path.dirname(__file__) + '\\'
# path_Main = "/home/huh/"
path_HS300_Rolling = path
path_HS300_Rolling_pic = path + "Loss-pic\\"

begin_date = "20180101"
end_date = "20240430"
today_date = str(date.today())
if Main_bactest:
    end_date = today_date

'''
price = None
date_all = None
date_train = None
date_last = None
date_backtest = None
date_test = None
date_GRU = None
index = None

price_X_train = None
price_y_train = None
price_X_test = None
price_y_test = None
price_X_drop = None
price_y_drop = None
price_X_backtest = None
history_MLP= None
model_MLP = None
y_pred_MLP_backtest = None
y_pred_MLP_all = None
bst = None
y_pred_GBDT_backtest = None
y_pred_GBDT_all = None
train_X = None
train_y = None
text_X = None
text_y = None
drop_X = None
drop_y = None
history_AGRU = None
model_AGRU = None
y_pred_AGRU_train = None
y_pred_AGRU_test = None
y_pred_AGRU_backtest = None
y_pred_AGRU_all = None

factor_df = None
factor_all = None
factor_df_backtest = None
ICIR = None
'''

class HS_300_Rolling():
    def __init__(self, Price, Price_Return, Path, Date_All, 
                 Date_Backtest, Stocks_code, Backtest_Date_Index) -> None:
        self.RAM_USAGE()
        self.price = Price.loc[Price["trade_date"].isin(Date_All), :]
        self.price_return = Price_Return.loc[Price_Return.index.isin(Date_All), :]
        self.path = Path
        self.backtest_date_index = Backtest_Date_Index
        self.batch_size = len(Stocks_code)
        self.Date_Spliting(Date_All, Date_Backtest)
        self.Data_Spliting()
        self.MLP()
        self.GBDT()
        self.create_sequences()
        self.AGRU()
        self.Create_factor_df()
        self.Ensemble_ICIR_weight()
        self.Ensemble_ICIR_max(self.MLP_ICIR,
                               self.GBDT_ICIR,
                               self.AGRU_ICIR,
                               type_list=["MLP", "GBDT", "AGRU"])
    def RAM_USAGE(self):
        # Get the memory usage of the IPython kernel in MB
        ram_usage = process.memory_info().rss / (1024 * 1024)
        print(f"RAM Usage: {ram_usage} MB")
    def Date_Spliting(self, Date_All, Date_Backtest):
        self.date_all = Date_All
        self.date_backtest = Date_Backtest
        self.date_backtest_Monday = [Date_Backtest[0]]
        self.date_last = [Date_All[-1]] # The last day of all, deciding the next stage's position.
        # Attention: len(date_all) >= 60
        self.date_train, date_test = train_test_split(Date_All, 
                                                      test_size=0.2, 
                                                      shuffle=False)
        self.date_test = date_test[:-11] # If len(date_test[:-11]) == 1, it will be error.
        self.date_drop = date_test[-11:]
        self.date_GRU = Date_All[29:]
    def Data_Spliting(self):
        print("*" * 60)
        print("Data spliting...")
        price = self.price
        # date_train, date_test = train_test_split(trade_date, test_size=0.2, shuffle=False)
        price_train = price.loc[price["trade_date"].isin(self.date_train), :]
        # date_stock = ["trade_date", "ts_code"]
        X_indexes = [
            "OPEN_processed",
            "HIGH_processed",
            "LOW_processed",
            "CLOSE_processed",
            "VWAP_processed",
            "VOLUME_processed",
        ]
        y_index = "Label_processed"
        self.price_X_train = price_train[X_indexes].values
        self.price_y_train = price_train[y_index].values

        price_test = price.loc[price["trade_date"].isin(self.date_test), :]
        price_X_test = price_test[X_indexes].values
        self.price_X_test = price_X_test
        self.price_y_test = price_test[y_index].values

        price_drop = price.loc[price["trade_date"].isin(self.date_drop), :]
        self.price_X_drop = price_drop[X_indexes].values
        # print(list(self.date_backtest))
        price_last = price.loc[price["trade_date"].isin(self.date_last), :]
        self.price_X_last = price_last[X_indexes].values

        self.price_X_all = price[X_indexes].values
        self.price_y_all = price[y_index].values

        self.RAM_USAGE()
        print("*" * 60)

        # del price_train
        # del price_test
        # del price_drop
    def MLP(self):
        print("*" * 60)
        print("MLP training...")
        # Define the MLP model
        model_MLP = tf.keras.Sequential(
            [
                tf.keras.layers.Dense(128, activation="relu"),
                tf.keras.layers.Dropout(0.05),
                tf.keras.layers.Dense(128, activation="relu"),
                tf.keras.layers.Dropout(0.05),
                tf.keras.layers.Dense(128, activation="relu"),
                tf.keras.layers.Dropout(0.05),
                tf.keras.layers.Dense(1),
            ]
        )

        # Compile the model
        optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
        model_MLP.compile(optimizer=optimizer, loss="mean_squared_error")

        # Train the model
        early_stopping_MLP = tf.keras.callbacks.EarlyStopping(
            monitor="loss", patience=50, restore_best_weights=True)
        history_MLP = model_MLP.fit(
            self.price_X_train,
            self.price_y_train,
            epochs=1000,
            callbacks=[early_stopping_MLP],
            validation_data=(self.price_X_test, self.price_y_test),
            batch_size=len(self.price_X_train),  # Set batch size as the training set size
        )

        # Evaluate the model
        self.y_pred_MLP_last = model_MLP.predict(self.price_X_last)
        self.y_pred_MLP_all = model_MLP.predict(self.price_X_all)
        mse_MLP_train = mean_squared_error(self.price_y_train, model_MLP.predict(self.price_X_train))
        mse_MLP_test = mean_squared_error(self.price_y_test, model_MLP.predict(self.price_X_test))

        print(f"MLP MSE(train): {mse_MLP_train}")
        print(f"MLP MSE(test): {mse_MLP_test}")
        # MLP MSE(train): 0.9924953226659671
        # MLP MSE(test): 0.9949950036401641

        def Draw_Loss():
            # Extract loss values from the history object
            training_loss_MLP = history_MLP.history['loss']
            validation_loss_MLP = history_MLP.history['val_loss']

            # Create a range of epochs for x-axis
            epochs = range(1, len(training_loss_MLP) + 1)

            # Plot the training and validation loss values
            plt.figure()
            plt.plot(epochs, training_loss_MLP, label='Training Loss')
            plt.plot(epochs, validation_loss_MLP, label='Validation Loss')
            plt.title('Training and Validation Loss')
            plt.xlabel('Epochs')
            plt.ylabel('Loss')
            plt.legend()
            plt.savefig(path_HS300_Rolling_pic + f'MLP_{self.backtest_date_index}.png')
            # plt.show()
        Draw_Loss()
        # self.history_MLP = history_MLP
        print("*" * 60)
    def GBDT(self):
        print("*" * 60)
        print("GBDT training...")
        params = {
            "learning_rate": 0.01,
            "max_depth": 64,
            "max_leaves": 512,
            "min_child_weight": 512,
            "subsample": 0.7,
            "colsample_bytree": 0.7,
            "objective": "reg:squarederror",
        }

        dtrain = xgb.DMatrix(self.price_X_train, label=self.price_y_train)
        dtest = xgb.DMatrix(self.price_X_test, label=self.price_y_test)
        dlast = xgb.DMatrix(self.price_X_last)
        dall = xgb.DMatrix(self.price_X_all)

        evals = [(dtest, "eval"), (dtrain, "train")]
        num_round = 200

        bst = xgb.train(
            params, 
            dtrain, 
            num_round, 
            evals, 
            early_stopping_rounds=50, 
            verbose_eval=True
        )

        self.y_pred_GBDT_last = bst.predict(dlast)
        self.y_pred_GBDT_all = bst.predict(dall)
        mse_GBDT_train = mean_squared_error(self.price_y_train, bst.predict(dtrain))
        mse_GBDT_test = mean_squared_error(self.price_y_test, bst.predict(dtest))

        print(f"GBDT MSE(train): {mse_GBDT_train}")
        print(f"GBDT MSE(test): {mse_GBDT_test}")
        # GBDT MSE(train): 0.9476405621114625
        # GBDT MSE(test): 1.002617615102808

        # train_rmse = []
        # eval_rmse = []
        # def Draw_Loss():
        #     plt.plot(train_rmse, label='Training RMSE')
        #     plt.plot(eval_rmse, label='Validation RMSE')
        #     plt.xlabel('epochs')
        #     plt.ylabel('RMSE')
        #     plt.title('Training and Validation RMSE')
        #     plt.legend()
        #     plt.show()
        self.RAM_USAGE()
        self.bst = bst
        print("*" * 60)
    def create_sequences(self, Sequence_length=30):
        print("*" * 60)
        print("Creating sequences...")
        # Create sequences of variable length for each stock
        """
        sequence length: number of time steps in the entire sequence.
        In this paper, sequence length = 30.

        type: 'train' or 'test'

        begin index: index of the first sequence
        end index: index of the last sequence
        """
        def Sequences(df, sequence_length, begin_index, end_index):
            sequences = []
            labels = []
            for stock in df["ts_code"].unique():
                # Single stock dataframe
                stock_df = df[df["ts_code"] == stock].reset_index(drop=True)
                for i in range(sequence_length + begin_index, end_index + 2):
                    # The last sequence is included.
                    seq = stock_df.iloc[i - sequence_length : i][
                        [
                            "OPEN_processed",
                            "HIGH_processed",
                            "LOW_processed",
                            "CLOSE_processed",
                            "VWAP_processed",
                            "VOLUME_processed",
                        ]
                    ].values
                    label = stock_df.iloc[i - 1]["Label_processed"]
                    sequences.append(seq)
                    labels.append(label)
            return np.array(sequences), np.array(labels)
        # Set the desired sequence length
        sequence_length = Sequence_length
        train_X, train_y = Sequences(
            self.price, 
            sequence_length, 
            0, 
            len(self.date_train) - 1
        )
        # Note: len(date_train) + len(date_test) is the length of Modeling date.
        test_X, test_y = Sequences(
            self.price,
            sequence_length,
            len(self.date_train) - sequence_length + 1,
            len(self.date_train) + len(self.date_test) - 1,
        )
        drop_X, _ = Sequences(
            self.price,
            sequence_length,
            len(self.date_train) + len(self.date_test) - sequence_length + 1,
            len(self.date_all) - 1,
        )
        last_X, _ = Sequences(
            self.price,
            sequence_length,
            len(self.date_all) - sequence_length,
            len(self.date_all) - 1,
        )
        self.train_X = train_X
        self.train_y = train_y
        self.test_X = test_X
        self.test_y = test_y
        self.drop_X = drop_X
        self.last_X = last_X
        # self.last_y = last_y
        print("train_X:", train_X.shape)
        print("test_X:", test_X.shape)
        print("drop_X:", drop_X.shape)
        print("last_X:", last_X.shape)
        print("*" * 60)
    def AGRU(self):
        print("*" * 60)
        print("AGRU training...")
        # Define the AGRU model
        inputs = Input(shape=(30, 6))
        gru = GRU(units=6, return_sequences=True)(inputs)
        att = Attention()([gru, gru])
        gru = GRU(units=6)(att)
        outputs = Dense(1)(gru)

        model_AGRU = Model(inputs=inputs, outputs=outputs)

        # Compile the model
        model_AGRU.compile(optimizer=Adam(learning_rate=1e-3), 
                           loss="mean_squared_error")

        # Define early stopping criteria
        early_stopping_AGRU = EarlyStopping(monitor="loss", 
                                            patience=20, 
                                            restore_best_weights=True)

        # Train the model
        history_AGRU = model_AGRU.fit(
            self.train_X,
            self.train_y,
            epochs=200,
            batch_size=self.batch_size,
            validation_data=(self.test_X, self.test_y),
            callbacks=[early_stopping_AGRU],
        )

        # Evaluate the model
        loss = model_AGRU.evaluate(self.test_X, self.test_y)
        print("Model trained with loss:", loss)
        # Evaluate the model on test data
        y_pred_AGRU_train = model_AGRU.predict(self.train_X)
        self.y_pred_AGRU_train = y_pred_AGRU_train
        y_pred_AGRU_test = model_AGRU.predict(self.test_X)
        self.y_pred_AGRU_test = y_pred_AGRU_test
        y_pred_AGRU_drop = model_AGRU.predict(self.drop_X)
        self.y_pred_AGRU_last = model_AGRU.predict(self.last_X)
        # y_pred_AGRU_backtest = np.concatenate(
        #     (y_pred_AGRU_test.reshape(-1), 
        #      y_pred_AGRU_drop.reshape(-1))
        # )
        # self.y_pred_AGRU_backtest = y_pred_AGRU_backtest
        self.y_pred_AGRU_all = np.concatenate((np.concatenate((y_pred_AGRU_train.reshape(-1), 
                                                               y_pred_AGRU_test.reshape(-1))),
                                                               y_pred_AGRU_drop.reshape(-1)))
        mse_AGRU_train = mean_squared_error(self.train_y, y_pred_AGRU_train)
        mse_AGRU_test = mean_squared_error(self.test_y, y_pred_AGRU_test)
        print(f"AGRU MSE(train): {mse_AGRU_train}")
        print(f"AGRU MSE(test): {mse_AGRU_test}")
        # AGRU MSE(train): 0.9896615693578518
        # AGRU MSE(test): 0.9960141348406323

        def Draw_Loss():
            # Extract loss values from the history object
            training_loss_AGRU = history_AGRU.history['loss']
            validation_loss_AGRU = history_AGRU.history['val_loss']

            # Create a range of epochs for x-axis
            epochs = range(1, len(training_loss_AGRU) + 1)

            # Plot the training and validation loss values
            plt.figure()
            plt.plot(epochs, training_loss_AGRU, label='Training Loss')
            plt.plot(epochs, validation_loss_AGRU, label='Validation Loss')
            plt.title('Training and Validation Loss')
            plt.xlabel('Epochs')
            plt.ylabel('Loss')
            plt.legend()
            plt.savefig(path_HS300_Rolling_pic + f'AGRU_{self.backtest_date_index}.png')
            # plt.show()
        Draw_Loss()
        print("*" * 60)
    def Create_factor_df(self):
        print("*" * 60)
        print("Creating factor df...")
        price = self.price
        price_return = self.price_return
        def Create_Factor_all(y_pred_all):
            # This is for y_pred_all.
            factor_df = price[["trade_date", "ts_code"]]  # Date and stock
            factor_df.loc[:, "Factor_values"] = y_pred_all  # Add factor values
            factor_df = factor_df.pivot(
                index="trade_date", columns="ts_code", values="Factor_values"
            )  # Transfer to factor dataframe
            return factor_df

        def Create_Factor_backtest(y_pred_backtest):
            # This is for y_pred_all.
            factor_df = price.loc[price["trade_date"].isin(self.date_last), :][
                ["trade_date", "ts_code"]]  # Date and stock
            factor_df.loc[:, "Factor_values"] = y_pred_backtest  # Add factor values
            factor_df = factor_df.pivot(
                index="trade_date", columns="ts_code", values="Factor_values"
            )  # Transfer to factor dataframe
            return factor_df

        def Create_ICIR_df(factor_DF):
            # This is for y_pred_all.
            IC_series = factor_DF.corrwith(price_return, axis=1, method="spearman")
            ICIR_df = IC_series.shift(1).rolling(60).apply(lambda x: x.mean() / x.std())
            ICIR_df = ICIR_df[
                ICIR_df.index.isin(self.date_last)
            ]  # Select the ICIR in the test date.
            return ICIR_df

        MLP_factor_all = Create_Factor_all(self.y_pred_MLP_all)
        self.MLP_factor_backtest = Create_Factor_backtest(self.y_pred_MLP_last)
        GBDT_factor_all = Create_Factor_all(self.y_pred_GBDT_all)
        self.GBDT_factor_backtest = Create_Factor_backtest(self.y_pred_GBDT_last)

        AGRU_factor_all = price.loc[price["trade_date"].isin(self.date_GRU), :].reset_index(
            drop=True)
        AGRU_factor_all = AGRU_factor_all[["trade_date", "ts_code"]]  # Date and stock
        AGRU_factor_all.loc[:, "AGRU"] = self.y_pred_AGRU_all  # Add factor values
        AGRU_factor_all = AGRU_factor_all.pivot(
            index="trade_date", columns="ts_code", values="AGRU"
        )  # Transfer to factor dataframe
        self.AGRU_factor_backtest = Create_Factor_backtest(self.y_pred_AGRU_last)

        # Use abs()
        MLP_ICIR = Create_ICIR_df(MLP_factor_all)
        self.MLP_ICIR = MLP_ICIR.abs()
        GBDT_ICIR = Create_ICIR_df(GBDT_factor_all)
        self.GBDT_ICIR = GBDT_ICIR.abs()
        AGRU_ICIR = Create_ICIR_df(AGRU_factor_all)
        self.AGRU_ICIR = AGRU_ICIR.abs()
        print("*" * 60)
    def Ensemble_ICIR_weight(self):
        # Ensemble the models based on ICIR weight
        SUM_ICIR = self.MLP_ICIR + self.GBDT_ICIR + self.AGRU_ICIR
        MLP_ratio = self.MLP_ICIR / SUM_ICIR
        GBDT_ratio = self.GBDT_ICIR / SUM_ICIR
        AGRU_ratio = self.AGRU_ICIR / SUM_ICIR

        MLP_weight = self.MLP_factor_backtest.multiply(MLP_ratio, axis=0)
        GBDT_weight = self.GBDT_factor_backtest.multiply(GBDT_ratio, axis=0)
        AGRU_weight = self.AGRU_factor_backtest.multiply(AGRU_ratio, axis=0)

        Ensemble_weight_factor_df = MLP_weight + GBDT_weight + AGRU_weight
        Ensemble_weight_factor_df_Monday = Ensemble_weight_factor_df.copy()
        Ensemble_weight_factor_df_Monday.index = self.date_backtest_Monday
        Ensemble_weight_factor_df_Monday = Ensemble_weight_factor_df_Monday.rename_axis(index='trade_date')
        Ensemble_weight_factor_df_Monday.to_csv(self.path + 
                                                f'Ensemble_weight_factor_df_Monday_{self.backtest_date_index}.csv')
        self.Ensemble_weight_factor_df_Monday = Ensemble_weight_factor_df_Monday
    def Ensemble_ICIR_max(self, *ICIR_df, type_list):
        # Ensemble the models based on max ICIR
        # Use: Ensemble_ICIR_max(MLP_ICIR, GBDT_ICIR, AGRU_ICIR,
        #                        type_list=["MLP", "GBDT", "AGRU"])

        # Concatenate ICIR DataFrames into Combine_df
        # for ICIR in ICIR_df:
        #     Combine_df = pd.concat([Combine_df, ICIR], axis=1)
        Combine_df = pd.concat(ICIR_df, axis=1)
        Combine_df.columns = type_list
        # print(Combine_df)
        Combine_ranks = Combine_df.rank(axis=1, ascending=False)

        # Choose max ICIR and assign it as 1, others as 0
        Keep_df = Combine_ranks.applymap(lambda x: 1 if x == 1 else 0)

        Ensemble_max = pd.DataFrame()
        for TYPE in type_list:
            # Dynamically create variable names using exec()
            keepp = Keep_df[TYPE]
            # Find factor df use getattr()
            factor_df = getattr(self, f'{TYPE}_factor_backtest')
            positionn = factor_df.where(keepp == 1, 0)
            Ensemble_max = Ensemble_max.add(positionn, fill_value=0)
        Ensemble_max_factor_df = Ensemble_max
        Ensemble_max_factor_df_Monday = Ensemble_max_factor_df.copy()
        Ensemble_max_factor_df_Monday.index = self.date_backtest_Monday
        Ensemble_max_factor_df_Monday = Ensemble_max_factor_df_Monday.rename_axis(index='trade_date')
        Ensemble_max_factor_df_Monday.to_csv(self.path + 
                                             f'Ensemble_max_factor_df_Monday_{self.backtest_date_index}.csv')
        self.Ensemble_max_factor_df_Monday = Ensemble_max_factor_df_Monday

if not Online:
    date_all = pd.read_csv(path_HS300_Rolling + 'date_all.csv', 
                           index_col=0, 
                           parse_dates=True).index
    date_train, date_backtest = train_test_split(date_all, test_size=0.2, shuffle=False)

    # Load the stocks code from the file
    with open(path_HS300_Rolling + 'stocks_code.pkl', 'rb') as file:
        stocks_code = pickle.load(file)

    price = pd.read_csv(path_HS300_Rolling + "price_processed.csv", 
                        parse_dates=["trade_date"])
    price_return = pd.read_csv(path_HS300_Rolling + "price_return.csv", 
                        parse_dates=["trade_date"],
                        index_col="trade_date")
    # benchmark = pd.read_csv(path_HS300_Rolling + "benchmark.csv", 
    #                         parse_dates=["trade_date"])

# Create a DataFrame with the DatetimeIndex
date_all_df = pd.DataFrame(index=date_all)
date_backtest_df = pd.DataFrame(index=date_backtest)
date_all_df["Monday"] = (date_all_df.index.dayofweek == 0).astype(int)  # Add Monday
date_backtest_df["Monday"] = (date_backtest_df.index.dayofweek == 0).astype(int)  # Add Monday
date_backtest_Monday_df = date_backtest_df[date_backtest_df["Monday"] == 1]

# Check if the first backtest day is Monday
First_Monday = bool(date_backtest_df.values[0])
rolling_length = len(date_backtest_Monday_df)

if __name__ == '__main__':
    # Rolling training, the whole training step is not included in it.
    begin_index = -1
    end_index = rolling_length - 1
    true_step = 0
    for i in tqdm(range(begin_index, end_index + 1), desc="Rolling Progress", ncols=100):
        if i == -1 and not First_Monday:
            # The first backtest is the interval from the beginning to the first Monday.
            # Because the first day is not Monday, for corresponding date_backtest demand consideration,
            # we make the first day of date_backtest as a first backtest day in the first rolling step.
            backtest_begin = date_backtest_df.index[0]
            Next_backtest_begin = date_backtest_Monday_df.index[0]
        elif i == -1 and First_Monday: continue
        else:
            backtest_begin = date_backtest_Monday_df.index[i]
            if i < rolling_length - 1:
                Next_backtest_begin = date_backtest_Monday_df.index[i + 1]
            # If it is the last week.
            else: Next_backtest_begin = date_backtest_Monday_df.index[-1] + timedelta(weeks=1)
        true_step += 1
        print("*" * 60)
        print(f"This is the {i} th rolling step. Finished {round(true_step/(rolling_length+1)*100, 2)}% of all.")
        print("*" * 60)
        date_all_rolling = date_all_df[date_all_df.index < backtest_begin].index
        date_backtest_rolling = date_all_df[(date_all_df.index >= backtest_begin) & 
                                            (date_all_df.index < Next_backtest_begin)].index
        HS300_ROLLING = HS_300_Rolling(Price=price, 
                                    Price_Return=price_return,
                                    Path=path_HS300_Rolling,
                                    Date_All=date_all_rolling, 
                                    Date_Backtest=date_backtest_rolling, 
                                    Stocks_code=stocks_code,
                                    Backtest_Date_Index=i)
        # Clear the output of the current cell
        clear_output(wait=True)
        # print(date_all_rolling)
        # print(date_backtest_rolling)