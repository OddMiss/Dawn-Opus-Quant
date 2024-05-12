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

def Industry(stock_codes):
    SWClass_all = pd.read_csv(path + 'SWClass_all.csv')
    return SWClass_all[SWClass_all['股票代码'].isin(stock_codes)]