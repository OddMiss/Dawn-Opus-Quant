# Dawn-Opus-Quant
Quantitative research with Dawn Opus Asset Management. I mainly focused on machine learning research, selecting a quantitative report on its application as factors.

## Stage 1: Feature Engineering
Implementation of 30 standardized factors
```python
factor_def_dict = \
{'alpha191_022': ('MEAN(((CLOSE_DF-MEAN(CLOSE_DF,6))/MEAN(CLOSE_DF,6)-DELAY((CLOSE_DF-MEAN(CLOSE_DF,6))/MEAN(CLOSE_DF,6),3)),12)', 22),
'alpha_01': ('-1 * CORR(RANK(DELTA(LOG(VOLUME_DF), 1)), RANK(((CLOSE_DF - OPEN_DF) / OPEN_DF)), 6)', 22)}
'alpha_02': ('-1 * DELTA((((CLOSE_DF - LOW_DF) - (HIGH_DF - CLOSE_DF)) / (HIGH_DF - LOW_DF)), 1)', 22),
'alpha_06': ('RANK(SIGN(DELTA((((OPEN_DF * 0.85) + (HIGH_DF * 0.15))), 4))) * -1', 22),
'alpha_14': ('CLOSE_DF - DELAY(CLOSE_DF, 5)', 22)
'alpha_15': ('OPEN_DF / DELAY(CLOSE_DF, 1) - 1', 22),
'alpha_18': ('CLOSE_DF / DELAY(CLOSE_DF, 5)', 22),
'alpha_20': ('(CLOSE_DF - DELAY(CLOSE_DF, 6)) / DELAY(CLOSE_DF, 6) * 100', 22),
'alpha_29': ('(CLOSE_DF - DELAY(CLOSE_DF, 6)) / DELAY(CLOSE_DF, 6) * VOLUME_DF', 22),
'alpha_31': ('(CLOSE_DF - MEAN(CLOSE_DF, 12)) / MEAN(CLOSE_DF, 12) * 100', 22),
'alpha_32': ('-1 * SUM(RANK(CORR(RANK(HIGH_DF), RANK(VOLUME_DF), 3)), 3)', 22),
'alpha_34': ('MEAN(CLOSE_DF, 12) / CLOSE_DF', 22),
'alpha_42': ('(-1 * RANK(STD2(HIGH_DF, 10))) * CORR(HIGH_DF, VOLUME_DF, 10)', 22),
'alpha_46': ('(MEAN(CLOSE_DF, 3) + MEAN(CLOSE_DF, 6) + MEAN(CLOSE_DF, 12) + MEAN(CLOSE_DF, 24)) / (4 * CLOSE_DF)', 22),
'alpha_53': ('COUNT(CLOSE_DF > DELAY(CLOSE_DF, 1), 12) / 12 * 100', 22),
'alpha_54': ('-1 * RANK((STD1(ABS(CLOSE_DF - OPEN_DF)) + (CLOSE_DF - OPEN_DF)) + CORR(CLOSE_DF, OPEN_DF, 10))', 22),
'alpha_58': ('COUNT(CLOSE_DF > DELAY(CLOSE_DF, 1), 20) / 20 * 100', 22),
'alpha_62': ('(-1 * CORR(HIGH_DF, RANK(VOLUME_DF), 5))', 22),
'alpha_65': ('MEAN(CLOSE_DF, 6) / CLOSE_DF', 22),
'alpha_70': ('STD2(AMOUNT_DF, 6)', 22),
'alpha_71': ('(CLOSE_DF - MEAN(CLOSE_DF, 24)) / MEAN(CLOSE_DF, 24) * 100', 22),
'alpha_76': ('STD2(ABS((CLOSE_DF / DELAY(CLOSE_DF, 1) - 1)) / VOLUME_DF, 20) / MEAN(ABS((CLOSE_DF / DELAY(CLOSE_DF, 1)-  1)) / VOLUME_DF, 20) ', 22),
'alpha_80': ('(VOLUME_DF - DELAY(VOLUME_DF, 5)) / DELAY(VOLUME_DF, 5) * 100', 22),
'alpha_88': ('(CLOSE_DF - DELAY(CLOSE_DF, 20)) / DELAY(CLOSE_DF, 20) * 100', 22),
'alpha_95': ('STD2(AMOUNT_DF, 20)', 22),
'alpha_97': ('STD2(VOLUME_DF, 10)', 22),
'alpha_100': ('STD2(VOLUME_DF, 20)', 22),
'alpha_103': ('((20 - LOWDAY(LOW_DF, 20)) / 20) * 100', 22),
'alpha_104': ('-1 * (DELTA(CORR(HIGH_DF, VOLUME_DF, 5), 5) * RANK(STD2(CLOSE_DF, 20)))', 22),
'alpha_105': ('(-1 * CORR(RANK(OPEN_DF), RANK(VOLUME_DF), 10)) ', 22),
'alpha_106': ('CLOSE_DF - DELAY(CLOSE_DF, 20)', 22)}
```

## Stage 2: Quantitative Framework with Machine learning
1. Data collection: All A stocks.
2. Data cleaning: Clear inappropriate stocks like ST, ST* etc.
3. Label making: VWAP ROI of between T and T + 11.
4. Data preprocessing: 3MAD, z-score etc.
5. ROI dataframe making: HS300, CS500, CS1000 etc.
6. Modeling: MLP, GBDT, AGRU.
7. Ensembling: according to past 60 days' ICIR.
8. Backtesting.

## Stage 3: Offline internship: 2024.7.7~2024.8.10
1. Analysis the convertible bond (CB) with research report and write operators
```python
# 期权信号算子
def OPT_SIGNAL(Delta_DF, Theta_DF, Gamma_DF, Vol_DF, CB_ANAL_CON_DF, CB_CLOSE_DF):
    res_DF = Delta_DF * 0.05 + Theta_DF + 0.5 * (Vol_DF**2) * Gamma_DF
    return res_DF
MERGE_DF["Opt_Signal"] = OPT_SIGNAL(
    MERGE_DF["Delta"],
    MERGE_DF["Theta"],
    MERGE_DF["Gamma"],
    MERGE_DF["annual_vol"],
    MERGE_DF["CB_ANAL_CONVVALUE"],
    MERGE_DF["CB_close"]
)

# Gamma-Theta信号算子
def GAMMA_THETA_SIGNAL(Theta_DF, Gamma_DF, Vol_DF, Stock_Price_DF, CB_opt_price_DF):
    res_DF = (Theta_DF + 0.5 * ((Stock_Price_DF * Vol_DF) ** 2) * Gamma_DF) / CB_opt_price_DF
    return res_DF
MERGE_DF["Signal"] = GAMMA_THETA_SIGNAL(
    MERGE_DF["Theta"], 
    MERGE_DF["Gamma"], 
    MERGE_DF["annual_vol"], 
    MERGE_DF["Stock_close"],
    MERGE_DF["CB_optionvalue"] / MERGE_DF["CB_ANAL_CONVRATIO"]
)

# theta和gamma信号的算子
def CAL_GREEKS(S_DF, K_DF, r: float, T_DF, Sigma_DF, Option_type: str, Greeks_type: str):
    d1_DF = (np.log(S_DF / K_DF) + (r + Sigma_DF**2 / 2) * T_DF) / (Sigma_DF * np.sqrt(T_DF))
    d2_DF = d1_DF - Sigma_DF * np.sqrt(T_DF)

    d1_DF = (np.log(S_DF / K_DF) + (r + 0.5 * Sigma_DF**2)) / ((T_DF**0.5) * Sigma_DF)
    d2_DF = (np.log(S_DF / K_DF) + (r - 0.5 * Sigma_DF**2)) / ((T_DF**0.5) * Sigma_DF)
    nd1_DF, nd2_DF = NORM_CDF(d1_DF), NORM_CDF(d2_DF)
    if Greeks_type == "Gamma":
        gamma_DF = nd1_DF / (S_DF * Sigma_DF * (T_DF**0.5))
        return gamma_DF

    elif Greeks_type == "Theta":
        if Option_type == "Call":
            theta_DF = -0.5 * S_DF * Sigma_DF * nd1_DF / (T_DF**0.5) - r * K_DF * np.exp(-r * T_DF) * nd2_DF
        else:
            raise ValueError("Invalid option type. Must be 'Call' or 'Put'.")
        return theta_DF
    elif Greeks_type == "Delta":
        delta_DF = NORM_CDF(d1_DF)
        return delta_DF
    else:
        raise ValueError("Invalid Greeks type. Must be 'Call' or 'Put'.")

# 计算隐含波动率算子
def CALCULATE_IV(Option_price_DF, S_DF, K_DF, T_DF, r: float):
    def BS_CB(S, K, T, sigma, r):
        # 不带赎回
        d1 = (np.log(S / float(K)) + (r + 0.5 * (sigma**2)) * T) / (
            np.sqrt(T) * float(sigma))
        d2 = d1 - sigma * np.sqrt(T)
        res = S * ss.norm.cdf(d1) - K * np.exp(-r * T) * (ss.norm.cdf(d2))
        return res

    def Loop_Check(row):
        count = 0  # 计数器
        top = 1  # 波动率上限
        floor = 0  # 波动率下限
        sigma = (floor + top) / 2  # 波动率初始值
        Option_price = row["Option_price"]
        option_price_est = row["option_price_est"]
        S = row["S"]
        K = row["K"]
        T = row["T"]
        while Option_price - option_price_est > 0.000001:
            option_price_est = BS_CB(S, K, T, sigma, r) * K / 100
            # 根据价格判断波动率是被低估还是高估，并对波动率做修正
            count += 1
            if count > 500:
                # 时间价值为0的期权是算不出隐含波动率的，因此迭代到一定次数就不再迭代了
                sigma = 0
                break
            if Option_price - option_price_est > 0:  # f(x)>0
                floor = sigma
                sigma = (sigma + top) / 2
            else:
                top = sigma
                sigma = (sigma + floor) / 2
        return sigma

    Option_price_DF = Option_price_DF * K_DF / 100  # 除以转股比例
    # Option_Len = len(Option_price_DF)
    option_price_est_DF = pd.Series(0, index=Option_price_DF.index, 
                                    name="option_price_est")  # 期权价格估计值

    # 改变名称，方便后续使用
    Option_price_DF.name = "Option_price"
    S_DF.name = "S"
    K_DF.name = "K"
    T_DF.name = "T"

    MERGE_DF = pd.concat(
        [Option_price_DF, option_price_est_DF, 
         S_DF, K_DF, T_DF], axis=1)
    return MERGE_DF.apply(Loop_Check, axis=1)
```
3. Use DTW algorithm to financial market forecast
4. Python financial factors writing (An example as follows)

### 账面杠杆(7.21)

(Market Leverage)

$$账面杠杆=\frac{最近报告期总资产}{最近同期股东权益总计}$$

表示资产总额是股东权益总额的多少倍，作者测试结果表明平均回报率与账面杠杆负相关；

Barra中的账面杠杆=最近报告期的(非流动负债合计+优先股账面价值+普通股账面价值)/最近报告期的普通股账面价值
```python
# Market leverage
def Market_Leverage(Date, common_param):
    Stock_list = common_param["Stock_list"]
    Financial_DF = xtdata.get_financial_data(Stock_list, ["Balance"])
    # Market leverage
    All_data = pd.DataFrame()
    for asset in Financial_DF.keys():
        temp = Financial_DF[asset]["Balance"][
            [
                "m_timetag", 
                "m_anntime", 
                "total_equity", 
                "tot_liab_shrhldr_eqy"
            ]
        ]
        temp["ts_code"] = asset
        temp = temp[
            [
                "ts_code",
                "m_timetag",
                "m_anntime",
                "total_equity",
                "tot_liab_shrhldr_eqy"
            ]
        ]
        temp = temp[temp.m_anntime == Date]
        All_data = pd.concat([All_data, temp], ignore_index=True)
    if All_data.empty: return None

    All_data["Year"] = All_data["m_timetag"].apply(lambda x: x[:4]).astype(int)
    All_data["Month"] = All_data["m_timetag"].apply(lambda x: x[4:6]).astype(int)
    All_data["m_timetag"] = pd.to_datetime(All_data["m_timetag"])
    All_data["m_anntime"] = pd.to_datetime(All_data["m_anntime"])
    # All_data = All_data[All_data.m_timetag == Date]

    All_data["Market_Leverage"] = All_data["tot_liab_shrhldr_eqy"] / All_data["total_equity"]
    All_data["Market_Leverage"] = All_data["Market_Leverage"].replace([np.inf, -np.inf], np.nan)
    All_data = All_data[["m_timetag", "ts_code", "Market_Leverage"]]
    All_data = All_data.rename(columns={"m_timetag": "trade_date"})
    All_data.reset_index(drop=True, inplace=True)
    return All_data
```
