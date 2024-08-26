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
1. Analysis the convertible bond (CB) with research report
2. Use DTW algorithm to financial market forecast
3. Python financial factors writing
