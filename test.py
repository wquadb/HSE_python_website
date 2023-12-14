import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import pandas as pd
import numpy as np
import pandas_ta as ta
import os


def get_data():
    assert 'YNDX.csv' in os.listdir(), "You need to get the dataset first"

    yndxdf = pd.read_csv('FLOT.csv', sep=';')

    yndxdf['name'] = 'Yandex'

    return yndxdf

def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocessing the dataframe
    :param df: dataframe, *index must be datetime
    :return: dataframe with datetime index
    """

    df['tradedate'] = pd.to_datetime(df['tradedate'], yearfirst=True, format='%Y-%m-%d')
    df.set_index('tradedate', inplace=True)

    return df

def inita():
    yndxdf = get_data()
    yndxdf = preprocess(yndxdf)
    return yndxdf

def add_corr(df_origin: pd.DataFrame,periods: list = [],memories: dict = {},cut: dict = {},mine: bool = False) -> pd.DataFrame:
    """
    Calculating correlation between two columns and append to a new one
    :param df_origin: dataframe, *index must be datetime
    :param periods: list of all periods of RSI that are in the dataframe
    :param memories: dictionary of all periods of RSI and their memory
    :param cut: dictionary of config of cutting the correlation
    :return: dataframe with correlation columns
    """

    assert len(periods) > 0 and len(periods) <= 5, "You need to choose at least 1 period, at most 5 periods"
    assert all([period in memories.keys() for period in periods]), "You need to choose periods for each period"
    assert all([f'ta_rsi_{period}' in df_origin.columns for period in periods]), "You need to calculate RSI first"

    go = {0: 'ta', 1: 'my'}

    df = df_origin.copy()

    for period in periods:

        memory = memories[period]

        for i in range(2):
            if not mine:
                if i == 1:
                    break

            result_df, corr = df.copy(), dict()
            result_corr, timesteps = [], [i for i in df.index]

            col_1, col_2 = f'close', f'{go[i]}_rsi_{period}'
            new_col, beta = f'corr_{go[i]}_rsi_{period}', 0

            for k in range(len(timesteps)):
                tempdf = df.loc[timesteps[max(0, k - memory)]:timesteps[k]].copy()
                result_corr.append(tempdf[col_1].corr(tempdf[col_2]))

            if len(timesteps) != len(result_corr):
                raise ValueError("num of rows of df and calculated correlation must be equal")
            else:

                corr = {x: i for i, x in zip(result_corr, timesteps)}

                if cut and (go[i] in cut['who']) and (cut['how'] == 'front' or cut['how'] == 'all'):
                    for x in corr:
                        if beta < cut['amount']:
                            corr[x] = np.NaN
                        beta += 1
                if cut and (go[i] in cut['who']) and (cut['how'] == 'back' or cut['how'] == 'all'):
                    for x in corr:
                        if beta > len(corr) - cut['amount']:
                            corr[x] = np.NaN
                        beta += 1

                result_df[new_col] = corr

            df = result_df

    return result_df
def add_rsi(df_origin: pd.DataFrame, periods: list = []) -> pd.DataFrame:
    """
    Calculating RSI (Relative Strength Ratio) indicator
    and adding it to the dataframe
    """

    df = df_origin.copy()

    for period in periods:
        assert period > 0, "Period must be greater than 0"
        assert period < len(df), "Period must be less than the length of dataframe"

        # rsi
        accurate_rsi = ta.rsi(df['close'], length=period)

        result_df = df.copy()
        result_df[f'ta_rsi_{period}'] = accurate_rsi

        df = result_df

    return result_df


# Load data
data = inita()


# Set up the plot


import pandas as pd
import numpy as np

df = data['2020-10-01':'2024-01-01']

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import mplfinance as mpf

def plot_time_series_correlation(series1, series2, method='pearson'):
    """
    Plots the correlation between two pandas.Series objects with time indices.

    :param series1: pandas.Series
    :param series2: pandas.Series
    :param method: Method of correlation (default is 'pearson')
    """
    # Ensure the series are aligned in time
    aligned_series1, aligned_series2 = series1.align(series2, join='inner')

    # Calculate correlation
    correlation = aligned_series1.corr(aligned_series2, method=method)

    # Plotting
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=aligned_series1, y=aligned_series2)
    plt.title(f'Yandex 2021/09/01 : 2022/01/01 | Time Series Correlation ({method.title()}): {correlation:.2f}')
    plt.xlabel('RSI')
    plt.ylabel('Price')
    plt.grid(True)
    plt.show()

# Example usage:
# plot_time_series_correlation(df['close'], df['ta_rsi_14'])

def monthly_candles(df):
    """
    Changing candle's period to month
    """

    # Ensure the index is a datetime index
    df.index = pd.to_datetime(df.index)

    monthly_df = df.resample('M').agg(
        {
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum',
            'name': 'first'
        }
    )

    return monthly_df
def mpl_candlechart(df):
    """
    Plotting candlestick chart with volume bars using mplfinance library
    """

    mpf.plot(
        df,
        title='sovcomflot',
        type='candle',
        mav=(3, 7),
        volume=True,
        show_nontrading=True,
        style='yahoo',
        figsize=(10, 6),
        savefig='flot2123.png'

    )

mpl_candlechart(monthly_candles(data))
