from flask import Flask, render_template, request, redirect
import pandas as pd
import numpy as np
import pandas_ta as ta
import yfinance as yf
import json
import datetime as dt



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

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('terms.html')

@app.route('/interactive', methods=['GET', 'POST'])
def feature():
    if request.method == 'POST':
        cal = {}
        index = request.form.get('text')
        date = request.form.get('date')
        data = yf.Ticker(index).history(start=date, period='1d')
        df = pd.DataFrame(data)
        for i in range(1, 5):
            perR = request.form.get(f'PRSI{i}')
            perD = request.form.get(f'PDRSI{i}')
            if not perR or not perD:
                break
            else:
                cal[int(perR)] = int(perD)
        df.columns = map(str.lower, df.columns)
        df['x'] = df.index
        df['x'] = df['x'].apply(lambda x: x.strftime('%Y-%m-%d'))
        df.reset_index(drop=True, inplace=True)
        df = add_rsi(df, periods=cal.keys())
        df = add_corr(df, periods=cal.keys(), memories=cal)
        table = df.to_json()
        ma = [i for i in cal.keys()]
        sup = {i: ma[i] for i in range(len(ma))}
        cup = {i: cal[ma[i]] for i in range(len(ma))}
        r = json.dumps(table)
        table = json.loads(r)
        sup = json.dumps(sup)
        cup = json.dumps(cup)
        sup = json.loads(sup)
        cup = json.loads(cup)
        return render_template('interactive.html', table=table, index=index, sup=sup, cup=cup, n=len(sup))
    else:
        return render_template('interactive.html')

@app.route('/index')
def route2():
    return render_template('index.html')


if __name__ == '__main__':
    app.run(debug=True)
