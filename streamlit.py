from streamlit import *
import streamlit as st
import pandas as pd
import plotly.express as px
import os
# import yfinance as yf


def get_data():
    assert 'YNDX.csv' and 'FLOT.csv' in os.listdir(), "You need to get the datasets first"
    
    yndxdf = pd.read_csv('YNDX.csv', sep=';')
    flotdf = pd.read_csv('FLOT.csv', sep=';')
    
    yndxdf['name'], flotdf['name'] = 'Yandex', 'Sovcomflot'
    
    return yndxdf, flotdf

def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    df['tradedate'] = pd.to_datetime(df['tradedate'], yearfirst=True, format='%Y-%m-%d')
    df.set_index('tradedate', inplace=True)
    df.drop(['boardid', 'waval'], axis=1, inplace=True)
    df['vol'] = df['volume'] * df['close']
    
    return df

yndxdf, flotdf = get_data()

st.markdown('# Yandex and Sovcomflot stocks streamlit')
st.markdown('Here I will show how dataframes changing by data refactoring and visualize currency trading volumes over time.')
st.markdown('## Dataframes visualization')
st.markdown('### Yndx and flot datagrames before cleaning')
st.markdown('#### Yandex df')
st.dataframe(yndxdf, height=200)
st.markdown('#### Sovcomflot df')
st.dataframe(flotdf, height=200)

yndxdf, flotdf = preprocess(yndxdf), preprocess(flotdf)

st.markdown('### Yndx and flot dataframes after cleaning')
st.markdown('#### Yandex df')
st.dataframe(yndxdf, height=200)
st.markdown('#### Sovcomflot df')
st.dataframe(flotdf, height=200)

st.markdown('### Yndx and flot volumes visualization')
volumes_df = pd.DataFrame({'Yandex': yndxdf['vol'], 'Sovcomflot': flotdf['vol']})
st.area_chart(volumes_df)

st.markdown('### Yndx and flot close visualization')
volumes_df = pd.DataFrame({'Yandex': yndxdf['close']})
st.area_chart(volumes_df)

st.markdown('### Yndx and flot close visualization')
volumes_df = pd.DataFrame({'Sovcomflot': flotdf['close']})
st.area_chart(volumes_df)

st.plotly_chart(px.histogram(yndxdf, x='close', nbins=100, title='Yandex close prices distribution'))

st.plotly_chart(px.histogram(flotdf, x='close', nbins=100, title='Sovcomflot close prices distribution'))
