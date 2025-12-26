# Risk On/Off analysis

import yfinance as yf
import pandas as pd
import numpy as np
from pandas_datareader import data as web
import seaborn as sb
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import datetime as dt
from fredapi import Fred
import pandas as pd
import numpy as np
import urllib.request

#Fred access variables
fred_api_key= "Your own key"   #use your own Fred api key
fred = Fred(api_key=fred_api_key)

#Global variables
Backtest_start = '2005-01-01'
Backtest_end = '2025-01-01'
fred_tickers = ['GDP', 'CPIAUCSL']

#Function to use if you don't have fred api key

#def fetching_fred (fred_tickers):
#  data = web.DataReader(fred_tickers, 'fred', start=Backtest_start, end=Backtest_end)
#  data_m = data.resample('MS')      #monthly data (month start)
#  data_m= data_m.asfreq()           #define empty lines
#  data_m = data_m.ffill()           #forward fill for GDP in months
#  return data_m

#Inflation and growth data gathering and cleaning -> everything monthly
def fetching_fred(fred_tickers):
    data = pd.DataFrame()
    opener = urllib.request.build_opener()
    opener.addheaders = [("User-Agent", "Mozilla/5.0")]
    urllib.request.install_opener(opener)
    for ticker in fred_tickers:
        s = fred.get_series(ticker, observation_start=Backtest_start, observation_end=Backtest_end)
        s.index = pd.to_datetime(s.index)
        data[ticker] = s
        
    data = data.sort_index()
    data_m = data.resample("MS").asfreq().ffill()
    return data_m

#Regime function, with a yoy calculation before, regime is determined for each month of the period
def compute_macro_regim(data_m):
    yoy = data_m.copy()

    yoy["GDP_YoY"] = yoy["GDP"].pct_change(12) * 100
    yoy["CPI_YoY"] = yoy["CPIAUCSL"].pct_change(12) * 100

    rising_infl = yoy["CPI_YoY"].diff() > 0
    rising_gdp  = yoy["GDP_YoY"].diff() > 0

    yoy["GDP_direction"] = np.where(rising_gdp, "rising growth", "falling growth")
    yoy["CPI_direction"] = np.where(rising_infl, "rising inflation", "falling inflation")

    Regime = pd.DataFrame(index=yoy.index)
    Regime["Regime"] = yoy["GDP_direction"] + " and " + yoy["CPI_direction"]

    return Regime

# basket of assets
asset_mapping = {
    'SPY' : 'US Equities (SP500)',
    'QQQ' : 'US Equities (Large Cap)',
    'EFA' : 'US Equities (Mid Cap)',
    'EEM' : 'US Equities (Small Cap)',
    'TLT' : 'US Treasury (2 yr)',
    'HYG' : 'High Yield Bond',
    'GLD' : 'Gold',
    'DBC' : 'Commodities'
}

asset_tickers = list(asset_mapping.keys())
asset_tickers

#access to assets data, had to add a security with the close column because of a few problems occuring
def access_asset_prices(tickers, start_date, end_date):
        raw = yf.download(tickers, start=start_date, end=end_date, interval='1mo', progress=False)
        if isinstance(raw, pd.DataFrame) and 'Close' in raw.columns:
            close_prices = raw['Close']
        else:
            close_prices = raw
        clean_prices = close_prices.dropna(how='all')
        return clean_prices

access_asset_prices(asset_tickers,Backtest_start,Backtest_end)

#compute returns
def compute_assets (prices_df) :
    returns_df = prices_df.pct_change(12) * 100
    clean_returns = returns_df.dropna()
    return clean_returns


assets_yoy = compute_assets(access_asset_prices(asset_tickers,Backtest_start,Backtest_end))
assets_yoy

#Final computation of vol for each regime
def compute_vol_regime(Regime, assets_yoy):
    data = Regime.join(assets_yoy, how="inner")
    vol_regime = data.groupby("Regime").std()

    sb.heatmap(vol_regime, cmap="RdBu_r", annot=True, fmt=".2f",
               linewidths=0.5, cbar_kws={"label": "Volatility"})
    plt.title("Asset Volatility by Macroeconomic Regime")
    plt.ylabel("Macroeconomic Regime")
    plt.xlabel("Asset Class")
    plt.tight_layout()
    plt.show()

    return vol_regime

#Final computation of returns for each regime
def compute_return_regime(Regime, assets_yoy):
    data = Regime.join(assets_yoy, how="inner")
    return_regime = data.groupby("Regime").mean()

    sb.heatmap(return_regime, cmap="RdBu_r", center=0, annot=True, fmt=".2f",
               linewidths=0.5, cbar_kws={"label": "Mean Return"})
    plt.title("Asset Returns by Macroeconomic Regime")
    plt.ylabel("Macroeconomic Regime")
    plt.xlabel("Asset Class")
    plt.tight_layout()
    plt.show()

    return return_regime

#Final computation of normal returns for each regime
def compute_normal_return_regime(Regime, assets_yoy):
    data = Regime.join(assets_yoy, how="inner")

    mean_regime = data.groupby("Regime").mean()
    vol_regime  = data.groupby("Regime").std()

    norm_regime = mean_regime / vol_regime
    norm_regime = norm_regime.replace([np.inf, -np.inf], np.nan)

    sb.heatmap(norm_regime, cmap="RdBu_r", center=0, annot=True, fmt=".2f",
               linewidths=0.5, cbar_kws={"label": "Normalized Return (Mean/Vol)"})
    plt.title("Normalized Return by Macroeconomic Regime")
    plt.ylabel("Macroeconomic Regime")
    plt.xlabel("Asset Class")
    plt.tight_layout()
    plt.show()

    return norm_regime

 # only calling api once, otherwise there is a anti-bot of fred that stop the download
macro_m   = fetching_fred(fred_tickers)         
Regime_df = compute_macro_regim(macro_m)

#the 3 final heatmaps
vol = compute_vol_regime(Regime_df, assets_yoy)
ret = compute_return_regime(Regime_df, assets_yoy)
nor = compute_normal_return_regime(Regime_df, assets_yoy)