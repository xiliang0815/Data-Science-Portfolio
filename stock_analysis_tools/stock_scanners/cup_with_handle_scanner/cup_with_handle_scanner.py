import pandas as pd
import warnings
warnings.filterwarnings("ignore")

def downloand_stock_daily(stocks_to_dl, startdate, enddate):
    import fix_yahoo_finance as yf
    import pandas as pd
    from pandas_datareader import data as pdr
    yf.pdr_override()
    
    output_df = pd.DataFrame(columns=['Ticker', 'Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume'])
    
    for stock in stocks_to_dl:
        try:
            data_tmp = pdr.get_data_yahoo(stock, start=startdate, end=enddate)
            data_tmp.reset_index(inplace=True)
            data_tmp['Ticker'] = stock
            output_df = output_df.append(data_tmp, sort=False)
        except:
            next
    output_df.reset_index(drop=True, inplace=True)
    return output_df

def annual_max_min(df):
    from functools import reduce
    #output_df = pd.DataFrame(columns=['Ticker', '52wk_max', '52wk_min', '52wk_max_date', '52wk_min_date'])
    
    max_df = df.groupby('Ticker').apply(lambda x: x['High'].max()).reset_index(name='52wk_max')
    max_date_df = df.groupby('Ticker').apply(lambda x: x.loc[x['High'].argmax()]['Date']).reset_index(name='52wk_max_date')
    
    min_df = df.groupby('Ticker').apply(lambda x: x['Low'].min()).reset_index(name='52wk_min')
    min_date_df = df.groupby('Ticker').apply(lambda x: x.loc[x['Low'].argmin()]['Date']).reset_index(name='52wk_min_date')
    
    dfs_to_merge = [max_df, max_date_df, min_df, min_date_df]
    output_df = reduce(lambda left, right: pd.merge(left, right, on ='Ticker'), dfs_to_merge)
    return(output_df)

def annual_max_time_delta(annual_max_df):
    import datetime
    current_date = datetime.datetime.today().strftime('%Y-%m-%d')
    
    annual_max_df['52wk_max_date'] = pd.to_datetime(annual_max_df['52wk_max_date'])
    annual_max_df['current_date'] = current_date
    annual_max_df['current_date'] = pd.to_datetime(annual_max_df['current_date'])
    
    annual_max_df['max_time_delta_wks'] = (annual_max_df['current_date'] - annual_max_df['52wk_max_date']).dt.days/7
    return(annual_max_df)

def find_tailing_max(daily_df, annual_max_min_df):
    output_df = pd.DataFrame(columns=['Ticker', 'tailing_max_date', 'pct_change_to_52wk_max', 'time_delta_to_52wk_max_wk'])
    
    for stock in annual_max_min_df['Ticker']:
        max_52wk_date = annual_max_min_df[annual_max_min_df['Ticker'] == stock]['52wk_max_date'].values[0]
        #if the 52wk max date is today, skip
        if datetime.datetime.utcfromtimestamp(max_52wk_date.tolist()/1e9).strftime('%Y-%m-%d') != datetime.datetime.today().strftime('%Y-%m-%d'):
            max_52wk_price = annual_max_min_df[annual_max_min_df['Ticker'] == stock]['52wk_max'].values[0]
        
            post_52wk_max_df = daily_df[(daily_df['Ticker'] == stock) & (daily_df['Date']>max_52wk_date)].reset_index(drop=True)
            post_52wk_max_min_date =post_52wk_max_df.loc[post_52wk_max_df['Low'].argmin]['Date']
            tailing_max_df = daily_df[(daily_df['Ticker'] == stock) & (daily_df['Date']>post_52wk_max_min_date)]
        
            if len(tailing_max_df) !=0:
                tailing_max_date = tailing_max_df.loc[tailing_max_df['High'].argmax]['Date']
                tailing_max_price = tailing_max_df['High'].max()
                output_df = output_df.append(pd.DataFrame({'Ticker': [stock],
                                                          'tailing_max_date': [tailing_max_date],
                                                          'pct_change_to_52wk_max': [(tailing_max_price-max_52wk_price)/max_52wk_price],
                                                          'time_delta_to_52wk_max_wk': [(tailing_max_date - max_52wk_date).days/7]}))
            else: next
        else: next
    return(output_df)