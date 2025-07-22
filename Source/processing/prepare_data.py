from vnstock import Vnstock
from vnstock.explorer.tcbs import Company
import pandas as pd
import urllib3
from bs4 import BeautifulSoup
http = urllib3.PoolManager()
from deep_translator import GoogleTranslator
import time

def adjust_weekend_time(time):
    if time.weekday() == 5:  # Saturday (Monday is 0, Sunday is 6)
        return time + pd.Timedelta(days=2)
    elif time.weekday() == 6:  # Sunday
        return time + pd.Timedelta(days=1)
    else:
        return time

def translate_vi_to_en(text):
    try:
        translated = GoogleTranslator(source='vi', target='en').translate(text)
        return translated
    except Exception as e:
        return f"Error: {e}"
    
def get_news_cafef(symbol, start='2021-01-01', end='2025-01-01', translate=False):
    page_index = 1
    page_size = 30
    data_dicts = []

    start_date = pd.to_datetime(start).date()
    end_date = pd.to_datetime(end).date()
    while True:
        url = f"http://s.cafef.vn/Ajax/Events_RelatedNews_New.aspx?symbol={symbol}&floorID=0&configID=0&PageIndex={page_index}&PageSize={page_size}&Type=2"
        r = http.request('GET', url)
        soup = BeautifulSoup(r.data, "html.parser")
        data = soup.find("ul", {"class": "News_Title_Link"})
        if not data:
            break
        raw = data.find_all('li')
        if not raw:
            break
        keep_going = True
        for row in raw:
            symbol = symbol
            date = pd.to_datetime(row.span.text, dayfirst=True).date()
            if date > end_date:
                continue
            if date < start_date:
                keep_going = False
                break
            title = row.a.text
            data_dicts.append({'time': date, 'symbol': symbol, 'title': title})
        if not keep_going or len(raw) < page_size:
            break
        page_index += 1
        time.sleep(2)
    df = pd.DataFrame(data_dicts)
    df['time'] = df['time'].apply(adjust_weekend_time)
    df['title'] = df['title'].apply(lambda x: translate_vi_to_en(x))
    return df

def get_symbol_index(symbol, start, end):
    columns = ['time', 'symbol','symbol_price_change', 'symbol_price_change_ratio']
    pd.set_option('display.max_colwidth', None)
    stock = Vnstock().stock(symbol= symbol, source='VCI')
    cp = stock.quote.history(symbol=symbol, start= start, end= end, interval='1D')
    cp['symbol_price_change'] = cp['close'] - cp['open']
    cp['symbol_price_change_ratio'] = (cp['close'] - cp['open'])/cp['open']
    cp['symbol'] = symbol
    df = pd.DataFrame(cp, columns=columns)
    df['time'] = pd.to_datetime(df['time']).dt.date
    df = df[columns]
    return df

def VnId(start_date = '2021-01-01', end_date = '2025-01-01'):
    columns = ['time', 'VnId_volume', 'VnId_price_change_ratio', 'VnId_change_ratio']
    start_date = pd.to_datetime(start_date).date()
    end_date = pd.to_datetime(end_date).date()
    stock = Vnstock().stock(symbol= 'ACB', source='VCI')
    VnId = stock.quote.history(symbol='VNINDEX', start='2021-01-01', end='2025-03-31', interval='1D')
    VnId.drop_duplicates(subset=['time'], inplace=True)
    VnId['VnId_change_ratio'] = (VnId['high'] - VnId['low'])/VnId['close']
    VnId['VnId_price_change_ratio'] = (VnId['close'] - VnId['open'])/VnId['open']
    VnId['vol20'] = VnId['volume'].rolling(window=20).mean()
    VnId['mean_VnId_price_change_ratio'] = VnId['VnId_price_change_ratio'].rolling(window=20).mean()
    VnId['time'] = pd.to_datetime(VnId['time']).dt.date
    VnId.rename(columns={'volume': 'VnId_volume'}, inplace=True)
    VnId = VnId[columns]
    return VnId

def prepare_data(symbol, file = None, start = '2021-01-01', end = '2025-01-01'):
    start_date = pd.to_datetime(start).date()
    end_date = pd.to_datetime(end).date()
    data = []
    news = []
    for i in range(len(symbol)):
        j = str(symbol[i])
        symbol_price = get_symbol_index(j, start, end)
        data.append(symbol_price)
    symbol_price = pd.concat(data, ignore_index=True)
    symbol_price.to_csv(f'dataset/symbol.csv', index=False)
    VnId_df = VnId(start_date, end_date)
    if file is not None:
        cafef = pd.read_csv(file)
    else:
        for i in range(len(symbol)):
            j = str(symbol[i]) 
            cafef = get_news_cafef(j, start, end)
            news.append(cafef)
        cafef = pd.concat(news, ignore_index=True)
    cafef['time'] = pd.to_datetime(cafef['time']).dt.date
    cafef.to_csv(f'dataset/news.csv', index=False)
    news = pd.merge(cafef, symbol_price, on=['time', 'symbol'], how='left')
    news['time'] = pd.to_datetime(news['time']).dt.date
    merged_df = pd.merge(news, VnId_df, on='time', how='left')
    mask = (merged_df['time'] >= start_date) & (merged_df['time'] <= end_date)
    merged_df['title'] = merged_df['title'].apply(lambda x: translate_vi_to_en(x))
    merged_df = merged_df[mask]
    return merged_df

# pd.concat([
# get_news_cafef('ACB', start = '2021-01-01', end ='2025-01-01'),
# get_news_cafef('BID', start = '2021-01-01', end ='2025-01-01'),
# get_news_cafef('VCB', start = '2021-01-01', end ='2025-01-01'),   
# get_news_cafef('MBB', start = '2021-01-01', end ='2025-01-01'),
# ]).to_csv('dataset/news.csv', index=False)
# get_news_cafef('FPT', 1000,  '2021-01-01', '2025-03-31')
# get_news_cafef('ACB', 1000,  '2021-01-01', '2025-03-31')