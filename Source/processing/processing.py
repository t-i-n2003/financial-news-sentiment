from processing.prepare_data import prepare_data
import pandas as pd
from vnstock import Vnstock


def get_score(symbol, file, start_date = '2021 - 01 - 01', end_date = '2025 - 03 - 31'):
    df = prepare_data(symbol, file, start_date, end_date)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_colwidth', None)
    df.drop_duplicates(subset=['title'], inplace=True)
    revenue = [0.01, -0.015] #profit and loss
    df['symbol_price_change_ratio'] = df['symbol_price_change_ratio'].apply(
        lambda x: 1 if x > revenue[0] else (-1 if x < revenue[1] else 0)
    )
    df['symbol_price_change'] = df['symbol_price_change'].apply(
        lambda x: 1 if x > 0 else -1
    )
    df['VnId_volume_score'] = df.apply(
        lambda row: (
            1 if row['VnId_volume'] > row['vol20'] else 0.5),
        axis=1
    )
    df['VnId_price_change_ratio_score'] = df['VnId_price_change_ratio'].apply(
        lambda x: 1 if x > 0 else -1
    )
    df['VnId_change_ratio_score'] = df.apply(
        lambda x: 1 if x['VnId_change_ratio_score'] > x['mean_VnId_price_change_ratio'] else 0.5
    )

    df['VnId_score'] = df['VnId_volume_score'] * df['VnId_price_change_ratio_score'] * df['VnId_change_ratio_score']
    score=['title', 'symbol_price_change', 'symbol_price_change_ratio', 'VnId_score']
    df = pd.DataFrame(df, columns=score)
    weight = [0.4, 0.2, 0.4]
    df['score'] = (
        df['symbol_price_change_ratio'] * weight[0] + df['symbol_price_change']* weight[1] 
        + df['VnId_score'] * weight[2]
    )
    df.to_csv(f'dataset/score.csv', index=False)
    return df

#get_score('FPT', start_date='2021-01-01', end_date='2025-03-31')