import os
import warnings
import time
import pandas as pd
import numpy as np
from jugaad_data.nse import stock_df
from datetime import date, timedelta, datetime
import concurrent.futures

warnings.filterwarnings('ignore')

# ---------------- AUTO DATE (no input needed for automation) ----------------

def get_target_date():
    return date.today()

# ---------------- STOCK LIST ----------------

def get_tickers_from_file(filepath='companies.csv'):
    if not os.path.exists(filepath):
        print(f"File not found: {filepath}")
        return
    try:
        df = pd.read_csv(filepath)
        if 'NSE Code' not in df.columns:
            print("NSE Code column missing")
            return
        tickers = df['NSE Code'].dropna().astype(str).str.strip().tolist()
        return [t for t in tickers if t]
    except Exception as e:
        print(f"CSV Error: {e}")
        return

# ---------------- FETCH ANALYTICS ----------------

def fetch_stock_analytics(symbol, end_date, retries=3):
    for attempt in range(retries):
        try:
            # Fetches 120 days which is sufficient for a 60-day lookback + 20-day SMA calculation
            start_date = end_date - timedelta(days=120)
            df = stock_df(symbol=symbol, from_date=start_date, to_date=end_date, series='EQ')
            if df is None or df.empty or len(df) < 65:  # Increased minimum length for 60d lookback
                return None

            # required columns check
            required = ['DATE', 'CLOSE', 'PREV. CLOSE', 'VOLUME', 'NO OF TRADES']
            if not all(col in df.columns for col in required):
                return None

            df = df.sort_values('DATE').reset_index(drop=True)
            df['SYMBOL'] = symbol

            df['NO OF TRADES'].replace(0, np.nan, inplace=True)  # Avoid division errors

            df['Price Change %'] = (df['CLOSE'] - df['PREV. CLOSE']) / df['PREV. CLOSE'] * 100
            df['per_trade_volume'] = df['VOLUME'] / df['NO OF TRADES']
            df['PTV_SMA20'] = df['per_trade_volume'].rolling(20).mean()
            df['PTV/SMA20'] = df['per_trade_volume'] / df['PTV_SMA20']
            df['PTV_SMA20_spike'] = (df['PTV/SMA20'] > 1.15).astype(int)
            df['Count_PTVSMA>1.15_10d'] = df['PTV_SMA20_spike'].rolling(10).sum()
            df['vol_ratio_curr/prev'] = df['per_trade_volume'] / df['per_trade_volume'].shift(1)
            df['Total_Volume_Ratio'] = df['VOLUME'] / df['VOLUME'].shift(1)
            df['Vol_SMA20'] = df['VOLUME'].rolling(20).mean()
            df['Vol/SMA20'] = df['VOLUME'] / df['Vol_SMA20']

            # Days since higher volume
            volumes = df['VOLUME'].values
            days_since = []
            for i in range(len(volumes)):
                curr = volumes[i]
                days = np.nan
                for j in range(i - 1, -1, -1):
                    if volumes[j] >= curr:
                        days = i - j
                        break
                days_since.append(days)
            df['Days_Since_High_Vol'] = days_since

            return df
        except Exception:
            time.sleep(2)
    return None

# ---------------- WORKER FUNCTION ----------------

def process_single_ticker(symbol, target_date):
    """Wrapper function updated for 60-day PTV high and PTV/SMA20 >= 2 logic."""
    data = fetch_stock_analytics(symbol, target_date)
    time.sleep(np.random.uniform(0.8, 1.5))  # NSE safe delay

    if data is not None and not data.empty:
        last_60_days = data.tail(60)
        max_ptv_60d = last_60_days['per_trade_volume'].max()
        latest_row = last_60_days.iloc[-1]
        latest_ptv = latest_row['per_trade_volume']
        latest_ptv_sma_ratio = latest_row['PTV/SMA20']

        # New Logic: Highest PTV in 60 days AND PTV/SMA20 >= 2
        if latest_ptv >= max_ptv_60d and latest_ptv_sma_ratio >= 2:
            return data.tail(1), f"Processed {symbol} - Added (60d High + Ratio>=2)"
        else:
            return None, f"Skipped {symbol} - Failed 60d High or Ratio>=2"
    else:
        return None, f"Skipped {symbol} - Failed to fetch or missing data"

# ---------------- MAIN ----------------

if __name__ == '__main__':
    tickers = get_tickers_from_file('companies.csv')
    if not tickers:
        print("No tickers found")
        exit()

    target_date = get_target_date()
    print(f"Running for date: {target_date}")
    print(f"{len(tickers)} stocks using multithreading...")

    master = []
    completed_count = 0

    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        future_to_symbol = {executor.submit(process_single_ticker, symbol, target_date): symbol for symbol in tickers}
        for future in concurrent.futures.as_completed(future_to_symbol):
            completed_count += 1
            symbol = future_to_symbol[future]
            try:
                data_row, message = future.result()
                print(f"{completed_count}/{len(tickers)} {message}")
                if data_row is not None:
                    master.append(data_row)
            except Exception as e:
                print(f"{completed_count}/{len(tickers)} Error processing {symbol}: {e}")

    if not master:
        print("No data processed or no stocks met the 60-day highest PTV and Ratio>=2 criteria.")
        exit()

    full_df = pd.concat(master, ignore_index=True)
    actual_date = full_df['DATE'].max()
    actual_date_str = str(actual_date)[:10]

    final_df = full_df[full_df['DATE'] == actual_date].copy()
    columns = ['SYMBOL', 'DATE', 'Price Change %', 'VOLUME', 'NO OF TRADES',
               'per_trade_volume', 'PTV/SMA20', 'Count_PTVSMA>1.15_10d',
               'vol_ratio_curr/prev', 'Total_Volume_Ratio', 'Vol/SMA20', 'Days_Since_High_Vol']
    final_df = final_df[[c for c in columns if c in final_df.columns]]

    if 'PTV/SMA20' in final_df.columns:
        final_df = final_df.sort_values(by='PTV/SMA20', ascending=False)

    numeric_cols = final_df.select_dtypes(include=np.number).columns
    final_df[numeric_cols] = final_df[numeric_cols].round(2)

    filename = f'BreakoutAnalysis60D_{actual_date_str}.csv'
    final_df.to_csv(filename, index=False)

    print('=' * 60)
    print(f"DONE for date: {actual_date_str}")
    print(f"Saved: {filename}")
    print(f"Criteria: PTV is 60-day high AND PTV/SMA20 >= 2")
    print('=' * 60)
