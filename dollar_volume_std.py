# Dollar Trading Volume Standard Deviation (std_dolvol)
# 3-month rolling standard deviation of log(dollar volume)
# Optimized multiprocessing across 20 firm quantiles

import pandas as pd
import numpy as np
import datetime as dt
import wrds
from dateutil.relativedelta import *
from pandas.tseries.offsets import *
import datetime
import pickle as pkl
import pyarrow.feather as feather
import multiprocessing as mp


###################
# WRDS Connection #
###################
import sys
import sqlalchemy.engine
sys.modules['sqlalchemy'].create_engine = sqlalchemy.engine.create_engine
wrds_connection = wrds.Connection(wrds_username='hd2570')


# Daily CRSP price and volume data (1959+)
daily_price_volume_df = wrds_connection.raw_sql("""
                    select a.permno, a.date, a.vol, a.prc
                    from crsp.dsf as a
                    where a.date > '01/01/1959'
                    """)


# Sort by firm identifier and date
daily_price_volume_df = daily_price_volume_df.sort_values(by=['permno', 'date'])


# Convert firm identifier to integer
daily_price_volume_df['permno'] = daily_price_volume_df['permno'].astype(int)


# Standardize date column
daily_price_volume_df['date'] = pd.to_datetime(daily_price_volume_df['date'])


# Identify closest trading day to month-end
daily_price_volume_df['monthend_date'] = daily_price_volume_df['date'] + MonthEnd(0)
daily_price_volume_df['days_to_monthend'] = daily_price_volume_df['monthend_date'] - daily_price_volume_df['date']
monthend_closest_df = daily_price_volume_df.groupby(['permno', 'monthend_date'])['days_to_monthend'].min()
monthend_closest_df = pd.DataFrame(monthend_closest_df).reset_index()
monthend_closest_df.rename(columns={'days_to_monthend': 'min_days_diff'}, inplace=True)
daily_price_volume_df = pd.merge(daily_price_volume_df, monthend_closest_df, how='left', on=['permno', 'monthend_date'])
daily_price_volume_df['monthend_flag'] = np.where(
    daily_price_volume_df['days_to_monthend'] == daily_price_volume_df['min_days_diff'], 1, np.nan
)


# Sequential month numbering for month-end observations
daily_price_volume_df['month_sequence'] = daily_price_volume_df[
    daily_price_volume_df['monthend_flag'] == 1
].groupby(['permno']).cumcount()


# Total months per firm
firm_month_counts = daily_price_volume_df[
    daily_price_volume_df['monthend_flag'] == 1
].groupby(['permno'])['month_sequence'].tail(1)
firm_month_counts = firm_month_counts.astype(int).reset_index(drop=True)


# Backfill month numbers within each month
daily_price_volume_df['month_sequence'] = daily_price_volume_df.groupby(['permno'])['month_sequence'].fillna(method='bfill')


# Firm processing metadata
firm_metadata_df = daily_price_volume_df.drop_duplicates(['permno'])[['permno']]
firm_metadata_df['permno'] = firm_metadata_df['permno'].astype(int)
firm_metadata_df = firm_metadata_df.reset_index()
firm_metadata_df.rename(columns={'index': 'firm_order_id'}, inplace=True)
firm_metadata_df['total_months'] = firm_month_counts


######################
# Dollar Volume Std Dev #
######################


def compute_dollar_volume_std(stock_price_vol_df, firm_info_df):
    """
    Compute 3-month rolling std dev of log(dollar trading volume)
    
    :param stock_price_vol_df: Daily price/volume dataframe
    :param firm_info_df: Firm metadata with month counts
    :return: Dataframe with std_dolvol estimates
    """
    for firm_id, month_total, progress_idx in zip(
        firm_info_df['permno'], 
        firm_info_df['total_months'], 
        range(firm_info_df['permno'].count() + 1)
    ):
        progress_pct = ((progress_idx + 1) / firm_info_df['permno'].count()) * 100
        print(f'Processing firm {firm_id} / Completed {progress_pct:.2f}%')
        
        for month_idx in range(month_total + 1):
            # Extract 3-month rolling window
            window_data = stock_price_vol_df[
                (stock_price_vol_df['permno'] == firm_id) & 
                (month_idx - 2 <= stock_price_vol_df['month_sequence']) & 
                (stock_price_vol_df['month_sequence'] <= month_idx)
            ]
            
            # Skip insufficient observations
            if len(window_data) < 21:
                continue
                
            if window_data['vol'].notna().sum() < 21:
                continue
                
            latest_index = window_data.tail(1).index
            
            # Dollar volume = shares * price, log transform, compute std dev
            price_vol_matrix = pd.DataFrame()
            price_vol_matrix[['prc', 'vol']] = window_data[['prc', 'vol']]
            log_dollar_vol = np.log(np.abs(price_vol_matrix['vol'] * price_vol_matrix['prc']))
            log_dollar_vol = log_dollar_vol.replace([np.inf, -np.inf], np.nan)
            dollar_vol_std = log_dollar_vol.std()
            
            stock_price_vol_df.loc[latest_index, 'std_dolvol'] = dollar_vol_std
    
    return stock_price_vol_df


def partition_firms_by_quantile(start_quantile, end_quantile, step_size):
    """
    Partition firms into quantiles for parallel processing
    
    :param start_quantile: Starting quantile (0)
    :param end_quantile: Ending quantile (1)
    :param step_size: Quantile step size
    :return: Dictionary of firm partitions and matching stock data
    """
    partition_dict = {}
    for quantile_idx, group_num in zip(np.arange(start_quantile, end_quantile, step_size), 
                                     range(int((end_quantile - start_quantile) / step_size))):
        print(f'Partitioning firms: {quantile_idx:.2f} to {quantile_idx + step_size:.2f}')
        
        if quantile_idx == 0:
            partition_dict[f'firm_group{group_num}'] = firm_metadata_df[
                firm_metadata_df['firm_order_id'] <= firm_metadata_df['firm_order_id'].quantile(quantile_idx + step_size)
            ]
            partition_dict[f'pricevol_group{group_num}'] = pd.merge(
                daily_price_volume_df, partition_dict[f'firm_group{group_num}'], 
                how='left', on='permno'
            ).dropna(subset=['firm_order_id'])
        else:
            partition_dict[f'firm_group{group_num}'] = firm_metadata_df[
                (firm_metadata_df['firm_order_id'].quantile(quantile_idx) < firm_metadata_df['firm_order_id']) & 
                (firm_metadata_df['firm_order_id'] <= firm_metadata_df['firm_order_id'].quantile(quantile_idx + step_size))
            ]
            partition_dict[f'pricevol_group{group_num}'] = pd.merge(
                daily_price_volume_df, partition_dict[f'firm_group{group_num}'], 
                how='left', on='permno'
            ).dropna(subset=['firm_order_id'])
    
    return partition_dict


def execute_parallel_dolvol_calculation(start_q, end_q, q_step):
    """
    Orchestrate parallel dollar volume std dev calculation
    
    :param start_q: Start quantile
    :param end_q: End quantile
    :param q_step: Quantile step
    :return: Complete std_dolvol dataframe
    """
    partitioned_data = partition_firms_by_quantile(start_q, end_q, q_step)
    process_pool = mp.Pool()
    result_dict = {}
    
    num_groups = int((end_q - start_q) / q_step)
    for group_idx in range(num_groups):
        result_dict[f'group{group_idx}'] = process_pool.apply_async(
            compute_dollar_volume_std, 
            (partitioned_data[f'pricevol_group{group_idx}'], partitioned_data[f'firm_group{group_idx}'],)
        )
    
    process_pool.close()
    process_pool.join()
    
    print('Combining partition results...')
    final_results = pd.DataFrame()
    for group_idx in range(num_groups):
        final_results = pd.concat([final_results, result_dict[f'group{group_idx}'].get()])
    
    return final_results


# Execute parallel dollar volume std dev calculation (20 processes)
if __name__ == '__main__':
    dolvol_results_df = execute_parallel_dolvol_calculation(0, 1, 0.05)


# Final processing and save
dolvol_results_df = dolvol_results_df.dropna(subset=['std_dolvol'])
dolvol_results_df = dolvol_results_df.reset_index(drop=True)
dolvol_results_df = dolvol_results_df[['permno', 'date', 'std_dolvol']]


feather.write_feather(dolvol_results_df, 'dollar_volume_std.feather')
print("dollar_volume_std.feather saved successfully")