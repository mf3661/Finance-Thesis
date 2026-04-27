# Stock Turnover Standard Deviation (std_turn)
# 3-month rolling standard deviation of turnover = vol / shrout
# Optimized multiprocessing across 20 firm quantiles (1959+)

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


# Daily CRSP volume and shares outstanding data (1959+)
daily_turnover_df = wrds_connection.raw_sql("""
                    select a.permno, a.date, a.vol, a.shrout
                    from crsp.dsf as a
                    where a.date > '01/01/1959'
                    """)


# Sort by firm identifier and date
daily_turnover_df = daily_turnover_df.sort_values(by=['permno', 'date'])


# Convert firm identifier to integer
daily_turnover_df['permno'] = daily_turnover_df['permno'].astype(int)


# Standardize date column
daily_turnover_df['date'] = pd.to_datetime(daily_turnover_df['date'])


# Convert shares outstanding to consistent units (thousands → units)
daily_turnover_df['shrout'] = daily_turnover_df['shrout'] * 1000


# Identify closest trading day to month-end
daily_turnover_df['monthend_date'] = daily_turnover_df['date'] + MonthEnd(0)
daily_turnover_df['days_to_monthend'] = daily_turnover_df['monthend_date'] - daily_turnover_df['date']
monthend_closest_df = daily_turnover_df.groupby(['permno', 'monthend_date'])['days_to_monthend'].min()
monthend_closest_df = pd.DataFrame(monthend_closest_df).reset_index()
monthend_closest_df.rename(columns={'days_to_monthend': 'min_days_diff'}, inplace=True)
daily_turnover_df = pd.merge(daily_turnover_df, monthend_closest_df, how='left', on=['permno', 'monthend_date'])
daily_turnover_df['monthend_flag'] = np.where(
    daily_turnover_df['days_to_monthend'] == daily_turnover_df['min_days_diff'], 1, np.nan
)


# Sequential month numbering for month-end observations
daily_turnover_df['month_sequence'] = daily_turnover_df[
    daily_turnover_df['monthend_flag'] == 1
].groupby(['permno']).cumcount()


# Total months per firm
firm_month_totals = daily_turnover_df[
    daily_turnover_df['monthend_flag'] == 1
].groupby(['permno'])['month_sequence'].tail(1)
firm_month_totals = firm_month_totals.astype(int).reset_index(drop=True)


# Backfill month numbers within each month
daily_turnover_df['month_sequence'] = daily_turnover_df.groupby(['permno'])['month_sequence'].fillna(method='bfill')


# Firm processing metadata
firm_processing_df = daily_turnover_df.drop_duplicates(['permno'])[['permno']]
firm_processing_df['permno'] = firm_processing_df['permno'].astype(int)
firm_processing_df = firm_processing_df.reset_index()
firm_processing_df.rename(columns={'index': 'firm_sequence_id'}, inplace=True)
firm_processing_df['total_months'] = firm_month_totals


######################
# Turnover Std Dev   #
######################


def compute_turnover_std_dev(turnover_data_df, firm_metadata_df):
    """
    Compute 3-month rolling std dev of stock turnover (vol / shrout)
    
    :param turnover_data_df: Daily volume/shrout dataframe
    :param firm_metadata_df: Firm metadata with month counts
    :return: Dataframe with std_turn estimates
    """
    for firm_id, month_count, progress_idx in zip(
        firm_metadata_df['permno'], 
        firm_metadata_df['total_months'], 
        range(firm_metadata_df['permno'].count() + 1)
    ):
        progress_pct = ((progress_idx + 1) / firm_metadata_df['permno'].count()) * 100
        print(f'Processing firm {firm_id} / Completed {progress_pct:.2f}%')
        
        for month_idx in range(month_count + 1):
            # Extract 3-month rolling window
            window_data = turnover_data_df[
                (turnover_data_df['permno'] == firm_id) & 
                (month_idx - 2 <= turnover_data_df['month_sequence']) & 
                (turnover_data_df['month_sequence'] <= month_idx)
            ]
            
            # Skip insufficient observations
            if len(window_data) < 21:
                continue
                
            if window_data['vol'].notna().sum() < 21:
                continue
                
            latest_index = window_data.tail(1).index
            
            # Turnover = volume / shares outstanding, compute std dev
            turnover_matrix = pd.DataFrame()
            turnover_matrix[['vol', 'shrout']] = window_data[['vol', 'shrout']]
            turnover_values = turnover_matrix['vol'] / turnover_matrix['shrout']
            turnover_std = turnover_values.std()
            
            turnover_data_df.loc[latest_index, 'std_turn'] = turnover_std
    
    return turnover_data_df


def partition_firms_by_quantile(start_q, end_q, q_step):
    """
    Partition firms into quantiles for parallel processing
    
    :param start_q: Starting quantile (0)
    :param end_q: Ending quantile (1)
    :param q_step: Quantile step size
    :return: Dictionary of firm partitions and matching turnover data
    """
    partition_dict = {}
    for quantile_idx, group_num in zip(np.arange(start_q, end_q, q_step), 
                                     range(int((end_q - start_q) / q_step))):
        print(f'Partitioning firms: {quantile_idx:.2f} to {quantile_idx + q_step:.2f}')
        
        if quantile_idx == 0:
            partition_dict[f'firm_group{group_num}'] = firm_processing_df[
                firm_processing_df['firm_sequence_id'] <= firm_processing_df['firm_sequence_id'].quantile(quantile_idx + q_step)
            ]
            partition_dict[f'turnover_group{group_num}'] = pd.merge(
                daily_turnover_df, partition_dict[f'firm_group{group_num}'], 
                how='left', on='permno'
            ).dropna(subset=['firm_sequence_id'])
        else:
            partition_dict[f'firm_group{group_num}'] = firm_processing_df[
                (firm_processing_df['firm_sequence_id'].quantile(quantile_idx) < firm_processing_df['firm_sequence_id']) & 
                (firm_processing_df['firm_sequence_id'] <= firm_processing_df['firm_sequence_id'].quantile(quantile_idx + q_step))
            ]
            partition_dict[f'turnover_group{group_num}'] = pd.merge(
                daily_turnover_df, partition_dict[f'firm_group{group_num}'], 
                how='left', on='permno'
            ).dropna(subset=['firm_sequence_id'])
    
    return partition_dict


def execute_parallel_turnover_calc(start_q, end_q, q_step):
    """
    Orchestrate parallel turnover std dev calculation
    
    :param start_q: Start quantile
    :param end_q: End quantile
    :param q_step: Quantile step
    :return: Complete std_turn dataframe
    """
    partitioned_data = partition_firms_by_quantile(start_q, end_q, q_step)
    worker_pool = mp.Pool()
    task_results = {}
    
    num_partitions = int((end_q - start_q) / q_step)
    for partition_id in range(num_partitions):
        task_results[f'partition{partition_id}'] = worker_pool.apply_async(
            compute_turnover_std_dev, 
            (partitioned_data[f'turnover_group{partition_id}'], partitioned_data[f'firm_group{partition_id}'],)
        )
    
    worker_pool.close()
    worker_pool.join()
    
    print('Combining partition results...')
    combined_results = pd.DataFrame()
    for partition_id in range(num_partitions):
        combined_results = pd.concat([combined_results, task_results[f'partition{partition_id}'].get()])
    
    return combined_results


# Execute parallel turnover std dev calculation (20 processes)
if __name__ == '__main__':
    turnover_std_results = execute_parallel_turnover_calc(0, 1, 0.05)


# Final processing and save
turnover_std_results = turnover_std_results.dropna(subset=['std_turn'])
turnover_std_results = turnover_std_results.reset_index(drop=True)
turnover_std_results = turnover_std_results[['permno', 'date', 'std_turn']]


feather.write_feather(turnover_std_results, 'turnover_std_dev.feather')
print("turnover_std_dev.feather saved successfully")