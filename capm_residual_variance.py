# CAPM Residual Variance Calculation
# Optimized multiprocessing implementation using 20 firm quantiles
# Computes 3-month rolling CAPM residual variance (rvar_capm)

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


# Daily CRSP + Market Factor Data (1959+)
daily_returns_df = wrds_connection.raw_sql("""
                      select a.permno, a.date, a.ret, a.vol, b.rf, b.mktrf
                      from crsp.dsf as a
                      left join ff.factors_daily as b
                      on a.date=b.date
                      where a.date >= '01/01/1959'
                      """)


# Sort by firm identifier and date
daily_returns_df = daily_returns_df.sort_values(by=['permno', 'date'])


# Convert firm identifier to integer
daily_returns_df['permno'] = daily_returns_df['permno'].astype(int)


# Standardize date column
daily_returns_df['date'] = pd.to_datetime(daily_returns_df['date'])


# Delisting returns data
delisting_data_df = wrds_connection.raw_sql("""
                     select permno, dlret, dlstdt 
                     from crsp.dsedelist
                     """)


delisting_data_df.permno = delisting_data_df.permno.astype(int)
delisting_data_df['dlstdt'] = pd.to_datetime(delisting_data_df['dlstdt'])
delisting_data_df['date'] = delisting_data_df['dlstdt']


# Merge delisting returns and compute adjusted excess returns
daily_returns_df = pd.merge(daily_returns_df, delisting_data_df, how='left', on=['permno', 'date'])
daily_returns_df['dlret'] = daily_returns_df['dlret'].fillna(0)
daily_returns_df['ret'] = daily_returns_df['ret'].fillna(0)
daily_returns_df['retadj'] = (1 + daily_returns_df['ret']) * (1 + daily_returns_df['dlret']) - 1
daily_returns_df['exret'] = daily_returns_df['retadj'] - daily_returns_df['rf']


# Identify closest trading day to month-end
daily_returns_df['monthend_date'] = daily_returns_df['date'] + MonthEnd(0)
daily_returns_df['days_to_monthend'] = daily_returns_df['monthend_date'] - daily_returns_df['date']
monthend_closest_df = daily_returns_df.groupby(['permno', 'monthend_date'])['days_to_monthend'].min()
monthend_closest_df = pd.DataFrame(monthend_closest_df).reset_index()
monthend_closest_df.rename(columns={'days_to_monthend': 'min_days_diff'}, inplace=True)
daily_returns_df = pd.merge(daily_returns_df, monthend_closest_df, how='left', on=['permno', 'monthend_date'])
daily_returns_df['monthend_indicator'] = np.where(
    daily_returns_df['days_to_monthend'] == daily_returns_df['min_days_diff'], 1, np.nan
)


# Sequential month numbering for month-end observations
daily_returns_df['month_sequence'] = daily_returns_df[
    daily_returns_df['monthend_indicator'] == 1
].groupby(['permno']).cumcount()


# Total months per firm
firm_month_totals = daily_returns_df[
    daily_returns_df['monthend_indicator'] == 1
].groupby(['permno'])['month_sequence'].tail(1)
firm_month_totals = firm_month_totals.astype(int).reset_index(drop=True)


# Backfill month numbers within each month
daily_returns_df['month_sequence'] = daily_returns_df.groupby(['permno'])['month_sequence'].fillna(method='bfill')


# Firm processing metadata
firm_processing_df = daily_returns_df.drop_duplicates(['permno'])[['permno']]
firm_processing_df['permno'] = firm_processing_df['permno'].astype(int)
firm_processing_df = firm_processing_df.reset_index()
firm_processing_df.rename(columns={'index': 'firm_sequence_id'}, inplace=True)
firm_processing_df['total_months'] = firm_month_totals


######################
# CAPM Residual Variance #
######################


def compute_capm_residual_variance(stock_returns_df, firm_metadata_df):
    """
    Calculate 3-month rolling CAPM residual variance for each firm-month
    
    :param stock_returns_df: Complete daily returns dataframe
    :param firm_metadata_df: Firm metadata with month counts
    :return: Dataframe with CAPM residual variance estimates
    """
    for firm_id, total_months, progress_idx in zip(
        firm_metadata_df['permno'], 
        firm_metadata_df['total_months'], 
        range(firm_metadata_df['permno'].count() + 1)
    ):
        progress_pct = ((progress_idx + 1) / firm_metadata_df['permno'].count()) * 100
        print(f'Processing firm {firm_id} / Completed {progress_pct:.2f}%')
        
        for month_idx in range(total_months + 1):
            # Extract 3-month rolling window (month_idx-2 to month_idx)
            window_data = stock_returns_df[
                (stock_returns_df['permno'] == firm_id) & 
                (month_idx - 2 <= stock_returns_df['month_sequence']) & 
                (stock_returns_df['month_sequence'] <= month_idx)
            ]
            
            # Skip if insufficient observations (minimum 21 days)
            if len(window_data) < 21:
                continue
                
            if window_data['vol'].notna().sum() < 21:
                continue
                
            window_length = len(window_data)
            latest_obs_index = window_data.tail(1).index
            
            # CAPM regression setup: exret ~ intercept + mktrf
            regressor_matrix = pd.DataFrame()
            regressor_matrix[['mktrf']] = window_data[['mktrf']]
            regressor_matrix['intercept'] = 1
            regressor_matrix = regressor_matrix[['intercept', 'mktrf']]
            regressor_matrix = np.mat(regressor_matrix)
            
            dependent_var = np.mat(window_data[['exret']])
            
            # Residuals: I - X(X'X)^(-1)X'
            residuals_matrix = (
                np.identity(window_length) - 
                regressor_matrix.dot(regressor_matrix.T.dot(regressor_matrix).I).dot(regressor_matrix.T)
            ).dot(dependent_var)
            
            residual_variance = residuals_matrix.var(ddof=1)
            stock_returns_df.loc[latest_obs_index, 'rvar'] = residual_variance
    
    return stock_returns_df


def partition_firms_by_quantile(start_q, end_q, q_step):
    """
    Partition firms into quantiles for parallel processing
    
    :param start_q: Starting quantile (typically 0)
    :param end_q: Ending quantile (typically 1)
    :param q_step: Quantile step size
    :return: Dictionary of partitioned firm and stock data
    """
    partition_dict = {}
    for quantile_pos, group_id in zip(np.arange(start_q, end_q, q_step), range(int((end_q - start_q) / q_step))):
        print(f'Partitioning firms: {quantile_pos:.2f} to {quantile_pos + q_step:.2f}')
        
        if quantile_pos == 0:
            partition_dict[f'firm_partition{group_id}'] = firm_processing_df[
                firm_processing_df['firm_sequence_id'] <= firm_processing_df['firm_sequence_id'].quantile(quantile_pos + q_step)
            ]
            partition_dict[f'returns_partition{group_id}'] = pd.merge(
                daily_returns_df, partition_dict[f'firm_partition{group_id}'], 
                how='left', on='permno'
            ).dropna(subset=['firm_sequence_id'])
        else:
            partition_dict[f'firm_partition{group_id}'] = firm_processing_df[
                (firm_processing_df['firm_sequence_id'].quantile(quantile_pos) < firm_processing_df['firm_sequence_id']) & 
                (firm_processing_df['firm_sequence_id'] <= firm_processing_df['firm_sequence_id'].quantile(quantile_pos + q_step))
            ]
            partition_dict[f'returns_partition{group_id}'] = pd.merge(
                daily_returns_df, partition_dict[f'firm_partition{group_id}'], 
                how='left', on='permno'
            ).dropna(subset=['firm_sequence_id'])
    
    return partition_dict


def execute_parallel_residual_variance(start_q, end_q, q_step):
    """
    Orchestrate parallel CAPM residual variance calculation
    
    :param start_q: Starting quantile
    :param end_q: Ending quantile  
    :param q_step: Quantile step size
    :return: Complete residual variance dataframe
    """
    partitioned_data = partition_firms_by_quantile(start_q, end_q, q_step)
    worker_pool = mp.Pool()
    task_results = {}
    
    num_partitions = int((end_q - start_q) / q_step)
    for partition_id in range(num_partitions):
        task_results[f'partition{partition_id}'] = worker_pool.apply_async(
            compute_capm_residual_variance, 
            (partitioned_data[f'returns_partition{partition_id}'], partitioned_data[f'firm_partition{partition_id}'],)
        )
    
    worker_pool.close()
    worker_pool.join()
    
    print('Combining partition results...')
    combined_results = pd.DataFrame()
    for partition_id in range(num_partitions):
        combined_results = pd.concat([combined_results, task_results[f'partition{partition_id}'].get()])
    
    return combined_results


# Execute parallel CAPM residual variance calculation (20 processes)
if __name__ == '__main__':
    capm_residual_df = execute_parallel_residual_variance(0, 1, 0.05)


# Final processing and output
capm_residual_df = capm_residual_df.dropna(subset=['rvar'])
capm_residual_df = capm_residual_df.rename(columns={'rvar': 'rvar_capm'})
capm_residual_df = capm_residual_df.reset_index(drop=True)
capm_residual_df = capm_residual_df[['permno', 'date', 'rvar_capm']]


feather.write_feather(capm_residual_df, 'capm_residual_variance.feather')
print("capm_residual_variance.feather saved successfully")