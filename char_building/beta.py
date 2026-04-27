# Fama-French 3-Factor Residual Beta Calculation
# Optimized for multiprocessing across firm quantiles
# Uses 20 processes by default - adjust based on CPU cores

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


# Daily CRSP + Fama-French Factors
stock_returns_df = wrds_connection.raw_sql("""
                      select a.permno, a.date, a.ret, a.vol, b.rf, b.mktrf, b.smb, b.hml
                      from crsp.dsf as a
                      left join ff.factors_daily as b
                      on a.date=b.date
                      where a.date > '01/01/2015'
                      """)


# Sort by firm and date
stock_returns_df = stock_returns_df.sort_values(by=['permno', 'date'])


# Convert permno to integer
stock_returns_df['permno'] = stock_returns_df['permno'].astype(int)


# Standardize dates to month-end
stock_returns_df['date'] = pd.to_datetime(stock_returns_df['date'])


# Delisting returns
delisting_returns_df = wrds_connection.raw_sql("""
                     select permno, dlret, dlstdt 
                     from crsp.dsedelist
                     """)


delisting_returns_df.permno = delisting_returns_df.permno.astype(int)
delisting_returns_df['dlstdt'] = pd.to_datetime(delisting_returns_df['dlstdt'])
delisting_returns_df['date'] = delisting_returns_df['dlstdt']


# Merge delisting returns and compute adjusted returns
stock_returns_df = pd.merge(stock_returns_df, delisting_returns_df, how='left', on=['permno', 'date'])
stock_returns_df['dlret'] = stock_returns_df['dlret'].fillna(0)
stock_returns_df['ret'] = stock_returns_df['ret'].fillna(0)
stock_returns_df['retadj'] = (1 + stock_returns_df['ret']) * (1 + stock_returns_df['dlret']) - 1
stock_returns_df['exret'] = stock_returns_df['retadj'] - stock_returns_df['rf']


# Identify month-end trading days (closest to month-end)
stock_returns_df['monthend'] = stock_returns_df['date'] + MonthEnd(0)
stock_returns_df['date_diff'] = stock_returns_df['monthend'] - stock_returns_df['date']
month_end_diff_df = stock_returns_df.groupby(['permno', 'monthend'])['date_diff'].min()
month_end_diff_df = pd.DataFrame(month_end_diff_df).reset_index()
month_end_diff_df.rename(columns={'date_diff': 'min_diff'}, inplace=True)
stock_returns_df = pd.merge(stock_returns_df, month_end_diff_df, how='left', on=['permno', 'monthend'])
stock_returns_df['month_end_flag'] = np.where(stock_returns_df['date_diff'] == stock_returns_df['min_diff'], 1, np.nan)


# Create monthly sequence numbering
stock_returns_df['monthly_sequence'] = stock_returns_df[stock_returns_df['month_end_flag'] == 1].groupby(['permno']).cumcount()


# Get total months per firm
firm_month_counts = stock_returns_df[stock_returns_df['month_end_flag'] == 1].groupby(['permno'])['monthly_sequence'].tail(1)
firm_month_counts = firm_month_counts.astype(int).reset_index(drop=True)


# Propagate month numbers backward within each month
stock_returns_df['monthly_sequence'] = stock_returns_df.groupby(['permno'])['monthly_sequence'].fillna(method='bfill')


# Create firm list with processing metadata
firm_list_df = stock_returns_df.drop_duplicates(['permno'])[['permno']]
firm_list_df['permno'] = firm_list_df['permno'].astype(int)
firm_list_df = firm_list_df.reset_index()
firm_list_df.rename(columns={'index': 'firm_index'}, inplace=True)
firm_list_df['total_months'] = firm_month_counts


######################
# Beta Calculation   #
######################


def compute_firm_beta(stock_data_df, firm_metadata_df):
    """
    Compute rolling 3-month Fama-French beta for each firm-month
    
    :param stock_data_df: Full stock returns dataframe
    :param firm_metadata_df: Firm list with month counts
    :return: Dataframe with beta estimates
    """
    for firm_id, month_count, progress_idx in zip(
        firm_metadata_df['permno'], 
        firm_metadata_df['total_months'], 
        range(firm_metadata_df['permno'].count() + 1)
    ):
        progress_pct = ((progress_idx + 1) / firm_metadata_df['permno'].count()) * 100
        print(f'Processing firm {firm_id} / Completed {progress_pct:.2f}%')
        
        for month_idx in range(month_count + 1):
            # 3-month rolling window (i-2 to i)
            window_data = stock_data_df[
                (stock_data_df['permno'] == firm_id) & 
                (month_idx - 2 <= stock_data_df['monthly_sequence']) & 
                (stock_data_df['monthly_sequence'] <= month_idx)
            ]
            
            # Require minimum 21 observations
            if len(window_data) < 21:
                continue
                
            if window_data['vol'].notna().sum() < 21:
                continue
                
            window_size = len(window_data)
            latest_index = window_data.tail(1).index
            
            # Fama-French regression: exret ~ mktrf
            X_matrix = np.mat(window_data[['mktrf']])
            Y_matrix = np.mat(window_data[['exret']])
            ones_matrix = np.mat(np.ones(window_size)).T
            M_matrix = np.identity(window_size) - ones_matrix.dot((ones_matrix.T.dot(ones_matrix)).I).dot(ones_matrix.T)
            
            beta_estimate = (X_matrix.T.dot(M_matrix).dot(X_matrix)).I.dot(X_matrix.T.dot(M_matrix).dot(Y_matrix))
            stock_data_df.loc[latest_index, 'beta'] = beta_estimate
    
    return stock_data_df


def split_dataframe_by_quantile(start_quantile, end_quantile, quantile_step):
    """
    Split firms into quantiles for parallel processing
    
    :param start_quantile: Starting quantile (usually 0)
    :param end_quantile: Ending quantile (usually 1)  
    :param quantile_step: Step size between quantiles
    :return: Dictionary of firm lists and stock data by quantile
    """
    quantile_dict = {}
    for quantile_idx, group_num in zip(np.arange(start_quantile, end_quantile, quantile_step), range(int((end_quantile - start_quantile) / quantile_step))):
        print(f'Splitting firms: {quantile_idx:.2f} to {quantile_idx + quantile_step:.2f}')
        
        if quantile_idx == 0:
            quantile_dict[f'firm_group{group_num}'] = firm_list_df[firm_list_df['firm_index'] <= firm_list_df['firm_index'].quantile(quantile_idx + quantile_step)]
            quantile_dict[f'stock_group{group_num}'] = pd.merge(
                stock_returns_df, quantile_dict[f'firm_group{group_num}'], 
                how='left', on='permno'
            ).dropna(subset=['firm_index'])
        else:
            quantile_dict[f'firm_group{group_num}'] = firm_list_df[
                (firm_list_df['firm_index'].quantile(quantile_idx) < firm_list_df['firm_index']) & 
                (firm_list_df['firm_index'] <= firm_list_df['firm_index'].quantile(quantile_idx + quantile_step))
            ]
            quantile_dict[f'stock_group{group_num}'] = pd.merge(
                stock_returns_df, quantile_dict[f'firm_group{group_num}'], 
                how='left', on='permno'
            ).dropna(subset=['firm_index'])
    
    return quantile_dict


def process_parallel_beta_calculation(start_quantile, end_quantile, quantile_step):
    """
    Main multiprocessing orchestration
    
    :param start_quantile: Starting quantile
    :param end_quantile: Ending quantile
    :param quantile_step: Quantile step size
    :return: Complete beta dataframe
    """
    quantile_data_dict = split_dataframe_by_quantile(start_quantile, end_quantile, quantile_step)
    process_pool = mp.Pool()
    process_results = {}
    
    num_groups = int((end_quantile - start_quantile) / quantile_step)
    for group_idx in range(num_groups):
        process_results[f'group{group_idx}'] = process_pool.apply_async(
            compute_firm_beta, 
            (quantile_data_dict[f'stock_group{group_idx}'], quantile_data_dict[f'firm_group{group_idx}'],)
        )
    
    process_pool.close()
    process_pool.join()
    
    print('Concatenating results...')
    final_results = pd.DataFrame()
    for group_idx in range(num_groups):
        final_results = pd.concat([final_results, process_results[f'group{group_idx}'].get()])
    
    return final_results


# Execute parallel beta calculation (20 processes for 0.05 quantiles)
if __name__ == '__main__':
    beta_results_df = process_parallel_beta_calculation(0, 1, 0.05)


# Final cleanup and save
beta_results_df = beta_results_df.dropna(subset=['beta'])
beta_results_df = beta_results_df.reset_index(drop=True)
beta_results_df = beta_results_df[['permno', 'date', 'beta']]


feather.write_feather(beta_results_df, 'firm_beta_estimates.feather')
print("firm_beta_estimates.feather saved successfully")