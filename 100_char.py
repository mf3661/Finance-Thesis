import pandas as pd
import numpy as np
import wrds
from pandas.tseries.offsets import *
import pyarrow.feather as feather
from functions import *

###################
# Connect to WRDS #
###################
import sys
import sqlalchemy.engine
sys.modules['sqlalchemy'].create_engine = sqlalchemy.engine.create_engine
wrds.Connection.load_library_list = lambda self: None
wrds_connection = wrds.Connection(wrds_username='hd2570')
print(f"Connected to WRDS successfully!")
#######################################################################################################################
#                                                    TTM functions                                                    #
#######################################################################################################################


def compute_ttm4(column_name, df):
    """

    :param column_name: variables' name
    :param df: dataframe
    :return: compute_ttm4
    """
    lagged_df = pd.DataFrame()
    for i in range(1, 4):
        lagged_df['%(column_name)s%(lagged_df)s' % {'column_name': column_name, 'lagged_df': i}] = df.groupby('permno')['%s' % column_name].shift(i)
    computed_result = df['%s' % column_name] + lagged_df['%s1' % column_name] + lagged_df['%s2' % column_name] + lagged_df['%s3' % column_name]
    return computed_result


def compute_ttm12(column_name, df):
    """

    :param column_name: variables' name
    :param df: dataframe
    :return: compute_ttm12
    """
    lagged_df = pd.DataFrame()
    for i in range(1, 12):
        lagged_df['%(column_name)s%(lagged_df)s' % {'column_name': column_name, 'lagged_df': i}] = df.groupby('permno')['%s' % column_name].shift(i)
    computed_result = df['%s' % column_name] + lagged_df['%s1' % column_name] + lagged_df['%s2' % column_name] + lagged_df['%s3' % column_name] + \
             lagged_df['%s4' % column_name] + lagged_df['%s5' % column_name] + lagged_df['%s6' % column_name] + lagged_df['%s7' % column_name] + \
             lagged_df['%s8' % column_name] + lagged_df['%s9' % column_name] + lagged_df['%s10' % column_name] + lagged_df['%s11' % column_name]
    return computed_result


#######################################################################################################################
#                                                  Compustat Block                                                    #
#######################################################################################################################
print("Loading Compustat Annual data...")
compustat_df = wrds_connection.raw_sql("""
                    /*header info*/
                    select c.gvkey, f.cusip, f.datadate, f.fyear, c.cik, substr(c.sic,1,2) as sic2, c.sic, c.naics,
                    
                    /*firm variables*/
                    /*income statement*/
                    f.sale, f.revt, f.cogs, f.xsga, f.dp, f.xrd, f.xad, f.ib, f.ebitda,
                    f.ebit, f.nopi, f.spi, f.pi, f.txp, f.ni, f.txfed, f.txfo, f.txt, f.xint,
                    
                    /*CF statement and others*/
                    f.capx, f.oancf, f.dvt, f.ob, f.gdwlia, f.gdwlip, f.gwo, f.mib, f.oiadp, f.ivao, f.xpp, f.xacc,
                    
                    /*assets*/
                    f.rect, f.act, f.che, f.ppegt, f.invt, f.at, f.aco, f.intan, f.ao, f.ppent, f.gdwl, f.fatb, f.fatl,
                    
                    /*liabilities*/
                    f.lct, f.dlc, f.dltt, f.lt, f.dm, f.dcvt, f.cshrc, 
                    f.dcpstk, f.pstk, f.ap, f.lco, f.lo, f.drc, f.drlt, f.txdi,
                    
                    /*equity and other*/
                    f.ceq, f.scstkc, f.emp, f.csho, f.seq, f.txditc, f.pstkrv, f.pstkl, f.np, f.txdc, f.dpc, f.ajex, f.conm,
                    
                    /*market*/
                    abs(f.prcc_f) as prcc_f
                    
                    from compustat_df.funda as f
                    left join compustat_df.company as c
                    on f.gvkey = c.gvkey
                    
                    /*get consolidated, standardized, industrial format statements*/
                    where f.indfmt = 'INDL' 
                    and f.datafmt = 'STD'
                    and f.popsrc = 'D'
                    and f.consol = 'C'
                    and f.datadate >= '01/01/2015'
                    """)

# convert datadate to date fmt
compustat_df['datadate'] = pd.to_datetime(compustat_df['datadate'])

# sort and clean up
compustat_df = compustat_df.sort_values(by=['gvkey', 'datadate']).drop_duplicates().reset_index(drop=True)

# clean up csho
compustat_df['csho'] = np.where(compustat_df['csho'] == 0, np.nan, compustat_df['csho'])

# calculate Compustat market equity
compustat_df['mve_f'] = compustat_df['csho'] * compustat_df['prcc_f']

# do some clean up. several variables have lots of missing values
condition_list = [compustat_df['drc'].notna() & compustat_df['drlt'].notna(),
            compustat_df['drc'].notna() & compustat_df['drlt'].isnull(),
            compustat_df['drlt'].notna() & compustat_df['drc'].isnull()]
choice_list = [compustat_df['drc'] + compustat_df['drlt'],
              compustat_df['drc'],
              compustat_df['drlt']]
compustat_df['dr'] = np.select(condition_list, choice_list, default=np.nan)

condition_list = [compustat_df['dcvt'].isnull() & compustat_df['dcpstk'].notna() & compustat_df['pstk'].notna() & compustat_df['dcpstk'] > compustat_df['pstk'],
            compustat_df['dcvt'].isnull() & compustat_df['dcpstk'].notna() & compustat_df['pstk'].isnull()]
choice_list = [compustat_df['dcpstk'] - compustat_df['pstk'],
              compustat_df['dcpstk']]
compustat_df['dc'] = np.select(condition_list, choice_list, default=np.nan)
compustat_df['dc'] = np.where(compustat_df['dc'].isnull(), compustat_df['dcvt'], compustat_df['dc'])

compustat_df['xint0'] = np.where(compustat_df['xint'].isnull(), 0, compustat_df['xint'])
# compustat_df['xsga0'] = np.where(compustat_df['xsga'].isnull, 0, 0)

compustat_df['ceq'] = np.where(compustat_df['ceq'] == 0, np.nan, compustat_df['ceq'])
compustat_df['at'] = np.where(compustat_df['at'] == 0, np.nan, compustat_df['at'])
compustat_df = compustat_df.dropna(subset=['at']).reset_index(drop=True)

feather.write_feather(compustat_df, 'checkpoint_compustat.feather')
print("SAVED: checkpoint_compustat.feather")
#######################################################################################################################
#                                                       CRSP Block                                                    #
#######################################################################################################################
# Create a CRSP Subsample with Monthly Stock and Event Variables
# Restrictions will be applied later
# Select variables from the CRSP monthly stock and event datasets
print("Loading CRSP data...")
crsp_df = wrds_connection.raw_sql("""
                      select a.prc, a.ret, a.retx, a.shrout, a.vol, a.cfacpr, a.cfacshr, a.date, a.permno, a.permco,
                      b.ticker, b.ncusip, b.shrcd, b.exchcd, b.comnam
                      from crsp_df.msf as a
                      left join crsp_df.msenames as b
                      on a.permno=b.permno
                      and b.namedt<=a.date
                      and a.date<=b.nameendt
                      where a.date >= '01/01/2015'
                      and b.exchcd between 1 and 3
                      """)

# change variable format to int
crsp_df[['permco', 'permno', 'shrcd', 'exchcd']] = crsp_df[['permco', 'permno', 'shrcd', 'exchcd']].astype(int)

# Line up date to be end_lag of month
crsp_df['date'] = pd.to_datetime(crsp_df['date'])
crsp_df['monthend'] = crsp_df['date'] + MonthEnd(0)  # set all the date to the standard end_lag date of month

crsp_df = crsp_df.dropna(subset=['prc']).reset_index(drop=True)
crsp_df['me'] = crsp_df['prc'].abs() * crsp_df['shrout']  # calculate market equity

# if Market Equity is Nan then let return equals to 0
crsp_df['ret'] = np.where(crsp_df['me'].isnull(), 0, crsp_df['ret'])
crsp_df['retx'] = np.where(crsp_df['me'].isnull(), 0, crsp_df['retx'])

# impute me
crsp_df = crsp_df.sort_values(by=['permno', 'date']).drop_duplicates().reset_index(drop=True)
crsp_df['me'] = np.where(crsp_df['permno'] == crsp_df['permno'].shift(1), crsp_df['me'].fillna(method='ffill'), crsp_df['me'])

# Aggregate Market Cap
'''
There are cases when the same firm (permco) has two or more securities (permno) at same date.
For the purpose of ME for the firm, we aggregated all ME for a given permco, date.
This aggregated ME will be assigned to the permno with the largest ME.
'''
# sum of me across different permno belonging to same permco a given date
crsp_market_equity_sum_df = crsp_df.groupby(['monthend', 'permco'])['me'].sum().reset_index()
# largest mktcap within a permco/date
crsp_market_equity_max_df = crsp_df.groupby(['monthend', 'permco'])['me'].max().reset_index()
# join by monthend/maxme to find the permno
crsp_largest_security_df = pd.merge(crsp_df, crsp_market_equity_max_df, how='inner', on=['monthend', 'permco', 'me']).reset_index(drop=True)
# drop me column and replace with the sum me
crsp_largest_security_df = crsp_largest_security_df.drop(['me'], axis=1)
# join with sum of me to get the correct market cap info
crsp_aligned_market_df = pd.merge(crsp_largest_security_df, crsp_market_equity_sum_df, how='inner', on=['monthend', 'permco']).reset_index(drop=True)
# sort by permno and date and also drop duplicates
crsp_aligned_market_df = crsp_aligned_market_df.sort_values(by=['permno', 'monthend']).drop_duplicates().reset_index(drop=True)

feather.write_feather(crsp_aligned_market_df, 'checkpoint_crsp_raw.feather')
print("SAVED: checkpoint_crsp_raw.feather")

#######################################################################################################################
#                                                        CCM Block                                                    #
#######################################################################################################################
# merge CRSP and Compustat
# reference: https://wrds-www.wharton.upenn.edu/pages/support/applications/linking-databases/linking-crsp_df-and-compustat/
ccm_link_df = wrds_connection.raw_sql("""
                  select gvkey, lpermno as permno, linktype, linkprim, 
                  linkdt, linkenddt
                  from crsp_df.ccmxpf_linktable
                  where substr(linktype,1,1)='L'
                  and (linkprim ='C' or linkprim='P')
                  """)

ccm_link_df['linkdt'] = pd.to_datetime(ccm_link_df['linkdt'])
ccm_link_df['linkenddt'] = pd.to_datetime(ccm_link_df['linkenddt'])

# if linkenddt is missing then set to today date
ccm_link_df['linkenddt'] = ccm_link_df['linkenddt'].fillna(pd.to_datetime('today'))

# merge ccm_link_df and compustat_df
compustat_df = pd.read_feather('checkpoint_compustat.feather')
ccm_merged_df = pd.merge(compustat_df, ccm_link_df, how='left', on=['gvkey'])

# we can only get the accounting data after the firm public their report
# for annual data, we use 4, 5 or 6 months lagged data, now we follow Hou, Xue and Zhang (2015) use 4 months lagged_df
ccm_merged_df['yearend'] = ccm_merged_df['datadate'] + YearEnd(0)
ccm_merged_df['jdate'] = ccm_merged_df['datadate'] + MonthEnd(4)

# set link date bounds
ccm_linked_df = ccm_merged_df[(ccm_merged_df['jdate'] >= ccm_merged_df['linkdt']) & (ccm_merged_df['jdate'] <= ccm_merged_df['linkenddt'])].reset_index(drop=True)

# link compustat_df and crsp_df
crsp_aligned_market_df = pd.read_feather('checkpoint_crsp_raw.feather')
crsp_aligned_market_df = crsp_aligned_market_df.rename(columns={'monthend': 'jdate'})
annual_characteristics_df = pd.merge(crsp_aligned_market_df, ccm_linked_df, how='inner', on=['permno', 'jdate'])

# filter exchcd & shrcd and at least more than 1 year data
annual_characteristics_df = annual_characteristics_df[((annual_characteristics_df['exchcd'] == 1) | (annual_characteristics_df['exchcd'] == 2) | (annual_characteristics_df['exchcd'] == 3)) &
                      ((annual_characteristics_df['shrcd'] == 10) | (annual_characteristics_df['shrcd'] == 11))].reset_index(drop=True)

# process Market Equity
'''
Note: me is CRSP market equity, mve_f is Compustat market equity. Please choose the me below.
'''
annual_characteristics_df['me'] = annual_characteristics_df['me'] / 1000  # CRSP ME
# annual_characteristics_df['me'] = annual_characteristics_df['mve_f']  # Compustat ME

# there are some ME equal to zero since this company do not have price or shares data, we drop these observations
annual_characteristics_df['me'] = np.where(annual_characteristics_df['me'] == 0, np.nan, annual_characteristics_df['me'])
annual_characteristics_df = annual_characteristics_df.dropna(subset=['me']).reset_index(drop=True)

# count single stock years
annual_characteristics_df['count'] = annual_characteristics_df.groupby(['gvkey']).cumcount() + 1

# deal with the duplicates
annual_characteristics_df.loc[annual_characteristics_df.groupby(['datadate', 'permno', 'linkprim'], as_index=False).nth([0]).index, 'temp'] = 1
annual_characteristics_df = annual_characteristics_df[annual_characteristics_df['temp'].notna()].reset_index(drop=True)

annual_characteristics_df.loc[annual_characteristics_df.groupby(['permno', 'yearend', 'datadate'], as_index=False).nth([-1]).index, 'temp'] = 1
annual_characteristics_df = annual_characteristics_df[annual_characteristics_df['temp'].notna()].reset_index(drop=True)

annual_characteristics_df = annual_characteristics_df.sort_values(by=['permno', 'jdate']).reset_index(drop=True)

# fama-french 49 industry
annual_characteristics_df['sic'] = annual_characteristics_df['sic'].astype(int)
annual_characteristics_df['ffi49'] = ffi49(annual_characteristics_df)
annual_characteristics_df['ffi49'] = annual_characteristics_df['ffi49'].fillna(49)
annual_characteristics_df['ffi49'] = annual_characteristics_df['ffi49'].astype(int)
#######################################################################################################################
#                                                  Annual Variables                                                   #
#######################################################################################################################
# preferrerd stock
annual_characteristics_df['ps'] = np.where(annual_characteristics_df['pstkrv'].isnull(), annual_characteristics_df['pstkl'], annual_characteristics_df['pstkrv'])
annual_characteristics_df['ps'] = np.where(annual_characteristics_df['ps'].isnull(), annual_characteristics_df['pstk'], annual_characteristics_df['ps'])
annual_characteristics_df['ps'] = np.where(annual_characteristics_df['ps'].isnull(), 0, annual_characteristics_df['ps'])

annual_characteristics_df['txditc'] = annual_characteristics_df['txditc'].fillna(0)

# book equity
annual_characteristics_df['be'] = annual_characteristics_df['seq'] + annual_characteristics_df['txditc'] - annual_characteristics_df['ps']
annual_characteristics_df['be'] = np.where(annual_characteristics_df['be'] > 0, annual_characteristics_df['be'], np.nan)

# acc
annual_characteristics_df['act_l1'] = annual_characteristics_df.groupby(['permno'])['act'].shift(1)
annual_characteristics_df['lct_l1'] = annual_characteristics_df.groupby(['permno'])['lct'].shift(1)
annual_characteristics_df['at_l1'] = annual_characteristics_df.groupby(['permno'])['at'].shift(1)

# #################### Add np lagged_df (also fixed row 272 below) on 2025.02.23 ####################
# annual_characteristics_df['np_l1'] = annual_characteristics_df.groupby(['permno'])['np'].shift(1) 

# condition_list = [annual_characteristics_df['np'].isnull(),
#             annual_characteristics_df['act'].isnull() | annual_characteristics_df['lct'].isnull()]
# choice_list = [((annual_characteristics_df['act'] - annual_characteristics_df['lct']) - (annual_characteristics_df['act_l1'] - annual_characteristics_df['lct_l1']) / (annual_characteristics_df['be'])),
#               (annual_characteristics_df['ib'] - annual_characteristics_df['oancf']) / (annual_characteristics_df['be'])] ##### Delete "10*" on 2025.02.26 #####
# annual_characteristics_df['acc'] = np.select(condition_list,
#                              choice_list,
#                              default=((annual_characteristics_df['act'] - annual_characteristics_df['lct'] + annual_characteristics_df['np']) -
#                                       (annual_characteristics_df['act_l1'] - annual_characteristics_df['lct_l1'] + annual_characteristics_df['np_l1'])) / (annual_characteristics_df['be']))

#################### Add Sloan(1996) or HXZ and GHZ operating accruals on 2025.02.28 ####################
annual_characteristics_df['che_l1'] = annual_characteristics_df.groupby(['permno'])['che'].shift(1)
annual_characteristics_df['dlc_l1'] = annual_characteristics_df.groupby(['permno'])['dlc'].shift(1)
annual_characteristics_df['txp_l1'] = annual_characteristics_df.groupby(['permno'])['txp'].shift(1)

annual_characteristics_df['acc'] = np.where(annual_characteristics_df['oancf'].isnull(),
                            ((annual_characteristics_df['act'] - annual_characteristics_df['act_l1']) - (annual_characteristics_df['che'] - annual_characteristics_df['che_l1']) -
                            (annual_characteristics_df['lct'] - annual_characteristics_df['lct_l1']) + (annual_characteristics_df['dlc'] - annual_characteristics_df['dlc_l1']) +
                            (annual_characteristics_df['txp'] - annual_characteristics_df['txp_l1']).fillna(0) - annual_characteristics_df['dp']) / ((annual_characteristics_df['at'] + annual_characteristics_df['at_l1']) / 2),
                            (annual_characteristics_df['ib'] - annual_characteristics_df['oancf']) / ((annual_characteristics_df['at'] + annual_characteristics_df['at_l1']) / 2))

# absacc
annual_characteristics_df['absacc'] = abs(annual_characteristics_df['acc'])

# agr
annual_characteristics_df['agr'] = (annual_characteristics_df['at'] - annual_characteristics_df['at_l1']) / annual_characteristics_df['at_l1']

# bm
# annual_characteristics_df['bm'] = annual_characteristics_df['be'] / annual_characteristics_df['me']

# cfp
# condition_list = [annual_characteristics_df['dp'].isnull(),
#             annual_characteristics_df['ib'].isnull()]
# choice_list = [annual_characteristics_df['ib']/annual_characteristics_df['me'],
#               np.nan]
# annual_characteristics_df['cfp'] = np.select(condition_list, choice_list, default=(annual_characteristics_df['ib']+annual_characteristics_df['dp'])/annual_characteristics_df['me'])

# ep
# annual_characteristics_df['ep'] = annual_characteristics_df['ib']/annual_characteristics_df['me']

# ni
annual_characteristics_df['csho_l1'] = annual_characteristics_df.groupby(['permno'])['csho'].shift(1)
annual_characteristics_df['ajex_l1'] = annual_characteristics_df.groupby(['permno'])['ajex'].shift(1)
annual_characteristics_df['ni'] = np.where(annual_characteristics_df['gvkey'] != annual_characteristics_df['gvkey'].shift(1),
                           np.nan,
                           np.log(annual_characteristics_df['csho'] * annual_characteristics_df['ajex']).replace(-np.inf, 0) -
                           np.log(annual_characteristics_df['csho_l1'] * annual_characteristics_df['ajex_l1']).replace(-np.inf, 0))

# op
annual_characteristics_df['cogs0'] = np.where(annual_characteristics_df['cogs'].isnull(), 0, annual_characteristics_df['cogs'])
annual_characteristics_df['xint0'] = np.where(annual_characteristics_df['xint'].isnull(), 0, annual_characteristics_df['xint'])
annual_characteristics_df['xsga0'] = np.where(annual_characteristics_df['xsga'].isnull(), 0, annual_characteristics_df['xsga'])

condition_list = [annual_characteristics_df['revt'].isnull(), annual_characteristics_df['be'].isnull()]
choice_list = [np.nan, np.nan]
annual_characteristics_df['op'] = np.select(condition_list, choice_list,
                            default=(annual_characteristics_df['revt'] - annual_characteristics_df['cogs0'] - annual_characteristics_df['xsga0'] - annual_characteristics_df['xint0']) / annual_characteristics_df['be'])

# rsup
annual_characteristics_df['sale_l1'] = annual_characteristics_df.groupby(['permno'])['sale'].shift(1)
# annual_characteristics_df['rsup'] = (annual_characteristics_df['sale']-annual_characteristics_df['sale_l1'])/annual_characteristics_df['me']

# cash
annual_characteristics_df['cash'] = annual_characteristics_df['che'] / annual_characteristics_df['at']

# lev
# annual_characteristics_df['lev'] = annual_characteristics_df['lt']/annual_characteristics_df['me']

# sp
# annual_characteristics_df['sp'] = annual_characteristics_df['sale']/annual_characteristics_df['me']

# rd_sale
annual_characteristics_df['xrd0'] = np.where(annual_characteristics_df['xrd'].isnull(), 0, annual_characteristics_df['xrd'])
annual_characteristics_df['rd_sale'] = annual_characteristics_df['xrd0']/annual_characteristics_df['sale']

# rdm
# annual_characteristics_df['rdm'] = annual_characteristics_df['xrd']/annual_characteristics_df['me']

# adm hxz adm
# annual_characteristics_df['adm'] = annual_characteristics_df['xad']/annual_characteristics_df['me']

# gma
annual_characteristics_df['gma'] = (annual_characteristics_df['revt'] - annual_characteristics_df['cogs']) / annual_characteristics_df['at_l1']

# chcsho
annual_characteristics_df['chcsho'] = (annual_characteristics_df['csho'] / annual_characteristics_df['csho_l1']) - 1

# lgr
annual_characteristics_df['lt_l1'] = annual_characteristics_df.groupby(['permno'])['lt'].shift(1)
annual_characteristics_df['lgr'] = (annual_characteristics_df['lt'] / annual_characteristics_df['lt_l1']) - 1

#################### Follow Hafzalla, Lundholm, and Van Winkle (2011) and GHZ on 2025.02.28 ####################
# pctacc
condition_list = [annual_characteristics_df['ib'] == 0,
            annual_characteristics_df['oancf'].isnull(),
            annual_characteristics_df['oancf'].isnull() & annual_characteristics_df['ib'] == 0]
choice_list = [(annual_characteristics_df['ib'] - annual_characteristics_df['oancf']) / 0.01,
              ((annual_characteristics_df['act'] - annual_characteristics_df['act_l1']) - (annual_characteristics_df['che'] - annual_characteristics_df['che_l1'])) -
              ((annual_characteristics_df['lct'] - annual_characteristics_df['lct_l1']) - (annual_characteristics_df['dlc']) - annual_characteristics_df['dlc_l1'] -
               ((annual_characteristics_df['txp'] - annual_characteristics_df['txp_l1']).fillna(0) - annual_characteristics_df['dp'])) / annual_characteristics_df['ib'].abs(),
              ((annual_characteristics_df['act'] - annual_characteristics_df['act_l1']) - (annual_characteristics_df['che'] - annual_characteristics_df['che_l1'])) -
              ((annual_characteristics_df['lct'] - annual_characteristics_df['lct_l1']) - (annual_characteristics_df['dlc']) - annual_characteristics_df['dlc_l1'] -
               ((annual_characteristics_df['txp'] - annual_characteristics_df['txp_l1']).fillna(0) - annual_characteristics_df['dp'])) / 0.01]
annual_characteristics_df['pctacc'] = np.select(condition_list, choice_list,
                                default=(annual_characteristics_df['ib'] - annual_characteristics_df['oancf']) / annual_characteristics_df['ib'].abs())

# sgr
annual_characteristics_df['sgr'] = (annual_characteristics_df['sale'] / annual_characteristics_df['sale_l1']) - 1

# chato
annual_characteristics_df['at_l2'] = annual_characteristics_df.groupby(['permno'])['at'].shift(2)
annual_characteristics_df['chato'] = (annual_characteristics_df['sale'] / ((annual_characteristics_df['at'] + annual_characteristics_df['at_l1']) / 2)) - \
                     (annual_characteristics_df['sale_l1'] / ((annual_characteristics_df['at'] + annual_characteristics_df['at_l2']) / 2))

# chtx
annual_characteristics_df['txt_l1'] = annual_characteristics_df.groupby(['permno'])['txt'].shift(1)
annual_characteristics_df['chtx'] = (annual_characteristics_df['txt'] - annual_characteristics_df['txt_l1']) / annual_characteristics_df['at_l1']

# noa
annual_characteristics_df['noa'] = ((annual_characteristics_df['at'] - annual_characteristics_df['che'] - annual_characteristics_df['ivao'].fillna(0)) -
                    (annual_characteristics_df['at'] - annual_characteristics_df['dlc'].fillna(0) - annual_characteristics_df['dltt'].fillna(0) - annual_characteristics_df['mib'].fillna(0)
                     - annual_characteristics_df['pstk'].fillna(0) - annual_characteristics_df['ceq']) / annual_characteristics_df['at_l1'])

# rna
annual_characteristics_df['noa_l1'] = annual_characteristics_df.groupby(['permno'])['noa'].shift(1)
annual_characteristics_df['rna'] = annual_characteristics_df['oiadp'] / annual_characteristics_df['noa_l1']

# pm
annual_characteristics_df['pm'] = annual_characteristics_df['oiadp'] / annual_characteristics_df['sale']

# ato
annual_characteristics_df['ato'] = annual_characteristics_df['sale'] / annual_characteristics_df['noa_l1']

# depr
annual_characteristics_df['depr'] = annual_characteristics_df['dp'] / annual_characteristics_df['ppent']

# invest
annual_characteristics_df['ppent_l1'] = annual_characteristics_df.groupby(['permno'])['ppent'].shift(1)
annual_characteristics_df['invt_l1'] = annual_characteristics_df.groupby(['permno'])['invt'].shift(1)

annual_characteristics_df['invest'] = np.where(annual_characteristics_df['ppegt'].isnull(), ((annual_characteristics_df['ppent'] - annual_characteristics_df['ppent_l1']) +
                                                             (annual_characteristics_df['invt'] - annual_characteristics_df['invt_l1'])) / annual_characteristics_df['at_l1'],
                               ((annual_characteristics_df['ppegt'] - annual_characteristics_df['ppent_l1']) + (annual_characteristics_df['invt'] - annual_characteristics_df['invt_l1'])) / annual_characteristics_df['at_l1'])

# egr
annual_characteristics_df['ceq_l1'] = annual_characteristics_df.groupby(['permno'])['ceq'].shift(1)
annual_characteristics_df['egr'] = ((annual_characteristics_df['ceq'] - annual_characteristics_df['ceq_l1']) / annual_characteristics_df['ceq_l1'])

# cashdebt
annual_characteristics_df['cashdebt'] = (annual_characteristics_df['ib'] + annual_characteristics_df['dp']) / ((annual_characteristics_df['lt'] + annual_characteristics_df['lt_l1']) / 2)

# rd
# if ((xrd/at)-(lagged_df(xrd/lagged_df(at))))/(lagged_df(xrd/lagged_df(at))) >.05 then rd=1 else rd=0
annual_characteristics_df['xrd/at_l1'] = annual_characteristics_df['xrd0']/annual_characteristics_df['at_l1']
annual_characteristics_df['xrd/at_l1_l1'] = annual_characteristics_df.groupby(['permno'])['xrd/at_l1'].shift(1)
annual_characteristics_df['rd'] = np.where(((annual_characteristics_df['xrd0']/annual_characteristics_df['at'])-
                            (annual_characteristics_df['xrd/at_l1_l1']))/annual_characteristics_df['xrd/at_l1_l1']>0.05, 1, 0)

# roa
annual_characteristics_df['roa'] = annual_characteristics_df['ib'] / annual_characteristics_df['at_l1'] ##### Debug on 2025.02.23 #####

# roe
annual_characteristics_df['roe'] = annual_characteristics_df['ib'] / annual_characteristics_df['ceq_l1']

# dy
# annual_characteristics_df['dy'] = annual_characteristics_df['dvt']/annual_characteristics_df['me']

# roic
annual_characteristics_df['roic'] = (annual_characteristics_df['ebit'] - annual_characteristics_df['nopi']) / (annual_characteristics_df['ceq'] + annual_characteristics_df['lt'] - annual_characteristics_df['che'])

# chinv
annual_characteristics_df['chinv'] = (annual_characteristics_df['invt'] - annual_characteristics_df['invt_l1']) / ((annual_characteristics_df['at'] + annual_characteristics_df['at_l2']) / 2)

# pchsale_pchinvt
annual_characteristics_df['pchsale_pchinvt'] = ((annual_characteristics_df['sale'] - annual_characteristics_df['sale_l1']) / annual_characteristics_df['sale_l1']) \
                               - ((annual_characteristics_df['invt'] - annual_characteristics_df['invt_l1']) / annual_characteristics_df['invt_l1'])

# pchsale_pchrect
annual_characteristics_df['rect_l1'] = annual_characteristics_df.groupby(['permno'])['rect'].shift(1)
annual_characteristics_df['pchsale_pchrect'] = ((annual_characteristics_df['sale'] - annual_characteristics_df['sale_l1']) / annual_characteristics_df['sale_l1']) \
                               - ((annual_characteristics_df['rect'] - annual_characteristics_df['rect_l1']) / annual_characteristics_df['rect_l1'])

# pchgm_pchsale
annual_characteristics_df['cogs_l1'] = annual_characteristics_df.groupby(['permno'])['cogs'].shift(1)
annual_characteristics_df['pchgm_pchsale'] = (((annual_characteristics_df['sale'] - annual_characteristics_df['cogs'])
                               - (annual_characteristics_df['sale_l1'] - annual_characteristics_df['cogs_l1'])) / (annual_characteristics_df['sale_l1'] - annual_characteristics_df['cogs_l1'])) \
                             - ((annual_characteristics_df['sale'] - annual_characteristics_df['sale_l1']) / annual_characteristics_df['sale'])

# pchsale_pchxsga
annual_characteristics_df['xsga_l1'] = annual_characteristics_df.groupby(['permno'])['xsga'].shift(1)
annual_characteristics_df['pchsale_pchxsga'] = ((annual_characteristics_df['sale'] - annual_characteristics_df['sale_l1']) / annual_characteristics_df['sale_l1']) \
                               - ((annual_characteristics_df['xsga'] - annual_characteristics_df['xsga_l1']) / annual_characteristics_df['xsga_l1'])

# pchdepr
annual_characteristics_df['dp_l1'] = annual_characteristics_df.groupby(['permno'])['dp'].shift(1)
annual_characteristics_df['pchdepr'] = ((annual_characteristics_df['dp'] / annual_characteristics_df['ppent']) - (annual_characteristics_df['dp_l1']
                                                                  / annual_characteristics_df['ppent_l1'])) \
                       / (annual_characteristics_df['dp_l1'] / annual_characteristics_df['ppent'])

# chadv
annual_characteristics_df['xad_l1'] = annual_characteristics_df.groupby(['permno'])['xad'].shift(1)
annual_characteristics_df['chadv'] = np.log(annual_characteristics_df['xad'] + 1) - np.log(annual_characteristics_df['xad_l1'] + 1)

# pchcapx
annual_characteristics_df['capx_l1'] = annual_characteristics_df.groupby(['permno'])['capx'].shift(1)
annual_characteristics_df['pchcapx'] = (annual_characteristics_df['capx'] - annual_characteristics_df['capx_l1']) / annual_characteristics_df['capx_l1']

# grcapx
annual_characteristics_df['capx_l2'] = annual_characteristics_df.groupby(['permno'])['capx'].shift(2)
annual_characteristics_df['grcapx'] = (annual_characteristics_df['capx'] - annual_characteristics_df['capx_l2']) / annual_characteristics_df['capx_l2']

# grGW
annual_characteristics_df['gdwl_l1'] = annual_characteristics_df.groupby(['permno'])['gdwl'].shift(1)
annual_characteristics_df['grGW'] = (annual_characteristics_df['gdwl'] - annual_characteristics_df['gdwl_l1']) / annual_characteristics_df['gdwl']
condition_list = [(annual_characteristics_df['gdwl'] == 0) | (annual_characteristics_df['gdwl'].isnull()),
            (annual_characteristics_df['gdwl'].notna()) & (annual_characteristics_df['gdwl'] != 0) & (annual_characteristics_df['grGW'].isnull())]
choice_list = [0, 1]
annual_characteristics_df['grGW'] = np.select(condition_list, choice_list, default=annual_characteristics_df['grGW'])

# currat
annual_characteristics_df['currat'] = annual_characteristics_df['act'] / annual_characteristics_df['lct']

# pchcurrat
annual_characteristics_df['pchcurrat'] = ((annual_characteristics_df['act'] / annual_characteristics_df['lct']) - (annual_characteristics_df['act_l1'] / annual_characteristics_df['lct_l1'])) \
                         / (annual_characteristics_df['act_l1'] / annual_characteristics_df['lct_l1'])

# quick
annual_characteristics_df['quick'] = (annual_characteristics_df['act'] - annual_characteristics_df['invt']) / annual_characteristics_df['lct']

# pchquick
annual_characteristics_df['pchquick'] = ((annual_characteristics_df['act'] - annual_characteristics_df['invt']) / annual_characteristics_df['lct']
                         - (annual_characteristics_df['act_l1'] - annual_characteristics_df['invt_l1']) / annual_characteristics_df['lct_l1']) \
                        / ((annual_characteristics_df['act_l1'] - annual_characteristics_df['invt_l1']) / annual_characteristics_df['lct_l1'])

# salecash
annual_characteristics_df['salecash'] = annual_characteristics_df['sale'] / annual_characteristics_df['che']

# salerec
annual_characteristics_df['salerec'] = annual_characteristics_df['sale'] / annual_characteristics_df['rect']

# saleinv
annual_characteristics_df['saleinv'] = annual_characteristics_df['sale'] / annual_characteristics_df['invt']

# pchsaleinv
annual_characteristics_df['pchsaleinv'] = ((annual_characteristics_df['sale'] / annual_characteristics_df['invt']) - (annual_characteristics_df['sale_l1'] / annual_characteristics_df['invt_l1'])) \
                          / (annual_characteristics_df['sale_l1'] / annual_characteristics_df['invt_l1'])

# realestate
annual_characteristics_df['realestate'] = (annual_characteristics_df['fatb'] + annual_characteristics_df['fatl']) / annual_characteristics_df['ppegt']
annual_characteristics_df['realestate'] = np.where(annual_characteristics_df['ppegt'].isnull(),
                                   (annual_characteristics_df['fatb'] + annual_characteristics_df['fatl']) / annual_characteristics_df['ppent'], annual_characteristics_df['realestate'])

# obklg
annual_characteristics_df['obklg'] = annual_characteristics_df['ob'] / ((annual_characteristics_df['at'] + annual_characteristics_df['at_l1']) / 2)

# chobklg
annual_characteristics_df['ob_l1'] = annual_characteristics_df.groupby(['permno'])['ob'].shift(1)
annual_characteristics_df['chobklg'] = (annual_characteristics_df['ob'] - annual_characteristics_df['ob_l1']) / ((annual_characteristics_df['at'] + annual_characteristics_df['at_l1']) / 2)

# grltnoa
annual_characteristics_df['aco_l1'] = annual_characteristics_df.groupby(['permno'])['aco'].shift(1)
annual_characteristics_df['intan_l1'] = annual_characteristics_df.groupby(['permno'])['intan'].shift(1)
annual_characteristics_df['ao_l1'] = annual_characteristics_df.groupby(['permno'])['ao'].shift(1)
annual_characteristics_df['ap_l1'] = annual_characteristics_df.groupby(['permno'])['ap'].shift(1)
annual_characteristics_df['lco_l1'] = annual_characteristics_df.groupby(['permno'])['lco'].shift(1)
annual_characteristics_df['lo_l1'] = annual_characteristics_df.groupby(['permno'])['lo'].shift(1)
annual_characteristics_df['rect_l1'] = annual_characteristics_df.groupby(['permno'])['rect'].shift(1)

annual_characteristics_df['grltnoa'] = ((annual_characteristics_df['rect']+annual_characteristics_df['invt']+annual_characteristics_df['ppent']+annual_characteristics_df['aco']+annual_characteristics_df['intan']+
                       annual_characteristics_df['ao']-annual_characteristics_df['ap']-annual_characteristics_df['lco']-annual_characteristics_df['lo'])
                        -(annual_characteristics_df['rect_l1']+annual_characteristics_df['invt_l1']+annual_characteristics_df['ppent_l1']+annual_characteristics_df['aco_l1']
                       +annual_characteristics_df['intan_l1']+annual_characteristics_df['ao_l1']-annual_characteristics_df['ap_l1']-annual_characteristics_df['lco_l1']
                       -annual_characteristics_df['lo_l1'])
                        -(annual_characteristics_df['rect']-annual_characteristics_df['rect_l1']+annual_characteristics_df['invt']-annual_characteristics_df['invt_l1']
                          +annual_characteristics_df['aco']-annual_characteristics_df['aco_l1']
                          -(annual_characteristics_df['ap']-annual_characteristics_df['ap_l1']+annual_characteristics_df['lco']-annual_characteristics_df['lco_l1'])-annual_characteristics_df['dp']))\
                       /((annual_characteristics_df['at']+annual_characteristics_df['at_l1'])/2)

# conv
annual_characteristics_df['conv'] = annual_characteristics_df['dc']/annual_characteristics_df['dltt']

# convind
annual_characteristics_df['convind'] = np.where(((annual_characteristics_df['dc'].notna()) & (annual_characteristics_df['dc'] != 0)) | ((annual_characteristics_df['cshrc'].notna()) & (annual_characteristics_df['cshrc'] != 0)), 1, 0)

# chdrc
annual_characteristics_df['dr_l1'] = annual_characteristics_df.groupby(['permno'])['dr'].shift(1)
annual_characteristics_df['chdrc'] = (annual_characteristics_df['dr']-annual_characteristics_df['dr_l1'])/((annual_characteristics_df['at']+annual_characteristics_df['at_l1'])/2)

# rdbias
annual_characteristics_df['xrd_l1'] = annual_characteristics_df.groupby(['permno'])['xrd0'].shift(1)
annual_characteristics_df['rdbias'] = (annual_characteristics_df['xrd0']/annual_characteristics_df['xrd_l1'])-1-annual_characteristics_df['ib']/annual_characteristics_df['ceq_l1']

# operprof
annual_characteristics_df['operprof'] = (annual_characteristics_df['revt']-annual_characteristics_df['cogs']-annual_characteristics_df['xsga0']-annual_characteristics_df['xint0'])/annual_characteristics_df['ceq_l1']

# cfroa
annual_characteristics_df['cfroa'] = annual_characteristics_df['oancf']/((annual_characteristics_df['at']+annual_characteristics_df['at_l1'])/2)
annual_characteristics_df['cfroa'] = np.where(annual_characteristics_df['oancf'].isnull(),
                              (annual_characteristics_df['ib'] + annual_characteristics_df['dp'])/((annual_characteristics_df['at']+annual_characteristics_df['at_l1'])/2),
                              annual_characteristics_df['cfroa'])

# xrdint
annual_characteristics_df['xrdint'] = annual_characteristics_df['xrd0']/((annual_characteristics_df['at']+annual_characteristics_df['at_l1'])/2)

# capxint
annual_characteristics_df['capxint'] = annual_characteristics_df['capx']/((annual_characteristics_df['at']+annual_characteristics_df['at_l1'])/2)

# xadint
annual_characteristics_df['xadint'] = annual_characteristics_df['xad']/((annual_characteristics_df['at']+annual_characteristics_df['at_l1'])/2)

# chpm
annual_characteristics_df['ib_l1'] = annual_characteristics_df.groupby(['permno'])['ib'].shift(1)
annual_characteristics_df['chpm'] = (annual_characteristics_df['ib']/annual_characteristics_df['sale'])-(annual_characteristics_df['ib_l1']/annual_characteristics_df['sale_l1'])

# ala
annual_characteristics_df['gdwl'] = np.where(annual_characteristics_df['gdwl'].isnull(), 0, annual_characteristics_df['gdwl'])
annual_characteristics_df['intan'] = np.where(annual_characteristics_df['intan'].isnull(), 0, annual_characteristics_df['intan'])
annual_characteristics_df['ala'] = annual_characteristics_df['che']+0.75*(annual_characteristics_df['act']-annual_characteristics_df['che'])-\
                   0.5*(annual_characteristics_df['at']-annual_characteristics_df['act']-annual_characteristics_df['gdwl']-annual_characteristics_df['intan'])

# alm
annual_characteristics_df['alm'] = annual_characteristics_df['ala']/(annual_characteristics_df['at']+annual_characteristics_df['prcc_f']*annual_characteristics_df['csho']-annual_characteristics_df['ceq'])

# hire
annual_characteristics_df['emp_l1'] = annual_characteristics_df.groupby(['permno'])['emp'].shift(1)
annual_characteristics_df['hire'] = (annual_characteristics_df['emp'] - annual_characteristics_df['emp_l1'])/annual_characteristics_df['emp_l1']
annual_characteristics_df['hire'] = np.where((annual_characteristics_df['emp'].isnull()) | (annual_characteristics_df['emp_l1'].isnull()), 0, annual_characteristics_df['hire'])

# herf
industry_temp_df = annual_characteristics_df.groupby(['datadate', 'ffi49'], as_index=False)['sale'].sum()
industry_temp_df = industry_temp_df.rename(columns={'sale': 'indsale'})
annual_characteristics_df = pd.merge(annual_characteristics_df, industry_temp_df, how='left', on=['datadate', 'ffi49']).reset_index(drop=True)
annual_characteristics_df['herf'] = (annual_characteristics_df['sale']/annual_characteristics_df['indsale'])*(annual_characteristics_df['sale']/annual_characteristics_df['indsale'])
industry_temp_df = annual_characteristics_df.groupby(['datadate', 'ffi49'], as_index=False)['herf'].sum()
annual_characteristics_df = annual_characteristics_df.drop(['herf'], axis=1)
annual_characteristics_df = pd.merge(annual_characteristics_df, industry_temp_df, how='left', on=['datadate', 'ffi49']).reset_index(drop=True)

# age
annual_characteristics_df['age'] = annual_characteristics_df['count'].copy()

# cashpr
# annual_characteristics_df['cashpr'] = ((annual_characteristics_df['me'] + annual_characteristics_df['dltt'] - annual_characteristics_df['at']) / annual_characteristics_df['che'])

# chempia
industry_temp_df = annual_characteristics_df.groupby(['datadate', 'ffi49'], as_index=False)['hire'].mean()
industry_temp_df = industry_temp_df.rename(columns={'hire': 'hire_ind'})
annual_characteristics_df = pd.merge(annual_characteristics_df, industry_temp_df, how='left', on=['datadate', 'ffi49']).reset_index(drop=True)
annual_characteristics_df['chempia'] = annual_characteristics_df['hire'] - annual_characteristics_df['hire_ind']

# chpmia
industry_temp_df = annual_characteristics_df.groupby(['datadate', 'ffi49'], as_index=False)['chpm'].mean()
industry_temp_df = industry_temp_df.rename(columns={'chpm': 'chpm_ind'})
annual_characteristics_df = pd.merge(annual_characteristics_df, industry_temp_df, how='left', on=['datadate', 'ffi49']).reset_index(drop=True)
annual_characteristics_df['chpmia'] = annual_characteristics_df['chpm'] - annual_characteristics_df['chpm_ind']

# chatoia
industry_temp_df = annual_characteristics_df.groupby(['datadate', 'ffi49'], as_index=False)['chato'].mean()
industry_temp_df = industry_temp_df.rename(columns={'chato': 'chato_ind'})
annual_characteristics_df = pd.merge(annual_characteristics_df, industry_temp_df, how='left', on=['datadate', 'ffi49']).reset_index(drop=True)
annual_characteristics_df['chatoia'] = annual_characteristics_df['chato'] - annual_characteristics_df['chato_ind']

# divi
annual_characteristics_df['dvt_l1'] = annual_characteristics_df.groupby(['permno'])['dvt'].shift(1)
annual_characteristics_df['divi'] = np.where(((annual_characteristics_df['dvt'].notna()) & (annual_characteristics_df['dvt'] > 0) & ((annual_characteristics_df['dvt_l1'] == 0) | (annual_characteristics_df['dvt_l1'].isnull()))), 1, 0)

# divo
annual_characteristics_df['divo'] = np.where(((annual_characteristics_df['dvt'].isnull()) | (annual_characteristics_df['dvt'] == 0) & ((annual_characteristics_df['dvt_l1'] > 0) | (annual_characteristics_df['dvt_l1'].notna()))), 1, 0)

# Mohanram (2005) score (Annual Related)
industry_temp_df = annual_characteristics_df.groupby(['fyear', 'ffi49'], as_index=False)['roa'].median().rename(columns={'roa': 'md_roa'})
annual_characteristics_df = pd.merge(annual_characteristics_df, industry_temp_df, how='left', on=['fyear', 'ffi49']).reset_index(drop=True)

industry_temp_df = annual_characteristics_df.groupby(['fyear', 'ffi49'], as_index=False)['cfroa'].median().rename(columns={'cfroa': 'md_cfroa'})
annual_characteristics_df = pd.merge(annual_characteristics_df, industry_temp_df, how='left', on=['fyear', 'ffi49']).reset_index(drop=True)

industry_temp_df = annual_characteristics_df.groupby(['fyear', 'ffi49'], as_index=False)['oancf'].median().rename(columns={'oancf': 'md_oancf'})
annual_characteristics_df = pd.merge(annual_characteristics_df, industry_temp_df, how='left', on=['fyear', 'ffi49']).reset_index(drop=True)

industry_temp_df = annual_characteristics_df.groupby(['fyear', 'ffi49'], as_index=False)['xrdint'].median().rename(columns={'xrdint': 'md_xrdint'})
annual_characteristics_df = pd.merge(annual_characteristics_df, industry_temp_df, how='left', on=['fyear', 'ffi49']).reset_index(drop=True)

industry_temp_df = annual_characteristics_df.groupby(['fyear', 'ffi49'], as_index=False)['capxint'].median().rename(columns={'capxint': 'md_capxint'})
annual_characteristics_df = pd.merge(annual_characteristics_df, industry_temp_df, how='left', on=['fyear', 'ffi49']).reset_index(drop=True)

industry_temp_df = annual_characteristics_df.groupby(['fyear', 'ffi49'], as_index=False)['xadint'].median().rename(columns={'xadint': 'md_xadint'})
annual_characteristics_df = pd.merge(annual_characteristics_df, industry_temp_df, how='left', on=['fyear', 'ffi49']).reset_index(drop=True)

annual_characteristics_df['m1'] = np.where(annual_characteristics_df['roa'] > annual_characteristics_df['md_roa'], 1, 0)
annual_characteristics_df['m2'] = np.where(annual_characteristics_df['cfroa'] > annual_characteristics_df['md_cfroa'], 1, 0)
annual_characteristics_df['m3'] = np.where(annual_characteristics_df['oancf'] > annual_characteristics_df['md_oancf'], 1, 0)
annual_characteristics_df['m4'] = np.where(annual_characteristics_df['xrdint'] > annual_characteristics_df['md_xrdint'], 1, 0)
annual_characteristics_df['m5'] = np.where(annual_characteristics_df['capxint'] > annual_characteristics_df['md_capxint'], 1, 0)
annual_characteristics_df['m6'] = np.where(annual_characteristics_df['xadint'] > annual_characteristics_df['md_xadint'], 1, 0)

# pchcapx_ia
industry_temp_df = annual_characteristics_df.groupby(['datadate', 'ffi49'], as_index=False)['pchcapx'].mean()
industry_temp_df = industry_temp_df.rename(columns={'pchcapx': 'pchcapx_ind'})
annual_characteristics_df = pd.merge(annual_characteristics_df, industry_temp_df, how='left', on=['datadate', 'ffi49']).reset_index(drop=True)
annual_characteristics_df['pchcapx_ia'] = annual_characteristics_df['pchcapx'] - annual_characteristics_df['pchcapx_ind']

# secured
annual_characteristics_df['secured'] = annual_characteristics_df['dm'] / annual_characteristics_df['dltt']

# securedind
annual_characteristics_df['securedind'] = np.where((annual_characteristics_df['dm'].notna()) & (annual_characteristics_df['dm'] != 0), 1, 0)

# sin
annual_characteristics_df['sin'] = np.where(((2100 <= annual_characteristics_df['sic']) & (annual_characteristics_df['sic'] <= 2199)) |
                            ((2080 <= annual_characteristics_df['sic']) & (annual_characteristics_df['sic'] <= 2085)) |
                            (annual_characteristics_df['naics'] == '7132') |
                            (annual_characteristics_df['naics'] == '71312') |
                            (annual_characteristics_df['naics'] == '713210') |
                            (annual_characteristics_df['naics'] == '71329') |
                            (annual_characteristics_df['naics'] == '713290') |
                            (annual_characteristics_df['naics'] == '72112') |
                            (annual_characteristics_df['naics'] == '721120'), 1, 0)

# tang
annual_characteristics_df['tang'] = (annual_characteristics_df['che'] + annual_characteristics_df['rect'] * 0.715 + annual_characteristics_df['invt'] * 0.547 + annual_characteristics_df['ppent'] * 0.535) / annual_characteristics_df['at']

# tb, Lev and Nissim (2004)
condition_list = [annual_characteristics_df['fyear'] <= 1978,
            (1979 <= annual_characteristics_df['fyear']) & (annual_characteristics_df['fyear'] <= 1986),
            annual_characteristics_df['fyear'] == 1987,
            (1988 <= annual_characteristics_df['fyear']) & (annual_characteristics_df['fyear'] <= 1992),
            1993 <= annual_characteristics_df['fyear']]
choice_list = [0.48, 0.46, 0.4, 0.34, 0.35]
annual_characteristics_df['tr'] = np.select(condition_list, choice_list, np.nan)

annual_characteristics_df['tb_1'] = ((annual_characteristics_df['txfo'] + annual_characteristics_df['txfed']) / annual_characteristics_df['tr']) / annual_characteristics_df['ib']
annual_characteristics_df['tb_1'] = np.where((annual_characteristics_df['txfo'].isnull()) | (annual_characteristics_df['txfed'].isnull()),
                             ((annual_characteristics_df['txt'] - annual_characteristics_df['txdi']) / annual_characteristics_df['tr']) / annual_characteristics_df['ib'],
                             annual_characteristics_df['tb_1'])
annual_characteristics_df['tb_1'] = np.where((((annual_characteristics_df['txfo'] + annual_characteristics_df['txfed'] > 0) | (annual_characteristics_df['txt'] > annual_characteristics_df['txdi'])) & annual_characteristics_df['ib'] <= 0), 1, annual_characteristics_df['tb_1'])

industry_temp_df = annual_characteristics_df.groupby(['datadate', 'ffi49'], as_index=False)['tb_1'].mean()
industry_temp_df = industry_temp_df.rename(columns={'tb_1': 'tb_1_ind'})
annual_characteristics_df = pd.merge(annual_characteristics_df, industry_temp_df, how='left', on=['datadate', 'ffi49'])
annual_characteristics_df['tb'] = annual_characteristics_df['tb_1'] - annual_characteristics_df['tb_1_ind']

print("Finish Annual Variables Calculation! \n")

# opa from Ball el al. (2016)
condition_list = [annual_characteristics_df['revt'].isnull(), annual_characteristics_df['at'].isnull()]
choice_list = [np.nan, np.nan]
annual_characteristics_df['opa'] = np.select(condition_list, choice_list,
                          default=(annual_characteristics_df['revt'] - annual_characteristics_df['cogs0'] - annual_characteristics_df['xsga0'] + annual_characteristics_df['xrd0'])/annual_characteristics_df['at'])

# cop from Ball el al. (2016)
annual_characteristics_df['xpp_l1'] = annual_characteristics_df.groupby(['permno'])['xpp'].shift(1)
annual_characteristics_df['xacc_l1'] = annual_characteristics_df.groupby(['permno'])['xacc'].shift(1)

condition_list = [annual_characteristics_df['revt'].isnull(), annual_characteristics_df['at'].isnull()]
choice_list = [np.nan, np.nan]
annual_characteristics_df['cop'] = np.select(condition_list, choice_list,
                          default=(annual_characteristics_df['opa']
                                   - (annual_characteristics_df['rect']-annual_characteristics_df['rect_l1']) 
                                   - (annual_characteristics_df['invt']-annual_characteristics_df['invt_l1'])
                                   - (annual_characteristics_df['xpp']-annual_characteristics_df['xpp_l1'])
                                   + (annual_characteristics_df['drc']+annual_characteristics_df['drlt'])
                                   + (annual_characteristics_df['ap'] - annual_characteristics_df['ap_l1'])
                                   + (annual_characteristics_df['xacc']-annual_characteristics_df['xacc_l1']))/annual_characteristics_df['at'])

#######################################################################################################################
#                                              Compustat Quarterly Raw Info                                           #
#######################################################################################################################
compustat_df = wrds_connection.raw_sql("""
                    /*header info*/
                    select c.gvkey, f.cusip, f.datadate, f.fyearq,  substr(c.sic,1,2) as sic2, c.sic, f.fqtr, f.rdq,

                    /*income statement*/
                    f.ibq, f.saleq, f.txtq, f.revtq, f.cogsq, f.xsgaq, f.revty, f.cogsy, f.saley,

                    /*balance sheet items*/
                    f.atq, f.actq, f.cheq, f.lctq, f.dlcq, f.ppentq, f.ppegtq, f.txpq, f.drcq, f.drltq, f.xaccq,

                    /*others*/
                    abs(f.prccq) as prccq, abs(f.prccq)*f.cshoq as mveq_f, f.ceqq, f.seqq, f.pstkq, f.ltq,
                    f.pstkrq, f.gdwlq, f.intanq, f.mibq, f.oiadpq, f.ivaoq, f.conm,
                    
                    /* v3 my formula add*/
                    f.ajexq, f.cshoq, f.txditcq, f.npq, f.xrdy, f.xrdq, f.dpq, f.xintq, f.invtq, f.scstkcy, f.niq,
                    f.oancfy, f.dlttq, f.rectq, f.acoq, f.apq, f.lcoq, f.loq, f.aoq

                    from compustat_df.fundq as f
                    left join compustat_df.company as c
                    on f.gvkey = c.gvkey

                    /*get consolidated, standardized, industrial format statements*/
                    where f.indfmt = 'INDL' 
                    and f.datafmt = 'STD'
                    and f.popsrc = 'D'
                    and f.consol = 'C'
                    and f.datadate >= '01/01/2015'
                    """)

# compustat_df['cusip6'] = compustat_df['cusip'].str.strip().str[0:6]
compustat_df = compustat_df.dropna(subset=['ibq']).reset_index(drop=True)

# sort and clean up
compustat_df = compustat_df.sort_values(by=['gvkey', 'datadate']).drop_duplicates().reset_index(drop=True)
compustat_df['cshoq'] = np.where(compustat_df['cshoq'] == 0, np.nan, compustat_df['cshoq'])
compustat_df['ceqq'] = np.where(compustat_df['ceqq'] == 0, np.nan, compustat_df['ceqq'])
compustat_df['atq'] = np.where(compustat_df['atq'] == 0, np.nan, compustat_df['atq'])
compustat_df = compustat_df.dropna(subset=['atq']).reset_index(drop=True)

# convert datadate to date fmt
compustat_df['datadate'] = pd.to_datetime(compustat_df['datadate'])

# merge ccm_link_df and compustat_df
# Lag rule: Following Hou, Xue and Zhang (2015), We use earnings immediately after the announcement day
# For those data with missing announcement date record, we straightly let the data available after 4 month
ccm_merged_df = pd.merge(compustat_df, ccm_link_df, how='left', on=['gvkey']).reset_index(drop=True)
ccm_merged_df['yearend'] = ccm_merged_df['datadate'] + YearEnd(0)
ccm_merged_df['jdate'] = ccm_merged_df['datadate'] + MonthEnd(4)  # we change quarterly lagged_df here

# deal with ibq to make it as up-to-date as possible
ccm_merged_df['rdq'] = pd.to_datetime(ccm_merged_df['rdq']) + MonthEnd(0)
ccm_merged_df['rdq'] = np.where(ccm_merged_df['rdq'].isnull(), ccm_merged_df['jdate'], ccm_merged_df['rdq'])
ccm_merged_df['rdq_temp'] = ccm_merged_df.groupby(['permno'])['rdq'].shift(-1)  # compare next quarter's announcement date with jdate
ccm_merged_df['rdq_temp'] = np.where(ccm_merged_df['rdq_temp'].isnull(), ccm_merged_df['jdate'], ccm_merged_df['rdq_temp'])  # if rdq is NaN, let it be jdate
ccm_merged_df['ibq_diff'] = ccm_merged_df['jdate'] - ccm_merged_df['rdq_temp']  # compare next quarter's announcement date with jdate
ccm_merged_df['ibq_diff'] = ccm_merged_df['ibq_diff'].dt.days
ccm_merged_df['ibq_new'] = ccm_merged_df.groupby(['permno'])['ibq'].shift(-1)  # next quarter's ibq
ccm_merged_df = ccm_merged_df.rename(columns={'ibq': 'ibq_old'})  # original ibq
'''
if the announcement date is same or in front of jdate, we can use the up-to-date ibq.
otherwise, we consider the up-to-date ibq is not available and still use the lagged_df-4-months ibq
'''
ccm_merged_df['ibq'] = np.where(ccm_merged_df['ibq_diff'] >= 0, ccm_merged_df['ibq_new'], ccm_merged_df['ibq_old'])
ccm_merged_df['ibq'] = np.where(ccm_merged_df['ibq'].isnull(), ccm_merged_df['ibq_old'], ccm_merged_df['ibq'])  # for most recent record we can only use the lagged_df-4-months ibq

# set link date bounds
ccm_linked_df = ccm_merged_df[(ccm_merged_df['jdate'] >= ccm_merged_df['linkdt']) & (ccm_merged_df['jdate'] <= ccm_merged_df['linkenddt'])].reset_index(drop=True)

# merge ccm_linked_df and crsp_aligned_market_df
# crsp_aligned_market_df['jdate'] = crsp_aligned_market_df['monthend']
quarterly_characteristics_df = pd.merge(crsp_aligned_market_df, ccm_linked_df, how='inner', on=['permno', 'jdate']).reset_index(drop=True)

# filter exchcd & shrcd and at least one year data after the IPO
quarterly_characteristics_df = quarterly_characteristics_df[((quarterly_characteristics_df['exchcd'] == 1) | (quarterly_characteristics_df['exchcd'] == 2) | (quarterly_characteristics_df['exchcd'] == 3)) &
                      ((quarterly_characteristics_df['shrcd'] == 10) | (quarterly_characteristics_df['shrcd'] == 11))].reset_index(drop=True)

# process Market Equity
'''
Note: me is CRSP market equity, mveq_f is Compustat market equity. Please choose the me below.
'''
quarterly_characteristics_df['me'] = quarterly_characteristics_df['me'] / 1000  # CRSP ME
# quarterly_characteristics_df['me'] = quarterly_characteristics_df['mveq_f']  # Compustat ME

# there are some ME equal to zero since this company do not have price or shares data, we drop these observations
quarterly_characteristics_df['me'] = np.where(quarterly_characteristics_df['me'] == 0, np.nan, quarterly_characteristics_df['me'])
quarterly_characteristics_df = quarterly_characteristics_df.dropna(subset=['me']).reset_index(drop=True)

# deal with the duplicates
quarterly_characteristics_df.loc[quarterly_characteristics_df.groupby(['datadate', 'permno', 'linkprim'], as_index=False).nth([0]).index, 'temp'] = 1
quarterly_characteristics_df = quarterly_characteristics_df[quarterly_characteristics_df['temp'].notna()].reset_index(drop=True)

quarterly_characteristics_df.loc[quarterly_characteristics_df.groupby(['permno', 'yearend', 'datadate'], as_index=False).nth([-1]).index, 'temp'] = 1
quarterly_characteristics_df = quarterly_characteristics_df[quarterly_characteristics_df['temp'].notna()].reset_index(drop=True)

quarterly_characteristics_df = quarterly_characteristics_df.sort_values(by=['permno', 'jdate']).reset_index(drop=True)

# add industry code for quarterly data
quarterly_characteristics_df = quarterly_characteristics_df.dropna(subset=['sic']).reset_index(drop=True)  # gvkey 039750 does not have sic
quarterly_characteristics_df['sic'] = quarterly_characteristics_df['sic'].astype(int)
quarterly_characteristics_df['ffi49'] = ffi49(quarterly_characteristics_df)
quarterly_characteristics_df['ffi49'] = quarterly_characteristics_df['ffi49'].fillna(49)
quarterly_characteristics_df['ffi49'] = quarterly_characteristics_df['ffi49'].astype(int)
#######################################################################################################################
#                                                   Quarterly Variables                                               #
#######################################################################################################################
# prepare be
quarterly_characteristics_df['beq'] = np.where(quarterly_characteristics_df['seqq']>0, quarterly_characteristics_df['seqq']+quarterly_characteristics_df['txditcq']-quarterly_characteristics_df['pstkq'], np.nan)
quarterly_characteristics_df['beq'] = np.where(quarterly_characteristics_df['beq']<=0, np.nan, quarterly_characteristics_df['beq'])

# dy
# quarterly_characteristics_df['me_l1'] = quarterly_characteristics_df.groupby(['permno'])['me'].shift(1)
# quarterly_characteristics_df['retdy'] = quarterly_characteristics_df['ret'] - quarterly_characteristics_df['retx']
# quarterly_characteristics_df['mdivpay'] = quarterly_characteristics_df['retdy']*quarterly_characteristics_df['me_l1']
#
# quarterly_characteristics_df['dy'] = compute_ttm12(column_name='mdivpay', df=quarterly_characteristics_df)/quarterly_characteristics_df['me']

# chtx
quarterly_characteristics_df['txtq_l4'] = quarterly_characteristics_df.groupby(['permno'])['txtq'].shift(4)
quarterly_characteristics_df['atq_l4'] = quarterly_characteristics_df.groupby(['permno'])['atq'].shift(4)
quarterly_characteristics_df['chtx'] = (quarterly_characteristics_df['txtq']-quarterly_characteristics_df['txtq_l4'])/quarterly_characteristics_df['atq_l4']

# roa
quarterly_characteristics_df['atq_l1'] = quarterly_characteristics_df.groupby(['permno'])['atq'].shift(1)
quarterly_characteristics_df['roa'] = quarterly_characteristics_df['ibq']/quarterly_characteristics_df['atq_l1']

# cash
quarterly_characteristics_df['cash'] = quarterly_characteristics_df['cheq']/quarterly_characteristics_df['atq']

# acc
quarterly_characteristics_df['actq_l4'] = quarterly_characteristics_df.groupby(['permno'])['actq'].shift(4)
quarterly_characteristics_df['lctq_l4'] = quarterly_characteristics_df.groupby(['permno'])['lctq'].shift(4)

quarterly_characteristics_df['cheq_l4'] = quarterly_characteristics_df.groupby(['permno'])['cheq'].shift(4)
quarterly_characteristics_df['dlcq_l4'] = quarterly_characteristics_df.groupby(['permno'])['dlcq'].shift(4)
quarterly_characteristics_df['txpq_l4'] = quarterly_characteristics_df.groupby(['permno'])['txpq'].shift(4)

quarterly_characteristics_df['acc'] = np.where(quarterly_characteristics_df['oancfy'].isnull(),
                            ((quarterly_characteristics_df['actq'] - quarterly_characteristics_df['actq_l4']) - (quarterly_characteristics_df['cheq'] - quarterly_characteristics_df['cheq_l4']) -
                            (quarterly_characteristics_df['lctq'] - quarterly_characteristics_df['lctq_l4']) + (quarterly_characteristics_df['dlcq'] - quarterly_characteristics_df['dlcq_l4']) +
                            (quarterly_characteristics_df['txpq'] - quarterly_characteristics_df['txpq_l4']).fillna(0) - quarterly_characteristics_df['dpq']) / ((quarterly_characteristics_df['atq'] + quarterly_characteristics_df['atq_l4']) / 2),
                            (quarterly_characteristics_df['ibq'] - quarterly_characteristics_df['oancfy']) / ((quarterly_characteristics_df['atq'] + quarterly_characteristics_df['atq_l4']) / 2))

# absacc
quarterly_characteristics_df['absacc'] = abs(quarterly_characteristics_df['acc'])

# bm
# quarterly_characteristics_df['bm'] = quarterly_characteristics_df['beq']/quarterly_characteristics_df['me']

# cfp
quarterly_characteristics_df['ibq4'] = compute_ttm4('ibq', quarterly_characteristics_df)
quarterly_characteristics_df['dpq4'] = compute_ttm4('dpq', quarterly_characteristics_df)
# quarterly_characteristics_df['cfp'] = np.where(quarterly_characteristics_df['dpq'].isnull(),
#                             quarterly_characteristics_df['ibq4']/quarterly_characteristics_df['me'],
#                             (quarterly_characteristics_df['ibq4']+quarterly_characteristics_df['dpq4'])/quarterly_characteristics_df['me'])

# ep
# quarterly_characteristics_df['ep'] = quarterly_characteristics_df['ibq4']/quarterly_characteristics_df['me']

# agr
quarterly_characteristics_df['agr'] = (quarterly_characteristics_df['atq']-quarterly_characteristics_df['atq_l4'])/quarterly_characteristics_df['atq_l4']

# ni
quarterly_characteristics_df['cshoq_l4'] = quarterly_characteristics_df.groupby(['permno'])['cshoq'].shift(4)
quarterly_characteristics_df['ajexq_l4'] = quarterly_characteristics_df.groupby(['permno'])['ajexq'].shift(4)
quarterly_characteristics_df['ni'] = np.where(quarterly_characteristics_df['cshoq'].isnull(), np.nan,
                         np.log(quarterly_characteristics_df['cshoq']*quarterly_characteristics_df['ajexq']).replace(-np.inf, 0)-np.log(quarterly_characteristics_df['cshoq_l4']*quarterly_characteristics_df['ajexq_l4']))

# ope
quarterly_characteristics_df['xintq0'] = np.where(quarterly_characteristics_df['xintq'].isnull(), 0, quarterly_characteristics_df['xintq'])
quarterly_characteristics_df['xsgaq0'] = np.where(quarterly_characteristics_df['xsgaq'].isnull(), 0, quarterly_characteristics_df['xsgaq'])
quarterly_characteristics_df['cogsq0'] = np.where(quarterly_characteristics_df['cogsq'].isnull(), 0, quarterly_characteristics_df['cogsq'])
quarterly_characteristics_df['beq_l4'] = quarterly_characteristics_df.groupby(['permno'])['beq'].shift(4)

quarterly_characteristics_df['ope'] = (compute_ttm4('revtq', quarterly_characteristics_df)-compute_ttm4('cogsq0', quarterly_characteristics_df)-compute_ttm4('xsgaq0', quarterly_characteristics_df)-compute_ttm4('xintq0', quarterly_characteristics_df))/quarterly_characteristics_df['beq_l4']

# chcsho
quarterly_characteristics_df['chcsho'] = (quarterly_characteristics_df['cshoq']/quarterly_characteristics_df['cshoq_l4'])-1

# cashdebt
quarterly_characteristics_df['ltq_l4'] = quarterly_characteristics_df.groupby(['permno'])['ltq'].shift(4)
quarterly_characteristics_df['cashdebt'] = (compute_ttm4('ibq', quarterly_characteristics_df) + compute_ttm4('dpq', quarterly_characteristics_df))/((quarterly_characteristics_df['ltq']+quarterly_characteristics_df['ltq_l4'])/2)

# rd
quarterly_characteristics_df['xrdq4'] = compute_ttm4('xrdq', quarterly_characteristics_df)
quarterly_characteristics_df['xrdq4'] = np.where(quarterly_characteristics_df['xrdq4'].isnull(), quarterly_characteristics_df['xrdy'], quarterly_characteristics_df['xrdq4'])

quarterly_characteristics_df['xrdq4/atq_l4'] = quarterly_characteristics_df['xrdq4']/quarterly_characteristics_df['atq_l4']
quarterly_characteristics_df['xrdq4/atq_l4_l4'] = quarterly_characteristics_df.groupby(['permno'])['xrdq4/atq_l4'].shift(4)
quarterly_characteristics_df['rd'] = np.where(((quarterly_characteristics_df['xrdq4']/quarterly_characteristics_df['atq'])-quarterly_characteristics_df['xrdq4/atq_l4_l4'])/quarterly_characteristics_df['xrdq4/atq_l4_l4']>0.05, 1, 0)

#################### Follow Hafzalla, Lundholm, and Van Winkle (2011) and GHZ ####################


condition_list = [quarterly_characteristics_df['ibq'] == 0,
            quarterly_characteristics_df['oancfy'].isnull(),
            quarterly_characteristics_df['oancfy'].isnull() & quarterly_characteristics_df['ibq'] == 0]
choice_list = [(quarterly_characteristics_df['ibq'] - quarterly_characteristics_df['oancfy']) / 0.01,
              ((quarterly_characteristics_df['actq'] - quarterly_characteristics_df['actq_l4']) - (quarterly_characteristics_df['cheq'] - quarterly_characteristics_df['cheq_l4']) -
               (quarterly_characteristics_df['lctq'] - quarterly_characteristics_df['lctq_l4']) + (quarterly_characteristics_df['dlcq'] - quarterly_characteristics_df['dlcq_l4']) + 
               (quarterly_characteristics_df['txpq'] - quarterly_characteristics_df['txpq_l4']).fillna(0) - quarterly_characteristics_df['dpq']) / quarterly_characteristics_df['ibq'].abs(), 
               ((quarterly_characteristics_df['actq'] - quarterly_characteristics_df['actq_l4']) - (quarterly_characteristics_df['cheq'] - quarterly_characteristics_df['cheq_l4']) -
               (quarterly_characteristics_df['lctq'] - quarterly_characteristics_df['lctq_l4']) + (quarterly_characteristics_df['dlcq'] - quarterly_characteristics_df['dlcq_l4']) + 
               (quarterly_characteristics_df['txpq'] - quarterly_characteristics_df['txpq_l4']).fillna(0) - quarterly_characteristics_df['dpq']) / 0.01]
quarterly_characteristics_df['pctacc'] = np.select(condition_list, choice_list,
                                default=(quarterly_characteristics_df['ibq'] - quarterly_characteristics_df['oancfy']) / quarterly_characteristics_df['ibq'].abs())

# gma
quarterly_characteristics_df['revtq4'] = compute_ttm4('revtq', quarterly_characteristics_df)
quarterly_characteristics_df['cogsq4'] = compute_ttm4('cogsq', quarterly_characteristics_df)
quarterly_characteristics_df['gma'] = (quarterly_characteristics_df['revtq4']-quarterly_characteristics_df['cogsq4'])/quarterly_characteristics_df['atq_l4']

# lev
# quarterly_characteristics_df['lev'] = quarterly_characteristics_df['ltq']/quarterly_characteristics_df['me']

# rdm
# quarterly_characteristics_df['rdm'] = quarterly_characteristics_df['xrdq4']/quarterly_characteristics_df['me']

# sgr
quarterly_characteristics_df['saleq4'] = compute_ttm4('saleq', quarterly_characteristics_df)
quarterly_characteristics_df['saleq4'] = np.where(quarterly_characteristics_df['saleq4'].isnull(), quarterly_characteristics_df['saley'], quarterly_characteristics_df['saleq4'])

quarterly_characteristics_df['saleq4_l4'] = quarterly_characteristics_df.groupby(['permno'])['saleq4'].shift(4)
quarterly_characteristics_df['sgr'] = (quarterly_characteristics_df['saleq4']/quarterly_characteristics_df['saleq4_l4'])-1

# sp
# quarterly_characteristics_df['sp'] = quarterly_characteristics_df['saleq4']/quarterly_characteristics_df['me']

# invest
quarterly_characteristics_df['ppentq_l4'] = quarterly_characteristics_df.groupby(['permno'])['ppentq'].shift(4)
quarterly_characteristics_df['invtq_l4'] = quarterly_characteristics_df.groupby(['permno'])['invtq'].shift(4)
quarterly_characteristics_df['ppegtq_l4'] = quarterly_characteristics_df.groupby(['permno'])['ppegtq'].shift(4)

quarterly_characteristics_df['invest'] = np.where(quarterly_characteristics_df['ppegtq'].isnull(), ((quarterly_characteristics_df['ppentq']-quarterly_characteristics_df['ppentq_l4'])+
                                                            (quarterly_characteristics_df['invtq']-quarterly_characteristics_df['invtq_l4']))/quarterly_characteristics_df['atq_l4'],
                             ((quarterly_characteristics_df['ppegtq']-quarterly_characteristics_df['ppegtq_l4'])+(quarterly_characteristics_df['invtq']-quarterly_characteristics_df['invtq_l4']))/quarterly_characteristics_df['atq_l4'])

# rd_sale
quarterly_characteristics_df['rd_sale'] = quarterly_characteristics_df['xrdq4']/quarterly_characteristics_df['saleq4']

# lgr
quarterly_characteristics_df['lgr'] = (quarterly_characteristics_df['ltq']/quarterly_characteristics_df['ltq_l4'])-1

# depr
quarterly_characteristics_df['depr'] = compute_ttm4('dpq', quarterly_characteristics_df)/quarterly_characteristics_df['ppentq']

# egr
quarterly_characteristics_df['ceqq_l4'] = quarterly_characteristics_df.groupby(['permno'])['ceqq'].shift(4)
quarterly_characteristics_df['egr'] = (quarterly_characteristics_df['ceqq']-quarterly_characteristics_df['ceqq_l4'])/quarterly_characteristics_df['ceqq_l4']

# chpm
quarterly_characteristics_df['ibq4_l1'] = quarterly_characteristics_df.groupby(['permno'])['ibq4'].shift(1)
quarterly_characteristics_df['saleq4_l1'] = quarterly_characteristics_df.groupby(['permno'])['saleq4'].shift(1)

quarterly_characteristics_df['chpm'] = (quarterly_characteristics_df['ibq4']/quarterly_characteristics_df['saleq4'])-(quarterly_characteristics_df['ibq4_l1']/quarterly_characteristics_df['saleq4_l1'])

# chato
quarterly_characteristics_df['atq_l8'] = quarterly_characteristics_df.groupby(['permno'])['atq'].shift(8)
quarterly_characteristics_df['chato'] = (quarterly_characteristics_df['saleq4']/((quarterly_characteristics_df['atq']+quarterly_characteristics_df['atq_l4'])/2))-(quarterly_characteristics_df['saleq4_l4']/((quarterly_characteristics_df['atq_l4']+quarterly_characteristics_df['atq_l8'])/2))

# chatoia
industry_temp_df = quarterly_characteristics_df.groupby(['datadate', 'ffi49'], as_index=False)['chato'].mean()
industry_temp_df = industry_temp_df.rename(columns={'chato': 'chato_ind'})
quarterly_characteristics_df = pd.merge(quarterly_characteristics_df, industry_temp_df, how='left', on=['datadate', 'ffi49']).reset_index(drop=True)
quarterly_characteristics_df['chatoia'] = quarterly_characteristics_df['chato'] - quarterly_characteristics_df['chato_ind']

# noa
quarterly_characteristics_df['ivaoq'] = np.where(quarterly_characteristics_df['ivaoq'].isnull(), 0, 1)
quarterly_characteristics_df['dlcq'] = np.where(quarterly_characteristics_df['dlcq'].isnull(), 0, 1)
quarterly_characteristics_df['dlttq'] = np.where(quarterly_characteristics_df['dlttq'].isnull(), 0, 1)
quarterly_characteristics_df['mibq'] = np.where(quarterly_characteristics_df['mibq'].isnull(), 0, 1)
quarterly_characteristics_df['pstkq'] = np.where(quarterly_characteristics_df['pstkq'].isnull(), 0, 1)
quarterly_characteristics_df['noa'] = (quarterly_characteristics_df['atq']-quarterly_characteristics_df['cheq']-quarterly_characteristics_df['ivaoq'])-\
                 (quarterly_characteristics_df['atq']-quarterly_characteristics_df['dlcq']-quarterly_characteristics_df['dlttq']-quarterly_characteristics_df['mibq']-quarterly_characteristics_df['pstkq']-quarterly_characteristics_df['ceqq'])/quarterly_characteristics_df['atq_l4']

# rna
quarterly_characteristics_df['noa_l4'] = quarterly_characteristics_df.groupby(['permno'])['noa'].shift(4)
quarterly_characteristics_df['rna'] = quarterly_characteristics_df['oiadpq']/quarterly_characteristics_df['noa_l4']

# pm
quarterly_characteristics_df['pm'] = quarterly_characteristics_df['oiadpq']/quarterly_characteristics_df['saleq']

# ato
quarterly_characteristics_df['ato'] = quarterly_characteristics_df['saleq']/quarterly_characteristics_df['noa_l4']

# roe
quarterly_characteristics_df['ceqq_l1'] = quarterly_characteristics_df.groupby(['permno'])['ceqq'].shift(1)
quarterly_characteristics_df['roe'] = quarterly_characteristics_df['ibq']/quarterly_characteristics_df['ceqq_l1']

# grltnoa
quarterly_characteristics_df['rectq_l4'] = quarterly_characteristics_df.groupby(['permno'])['rectq'].shift(4)
quarterly_characteristics_df['acoq_l4'] = quarterly_characteristics_df.groupby(['permno'])['acoq'].shift(4)
quarterly_characteristics_df['apq_l4'] = quarterly_characteristics_df.groupby(['permno'])['apq'].shift(4)
quarterly_characteristics_df['lcoq_l4'] = quarterly_characteristics_df.groupby(['permno'])['lcoq'].shift(4)
quarterly_characteristics_df['loq_l4'] = quarterly_characteristics_df.groupby(['permno'])['loq'].shift(4)
quarterly_characteristics_df['invtq_l4'] = quarterly_characteristics_df.groupby(['permno'])['invtq'].shift(4)
quarterly_characteristics_df['ppentq_l4'] = quarterly_characteristics_df.groupby(['permno'])['ppentq'].shift(4)
quarterly_characteristics_df['atq_l4'] = quarterly_characteristics_df.groupby(['permno'])['atq'].shift(4)

quarterly_characteristics_df['grltnoa'] = ((quarterly_characteristics_df['rectq']+quarterly_characteristics_df['invtq']+quarterly_characteristics_df['ppentq']+quarterly_characteristics_df['acoq']+quarterly_characteristics_df['intanq']+
                       quarterly_characteristics_df['aoq']-quarterly_characteristics_df['apq']-quarterly_characteristics_df['lcoq']-quarterly_characteristics_df['loq'])-
                      (quarterly_characteristics_df['rectq_l4']+quarterly_characteristics_df['invtq_l4']+quarterly_characteristics_df['ppentq_l4']+quarterly_characteristics_df['acoq_l4']-quarterly_characteristics_df['apq_l4']-quarterly_characteristics_df['lcoq_l4']-quarterly_characteristics_df['loq_l4'])-\
                     (quarterly_characteristics_df['rectq']-quarterly_characteristics_df['rectq_l4']+quarterly_characteristics_df['invtq']-quarterly_characteristics_df['invtq_l4']+quarterly_characteristics_df['acoq']-
                      (quarterly_characteristics_df['apq']-quarterly_characteristics_df['apq_l4']+quarterly_characteristics_df['lcoq']-quarterly_characteristics_df['lcoq_l4'])-
                      compute_ttm4('dpq', quarterly_characteristics_df)))/((quarterly_characteristics_df['atq']+quarterly_characteristics_df['atq_l4'])/2)

# ala
quarterly_characteristics_df['gdwlq'] = np.where(quarterly_characteristics_df['gdwlq'].isnull(), 0, quarterly_characteristics_df['gdwlq'])
quarterly_characteristics_df['intanq'] = np.where(quarterly_characteristics_df['intanq'].isnull(), 0, quarterly_characteristics_df['intanq'])
quarterly_characteristics_df['ala'] = quarterly_characteristics_df['cheq'] + 0.75*(quarterly_characteristics_df['actq']-quarterly_characteristics_df['cheq'])+\
                 0.5*(quarterly_characteristics_df['atq']-quarterly_characteristics_df['actq']-quarterly_characteristics_df['gdwlq']-quarterly_characteristics_df['intanq'])

# rsup
quarterly_characteristics_df['saleq_l4'] = quarterly_characteristics_df.groupby(['permno'])['saleq'].shift(4)
# quarterly_characteristics_df['rsup'] = (quarterly_characteristics_df['saleq'] - quarterly_characteristics_df['saleq_l4'])/quarterly_characteristics_df['me']

# stdsacc
quarterly_characteristics_df['actq_l1'] = quarterly_characteristics_df.groupby(['permno'])['actq'].shift(1)
quarterly_characteristics_df['cheq_l1'] = quarterly_characteristics_df.groupby(['permno'])['cheq'].shift(1)
quarterly_characteristics_df['lctq_l1'] = quarterly_characteristics_df.groupby(['permno'])['lctq'].shift(1)
quarterly_characteristics_df['dlcq_l1'] = quarterly_characteristics_df.groupby(['permno'])['dlcq'].shift(1)

quarterly_characteristics_df['sacc'] = ((quarterly_characteristics_df['actq']-quarterly_characteristics_df['actq_l1'] - (quarterly_characteristics_df['cheq']-quarterly_characteristics_df['cheq_l1']))
                     -((quarterly_characteristics_df['lctq']-quarterly_characteristics_df['lctq_l1'])-(quarterly_characteristics_df['dlcq']-quarterly_characteristics_df['dlcq_l1'])))/quarterly_characteristics_df['saleq']
quarterly_characteristics_df['sacc'] = np.where(quarterly_characteristics_df['saleq']<=0, ((quarterly_characteristics_df['actq']-quarterly_characteristics_df['actq_l1'] - (quarterly_characteristics_df['cheq']-quarterly_characteristics_df['cheq_l1']))
                     -((quarterly_characteristics_df['lctq']-quarterly_characteristics_df['lctq_l1'])-(quarterly_characteristics_df['dlcq']-quarterly_characteristics_df['dlcq_l1'])))/0.01, quarterly_characteristics_df['sacc'])

def compute_chars_std(start_lag, end_lag, df, characteristic_name):
    """

    :param start_lag: Order of starting lagged_df
    :param end_lag: Order of ending lagged_df
    :param df: Dataframe
    :param characteristic_name: lagged_df characteristic_name
    :return: std of factor
    """
    lagged_df = pd.DataFrame()
    lag_column_list = []
    for i in range(start_lag, end_lag):
        lagged_df['chars_l%s' % i] = df.groupby(['permno'])['%s' % characteristic_name].shift(i)
        lag_column_list.append('chars_l%s' % i)
    computed_result = lagged_df[lag_column_list].std(axis=1)
    return computed_result

quarterly_characteristics_df['stdacc'] = compute_chars_std(0, 16, quarterly_characteristics_df, 'sacc')

# roavol
quarterly_characteristics_df['roavol'] = compute_chars_std(0, 16, quarterly_characteristics_df, 'roa')

# stdcf
quarterly_characteristics_df['scf'] = (quarterly_characteristics_df['ibq']/quarterly_characteristics_df['saleq']) - quarterly_characteristics_df['sacc']
quarterly_characteristics_df['scf'] = np.where(quarterly_characteristics_df['saleq']<=0, (quarterly_characteristics_df['ibq']/0.01) - quarterly_characteristics_df['sacc'], quarterly_characteristics_df['sacc'])

quarterly_characteristics_df['stdcf'] = compute_chars_std(0, 16, quarterly_characteristics_df, 'scf')

# cinvest
quarterly_characteristics_df['ppentq_l1'] = quarterly_characteristics_df.groupby(['permno'])['ppentq'].shift(1)
quarterly_characteristics_df['ppentq_l2'] = quarterly_characteristics_df.groupby(['permno'])['ppentq'].shift(2)
quarterly_characteristics_df['ppentq_l3'] = quarterly_characteristics_df.groupby(['permno'])['ppentq'].shift(3)
quarterly_characteristics_df['ppentq_l4'] = quarterly_characteristics_df.groupby(['permno'])['ppentq'].shift(4)
quarterly_characteristics_df['saleq_l1'] = quarterly_characteristics_df.groupby(['permno'])['saleq'].shift(1)
quarterly_characteristics_df['saleq_l2'] = quarterly_characteristics_df.groupby(['permno'])['saleq'].shift(2)
quarterly_characteristics_df['saleq_l3'] = quarterly_characteristics_df.groupby(['permno'])['saleq'].shift(3)

quarterly_characteristics_df['c_temp1'] = (quarterly_characteristics_df['ppentq_l1'] - quarterly_characteristics_df['ppentq_l2']) / quarterly_characteristics_df['saleq_l1']
quarterly_characteristics_df['c_temp2'] = (quarterly_characteristics_df['ppentq_l2'] - quarterly_characteristics_df['ppentq_l3']) / quarterly_characteristics_df['saleq_l2']
quarterly_characteristics_df['c_temp3'] = (quarterly_characteristics_df['ppentq_l3'] - quarterly_characteristics_df['ppentq_l4']) / quarterly_characteristics_df['saleq_l3']

quarterly_characteristics_df['cinvest'] = ((quarterly_characteristics_df['ppentq'] - quarterly_characteristics_df['ppentq_l1']) / quarterly_characteristics_df['saleq'])\
                       -(quarterly_characteristics_df[['c_temp1', 'c_temp2', 'c_temp3']].mean(axis=1))

quarterly_characteristics_df['c_temp1'] = (quarterly_characteristics_df['ppentq_l1'] - quarterly_characteristics_df['ppentq_l2']) / 0.01
quarterly_characteristics_df['c_temp2'] = (quarterly_characteristics_df['ppentq_l2'] - quarterly_characteristics_df['ppentq_l3']) / 0.01
quarterly_characteristics_df['c_temp3'] = (quarterly_characteristics_df['ppentq_l3'] - quarterly_characteristics_df['ppentq_l4']) / 0.01

quarterly_characteristics_df['cinvest'] = np.where(quarterly_characteristics_df['saleq']<=0, ((quarterly_characteristics_df['ppentq'] - quarterly_characteristics_df['ppentq_l1']) / 0.01)
                                -(quarterly_characteristics_df[['c_temp1', 'c_temp2', 'c_temp3']].mean(axis=1)), quarterly_characteristics_df['cinvest'])

quarterly_characteristics_df = quarterly_characteristics_df.drop(['c_temp1', 'c_temp2', 'c_temp3'], axis=1)

# nincr
quarterly_characteristics_df['ibq_l1'] = quarterly_characteristics_df.groupby(['permno'])['ibq'].shift(1)
quarterly_characteristics_df['ibq_l2'] = quarterly_characteristics_df.groupby(['permno'])['ibq'].shift(2)
quarterly_characteristics_df['ibq_l3'] = quarterly_characteristics_df.groupby(['permno'])['ibq'].shift(3)
quarterly_characteristics_df['ibq_l4'] = quarterly_characteristics_df.groupby(['permno'])['ibq'].shift(4)
quarterly_characteristics_df['ibq_l5'] = quarterly_characteristics_df.groupby(['permno'])['ibq'].shift(5)
quarterly_characteristics_df['ibq_l6'] = quarterly_characteristics_df.groupby(['permno'])['ibq'].shift(6)
quarterly_characteristics_df['ibq_l7'] = quarterly_characteristics_df.groupby(['permno'])['ibq'].shift(7)
quarterly_characteristics_df['ibq_l8'] = quarterly_characteristics_df.groupby(['permno'])['ibq'].shift(8)

quarterly_characteristics_df['nincr_temp1'] = np.where(quarterly_characteristics_df['ibq'] > quarterly_characteristics_df['ibq_l1'], 1, 0)
quarterly_characteristics_df['nincr_temp2'] = np.where(quarterly_characteristics_df['ibq_l1'] > quarterly_characteristics_df['ibq_l2'], 1, 0)
quarterly_characteristics_df['nincr_temp3'] = np.where(quarterly_characteristics_df['ibq_l2'] > quarterly_characteristics_df['ibq_l3'], 1, 0)
quarterly_characteristics_df['nincr_temp4'] = np.where(quarterly_characteristics_df['ibq_l3'] > quarterly_characteristics_df['ibq_l4'], 1, 0)
quarterly_characteristics_df['nincr_temp5'] = np.where(quarterly_characteristics_df['ibq_l4'] > quarterly_characteristics_df['ibq_l5'], 1, 0)
quarterly_characteristics_df['nincr_temp6'] = np.where(quarterly_characteristics_df['ibq_l5'] > quarterly_characteristics_df['ibq_l6'], 1, 0)
quarterly_characteristics_df['nincr_temp7'] = np.where(quarterly_characteristics_df['ibq_l6'] > quarterly_characteristics_df['ibq_l7'], 1, 0)
quarterly_characteristics_df['nincr_temp8'] = np.where(quarterly_characteristics_df['ibq_l7'] > quarterly_characteristics_df['ibq_l8'], 1, 0)

quarterly_characteristics_df['nincr'] = (quarterly_characteristics_df['nincr_temp1']
                      + (quarterly_characteristics_df['nincr_temp1']*quarterly_characteristics_df['nincr_temp2'])
                      + (quarterly_characteristics_df['nincr_temp1']*quarterly_characteristics_df['nincr_temp2']*quarterly_characteristics_df['nincr_temp3'])
                      + (quarterly_characteristics_df['nincr_temp1']*quarterly_characteristics_df['nincr_temp2']*quarterly_characteristics_df['nincr_temp3']*quarterly_characteristics_df['nincr_temp4'])
                      + (quarterly_characteristics_df['nincr_temp1']*quarterly_characteristics_df['nincr_temp2']*quarterly_characteristics_df['nincr_temp3']*quarterly_characteristics_df['nincr_temp4']*quarterly_characteristics_df['nincr_temp5'])
                      + (quarterly_characteristics_df['nincr_temp1']*quarterly_characteristics_df['nincr_temp2']*quarterly_characteristics_df['nincr_temp3']*quarterly_characteristics_df['nincr_temp4']*quarterly_characteristics_df['nincr_temp5']*quarterly_characteristics_df['nincr_temp6'])
                      + (quarterly_characteristics_df['nincr_temp1']*quarterly_characteristics_df['nincr_temp2']*quarterly_characteristics_df['nincr_temp3']*quarterly_characteristics_df['nincr_temp4']*quarterly_characteristics_df['nincr_temp5']*quarterly_characteristics_df['nincr_temp6']*quarterly_characteristics_df['nincr_temp7'])
                      + (quarterly_characteristics_df['nincr_temp1']*quarterly_characteristics_df['nincr_temp2']*quarterly_characteristics_df['nincr_temp3']*quarterly_characteristics_df['nincr_temp4']*quarterly_characteristics_df['nincr_temp5']*quarterly_characteristics_df['nincr_temp6']*quarterly_characteristics_df['nincr_temp7']*quarterly_characteristics_df['nincr_temp8']))

quarterly_characteristics_df = quarterly_characteristics_df.drop(['ibq_l1', 'ibq_l2', 'ibq_l3', 'ibq_l4', 'ibq_l5', 'ibq_l6', 'ibq_l7', 'ibq_l8', 'nincr_temp1',
                            'nincr_temp2', 'nincr_temp3', 'nincr_temp4', 'nincr_temp5', 'nincr_temp6', 'nincr_temp7',
                            'nincr_temp8'], axis=1)

# performance score
quarterly_characteristics_df['niq4'] = compute_ttm4(column_name='niq', df=quarterly_characteristics_df)
quarterly_characteristics_df['niq4_l4'] = quarterly_characteristics_df.groupby(['permno'])['niq4'].shift(4)
quarterly_characteristics_df['dlttq_l4'] = quarterly_characteristics_df.groupby(['permno'])['dlttq'].shift(4)
quarterly_characteristics_df['p_temp1'] = np.where(quarterly_characteristics_df['niq4']>0, 1, 0)
quarterly_characteristics_df['p_temp2'] = np.where(quarterly_characteristics_df['oancfy']>0, 1, 0)
quarterly_characteristics_df['p_temp3'] = np.where(quarterly_characteristics_df['niq4']/quarterly_characteristics_df['atq']>quarterly_characteristics_df['niq4_l4']/quarterly_characteristics_df['atq_l4'], 1, 0)
quarterly_characteristics_df['p_temp4'] = np.where(quarterly_characteristics_df['oancfy']>quarterly_characteristics_df['niq4'], 1, 0)
quarterly_characteristics_df['p_temp5'] = np.where(quarterly_characteristics_df['dlttq']/quarterly_characteristics_df['atq']<quarterly_characteristics_df['dlttq_l4']/quarterly_characteristics_df['atq_l4'], 1, 0)
quarterly_characteristics_df['p_temp6'] = np.where(quarterly_characteristics_df['actq']/quarterly_characteristics_df['lctq'] > quarterly_characteristics_df['actq_l4']/quarterly_characteristics_df['lctq_l4'], 1, 0)
quarterly_characteristics_df['cogsq4_l4'] = quarterly_characteristics_df.groupby(['permno'])['cogsq4'].shift(4)
quarterly_characteristics_df['p_temp7'] = np.where((quarterly_characteristics_df['saleq4']-quarterly_characteristics_df['cogsq4']/quarterly_characteristics_df['saleq4'])>(quarterly_characteristics_df['saleq4_l4']-quarterly_characteristics_df['cogsq4_l4']/quarterly_characteristics_df['saleq4_l4']), 1, 0)
quarterly_characteristics_df['p_temp8'] = np.where(quarterly_characteristics_df['saleq4']/quarterly_characteristics_df['atq']>quarterly_characteristics_df['saleq4_l4']/quarterly_characteristics_df['atq_l4'], 1, 0)
quarterly_characteristics_df['p_temp9'] = np.where(quarterly_characteristics_df['scstkcy']==0, 1, 0)

quarterly_characteristics_df['pscore'] = quarterly_characteristics_df['p_temp1']+quarterly_characteristics_df['p_temp2']+quarterly_characteristics_df['p_temp3']+quarterly_characteristics_df['p_temp4']\
                      +quarterly_characteristics_df['p_temp5']+quarterly_characteristics_df['p_temp6']+quarterly_characteristics_df['p_temp7']+quarterly_characteristics_df['p_temp8']\
                      +quarterly_characteristics_df['p_temp9']

quarterly_characteristics_df = quarterly_characteristics_df.drop(['p_temp1', 'p_temp2', 'p_temp3', 'p_temp4', 'p_temp5', 'p_temp6', 'p_temp7', 'p_temp8',
                            'p_temp9'], axis=1)


# opa from Ball el al. (2016)
quarterly_characteristics_df['opa'] = (compute_ttm4('revtq', quarterly_characteristics_df)-compute_ttm4('cogsq0', quarterly_characteristics_df)-compute_ttm4('xsgaq0', quarterly_characteristics_df)+quarterly_characteristics_df['xrdq4'])/quarterly_characteristics_df['atq_l4']

# cop from Ball el al. (2016)
# no quarterly xpp, borrow from annual data
quarterly_characteristics_df['year'] = quarterly_characteristics_df['jdate'].dt.year
annual_characteristics_df['year'] = annual_characteristics_df['jdate'].dt.year
annual_characteristics_df['jdate_a'] = annual_characteristics_df['jdate'].copy()
quarterly_characteristics_df = pd.merge(quarterly_characteristics_df, annual_characteristics_df[['permno', 'year', 'jdate_a', 'xpp', 'xpp_l1']], how='left', on=['permno', 'year'])

# if quarterly jdate is later than annual jdate, use the corresponding xpp, otherwise use the lagged_df-1 xpp (from one year before)
quarterly_characteristics_df['xpp'] = np.where(quarterly_characteristics_df['jdate'] > quarterly_characteristics_df['jdate_a'], quarterly_characteristics_df['xpp'], quarterly_characteristics_df['xpp_l1'])

quarterly_characteristics_df['xpp_l4'] = quarterly_characteristics_df.groupby(['permno'])['xpp'].shift(4)
quarterly_characteristics_df['xaccq_l4'] = quarterly_characteristics_df.groupby(['permno'])['xaccq'].shift(4)

quarterly_characteristics_df['cop'] = (quarterly_characteristics_df['opa'] - (quarterly_characteristics_df['rectq']-quarterly_characteristics_df['rectq_l4']) 
                                     - (quarterly_characteristics_df['invtq']-quarterly_characteristics_df['invtq_l4'])
                                     - (quarterly_characteristics_df['xpp']-quarterly_characteristics_df['xpp_l4'])
                                     + (quarterly_characteristics_df['drcq']+quarterly_characteristics_df['drltq'])
                                     + (quarterly_characteristics_df['apq'] - quarterly_characteristics_df['apq_l4'])
                                     + (quarterly_characteristics_df['xaccq']-quarterly_characteristics_df['xaccq_l4']))/quarterly_characteristics_df['atq']


#######################################################################################################################
#                                                       Momentum                                                      #
#######################################################################################################################
crsp_df = wrds_connection.raw_sql("""
                    select a.prc, a.ret, a.retx, a.shrout, a.vol, a.date, a.permno, a.permco
                    from crsp_df.msf as a
                    left join crsp_df.msenames as b
                    on a.permno=b.permno
                    and b.namedt<=a.date
                    and a.date<=b.nameendt
                    where a.date >= '01/01/2015'
                    and b.exchcd between 1 and 3
                    """)

crsp_df = crsp_df.dropna(subset=['ret', 'retx', 'prc']).reset_index(drop=True)

# change variable format to int
crsp_df[['permco', 'permno']] = crsp_df[['permco', 'permno']].astype(int)

# Line up date to be end_lag of month
crsp_df['date'] = pd.to_datetime(crsp_df['date'])
crsp_df['jdate'] = crsp_df['date'] + MonthEnd(0)  # set all the date to the standard end_lag date of month

crsp_df = crsp_df.dropna(subset=['prc']).reset_index(drop=True)
crsp_df['me'] = crsp_df['prc'].abs() * crsp_df['shrout']  # calculate market equity

# Aggregate Market Cap
'''
There are cases when the same firm (permco) has two or more securities (permno) at same date.
For the purpose of ME for the firm, we aggregated all ME for a given permco, date.
This aggregated ME will be assigned to the permno with the largest ME.
'''
# sum of me across different permno belonging to same permco a given date
crsp_market_equity_sum_df = crsp_df.groupby(['jdate', 'permco'])['me'].sum().reset_index()
# largest mktcap within a permco/date
crsp_market_equity_max_df = crsp_df.groupby(['jdate', 'permco'])['me'].max().reset_index()
# join by monthend/maxme to find the permno
crsp_largest_security_df = pd.merge(crsp_df, crsp_market_equity_max_df, how='inner', on=['jdate', 'permco', 'me']).reset_index(drop=True)
# drop me column and replace with the sum me
crsp_largest_security_df = crsp_largest_security_df.drop(['me'], axis=1)
# join with sum of me to get the correct market cap info
crsp_aligned_market_df = pd.merge(crsp_largest_security_df, crsp_market_equity_sum_df, how='inner', on=['jdate', 'permco']).reset_index(drop=True)
# sort by permno and date and also drop duplicates
crsp_aligned_market_df = crsp_aligned_market_df.sort_values(by=['permno', 'jdate']).drop_duplicates().reset_index(drop=True)

crsp_aligned_market_df['me'] = crsp_aligned_market_df['me']/1000 # CRSP ME in million unit

crsp_mom = crsp_aligned_market_df.copy()
crsp_mom = crsp_mom.sort_values(by=['permno', 'date']).reset_index(drop=True)

crsp_mom['permno'] = crsp_mom['permno'].astype(int)
crsp_mom['jdate'] = pd.to_datetime(crsp_mom['date']) + MonthEnd(0)
crsp_mom = crsp_mom.dropna(subset=['ret', 'retx', 'prc'])
crsp_mom = crsp_mom.sort_values(by=['permno', 'date'])

# add delisting return
dlret = wrds_connection.raw_sql("""
                     select permno, dlret, dlstdt 
                     from crsp_df.msedelist
                     """)

dlret.permno = dlret.permno.astype(int)
dlret['dlstdt'] = pd.to_datetime(dlret['dlstdt'])
dlret['jdate'] = dlret['dlstdt'] + MonthEnd(0)

# merge delisting return to crsp_df return
crsp_mom = pd.merge(crsp_mom, dlret, how='left', on=['permno', 'jdate']).reset_index(drop=True)
crsp_mom['dlret'] = crsp_mom['dlret'].fillna(0)
crsp_mom['ret'] = crsp_mom['ret'].fillna(0)
crsp_mom['retadj'] = (1 + crsp_mom['ret']) * (1 + crsp_mom['dlret']) - 1


def mom(start_lag, end_lag, df):
    """

    :param start_lag: Order of starting lagged_df
    :param end_lag: Order of ending lagged_df
    :param df: Dataframe
    :return: Momentum factor
    """
    lagged_df = pd.DataFrame()
    computed_result = 1
    for i in range(start_lag, end_lag):
        lagged_df['mom%s' % i] = df.groupby(['permno'])['ret'].shift(i)
        computed_result = computed_result * (1+lagged_df['mom%s' % i])
    computed_result = computed_result - 1
    return computed_result


def chmom(start_lag, end_lag, df):
    """

    :param start_lag: Order of starting lagged_df
    :param end_lag: Order of ending lagged_df
    :param df: Dataframe
    :return: Momentum factor
    """
    lagged_df = pd.DataFrame()
    result_first_half = 1
    result_second_half = 1
    for i in range(start_lag, end_lag):
        lagged_df['mom%s' % i] = df.groupby(['permno'])['ret'].shift(i)
        result_first_half = result_first_half * (1+lagged_df['mom%s' % i])
    lagged_df = pd.DataFrame()
    for i in range(start_lag + 6, end_lag + 6):
        lagged_df['mom%s' % i] = df.groupby(['permno'])['ret'].shift(i)
        result_second_half = result_second_half * (1 + lagged_df['mom%s' % i])
    result_first_half = result_first_half - 1
    result_second_half = result_second_half - 1
    computed_result = result_first_half - result_second_half
    return computed_result


crsp_mom['chmom'] = chmom(1, 12, crsp_mom)

crsp_mom['mom60m'] = mom(12, 60, crsp_mom)
crsp_mom['mom12m'] = mom(1, 12, crsp_mom)
crsp_mom['mom1m'] = crsp_mom['ret']
crsp_mom['mom6m'] = mom(1, 6, crsp_mom)
crsp_mom['mom36m'] = mom(12, 36, crsp_mom)
crsp_mom['seas1a'] = crsp_mom.groupby(['permno'])['ret'].shift(11)

crsp_mom['vol_l1'] = crsp_mom.groupby(['permno'])['vol'].shift(1)
crsp_mom['vol_l2'] = crsp_mom.groupby(['permno'])['vol'].shift(2)
crsp_mom['vol_l3'] = crsp_mom.groupby(['permno'])['vol'].shift(3)
crsp_mom['prc_l2'] = crsp_mom.groupby(['permno'])['prc'].shift(2)
crsp_mom['dolvol'] = np.log((crsp_mom['vol_l2']*100)*crsp_mom['prc_l2']).replace([np.inf, -np.inf], np.nan) 
crsp_mom['turn'] = ((crsp_mom['vol_l1']+crsp_mom['vol_l2']+crsp_mom['vol_l3'])/3/10)/crsp_mom['shrout']

# dy
crsp_mom['me_l1'] = crsp_mom.groupby(['permno'])['me'].shift(1)
crsp_mom['retdy'] = crsp_mom['ret'] - crsp_mom['retx']
crsp_mom['mdivpay'] = crsp_mom['retdy']*crsp_mom['me_l1']

crsp_mom['dy'] = compute_ttm12(column_name='mdivpay', df=crsp_mom)/crsp_mom['me']

# def moms(start_lag, end_lag, df):
#     """
#
#     :param start_lag: Order of starting lagged_df
#     :param end_lag: Order of ending lagged_df
#     :param df: Dataframe
#     :return: Momentum factor
#     """
#     lagged_df = pd.DataFrame()
#     computed_result = 1
#     for i in range(start_lag, end_lag):
#         lagged_df['moms%s' % i] = df.groupby['permno']['ret'].shift(i)
#         computed_result = computed_result + lagged_df['moms%s' % i]
#     computed_result = computed_result/11
#     return computed_result
#
#
# crsp_mom['moms12m'] = moms(1, 12, crsp_mom)

# populate the characteristic_name to monthly

# annual_characteristics_df
annual_characteristics_df = annual_characteristics_df.drop(['date', 'ret', 'retx', 'me', 'prc', 'shrout'], axis=1)
annual_characteristics_df = pd.merge(crsp_mom, annual_characteristics_df, how='left', on=['permno', 'jdate']).reset_index(drop=True)
annual_characteristics_df = annual_characteristics_df.sort_values(by=['permno', 'jdate']).reset_index(drop=True)
annual_characteristics_df['datadate'] = annual_characteristics_df.groupby(['permno'])['datadate'].fillna(method='ffill')
annual_characteristics_df[['permno1', 'datadate1']] = annual_characteristics_df[['permno', 'datadate']]  # avoid the bug of 'groupby' for py 3.8
annual_characteristics_df = annual_characteristics_df.groupby(['permno1', 'datadate1'], as_index=False).fillna(method='ffill')
annual_characteristics_df = annual_characteristics_df[((annual_characteristics_df['exchcd'] == 1) | (annual_characteristics_df['exchcd'] == 2) | (annual_characteristics_df['exchcd'] == 3)) &
                      ((annual_characteristics_df['shrcd'] == 10) | (annual_characteristics_df['shrcd'] == 11))].reset_index(drop=True)

# quarterly_characteristics_df
quarterly_characteristics_df = quarterly_characteristics_df.drop(['date', 'ret', 'retx', 'me', 'prc', 'shrout'], axis=1)
quarterly_characteristics_df = pd.merge(crsp_mom, quarterly_characteristics_df, how='left', on=['permno', 'jdate']).reset_index(drop=True)
quarterly_characteristics_df = quarterly_characteristics_df.sort_values(by=['permno', 'jdate']).reset_index(drop=True)
quarterly_characteristics_df['datadate'] = quarterly_characteristics_df.groupby(['permno'])['datadate'].fillna(method='ffill')
quarterly_characteristics_df[['permno1', 'datadate1']] = quarterly_characteristics_df[['permno', 'datadate']]  # avoid the bug of 'groupby' for py 3.8
quarterly_characteristics_df = quarterly_characteristics_df.groupby(['permno1', 'datadate1'], as_index=False).fillna(method='ffill')
quarterly_characteristics_df = quarterly_characteristics_df[((quarterly_characteristics_df['exchcd'] == 1) | (quarterly_characteristics_df['exchcd'] == 2) | (quarterly_characteristics_df['exchcd'] == 3)) &
                      ((quarterly_characteristics_df['shrcd'] == 10) | (quarterly_characteristics_df['shrcd'] == 11))].reset_index(drop=True)

#######################################################################################################################
#                                                    Monthly ME                                                       #
#######################################################################################################################

########################################
#                Annual                #
########################################

# bm
annual_characteristics_df['bm'] = annual_characteristics_df['be'] / annual_characteristics_df['me']

# bm_ia
industry_temp_df = annual_characteristics_df.groupby(['datadate', 'ffi49'], as_index=False)['bm'].mean()
industry_temp_df = industry_temp_df.rename(columns={'bm': 'bm_ind'})
annual_characteristics_df = pd.merge(annual_characteristics_df, industry_temp_df, how='left', on=['datadate', 'ffi49']).reset_index(drop=True)
annual_characteristics_df['bm_ia'] = annual_characteristics_df['bm'] - annual_characteristics_df['bm_ind']

# me_ia
industry_temp_df = annual_characteristics_df.groupby(['datadate', 'ffi49'], as_index=False)['me'].mean()
industry_temp_df = industry_temp_df.rename(columns={'me': 'me_ind'})
annual_characteristics_df = pd.merge(annual_characteristics_df, industry_temp_df, how='left', on=['datadate', 'ffi49']).reset_index(drop=True)
annual_characteristics_df['me_ia'] = annual_characteristics_df['me'] - annual_characteristics_df['me_ind']

# cfp
condition_list = [annual_characteristics_df['dp'].isnull(),
            annual_characteristics_df['ib'].isnull()]
choice_list = [annual_characteristics_df['ib']/annual_characteristics_df['me'],
              np.nan]
annual_characteristics_df['cfp'] = np.select(condition_list, choice_list, default=(annual_characteristics_df['ib']+annual_characteristics_df['dp'])/annual_characteristics_df['me'])

# cfp_ia
industry_temp_df = annual_characteristics_df.groupby(['datadate', 'ffi49'], as_index=False)['cfp'].mean()
industry_temp_df = industry_temp_df.rename(columns={'cfp': 'cfp_ind'})
annual_characteristics_df = pd.merge(annual_characteristics_df, industry_temp_df, how='left', on=['datadate', 'ffi49']).reset_index(drop=True)
annual_characteristics_df['cfp_ia'] = annual_characteristics_df['cfp'] - annual_characteristics_df['cfp_ind']

# ep
annual_characteristics_df['ep'] = annual_characteristics_df['ib']/annual_characteristics_df['me']

# rsup
# annual_characteristics_df['sale_l1'] = annual_characteristics_df.groupby(['permno'])['sale'].shift(1)
annual_characteristics_df['rsup'] = (annual_characteristics_df['sale']-annual_characteristics_df['sale_l1'])/annual_characteristics_df['me']

# lev
annual_characteristics_df['lev'] = annual_characteristics_df['lt']/annual_characteristics_df['me']

# sp
annual_characteristics_df['sp'] = annual_characteristics_df['sale']/annual_characteristics_df['me']

# rdm
annual_characteristics_df['rdm'] = annual_characteristics_df['xrd']/annual_characteristics_df['me']

# adm hxz adm
annual_characteristics_df['adm'] = annual_characteristics_df['xad']/annual_characteristics_df['me']

# dy
annual_characteristics_df['dy'] = annual_characteristics_df['dvt']/annual_characteristics_df['me']

# cashpr
annual_characteristics_df['cashpr'] = ((annual_characteristics_df['me'] + annual_characteristics_df['dltt'] - annual_characteristics_df['at']) / annual_characteristics_df['che'])

# indmom
industry_temp_df = annual_characteristics_df.groupby(['date', 'ffi49'], as_index=False)['mom12m'].mean().rename(columns={'mom12m': 'indmom'})
annual_characteristics_df = pd.merge(annual_characteristics_df, industry_temp_df, how='left', on=['date', 'ffi49']).reset_index(drop=True)

# Annual Accounting Variables
annual_chars_df = annual_characteristics_df[['cusip', 'ncusip', 'gvkey', 'permno', 'exchcd', 'shrcd', 'datadate', 'jdate', 'ticker', 'conm', 'comnam', 'prc', 'shrout',
                     'sic', 'ret', 'retx', 'retadj', 'acc', 'agr', 'bm', 'cfp', 'ep', 'ni', 'op',
                     'rsup', 'cash', 'chcsho',
                     'rd', 'cashdebt', 'pctacc', 'gma', 'lev', 'rdm', 'adm', 'sgr', 'sp', 'invest', 'roe',
                     'rd_sale', 'lgr', 'roa', 'depr', 'egr', 'chato', 'chtx', 'noa', 'rna', 'pm', 'ato', 'dy',
                     'roic', 'chinv', 'pchsale_pchinvt', 'pchsale_pchrect', 'pchgm_pchsale', 'pchsale_pchxsga',
                     'pchdepr', 'chadv', 'pchcapx', 'grcapx', 'grGW', 'currat', 'pchcurrat', 'quick', 'pchquick',
                     'salecash', 'salerec', 'saleinv', 'pchsaleinv', 'realestate', 'obklg', 'chobklg', 'grltnoa',
                     'conv', 'chdrc', 'rdbias', 'operprof', 'capxint', 'xadint', 'chpm', 'ala', 'alm',
                     'mom1m', 'mom6m', 'mom12m', 'mom60m', 'mom36m', 'seas1a', 'me', 'hire', 'herf', 'bm_ia',
                     'me_ia', 'turn', 'dolvol', 'absacc', 'age', 'cashpr', 'chatoia', 'chempia', 'chmom', 'chpmia',
                     'convind', 'divi', 'divo', 'secured', 'securedind', 'sin', 'cfp_ia', 'indmom', 'pchcapx_ia',
                     'tang', 'tb', 'm1', 'm2', 'm3', 'm4', 'm5', 'm6']]
annual_chars_df.reset_index(drop=True, inplace=True)

########################################
#               Quarterly              #
########################################
# bm
quarterly_characteristics_df['bm'] = quarterly_characteristics_df['beq']/quarterly_characteristics_df['me']

# bm_ia
industry_temp_df = quarterly_characteristics_df.groupby(['datadate', 'ffi49'], as_index=False)['bm'].mean()
industry_temp_df = industry_temp_df.rename(columns={'bm': 'bm_ind'})
quarterly_characteristics_df = pd.merge(quarterly_characteristics_df, industry_temp_df, how='left', on=['datadate', 'ffi49']).reset_index(drop=True)
quarterly_characteristics_df['bm_ia'] = quarterly_characteristics_df['bm'] - quarterly_characteristics_df['bm_ind']

# me_ia
industry_temp_df = quarterly_characteristics_df.groupby(['datadate', 'ffi49'], as_index=False)['me'].mean()
industry_temp_df = industry_temp_df.rename(columns={'me': 'me_ind'})
quarterly_characteristics_df = pd.merge(quarterly_characteristics_df, industry_temp_df, how='left', on=['datadate', 'ffi49']).reset_index(drop=True)
quarterly_characteristics_df['me_ia'] = quarterly_characteristics_df['me'] - quarterly_characteristics_df['me_ind']

# cfp
quarterly_characteristics_df['cfp'] = np.where(quarterly_characteristics_df['dpq'].isnull(),
                            quarterly_characteristics_df['ibq4']/quarterly_characteristics_df['me'],
                            (quarterly_characteristics_df['ibq4']+quarterly_characteristics_df['dpq4'])/quarterly_characteristics_df['me'])

# cfp_ia
industry_temp_df = quarterly_characteristics_df.groupby(['datadate', 'ffi49'], as_index=False)['cfp'].mean()
industry_temp_df = industry_temp_df.rename(columns={'cfp': 'cfp_ind'})
quarterly_characteristics_df = pd.merge(quarterly_characteristics_df, industry_temp_df, how='left', on=['datadate', 'ffi49']).reset_index(drop=True)
quarterly_characteristics_df['cfp_ia'] = quarterly_characteristics_df['cfp'] - quarterly_characteristics_df['cfp_ind']

# ep
quarterly_characteristics_df['ep'] = quarterly_characteristics_df['ibq4']/quarterly_characteristics_df['me']

# lev
quarterly_characteristics_df['lev'] = quarterly_characteristics_df['ltq']/quarterly_characteristics_df['me']

# rdm
quarterly_characteristics_df['rdm'] = quarterly_characteristics_df['xrdq4']/quarterly_characteristics_df['me']

# sp
quarterly_characteristics_df['sp'] = quarterly_characteristics_df['saleq4']/quarterly_characteristics_df['me']

# alm
quarterly_characteristics_df['alm'] = quarterly_characteristics_df['ala']/(quarterly_characteristics_df['atq']+quarterly_characteristics_df['me']-quarterly_characteristics_df['ceqq'])

# rsup
# quarterly_characteristics_df['saleq_l4'] = quarterly_characteristics_df.groupby(['permno'])['saleq'].shift(4)
quarterly_characteristics_df['rsup'] = (quarterly_characteristics_df['saleq'] - quarterly_characteristics_df['saleq_l4'])/quarterly_characteristics_df['me']

# sgrvol
quarterly_characteristics_df['sgrvol'] = compute_chars_std(0, 15, quarterly_characteristics_df, 'rsup')

# cashpr
quarterly_characteristics_df['cashpr'] = ((quarterly_characteristics_df['me'] + quarterly_characteristics_df['dlttq'] - quarterly_characteristics_df['atq']) / quarterly_characteristics_df['cheq'])

# indmom
industry_temp_df = quarterly_characteristics_df.groupby(['date', 'ffi49'], as_index=False)['mom12m'].mean().rename(columns={'mom12m': 'indmom'})
quarterly_characteristics_df = pd.merge(quarterly_characteristics_df, industry_temp_df, how='left', on=['date', 'ffi49']).reset_index(drop=True)

# Mohanram (2005) score (Quarterly Related)
industry_temp_df = quarterly_characteristics_df.groupby(['fyearq', 'fqtr', 'ffi49'], as_index=False)['roavol'].median().rename(columns={'roavol': 'md_roavol'})
quarterly_characteristics_df = pd.merge(quarterly_characteristics_df, industry_temp_df, how='left', on=['fyearq', 'fqtr', 'ffi49']).reset_index(drop=True)

industry_temp_df = quarterly_characteristics_df.groupby(['fyearq', 'fqtr', 'ffi49'], as_index=False)['sgrvol'].median().rename(columns={'sgrvol': 'md_sgrvol'})
quarterly_characteristics_df = pd.merge(quarterly_characteristics_df, industry_temp_df, how='left', on=['fyearq', 'fqtr', 'ffi49']).reset_index(drop=True)

quarterly_characteristics_df['m7'] = np.where(quarterly_characteristics_df['roavol'] < quarterly_characteristics_df['md_roavol'], 1, 0)
quarterly_characteristics_df['m8'] = np.where(quarterly_characteristics_df['sgrvol'] < quarterly_characteristics_df['md_sgrvol'], 1, 0)

# Quarterly Accounting Variables
quarterly_chars_df = quarterly_characteristics_df[['gvkey', 'permno', 'datadate', 'jdate', 'sic', 'exchcd', 'shrcd', 'ticker', 'conm', 'comnam', 'prc', 'shrout',
                     'ret', 'retx', 'retadj', 'acc', 'bm', 'cfp',
                     'ep', 'agr', 'ni', 'ope', 'opa', 'cop', 'cash', 'chcsho', 'rd', 'cashdebt', 'pctacc', 'gma', 'lev',
                     'rdm', 'sgr', 'sp', 'invest', 'rd_sale', 'lgr', 'roa', 'depr', 'egr', 'roe',
                     'chato', 'chpm', 'chtx', 'noa', 'rna', 'pm', 'ato', 'stdcf',
                     'grltnoa', 'ala', 'alm', 'rsup', 'stdacc', 'sgrvol', 'roavol', 'scf', 'cinvest',
                     'mom1m', 'mom6m', 'mom12m', 'mom60m', 'mom36m', 'seas1a', 'me', 'pscore', 'nincr',
                     'cfp_ia', 'bm_ia', 'me_ia', 'chatoia', 'chmom',
                     'turn', 'dolvol', 'cashpr', 'indmom', 'm7', 'm8']]
quarterly_chars_df.reset_index(drop=True, inplace=True)

print("Saving final datasets...")
feather.write_feather(annual_chars_df, 'FINAL_ANNUAL_DATA.feather')
feather.write_feather(quarterly_chars_df, 'FINAL_QUARTERLY_DATA.feather')

with open('chars_a_accounting.feather', 'wb') as f:
    feather.write_feather(annual_chars_df, f)

with open('chars_q_accounting.feather', 'wb') as f:
    feather.write_feather(quarterly_chars_df, f)
