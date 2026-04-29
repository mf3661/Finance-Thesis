# Asset Pricing Forest

## Code Descriptions

### 1_feature_construction

Builds the monthly firm-characteristic panel used as input for AP Tree construction, pruning, IPCA, and benchmark portfolio sorts.

#### 1_1_base_data

- `functions.py`: Shared helper functions for cleaning, merging, and saving intermediate datasets.
- `1_compustat_data.py`: Extracts and cleans Compustat accounting variables.
- `2_crsp_data.py`: Extracts and cleans CRSP stock return and market data.
- `3_ccm_link.py`: Links CRSP and Compustat using the CCM link table.
- `4_build_base.py`: Merges CRSP, Compustat, and link-table outputs into the base firm-month panel.

#### 1_2_features_part1

- `generate_features_part1.py`: Runs the first batch of characteristic construction scripts.
- `annual_features/*.py`: Constructs annual accounting-based characteristics, such as book-to-market, accruals, profitability, investment, leverage, cash, R&D, and asset growth.
- `quarterly_features/*.py`: Constructs quarterly accounting-based characteristics.
- `market_features/*.py`: Constructs market-based characteristics, including momentum, turnover, dollar volume, and dividend yield.
- `assemble/build_assemble_features.py`: Merges annual, quarterly, and market features.
- `assemble/f001_populate_to_monthly.py`: Expands lower-frequency accounting features to monthly observations.
- `assemble/f002_recompute_monthly_me_features.py`: Recomputes market-equity-related variables at monthly frequency.
- `assemble/f003_select_final_columns.py`: Selects the final feature columns for downstream analysis.

#### 1_3_features_part2

- `generate_features_part2.py`: Runs the second batch of additional characteristic scripts.
- `monthly_extra/*.py`: Constructs additional monthly characteristics, such as beta, residual variance, bid-ask spread, illiquidity, maximum return, zero trading, and real estate exposure.
- `quarterly_extra/*.py`: Constructs additional quarterly characteristics.
- `assemble_raw/`: Combines the extra feature outputs into the raw feature panel.

#### 1_4_rank_feature

- `generate_rank_features.py`: Cross-sectionally ranks firm characteristics by month and produces the ranked feature panel used for AP Tree splits.

#### Root-level files

- `100_chars.py`: Defines or organizes the full set of characteristics used in the empirical pipeline.
- `untitled.py`: Temporary or exploratory script; not part of the main production pipeline.

### 2_tree_construction

Constructs AP Tree candidate portfolios from ranked firm characteristics and prepares them for pruning.

#### 2_1_split

- `2_1_1_split.py`: Builds baseline AP Tree splits using ranked characteristics.
- `2_1_2_split_ew.py`: Builds equal-weighted tree split portfolios.
- `2_1_3_split_impute.py`: Builds tree splits using the imputed feature panel.
- `2_1_4_split_benchmark.py`: Builds benchmark sorting portfolios for comparison.
- `2_1_5_split_ptree.py`: Builds pruned-tree candidate splits for later selection.

#### 2_2_combine_portfolio

- `2_2_1_combine_portfolio.py`: Combines split-level outputs into portfolio return panels.
- `2_2_2_combine_ew.py`: Combines equal-weighted tree portfolio outputs.
- `2_2_3_combine_impute.py`: Combines imputed-data tree portfolio outputs.
- `2_2_4_combine_benchmark.py`: Combines benchmark portfolio outputs.
- `2_2_5_combine_ptree.py`: Combines pruned-tree candidate portfolio outputs.

#### 2_3_filter

- `2_3_1_filter.py`: Filters constructed tree portfolios before pruning.
- `2_3_2_filter_ew.py`: Filters equal-weighted portfolio outputs.
- `2_3_3_filter_impute.py`: Filters imputed-data portfolio outputs.
- `2_3_4_filter_benchmark.py`: Filters benchmark sorting portfolio outputs.
- `2_3_5_filter_ptree.py`: Filters pruned-tree candidate portfolios.

### 3_ap_prune

Implements the pruning stage of the AP Tree framework, selecting an optimal sparse subset of candidate portfolios using cross-validation and out-of-sample performance criteria.

#### 1_prune_cv

- `1_prune_cv.py`: Main pruning engine that performs cross-validation over candidate AP Tree portfolios and selects the optimal portfolio subset.
- `submit.sh`: Batch submission script for running pruning jobs in parallel on the cluster.
- `ensemble_first9_summary.py`: Summarizes results across the first nine repeated pruning runs.
- `ensemble_first9_summary_2.py`: Extended ensemble summary script for repeated-run robustness checks.
- `logs/`: Cluster job logs for pruning runs.

**Generated outputs**
- `ensemble_summary.csv`: Final pruning performance summary across runs.
- `ensemble_monthly_ret.csv`: Monthly long-short portfolio returns after pruning.
- `ensemble_monthly_ret_detail.csv`: Detailed monthly return decomposition.
- `ensemble_yearly_sr.csv`: Annual Sharpe ratios of pruned strategies.
- `ensemble_yearly_sr_detail.csv`: Detailed yearly Sharpe statistics.
- `ensemble_repeat_avg_sr.csv`: Average Sharpe ratio across repeated runs.
- `ensemble_first_9_repeats_summary.csv`: Summary of first nine repeated runs.
- `ensemble_first_9_repeats_timeseries.csv`: Return time series for first nine repeated runs.
- `plot_cumulative_return.png`: Cumulative return plot.
- `plot_drawdown.png`: Drawdown plot.
- `plot_monthly_return.png`: Monthly return visualization.
- `plot_yearly_sharpe.png`: Yearly Sharpe ratio visualization.

#### 2_selection

- `1_selection.py`: Final model selection script that chooses the preferred pruning specification after cross-validation.
- `submit.sh`: Batch submission script for final selection runs.
- `logs/`: Cluster logs for selection jobs.

### 4_ridge

Implements the ridge regression benchmark used to compare AP Trees against regularized linear prediction models in the cross-section of stock returns.

#### Main Scripts

- `4_1_ridge.py`: Main ridge regression pipeline. Estimates monthly cross-sectional ridge models using firm characteristics and forms long-short portfolios from predicted returns.
- `4_2_ridge_cv.py`: Selects the optimal ridge penalty parameter through cross-validation or rolling validation.
- `4_3_ridge_oos.py`: Generates out-of-sample portfolio returns and prediction results using the selected ridge specification.
- `4_4_ridge_summary.py`: Summarizes benchmark performance, including return, Sharpe ratio, alpha, and turnover statistics.
- `submit.sh`: Batch submission script for running ridge jobs in parallel on the cluster.
- `logs/`: Cluster job logs for ridge estimation runs.

#### Generated Outputs

- `ridge_monthly_ret.csv`: Monthly long-short returns from the ridge benchmark.
- `ridge_summary.csv`: Overall performance summary of the ridge strategy.
- `ridge_yearly_sr.csv`: Annual Sharpe ratio series.
- `ridge_predictions.csv`: Predicted cross-sectional returns from the ridge model.
- `plot_ridge_cumulative_return.png`: Cumulative return plot.
- `plot_ridge_drawdown.png`: Drawdown plot.

### 5_IPCA

Implements the Instrumented Principal Component Analysis (IPCA) benchmark used to compare AP Trees against characteristic-based latent factor models.

#### Main Scripts

- `5_1_ipca.py`: Main IPCA estimation pipeline. Uses firm characteristics as instruments to estimate latent factors and factor loadings.
- `5_2_ipca_cv.py`: Selects the optimal number of latent factors and model settings through cross-validation or rolling validation.
- `5_3_ipca_oos.py`: Generates out-of-sample expected returns and portfolio returns using the estimated IPCA model.
- `5_4_ipca_summary.py`: Summarizes benchmark performance, including return, Sharpe ratio, alpha, and turnover statistics.
- `submit.sh`: Batch submission script for running IPCA jobs in parallel on the cluster.
- `logs/`: Cluster job logs for IPCA estimation runs.

#### Generated Outputs

- `ipca_monthly_ret.csv`: Monthly long-short returns from the IPCA benchmark.
- `ipca_summary.csv`: Overall performance summary of the IPCA strategy.
- `ipca_yearly_sr.csv`: Annual Sharpe ratio series.
- `ipca_predictions.csv`: Predicted cross-sectional returns from the IPCA model.
- `plot_ipca_cumulative_return.png`: Cumulative return plot.
- `plot_ipca_drawdown.png`: Drawdown plot.

### 6_results

Stores the final empirical outputs used in the thesis, including benchmark comparisons, robustness checks, tables, and visualizations.

#### Main Contents

- `performance_summary.csv`: Consolidated comparison of all models, including AP Trees, Ridge, IPCA, and benchmark portfolios.
- `monthly_return_panel.csv`: Monthly return series for all strategies.
- `alpha_summary.csv`: FF3 / FF5 alpha regression results across strategies.
- `turnover_summary.csv`: Portfolio turnover and trading activity comparison.
- `robustness_checks.csv`: Sensitivity tests across alternative specifications and hyperparameters.
- `feature_importance.csv`: Leave-one-feature-out feature importance results.
- `selected_features.csv`: Characteristics most frequently retained in final AP Tree specifications.

#### Figures

- `plot_cumulative_returns.png`: Cumulative return comparison across models.
- `plot_drawdown_comparison.png`: Drawdown comparison across strategies.
- `plot_sharpe_comparison.png`: Sharpe ratio comparison.
- `plot_turnover_comparison.png`: Turnover comparison.
- `plot_feature_importance.png`: Feature importance visualization.
- `plot_rolling_performance.png`: Rolling out-of-sample performance over time.

#### Tables

- `table_main_results.tex`: Main thesis result table.
- `table_alpha_results.tex`: Alpha regression table.
- `table_robustness.tex`: Robustness test table.
- `table_feature_importance.tex`: Feature importance table.

### Notes

This folder contains finalized outputs only. Intermediate files, temporary diagnostics, and large raw backtest objects are excluded to keep the repository organized and reproducible.

## Extra Explanations
### Feature Importance

Feature importance is evaluated through a manual leave-one-feature-out procedure rather than a standalone script.

For each firm characteristic, we remove the feature from the input set and rerun the **entire AP Tree pipeline**, including:

1. Feature construction  
2. Tree construction  
3. Portfolio combination and filtering  
4. Pruning / cross-validation  
5. Out-of-sample portfolio evaluation  

The importance of a characteristic is measured by the change in final performance after exclusion, such as:

- Sharpe ratio decline  
- Return reduction  
- Alpha deterioration  
- Changes in selected tree structures  

We do not provide a dedicated feature-importance script because importance in our framework is **model-level rather than node-level**. The contribution of a characteristic affects every stage of the pipeline: candidate tree splits, portfolio interactions, pruning decisions, and final portfolio weights. As a result, meaningful importance must be evaluated by rerunning the full project after removing each feature.

This process is computationally expensive and depends on the user’s chosen specifications (sample period, hyperparameters, rolling windows, benchmark settings, etc.). A standalone script would therefore be misleading or incomplete. Instead, feature importance should be implemented as a full re-estimation exercise under the user’s preferred setup.

Compared with simple split-frequency or impurity metrics, this leave-one-feature-out approach provides a more economically relevant measure because it captures each characteristic’s contribution to final out-of-sample investment performance.



**Note:** AI help review and debug our code, and reformulate code. AI is not used in coding pipeline.

