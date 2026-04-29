# Asset Pricing Forest

This repository contains the empirical pipeline for constructing characteristic-based test assets and evaluating Asset Pricing Trees / Forests against standard benchmark models. The project starts from WRDS-style CRSP/Compustat data, constructs firm characteristics, builds tree-based portfolios, applies pruning and model selection, and finally reports out-of-sample performance, factor alphas, turnover, and importance measures.

All codes are under /code

## 1_feature_construction

This folder builds the monthly firm-characteristic panel used by the AP Tree / AP Forest pipeline and benchmark models.

### 1_1_base_data

- `functions.py`: Shared helper functions for data cleaning, merging, and saving.
- `1_compustat_data.py`: Cleans and organizes Compustat accounting data.
- `2_crsp_data.py`: Cleans and organizes CRSP stock return and market data.
- `3_ccm_link.py`: Links CRSP and Compustat using the CCM link table.
- `4_build_base.py`: Builds the base firm-month panel from CRSP, Compustat, and link-table data.

### 1_2_features_part1

- `generate_features_part1.py`: Runs the first batch of characteristic construction scripts.
- `annual_features/*.py`: Constructs annual accounting-based characteristics.
- `quarterly_features/*.py`: Constructs quarterly accounting-based characteristics.
- `market_features/*.py`: Constructs market-based characteristics.
- `assemble/build_assemble_features.py`: Combines annual, quarterly, and market feature blocks.
- `assemble/f001_populate_to_monthly.py`: Expands lower-frequency accounting features to the monthly panel.
- `assemble/f002_recompute_monthly_me_features.py`: Recomputes market-equity-related variables at monthly frequency.
- `assemble/f003_select_final_columns.py`: Selects the final columns used in downstream feature construction.

### 1_3_features_part2

- `generate_features_part2.py`: Runs the second batch of additional characteristic construction scripts.
- `monthly_extra/*.py`: Constructs additional monthly market-based characteristics.
- `quarterly_extra/*.py`: Constructs additional quarterly characteristics.
- `assemble_raw/`: Combines the additional features into the raw feature panel.

### 1_4_rank_feature

- `generate_rank_features.py`: Cross-sectionally ranks firm characteristics by month and creates the ranked feature panel.
- `by_year/no_impute`: Stores ranked features without additional imputation.
- `by_year/impute`: Stores ranked features with the imputed version used by selected robustness checks and benchmark models.

### Root-level scripts

- `100_chars.py`: Organizes the characteristic list used throughout the project.

## 2_tree_construction

This folder constructs candidate tree portfolios from ranked firm characteristics. The pipeline has three stages: split, combine, and filter.

### 2_1_split

- `2_1_1_split.py`: Builds the baseline AP Forest I candidate trees using no-impute ranked characteristics.
- `2_1_2_split_ew.py`: Builds the equal-weighted AP Forest candidate trees.
- `2_1_3_split_impute.py`: Builds the imputed-feature AP Forest candidate trees.
- `2_1_4_split_benchmark.py`: Builds the AP Tree / benchmark sorting candidate portfolios.
- `2_1_5_split_ptree.py`: Builds the AP Forest II / pruned-tree candidate portfolios.

### 2_2_combine_portfolio

- `2_2_1_combine_portfolio.py`: Combines AP Forest I split-level node returns into a portfolio return matrix.
- `2_2_2_combine_ew.py`: Combines equal-weighted AP Forest candidate portfolios.
- `2_2_3_combine_impute.py`: Combines imputed-feature AP Forest candidate portfolios.
- `2_2_4_combine_benchmark.py`: Combines AP Tree / benchmark candidate portfolios.
- `2_2_5_combine_ptree.py`: Combines AP Forest II / pruned-tree candidate portfolios.

### 2_3_filter

- `2_3_1_filter.py`: Filters AP Forest I candidate portfolios before pruning.
- `2_3_2_filter_ew.py`: Filters equal-weighted AP Forest candidate portfolios.
- `2_3_3_filter_impute.py`: Filters imputed-feature AP Forest candidate portfolios.
- `2_3_4_filter_benchmark.py`: Filters AP Tree / benchmark candidate portfolios.
- `2_3_5_filter_ptree.py`: Filters AP Forest II / pruned-tree candidate portfolios.

## 3_ap_prune

This folder implements the pruning stage. It selects sparse combinations of candidate tree portfolios using rolling train / validation / test windows.

### 3_1_prune_cv

- `3_1_1_prune_cv.py`: Runs pruning and cross-validation for AP Forest I.
- `3_1_2_prune_cv_ew.py`: Runs pruning and cross-validation for the equal-weighted AP Forest.
- `3_1_3_prune_cv_impute.py`: Runs pruning and cross-validation for the imputed-feature AP Forest.
- `3_1_4_prune_cv_benchmark.py`: Runs pruning and cross-validation for the AP Tree / benchmark version.
- `3_1_5_prune_cv_ptree.py`: Runs pruning and cross-validation for AP Forest II / pruned-tree portfolios.
- `submit*.sh`: Cluster submission scripts for running pruning jobs in parallel.
- `logs/`: Cluster logs for pruning jobs.

## 3_ridge

This folder implements the ridge regression benchmark.

### 3_1_run_ridge

- `3_1_1_ridge.py`: Fits rolling ridge models, selects the penalty parameter by validation MSE, predicts out-of-sample stock returns, and forms value-weighted long-short portfolios from the top and bottom predicted-return stocks.

## 3_IPCA

This folder implements the IPCA benchmark.

### 3_1_run_IPCA

- `3_1_1_ipca.py`: Fits rolling IPCA models, selects the number of factors by validation MSE, predicts out-of-sample stock returns using mean-factor prediction, and forms value-weighted long-short portfolios from the top and bottom predicted-return stocks.

## 4_results

This folder contains scripts for final empirical evaluation, figures, factor regressions, turnover, and importance analysis.

### 4_1_pnl

- `4_1_1_ap_forestI.py`: Constructs monthly AP Forest I out-of-sample returns from selected pruning results.
- `4_1_2_ap_forest_ew.py`: Constructs monthly equal-weighted AP Forest out-of-sample returns.
- `4_1_3_ap_forest_impute.py`: Constructs monthly imputed-feature AP Forest out-of-sample returns.
- `4_1_4_ap_forest_benchmark.py`: Constructs monthly AP Tree / benchmark out-of-sample returns.
- `4_1_5_ap_forestII.py`: Constructs monthly AP Forest II out-of-sample returns.
- `4_1_6_IPCA.py`: Aggregates IPCA out-of-sample returns into the common return format.
- `4_1_7_ridge.py`: Aggregates ridge out-of-sample returns into the common return format.
- `4_1_8_figure6.py`: Plots cumulative log returns for the main model comparison.
- `4_1_9_figure7.py`: Compares value-weighted and equal-weighted AP Forest performance.
- `4_1_10_figure8.py`: Compares no-impute and impute AP Forest performance.
- `4_1_11_figure9.py`: Provides sample code for plotting AP Forest performance under different ensemble-size settings.

### 4_2_FF3

- `4_2_1_FF3_regression.py`: Runs Fama-French three-factor regressions for all main strategy returns.

### 4_3_FF5

- `4_3_1_FF5_regression.py`: Runs Fama-French five-factor regressions for all main strategy returns.

### 4_4_turnover

- `4_4_1_turnover_ap_forestI.py`: Computes drift-adjusted turnover for AP Forest I.
- `4_4_4_turnover_IPCA.py`: Computes drift-adjusted turnover for IPCA.
- `4_4_5_turnover_ptree.py`: Computes drift-adjusted turnover for AP Forest II / pruned-tree portfolios.
- `4_4_5_turnover_ridge.py`: Computes drift-adjusted turnover for the ridge benchmark.

### 4_5_importance

- `4_5_1_interaction_importance.py`: Computes interaction importance for AP Forest I by decoding selected tree paths and aggregating selected node weights across rolling windows and repeats.

## Main Empirical Design

The main out-of-sample evaluation uses a rolling-window design. For each test year, the model uses an initial training period, a validation period for hyperparameter selection, and a one-year out-of-sample test period. The AP Forest models build many characteristic-based tree portfolios, prune the candidate set using validation performance, and combine selected portfolios into the final out-of-sample strategy.

The main AP Forest variants are:

- `AP forest I`: Baseline AP Forest using no-impute ranked characteristics.
- `AP forest EW`: Equal-weighted AP Forest robustness check.
- `AP forest Impute`: AP Forest using imputed ranked characteristics.
- `AP tree`: Benchmark AP Tree / sorting-based version.
- `AP forest II`: Pruned-tree version with a different split and selection structure.
- `IPCA`: Characteristic-based latent factor benchmark.
- `Ridge`: Regularized linear prediction benchmark.

## Notes

AI assistance was used to review, debug, and reformulate code. AI was not used to replace the empirical research design or final interpretation.
