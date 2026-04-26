import pandas as pd
import numpy as np
from ipca import InstrumentedPCA

print("Loading data...")
data = pd.read_feather('chars_raw_imputed.feather')

data['size'] = abs(data['prc'] * data['shrout'])
data = data.set_index(['permno', 'date'])

# Split
data_is = data[data.index.get_level_values('date') <= pd.Timestamp('2016-12-31')]
data_oos = data[data.index.get_level_values('date') > pd.Timestamp('2016-12-31')]

y_is = data_is['ret']
X_is = data_is[['abr','absacc','acc','adm','age','agr','alm','ato','baspread','beta','size']]

y_oos = data_oos['ret']
X_oos = data_oos[['abr','absacc','acc','adm','age','agr','alm','ato','baspread','beta','size']]

print("Fitting IPCA...")
regr = InstrumentedPCA(n_factors=1, intercept=False)
regr = regr.fit(X=X_is, y=y_is)

print("Predicting...")
X_pred = X_oos.loc[~X_oos.isna().any(axis=1)]

pre_y = regr.predict(
    X=X_pred,
    mean_factor=True,
    data_type='panel',
    label_ind=False
)

full_fitted = pd.Series(index=X_oos.index, dtype=float, name='fitted')
full_fitted.loc[X_pred.index] = pre_y.ravel()

result = pd.merge(full_fitted, y_oos, right_index=True, left_index=True, how='outer')

print("Saving output...")
result.to_feather("ipca_results.feather")

print("Done.")