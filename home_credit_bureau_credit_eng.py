# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
import os
import sys
import pandas as pd
import numpy as np
from tqdm import tqdm_notebook as tqdm
from functools import partial
from sklearn.externals import joblib
#%matplotlib inline
import seaborn as sns
from sklearn.linear_model import LinearRegression


from scipy.stats import skew, kurtosis, iqr
from tqdm import tqdm_notebook as tqdm
from sklearn.externals import joblib
import seaborn as sns
import os
import random
import sys
import multiprocessing as mp
from functools import reduce

import glob
import numpy as np
import pandas as pd
#from tqdm import tqdm
#import yaml
import glob
import numpy as np
import pandas as pd
from tqdm import tqdm
import yaml
#from attrdict import AttrDict
from functools import partial
###################
def chunk_groups(groupby_object, chunk_size):
    n_groups = groupby_object.ngroups
    group_chunk, index_chunk = [], []
    for i, (index, df) in enumerate(groupby_object):
        group_chunk.append(df)
        index_chunk.append(index)

        if (i + 1) % chunk_size == 0 or i + 1 == n_groups:
            group_chunk_, index_chunk_ = group_chunk.copy(), index_chunk.copy()
            group_chunk, index_chunk = [], []
            yield index_chunk_, group_chunk_


def parallel_apply(groups, func, index_name='Index', num_workers=2, chunk_size=100000):
    n_chunks = np.ceil(1.0 * groups.ngroups / chunk_size)
    indeces, features = [], []
    for index_chunk, groups_chunk in tqdm(chunk_groups(groups, chunk_size), total=n_chunks):
        with mp.pool.Pool(num_workers) as executor:
            features_chunk = executor.map(func, groups_chunk)
        features.extend(features_chunk)
        indeces.extend(index_chunk)

    features = pd.DataFrame(features)
    features.index = indeces
    features.index.name = index_name
    return features

def add_features_in_group(features, gr_, feature_name, aggs, prefix):
    for agg in aggs:
        if agg == 'sum':
            features['{}{}_sum'.format(prefix, feature_name)] = gr_[feature_name].sum()
        elif agg == 'mean':
            features['{}{}_mean'.format(prefix, feature_name)] = gr_[feature_name].mean()
        elif agg == 'max':
            features['{}{}_max'.format(prefix, feature_name)] = gr_[feature_name].max()
        elif agg == 'min':
            features['{}{}_min'.format(prefix, feature_name)] = gr_[feature_name].min()
        elif agg == 'std':
            features['{}{}_std'.format(prefix, feature_name)] = gr_[feature_name].std()
        elif agg == 'count':
            features['{}{}_count'.format(prefix, feature_name)] = gr_[feature_name].count()
        elif agg == 'skew':
            features['{}{}_skew'.format(prefix, feature_name)] = skew(gr_[feature_name])
        elif agg == 'kurt':
            features['{}{}_kurt'.format(prefix, feature_name)] = kurtosis(gr_[feature_name])
        elif agg == 'iqr':
            features['{}{}_iqr'.format(prefix, feature_name)] = iqr(gr_[feature_name])
        elif agg == 'median':
            features['{}{}_median'.format(prefix, feature_name)] = gr_[feature_name].median()
    return features

#sys.path.append('../')
#from src.utils import parallel_apply
#from src.feature_extraction import add_features_in_group

#DIR = '/mnt/ml-team/minerva/open-solutions/home-credit'
#description = pd.read_csv(os.path.join(DIR,'data/HomeCredit_columns_description.csv'),encoding = 'latin1')
application = pd.read_csv('../input/home-credit-application-engineering/application_eng.csv',usecols=['SK_ID_CURR'])
bureau = pd.read_csv('../input/home-credit-default-risk/bureau.csv')
bureau_balance = pd.read_csv('../input/home-credit-default-risk/bureau_balance.csv')

bureau_balance.fillna(0, inplace=True)
bureau_balance = bureau_balance.merge(bureau[['SK_ID_CURR', 'SK_ID_BUREAU']], on='SK_ID_BUREAU', how='right')
bureau_balance.shape

def _status_to_int(status):
    if status in ['X', 'C']:
        return 0
    if pd.isnull(status):
        return np.nan
    return int(status)
bureau_balance['bureau_balance_dpd_level'] = bureau_balance['STATUS'].apply(_status_to_int)
bureau_balance['bureau_balance_status_unknown'] = (bureau_balance['STATUS'] == 'X').astype(int)

groupby = bureau_balance.groupby(['SK_ID_CURR'])
features = pd.DataFrame({'SK_ID_CURR': bureau_balance['SK_ID_CURR'].unique()})

def last_k_installment_features(gr, periods):
    gr_ = gr.copy()

    features = {}
    for period in periods:
        if period > 10e10:
            period_name = 'all_installment_'
            gr_period = gr_.copy()
        else:
            period_name = 'last_{}_'.format(period)
            gr_period = gr_[gr_['MONTHS_BALANCE'] >= (-1) * period]

        features = add_features_in_group(features, gr_period, 'bureau_balance_dpd_level',
                                             ['sum', 'mean', 'max', 'std', 'skew', 'kurt'],
                                             period_name)
        features = add_features_in_group(features, gr_period, 'bureau_balance_status_unknown',
                                             ['sum', 'mean'],
                                             period_name)
    return features

func=partial(last_k_installment_features, periods=[6,12])
g = parallel_apply(groupby, func, index_name='SK_ID_CURR',
                   num_workers=2, chunk_size=10000).reset_index()
features = features.merge(g, on='SK_ID_CURR', how='left')

def trend_in_last_k_installment_features(gr, periods):
    gr_ = gr.copy()
    gr_.sort_values(['MONTHS_BALANCE'], ascending=False, inplace=True)

    features = {}
    for period in periods:
        gr_period = gr_[gr_['MONTHS_BALANCE'] >= (-1) * period]

        features = add_trend_feature(features, gr_period,
                                         'bureau_balance_dpd_level', '{}_period_trend_'.format(period)
                                         )
    return features

def add_trend_feature(features, gr, feature_name, prefix):
    y = gr[feature_name].values
    try:
        x = np.arange(0, len(y)).reshape(-1, 1)
        lr = LinearRegression()
        lr.fit(x, y)
        trend = lr.coef_[0]
    except:
        trend = np.nan
    features['{}{}'.format(prefix, feature_name)] = trend
    return features
    
    
func=partial(trend_in_last_k_installment_features, periods=[6])
g = parallel_apply(groupby, func, index_name='SK_ID_CURR',
                   num_workers=2, chunk_size=10000).reset_index()
features = features.merge(g, on='SK_ID_CURR', how='left')


def last_k_instalment_fractions(old_features, fraction_periods):
    features = old_features[['SK_ID_CURR']].copy()
    
    for short_period, long_period in fraction_periods:
        short_feature_names = _get_feature_names(old_features, short_period)
        long_feature_names = _get_feature_names(old_features, long_period)
        
        for short_feature, long_feature in zip(short_feature_names, long_feature_names):
            old_name_chunk = '_{}_'.format(short_period)
            new_name_chunk ='_{}by{}_fraction_'.format(short_period, long_period)
            fraction_feature_name = short_feature.replace(old_name_chunk, new_name_chunk)
            features[fraction_feature_name] = old_features[short_feature]/old_features[long_feature]
    return pd.DataFrame(features).fillna(0.0)

def _get_feature_names(features, period):
    return sorted([feat for feat in features.keys() if '_{}_'.format(period) in feat])
    
func=partial(last_k_instalment_fractions, fraction_periods=[(6, 12)])
g = parallel_apply(groupby, func, index_name='SK_ID_CURR',
                   num_workers=2, chunk_size=10000).reset_index()
features = features.merge(g, on='SK_ID_CURR', how='left')

application= application.merge(features, on='SK_ID_CURR',how='left')
print(application.shape)
application.to_csv('bureau_eng.csv',index=False)