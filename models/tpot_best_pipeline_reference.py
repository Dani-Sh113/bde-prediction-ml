# TPOT Best Pipeline
# Generated from TPOT optimization

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# Best pipeline structure:
# Pipeline(steps=[('minmaxscaler', MinMaxScaler()),
                ('variancethreshold',
                 VarianceThreshold(threshold=0.0003878045217)),
                ('featureunion-1',
                 FeatureUnion(transformer_list=[('skiptransformer',
                                                 SkipTransformer()),
                                                ('passthrough',
                                                 Passthrough())])),
                ('featureunion-2',
                 FeatureUnion(transformer_list=[('skiptransformer',
                                                 SkipTransformer()),
                                                ('passthrough',
                                                 Passthrough())])),
                ('lgbmregressor',
                 LGBMRegressor(boosting_type=np.str_('gbdt'), max_depth=3,
                               n_estimators=72, n_jobs=1, num_leaves=119,
                               random_state=42, verbose=-1))])

# NOTE: Use tpot.fitted_pipeline_ directly for predictions
# Example: predictions = tpot.fitted_pipeline_.predict(X_test)
