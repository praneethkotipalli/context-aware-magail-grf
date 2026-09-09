from sklearn.linear_model import RidgeCV
from sklearn.model_selection import GroupKFold, cross_val_score
import numpy as np
z = np.load('expert_features_cache.npz', allow_pickle=True)
X, eids = z['features'], z['episode_ids']
gkf = GroupKFold(n_splits=5)
for j, name in [(137,'T_norm'),(138,'dScore')]:
    r2 = cross_val_score(RidgeCV(alphas=[1,10,100,1000]), X[:,:137], X[:,j],
                          cv=gkf, groups=eids, scoring='r2')
    print(name, r2.mean(), r2)