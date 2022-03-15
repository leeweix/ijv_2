# -*- coding: utf-8 -*-
"""
Created on Wed Sep  8 20:43:05 2021

@author: kh722
"""

import sys
for place in sys.path:
        print(place)

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns 
from sklearn.decomposition import PCA

sns.set()
rng = np.random.RandomState(1)
# X = rng.randn(2, 200).T
z = range(200)

cm = plt.cm.get_cmap('jet')
X = np.dot(rng.rand(2, 2) , rng.randn(2, 200)).T
mappable = plt.scatter(X[:, 0], X[:, 1], c=z, marker='o',vmin=0, vmax=100, s=25, cmap=cm)
plt.colorbar(mappable)
plt.axis('equal')

pca = PCA(n_components=2)
trans_func = pca.fit(X)
num = np.array([[100, 50]])
n_f = trans_func.transform(num)
print(n_f)

