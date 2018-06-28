# -*- coding: utf-8 -*-
"""
Created on Wed Apr  6 13:41:25 2016

@author: irham
"""

'''
print(__doc__)

import matplotlib.pyplot as plt

from sklearn import datasets
from sklearn.decomposition import PCA
#from sklearn.lda import LDA

iris = datasets.load_iris()

X = iris.data
y = iris.target
target_names = iris.target_names

pca = PCA(n_components=4)
X_r = pca.fit(X).transform(X)

#lda = LDA(n_components=2)
#X_r2 = lda.fit(X, y).transform(X)

# Percentage of variance explained for each components
print('explained variance ratio (first two components): %s'
      % str(pca.explained_variance_ratio_))

'''

'''
plt.figure()
for c, i, target_name in zip("rgb", [0, 1, 2], target_names):
    plt.scatter(X_r[y == i, 0], X_r[y == i, 1], c=c, label=target_name)
plt.legend()
plt.title('PCA of IRIS dataset')

#plt.figure()
#for c, i, target_name in zip("rgb", [0, 1, 2], target_names):
#    plt.scatter(X_r2[y == i, 0], X_r2[y == i, 1], c=c, label=target_name)
#plt.legend()
#plt.title('LDA of IRIS dataset')

plt.show()
'''

'''
def norm2(x):
    for i in range(len(x)):
        for j in range(len(x[i])):
            
            x[i][j] = (x[i][j] - np.mean(x[:, j])) / (np.std(x[:, j])*np.sqrt(len(x)))

    return x

def norm3(x):
    for j in range(len(x[0])):
        
        x[:, j] = (x[:, j] / np.max(x[:, j]))
    
    return x
'''





import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
#from sklearn.lda import LDA

import numpy as np
from sklearn.preprocessing import normalize





data = np.loadtxt('result_for_pca_b.csv', skiprows=1, delimiter = ',')
#data = normalize(data, norm='l2')

'''
b = np.array(([5., 32., 1., 23., 45., 6., 7., 8., 31., 13.]))

data = np.array((np.linspace(1, 10., 10.),\
5*np.linspace(10., 1., 10.)+5., b,\
np.sin(b)))

data = data.transpose()
#data = normalize(data, norm='l2')
'''

def norm(x):
    for j in range(len(x[0])):
        
        x[:, j] = (x[:, j] - np.mean(x[:, j])) / (np.std(x[:, j])*np.sqrt(len(x)))
    
    return x


data = norm(data)

pca2 = PCA(n_components=5)
#data_r = pca2.fit(data).transform(data)
data_r = pca2.fit_transform(data)

print('explained variance ratio (first two components): \n%s'
      % str(pca2.explained_variance_ratio_))

print '\n========================================\n\n'

print '\nVariance (Eigenvalue)'
print pca2.explained_variance_

print '\nVariance Ratio (Proportion)'
print pca2.explained_variance_ratio_

#print '\nMean'
#print pca2.mean_

print '\nLO3, LO3/LO2, LfeII/LbHbeta, fwhm_bHbeta, LNII/LnHalpha, LSII/LnHalpha, LO3/LnHbeta, fwhm_OIII'

print '\nComponents'
print pca2.components_

print '\nCumulative Variance (Cumulative)'
print pca2.explained_variance_ratio_.cumsum()


o = open('hasil.csv', 'w')

o.write('---\tEV1\tEV2\tEV3\tEV4\tEV5\n')
o.write('Eigenvalue\t')
for i in pca2.explained_variance_:
    o.write(str(i) + '\t')

o.write('\n')

o.write('Proportion\t')
for i in pca2.explained_variance_ratio_:
    o.write(str(i) + '\t')

o.write('\n')

o.write('Cumulative\t')
for i in pca2.explained_variance_ratio_.cumsum():
    o.write(str(i) + '\t')

o.write('\n')


name = ['LO3', 'LO3/LO2', 'LFeII/LbHbeta', 'FWHM_bHbeta', 'LNII/LnHalpha', 'LSII/LnHalpha', 'LOIII/LnHbeta', 'FWHM_OIII']

j = 0

for i in range(len(name)):
    o.write(str(name[i]) + '\t')
    for j in range(5):
        o.write(str(pca2.components_[j][i]) + '\t')
    o.write('\n')

o.close()


'''
plt.figure()
plt.scatter(data_r[:, 0], data[:, 11])
plt.xlabel('Component 1')
plt.ylabel('FWHM bHbeta')


plt.figure()
plt.scatter(data_r[:, 1], data[:, 12])
plt.xlabel('Component 2')
plt.ylabel('LFeII/LbHbeta')
plt.savefig('new.eps', dpi = 300)
'''