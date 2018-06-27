# -*- coding: utf-8 -*-
"""
Created on Tue Jun 12 12:00:22 2018

@author: kenda
"""


import numpy as np
import matplotlib.pyplot as plt



k_start = 0
k_end = 4 * 10**-10
k_values = np.linspace(k_start, k_end, 100)

poly = []


q_start = -4 * 10**-10
q_end = 4 * 10**-10

k = 1.5 * 10**-6
beta = 8.0 * 10**-4

def polynomial_pos(kd,s_prec,n_prec):
    poly = []
    for i in range(100):
        coeff = [1, 2*kd[i], kd[i]**2 - k*beta*s_prec,(kd[i]*k*beta*(n_prec - s_prec))]
        poly.append(coeff)
    return poly

polyI_pos = polynomial_pos(k_values,.9*10**-10,.9*10**-10)
polyII_pos = polynomial_pos(k_values, .9*10**-10, .45*10**-10)
polyIII_pos = polynomial_pos(k_values, .9*10**-10, 1.35*10**-10)

def polynomial_neg(kd,s_prec,n_prec):
    poly = []
    for i in range(100):
        coeff = [1, -2*kd[i], kd[i]**2 - k*beta*n_prec,(kd[i]*k*beta*(n_prec - s_prec))]
        poly.append(coeff)
    return poly
polyI_neg = polynomial_neg(k_values,.9*10**-10,.9*10**-10)
polyII_neg = polynomial_neg(k_values, .9*10**-10, .45*10**-10)
polyIII_neg = polynomial_neg(k_values, .9*10**-10, 1.35*10**-10)

qI_pos = []
qI_neg = []

qII_pos = []
qII_neg = []

qIII_pos = []
qIII_neg = []

def qsolve(q,poly,nonneg):
    for i in range(100):
        sol = np.roots(poly[i])
        if nonneg:
            sol = [max(s, 0) for s in sol]
        else:
            sol = [min(s, 0) for s in sol]
        q.append(sol)
    return q

qI_pos = qsolve(qI_pos, polyI_pos, True) 
qI_neg =qsolve(qI_neg,polyI_neg, False)

qII_pos = qsolve(qII_pos, polyII_pos, True)
qII_neg = qsolve(qII_neg, polyII_neg, False)

qIII_pos = qsolve(qIII_pos, polyIII_pos, True)
qIII_neg = qsolve(qIII_neg, polyIII_neg, False)

for q in qI_pos:
    q.sort()
for q in qII_pos:
    q.sort()
for q in qII_neg:
    q.sort()
for q in qIII_pos:
    q.sort()
for q in qIII_neg:
    q.sort()    
    
ax = plt.subplot(1,1,1)
ax.plot(k_values,qI_pos,"k-",k_values,qI_neg,"k-",k_values,qII_pos,"r--",k_values[:15],qII_neg[:15],"r--", k_values[:30],qIII_pos[:30],"c-.",k_values,qIII_neg,"c-.")

ax.plot(.5 *10**-10,2.79*10**-10, "ks",1*10**-10, 2.29*10**-10,"ks", 2*10**-10, 1.29*10**-10, "ks", 3* 10**-10, .29 *10**-10, "ks" )
ax.plot(.5*10**-10, 2.92*10**-10, "rs", 1*10**-10, 2.59*10**-10,"rs", 2*10**-10,2.02*10**-10, "rs", 3*10**-10, 1.58*10**-10, "rs" )
ax.plot(.25*10**-10, 2.97*10**-10, "cs", .5*10**-10, 2.63*10**-10, "cs", .75*10**-10, 2.25*10**-10, "cs", 1*10**-10, 1.79*10**-10, "cs", 1.15*10**-10, 1.32*10**-10, "cs")
ax.plot(.5*10**-10, -2.79*10**-10, "kD", 1*10**-10, -2.29*10**-10, "kD", 2*10**-10, -1.29*10**-10, "kD", 3*10**-10, -.29*10**-10, "kD")
ax.plot(.5*10**-10, -3.6*10**-10, "cD", 1*10**-10, -3.22*10**-10, "cD", 2*10**-10, -2.53*10**-10, "cD", 3*10**-10, -1.95*10**-10, "cD")

plt.xlim(k_start,k_end)
plt.ylim(q_start,q_end)

plt.xlabel("K_d")
plt.ylabel("Q")
plt.title("Critical Flow Rooth Model")
plt.show()

