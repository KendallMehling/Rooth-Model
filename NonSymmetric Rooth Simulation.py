# -*- coding: utf-8 -*-
"""
Created on Tue Jun 26 13:36:01 2018

@author: kenda
"""
###Using the same treatment for flow characteristics I will set T1 = T3 to 
###Represent the Symmetric Temperature Disparity as Donce in Section C
###To reproduce Fig 6 to find various steady state flows as a function 
###Of K_d


import numpy as np
import matplotlib.pyplot as plt

time0 = 0  # 0 years in seconds
timeTempf =  9467280000  # 300 years in seconds
timeSalf = 2 * 31557600000  # 2,000 years in seconds
steps = 100001
Tdeltat = (timeTempf - time0) / (steps - 1)  # increment is equal to 3/1000 year in seconds
Sdeltat = (timeSalf - time0) / (steps - 1)  # increment is equal to 1/100 year in seconds
print("Temperature simulation: start (yrs):", time0, "-- end (yrs):", timeTempf / (60 * 60 * 24 * 365.25),
      "-- increment(yrs):", Tdeltat / (60 * 60 * 24 * 365.25))
print("\n\nSalinity simulation: start(yrs):", time0, "-- end (yrs):", timeSalf / (60 * 60 * 24 * 365.25),
      "-- increment (yrs):", Sdeltat / (60 * 60 * 24 * 365.25))


def euler_forward(initT1, initT2, initT3, initS1, initS2, initS3, kd):
    lam = 12.9 * 10 ** -10
    Te = 30
    Tp = 0
    n_prec = 1.35 * 10 ** -10 # Case II: Change this case to find various steady-states.
    s_prec = .9 * 10 ** -10
    k = 1.5 * 10 ** -6
    alpha = 1.5 * 10 ** -4
    beta = 8.0 * 10 ** -4

    T1 = np.zeros([steps])
    T2 = np.zeros([steps])
    T3 = np.zeros([steps])

    S1 = np.zeros([steps])
    S2 = np.zeros([steps])
    S3 = np.zeros([steps])

    T1[0] = initT1
    T2[0] = initT2
    T3[0] = initT3

    S1[0] = initS1
    S2[0] = initS2
    S3[0] = initS3

    q = np.zeros([steps])

    q[0] = k * (alpha * (T3[0] - T1[0]) - beta * (S3[0] - S1[0]))

    if q[0] >= 0:  # nonnegative flow solutions
        for step in range(steps - 1):
            T1[step + 1] = Tdeltat * (lam * (Tp - T1[step]) + kd * (T2[step] - T1[step]) + q[step] * (T2[step] - T1[step])) + T1[step]

            T2[step + 1] = Tdeltat * (lam * (Te - T2[step]) + kd * (T1[step] + T3[step] - 2 * T2[step]) + q[step] * (T3[step] - T2[step])) + T2[step]

            T3[step + 1] = Tdeltat * (lam * (Tp - T3[step]) + kd * (T2[step] - T3[step]) + q[step] * (T1[step] - T3[step])) + T3[step]

            S1[step + 1] = Sdeltat * (-n_prec + kd * (S2[step] - S1[step]) + q[step] * (S2[step] - S1[step])) + S1[step]

            S2[step + 1] = Sdeltat * (n_prec + s_prec + kd * (S1[step] + S3[step] - 2 * S2[step]) + q[step] * (S3[step] - S2[step])) + S2[step]

            S3[step + 1] = Sdeltat * (-s_prec + kd * (S2[step] - S3[step]) + q[step] * (S1[step] - S3[step])) + S3[step]

            q[step + 1] = k * (alpha * (T3[step + 1] - T1[step + 1] ) - beta * (S3[step + 1] - S1[step + 1]))

        return (T1, T2, T3), (S1, S2, S3), q
    else:  # negative flow solutions
        for step in range(steps - 1):
            T1[step + 1] = Tdeltat * (lam * (Tp - T1[step]) + kd * (T2[step] - T1[step]) + abs(q[step]) * (T3[step] - T1[step])) + T1[step]

            T2[step + 1] = Tdeltat * (lam * (Te - T2[step]) + kd * (T1[step] + T3[step] - 2 * T2[step]) + abs(q[step]) * (T1[step] - T2[step])) + T2[step]

            T3[step + 1] = Tdeltat * (lam * (Tp - T3[step]) + kd * (T2[step] - T3[step]) + abs(q[step]) * (T2[step] - T3[step])) + T3[step]
            
            S1[step + 1] = Sdeltat * (-n_prec + kd * (S2[step] - S1[step]) + abs(q[step]) * (S3[step] - S1[step])) + S1[step]

            S2[step + 1] = Sdeltat * (n_prec + s_prec + kd * (S1[step] + S3[step] - 2 * S2[step]) + abs(q[step]) * (S1[step] - S2[step])) + S2[step]

            S3[step + 1] = Sdeltat * (-s_prec + kd * (S2[step] - S3[step]) + abs(q[step]) * (S2[step] - S3[step])) + S3[step]

            q[step + 1] = k * (alpha * (T3[step + 1] - T1[step + 1]) - beta * (S3[step + 1] - S1[step + 1]))

        return (T1, T2, T3,), (S1, S2, S3), q 


###Using the reference values for kd and q as seen in Table 3 of Longworth Paper
### typical example: kd = 1*10**-10, q = 5.43*10**-10

### This program is helpful in finding the steady state flow solutions for various K_d
### Hopefully our outputs for various initial conditions will align with values found 
###in Figure 6 of the Longworth Paper. 


kdsim =  1*10**-10
timeTemp = np.linspace(time0, timeTempf / (1 * 10 ** 10), steps)
timeSal = np.linspace(time0, timeSalf / (1 * 10 ** 10), steps)

Temp, Sal, Q = euler_forward(1,10, .1 , 35, 35.7, 34.3, kdsim)
Temp = np.transpose(Temp)
Sal = np.transpose(Sal)






if Q[100000] - Q[99000] <= 1*10**-1:
    print(Q[100000])
else:
    print("Not a Steady State")
    

    
###For this simulation all outputs match the expected flows from 
### Longworth Table 3 All of Case Two Match Expectations.