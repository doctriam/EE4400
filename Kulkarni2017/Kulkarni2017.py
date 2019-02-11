#!/usr/bin/python
"""
Title: "Learning and Real-time Classification of Hand-written Digits with
    Spiking Neural Networks"
Original Author: Shruti Kulkarni
Python Author: Kenny Haynie
"""

import numpy
import matplotlib.pyplot as plt

t_sim = 0.1 # simulation time in seconds
dt = 0.1e-3 # time step in (milli)seconds
M = round(t_sim/dt) # number of points in t_array
t_ref = 3e-3 # reference time
t = numpy.arange(0.0, t_sim, dt) # time array

# Neuron Parameters
C = 300e-12 # membrane capacitance in Farads
gL = 30e-9 # membrane leak conductance in Siemens
EL = -70e-3 # resting potential in Volts
VT = 20e-3 # threshold potential in Volts

# Input Pixels
N_pixel = 256 # number of pixels
pix = numpy.arange(0, N_pixel) # 1-D array from 0 to 256
#pix = numpy.random.rand(1, 256)*N_pixel # randomized 1-D array from 0 to 256
w = 1.012e-10 # pixel weight
Ic = 2.7e-9 # minimum current for neuron to spike
Vm = numpy.zeros((N_pixel, M)) # zero array to hold Vm values
Y_spk = numpy.zeros((N_pixel, M)) # zero array to hold signum values
isref_n = numpy.zeros((N_pixel)) # last spike time array: see Eq. 10
I_in = Ic+pix*w # input current; Equation 6

for i in range(0, M-1):
    k1 = (1/C)*(I_in-(gL*(Vm[:, i]-EL))) # Equation 7
    k2 = (1/C)*(I_in-(gL*(Vm[:, i]+(k1*dt)-EL))) # Equation 8
    Vm[:, i+1] = Vm[:, i]+(dt*(k1+k2)/2) # Equation 9
    Vm[numpy.where(t[i]-isref_n < t_ref), i+1] = EL
    spind = numpy.sign(Vm[:, i+1]-VT) # find sign of difference b/t Vm and VT
    Vm[Vm[:, i+1] < EL, i+1] = EL # correct for under resting potential

    # Perform if over threshold
    if max(spind) > 0:
        resetfind_n = numpy.where(spind > 0) # find where Vm > VT
        isref_n[resetfind_n] = t[i] # mark time of reset
        Vm[resetfind_n, i] = VT # Set Vm to threshold
        Y_spk[resetfind_n, i] = 1 # Mark boolean true for spike at t[i]

[x, y] = numpy.where(Y_spk) # find where spike occured
plot1 = plt.plot(y/10, x, 'o') # convert y to ms
plt.title('Neuron Spike Trains')
plt.xlabel('Time (ms)')
plt.ylabel('Pixel Value')
plt.show()

spk_cnt = numpy.sum(Y_spk, axis=1)
spk_freq = spk_cnt/t_sim
plot2 = plt.plot(spk_freq)
plt.title('Spike Frequency vs Pixel Values')
plt.show()
