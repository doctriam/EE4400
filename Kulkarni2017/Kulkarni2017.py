##############################################################################
# TITLE:  Shruti Kulkarni Synapse Model
# DESCRIPTION:  https://arxiv.org/pdf/1711.03637.pdf - Section II
# VERSION:  1.0
# VERSION NOTES:
#     ""
# AUTHOR:  Kenny Haynie
##############################################################################

from numpy import *
import matplotlib.pyplot as plt

def membrane_potential():
    # Section II - Equation (1)
    # Evolution of membrane potential, V_m(t):
    # C*(dV_m(t)/dt)=-g_L(V_m(t)-E_L)+I(t)
    C=300e-12 # membrane capacitance in (pico)Farads
    g_L=30e-9 # membrane leak conductance in (nano)Siemens
    E_L=0 # leak reversal potential

    # Membrane potential in (milli)Volts
    # Value range pulled from Anwani 10/2018
#    V_m=random.random_sample(105)*(-0.1)
#
#    plt.plot(V_m)
#    plt.ylabel('test')
#    plt.xlabel('t')
#    plt.show()
    return

def post_synaptic_current():
    # k=input neuron, l=output neuron
    w_kl=1 # synaptic weight currently uncalculated

    tau_1=5e-3 # represents synaptic kernel time
    tau_2=1.25e-3 # represents synaptic kernel time

    c_k=0 # base

def pixel_current():
    k=random.randint(0,255) # pixel value
    I_p=101.2e-12 # scaling factor
    I_0=2.7e-9 # minimum current above which LIF neuron can generate a spike

    # Equation (4)
    # - current applied to neuron
    i_k = I_0 + k*I_p

    print("Pixel Current i(k) = %s nA" % (i_k*10**9))
    return

pixel_current()
