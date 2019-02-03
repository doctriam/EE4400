##############################################################################
# TITLE:  Shruti Kulkarni Synapse Model
# DESCRIPTION:  https://arxiv.org/pdf/1711.03637.pdf - Section II
# VERSION:  1.0
# VERSION NOTES:
#     ""
# AUTHOR:  Kenny Haynie
##############################################################################

from numpy import *
import random

def pixel_current():
    k=random.randint(0,255) # pixel value
    I_p=101.2e-12 # scaling factor
    I_0=2.7e-9 # minimum current above which LIF neuron can generate a spike

    # Equation (4)
    # - current applied to neuron
    i_k = I_0 + k*I_p

    print(i_k)
    return



