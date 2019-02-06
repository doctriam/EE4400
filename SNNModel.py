##############################################################################
# TITLE:  ""
# DESCRIPTION:  http://www.mjrlab.org/wp-content/uploads/2014/05/
#               network_python_tutorial2013.pdf
# AUTHOR:  Kenny Haynie
##############################################################################

from pylab import *
from scipy.sparse import csr_matrix
from scipy.linalg import circulant
from numpy import *
import sys

# System Controls
CURSOR_UP='\x1b[1A'
ERASE_LINE='\x1b[2K'

# Shared Parameters
tmax=1000 # time range
dt=0.5 # step size
E_L=-65 # membrane resting potential
b=0.2 # sensitivity; unknown origin for value
Iapp=10 # applied current
tr=array([200.,700])/dt # stm time
T=int(ceil(tmax/dt))

def main_menu():
    global tmax,dt,n,E_L,b,Iapp,tr,T

    # List of functions
    print("1. Neuron Model")
    print("2. Synapse Model")
    print("3. Excitatory and Inhibitory Neurons")
    print("4. Recurrent Spiking Neural Network with 1000 neurons")
    print("5. Ring Structure")
    print("0. Exit")

    # User Input
    menuOption=int(input("Choose a number from the menu above: "))

    # Menu options array
    options = {1: neuron_model,
               2: synapse_model,
               3: excit_inhib,
               4: recurrent_network,
               5: ring_structure
              }

    # Clear menu/prompt
    for n in range(0,len(options)+2):
        sys.stdout.write(CURSOR_UP)
        sys.stdout.write(ERASE_LINE)

    # Run menu option
    if menuOption in range(1,len(options)+1):
        options[menuOption]()
    else:
        sys.exit()

    #Reload menu
    main_menu()
    return

def neuron_model():
    a=0.02 # decay rate; possibly g_L/C_m
    d=8. # reset for u(t); unknown origin for value
    v=zeros(T) # vector for membrane potential
    u=zeros(T) # UNKNOWN vector
    v[0]=-70 # resting potential
    u[0]=-14 # steady state

    # For-loop over time
    for t in arange(T-1): # for all T
        if t>tr[0] and t<tr[1]: # for t in range of tr array
            I=Iapp # current is 10 (A?)
        else:
            I=0 # current is 0 outside of range of tr array

        if v[t]<35: # upper threshold for membrane potential
            # Follows Hodgkin-Huxley model
            dv=(0.04*v[t]+5)*v[t]+140-u[t]
            v[t+1]=v[t]+(dv+I)*dt
            du=a*(b*v[t]-u[t])
            u[t+1]=u[t]+dt*du
        else: # resets to E_L at t+1 when hitting upper threshold
            v[t]=35
            v[t+1]=E_L
            u[t+1]=u[t]+d

    # Plot
    figure(1)
    tvec=arange(0.,tmax,dt)
    plot(tvec,v,'b',label='Voltage trace')
    xlabel('Time [ms]')
    ylabel('Membrane voltage [mV]')
    title("""A single qIF neuron with current step input6""")
    show()
    return

def synapse_model():
    a=0.02 # decay rate; possibly g_L/C_m
    d=8. # reset for u(t); unknown origin for value
    tau_s=10 # synapse decay in ms
    rate_in=2 # input rate
    n_in=100 # number of inputs
    w_in=0.07 # input weight
    W_in=w_in*ones(n_in) # weight vector

    v=zeros(T)
    u=zeros(T)
    v[0]=-70
    u[0]=-14
    s_in=zeros(n_in) # synaptic variable
    E_in=zeros(n_in) # reverse potential
    prate=dt*rate_in*1e-3 # abbrev?

    # For Loop
    for t in arange(T-1):
        if t>tr[0] and t<tr[1]:
            p=uniform(size=n_in)<prate; # Get input Poisson Spikes
        else:
            p=0

        # Calculate input current
        s_in=(1-dt/tau_s)*s_in+p
        I=dot(W_in,s_in*E_in)
        I-=dot(W_in,s_in)*v[t]

        if v[t]<35:
            dv=(0.04*v[t]+5)*v[t]+140-u[t]
            v[t+1]=v[t]+(dv+I)*dt
            du=a*(b*v[t]-u[t])
            u[t+1]=u[t]+dt*du
        else:
            v[t]=35
            v[t+1]=E_L
            u[t+1]=u[t]+d

    # Plot
    figure(2)
    tvec=arange(0.,tmax,dt)
    plot(tvec,v,'b',label='Voltage trace')
    xlabel('Time [ms]')
    ylabel('Membrane voltage [mV]')
    title("""A single qIF neuron with %d Poisson inputs""" % n_in)
    show()
    return

def excit_inhib():
#    tmax=1000 # time range
#    dt=0.5 # step size
    n=1000 # number of neurons for multi-neuron functions
#    E_L=-65 # membrane resting potential
#    b=0.2 # sensitivity; unknown origin for value
#    Iapp=10 # applied current
#    tr=array([200.,700])/dt # stm time
#    T=int(ceil(tmax/dt))
    pinh=0.2 # probability of inhibited neuron
    inh=(uniform(size=n)<pinh) # whether inhibited
    exc=logical_not(inh) # whether excitatory
    a=inh.choose(0.02,0.1) # choose between inh or exc
    d=inh.choose(8,2) # choose between inh or exc
    tau_s=10
    rate_in=2
    n_in=100
    w_in=0.07
    pconn_in=0.1 # input connection probability
    C=uniform(size=(n,n_in))<pconn_in # array of connections
    W_in=C.choose(0,w_in) # weight matrix
    v=zeros((T,n)) # now a matrix
    u=zeros((T,n)) # now a matrix
    v[0]=-70 # set 1st row values
    u[0]=-14 # set 1st row values
    s_in=zeros(n_in)
    E_in=zeros(n_in)
    prate=dt*rate_in*1e-3

    # For Loop
    for t in arange(T-1):
        if t>tr[0] and t<tr[1]:
            p=uniform(size=n_in)<prate;
        else:
            p=0

        s_in=(1-dt/tau_s)*s_in+p
        I=W_in.dot(s_in*E_in)
        I-=W_in.dot(s_in)*v[t]

        # Changed to handle multiple inputs
        fired=v[t]>=35  # no more if statement; check for fired status
        dv=(0.04*v[t]+5)*v[t]+140-u[t]
        v[t+1]=v[t]+(dv+I)*dt
        du=a*(b*v[t]-u[t])
        u[t+1]=u[t]+dt*du
        v[t][fired]=35
        v[t+1][fired]=E_L
        u[t+1][fired]=u[t][fired]+d[fired]

    # Plot
    tspk,nspk=nonzero(v==35)
    idx_i=in1d(nspk,nonzero(inh)[0]) # determine inh/exc
    idx_e=logical_not(idx_i)

    figure(3)
    plot(tspk[idx_e]*dt,nspk[idx_e],'k.',
         label='Exc.',markersize=2)
    plot(tspk[idx_i]*dt,nspk[idx_i],'r.',
         label='Inh.',markersize=2)
    xlabel('Time[ms]')
    ylabel('Neuron number [\#]')
    xlim((0,tmax))
    title("""An unconnected network of %d qIF neurons""" % n)
    legend(loc='upper right')
    show()

def recurrent_network():
    n=1000 # number of neurons for multi-neuron functions
    pinh=0.2
    inh=(uniform(size=n)<pinh)
    exc=logical_not(inh)
    a=inh.choose(0.02,0.1)
    d=inh.choose(8,2)
    tau_s=10

    # Recurrent parameters
    w=0.005 # average recurrent weight
    pconn=0.1 # recurrent connection probability
    scaleEI=2 # scale I->E
    g_sc=0.002 # scale of gamma
    E=inh.choose(0,-85)

    # Weight matrix
    W=zeros((n,n))
    C=uniform(size=(n,n))
    idx=nonzero(C<pconn) # sparse connectivity
    W[idx]=gamma(w/g_sc,scale=g_sc,size=idx[0].size)
    W[ix_(exc,inh)]*=scaleEI # submat indexing
    W=csr_matrix(W) # make row sparse

    rate_in=2
    n_in=100
    w_in=0.07
    pconn_in=0.1
    C=uniform(size=(n,n_in))<pconn_in
    W_in=C.choose(0,w_in)

    v=zeros((T,n))
    u=zeros((T,n))
    v[0]=-70
    u[0]=-14
    s_in=zeros(n_in)
    E_in=zeros(n_in)
    prate=dt*rate_in*1e-3
    s=zeros(n)

    # For Loop
    for t in arange(T-1):
        if t>tr[0] and t<tr[1]:
            p=uniform(size=n_in)<prate;
        else:
            p=0

        s_in=(1-dt/tau_s)*s_in+p
        I=W_in.dot(s_in*E_in)
        I-=W_in.dot(s_in)*v[t]
        fired=v[t]>=35

        # Recurrent input
        s=(1-dt/tau_s)*s+fired
        Isyn=W.dot(s*E)-W.dot(s)*v[t]
        I+=Isyn

        dv=(0.04*v[t]+5)*v[t]+140-u[t]
        v[t+1]=v[t]+(dv+I)*dt
        du=a*(b*v[t]-u[t])
        u[t+1]=u[t]+dt*du
        v[t][fired]=35
        v[t+1][fired]=E_L
        u[t+1][fired]=u[t][fired]+d[fired]

    tspk,nspk=nonzero(v==35)
    idx_i=in1d(nspk,nonzero(inh)[0])
    idx_e=logical_not(idx_i)

    figure(4)
    plot(tspk[idx_e]*dt,nspk[idx_e],'k.',
         label='Exc.',markersize=2)
    plot(tspk[idx_i]*dt,nspk[idx_i],'r.',
         label='Inh.',markersize=2)
    xlabel('Time[ms]')
    ylabel('Neuron number[\#]')
    xlim((0,tmax))
    title("""A recurrent network of %d qIF neurons""" % n)
    legend(loc='upper right')
    show()

def ring_structure():
    n=1000 # number of neurons for multi-neuron functions
    pinh=0.2
    inh=(uniform(size=n)<pinh)
    exc=logical_not(inh)
    a=inh.choose(0.02,0.1)
    d=inh.choose(8,2)
    tau_s=10

    # Recurrent parameters
    width=pi/4 # half-width of the orientation tuning
    w=0.005
    pconn=0.4 # set higher connectivity probability
    scaleEI=2
    g_sc=0.002
    E=inh.choose(0,-85)

    # Weight matrix
    W=zeros((n,n))
    C=uniform(size=(n,n))
    idx=nonzero(C<pconn)
    W[idx]=gamma(w/g_sc,scale=g_sc,size=idx[0].size)
    W[ix_(exc,inh)]*=scaleEI
    theta=linspace(0,2*pi,n) # new
    R=circulant(cos(theta))>cos(width) # new
    W[:,exc]=where(R[:,exc],W[:,exc],0) # new
    W=csr_matrix(W)

    rate_in=2
    inwidth=pi/2 # new
    n_in=100
    w_in=0.07
    pconn_in=0.2 # set higher probablity
    C=uniform(size=(n,n_in))<pconn_in
    W_in=C.choose(0,w_in)
    W_in[int(n/2):,:]=0 # new

    v=zeros((T,n))
    u=zeros((T,n))
    v[0]=-70
    u[0]=-14
    s_in=zeros(n_in)
    E_in=zeros(n_in)
    prate=dt*rate_in*1e-3
    s=zeros(n)

    # For Loop
    for t in arange(T-1):
        if t>tr[0] and t<tr[1]:
            p=uniform(size=n_in)<prate;
        else:
            p=0

        s_in=(1-dt/tau_s)*s_in+p
        I=W_in.dot(s_in*E_in)
        I-=W_in.dot(s_in)*v[t]
        fired=v[t]>=35

        # Recurrent input
        s=(1-dt/tau_s)*s+fired
        Isyn=W.dot(s*E)-W.dot(s)*v[t]
        I+=Isyn

        dv=(0.04*v[t]+5)*v[t]+140-u[t]
        v[t+1]=v[t]+(dv+I)*dt
        du=a*(b*v[t]-u[t])
        u[t+1]=u[t]+dt*du
        v[t][fired]=35
        v[t+1][fired]=E_L
        u[t+1][fired]=u[t][fired]+d[fired]

    tspk,nspk=nonzero(v==35)
    idx_i=in1d(nspk,nonzero(inh)[0])
    idx_e=logical_not(idx_i)

    figure(5)
    plot(tspk[idx_e]*dt,nspk[idx_e],'k.',
         label='Exc.',markersize=2)
    plot(tspk[idx_i]*dt,nspk[idx_i],'r.',
         label='Inh.',markersize=2)
    xlabel('Time[ms]')
    ylabel('Neuron number[\#]')
    xlim((0,tmax))
    title("""A recurrent network of %d qIF neurons""" % n)
    legend(loc='upper right')
    show()

main_menu()
