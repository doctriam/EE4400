##############################################################################
# TITLE:  ""
# DESCRIPTION:  http://www.mjrlab.org/wp-content/uploads/2014/05/
#               network_python_tutorial2013.pdf
# AUTHOR:  Kenny Haynie
##############################################################################

from pylab import *
from scipy.sparse import csr_matrix
from scipy.linalg import circulant

def neuron_model():
    # Initialize paramaters
    tmax=1000 # t goes from 0 to 1000 ms
    dt=0.5 # in 0.5 ms intervals
    # Neuron network parameters
    a=0.02 # RS,IB: 0.02, FS: 0.1
    b=0.2 # RS,IB: 0.2
    c=-65 # RS,FS: -65 IB: -55
    d=8 # RS: 8, IB: 4, FS: 2

    # Input parameters
    Iapp=10 # Applied current
    tr=array([200.,700])/dt # stm time
    print(tr)

    # Reserve memory
    T=int(ceil(tmax/dt))
    print((ceil(tmax/dt)))
    v=zeros(T)
    u=zeros(T)
    v[0]=-70 # resting potential
    u[0]=-14 # steady state

    # For-loop over time
    for t in arange(T-1):
        if t>tr[0] and t<tr[1]:
            I=Iapp
        else:
            I=0

        if v[t]<35:
            dv=(0.04*v[t]+5)*v[t]+140-u[t]
            v[t+1]=v[t]+(dv+I)*dt
            du=a*(b*v[t]-u[t])
            u[t+1]=u[t]+dt*du
        else:
            v[t]=35
            v[t+1]=c
            u[t+1]=u[t]+d

    figure(1)
    tvec=arange(0.,tmax,dt)
    plot(tvec,v,'b',label='Voltage trace')
    xlabel('Time [ms]')
    ylabel('Membrane voltage [mV]')
    title("""A single qIF neuron with current step input6""")
    show()
    return

def synapse_model():
    # parameters
    tmax=1000
    dt=0.5
    a=0.02
    b=0.2
    c=-65
    d=8
    tau_s=10
    tr=array([200.,700])/dt
    rate_in=2
    n_in=100
    w_in=0.07
    W_in=w_in*ones(n_in)

    T=int(ceil(tmax/dt))
    v=zeros(T)
    u=zeros(T)
    v[0]=-70
    u[0]=-14
    s_in=zeros(n_in)
    E_in=zeros(n_in)
    prate=dt*rate_in*1e-3

    # For Loop
    for t in arange(T-1):
        if t>tr[0] and t<tr[1]:
            p=uniform(size=n_in)<prate; # Get input Poisson Spikes
        else:
            p=0

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
            v[t+1]=c
            u[t+1]=u[t]+d

    figure(2)
    tvec=arange(0.,tmax,dt)
    plot(tvec,v,'b',label='Voltage trace')
    xlabel('Time [ms]')
    ylabel('Membrane voltage [mV]')
    title("""A single qIF neuron with %d Poisson inputs""" % n_in)
    show()
    return

def excit_inhib():
    # parameters
    tmax=1000
    dt=0.5
    b=0.2
    c=-65
    tau_s=10
    tr=array([200.,700])/dt
    rate_in=2
    n_in=100
    w_in=0.07

    T=int(ceil(tmax/dt))
    s_in=zeros(n_in)
    E_in=zeros(n_in)
    prate=dt*rate_in*1e-3

    # New from synapse_model
    n=1000 # number of neurons
    pinh=0.2 # probability of inhibited neuron
    inh=(uniform(size=n)<pinh) # whether inhibited
    exc=logical_not(inh)
    a=inh.choose(0.02,0.1)
    d=inh.choose(8,2)
    pconn_in=0.1
    C=uniform(size=(n,n_in))<pconn_in
    W_in=C.choose(0,w_in)
    v=zeros((T,n))
    u=zeros((T,n))
    v[0]=-70
    u[0]=-14

    # For Loop
    for t in arange(T-1):
        if t>tr[0] and t<tr[1]:
            p=uniform(size=n_in)<prate; # Get input Poisson Spikes
        else:
            p=0

        s_in=(1-dt/tau_s)*s_in+p

        # Changed to handle multiple inputs
        I=W_in.dot(s_in*E_in)
        I-=W_in.dot(s_in)*v[t]
        fired=v[t]>=35
        dv=(0.04*v[t]+5)*v[t]+140-u[t]
        v[t+1]=v[t]+(dv+I)*dt
        du=a*(b*v[t]-u[t])
        u[t+1]=u[t]+dt*du
        v[t][fired]=35
        v[t+1][fired]=c
        u[t+1][fired]=u[t][fired]+d[fired]

    tspk,nspk=nonzero(v==35)
    idx_i=in1d(nspk,nonzero(inh)[0])
    idx_e=logical_not(idx_i)

    figure(3)
    plot(tspk[idx_e]*dt,nspk[idx_e],'k.',
         label='Exc.',markersize=2)
    plot(tspk[idx_i]*dt,nspk[idx_i],'r.',
         label='Inh.',markersize=2)
    xlabel('Time[ms]')
    ylabel('neuronnumber[\#]')
    xlim((0,tmax))
    title("""An unconnected network of %d qIF neurons""" % n)
    legend(loc='upper right')
    show()

def recurrent_network():
    # parameters
    tmax=1000
    dt=0.5

    n=1000
    pinh=0.2
    inh=(uniform(size=n)<pinh)
    exc=logical_not(inh)
    a=inh.choose(0.02,0.1)
    b=0.2
    c=-65
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

    tr=array([200.,700])/dt
    rate_in=2
    n_in=100
    w_in=0.07
    pconn_in=0.1
    C=uniform(size=(n,n_in))<pconn_in
    W_in=C.choose(0,w_in)

    T=int(ceil(tmax/dt))
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
        v[t+1][fired]=c
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
    ylabel('neuronnumber[\#]')
    xlim((0,tmax))
    title("""A recurrent network of %d qIF neurons""" % n)
    legend(loc='upper right')
    show()

def ring_structure():
    # parameters
    tmax=1000
    dt=0.5

    n=1000
    pinh=0.2
    inh=(uniform(size=n)<pinh)
    exc=logical_not(inh)
    a=inh.choose(0.02,0.1)
    b=0.2
    c=-65
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

    tr=array([200.,700])/dt
    rate_in=2
    inwidth=pi/2 # new
    n_in=100
    w_in=0.07
    pconn_in=0.2 # set higher probablity
    C=uniform(size=(n,n_in))<pconn_in
    W_in=C.choose(0,w_in)
    W_in[int(n/2):,:]=0 # new

    T=int(ceil(tmax/dt))
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
        v[t+1][fired]=c
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
    ylabel('neuronnumber[\#]')
    xlim((0,tmax))
    title("""A recurrent network of %d qIF neurons""" % n)
    legend(loc='upper right')
    show()

neuron_model()
synapse_model()
excit_inhib()
recurrent_network()
ring_structure()
