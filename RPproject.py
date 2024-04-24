import numpy as np
import matplotlib.pyplot as plt
import sirtipy as sp
import astropy.units as u
import astropy.constants as const

def Kroupa01_IMF(M):
    if M < 0.08:
        return M**-0.3
    if M < 0.5:
        return M**-1.3
    if M < 1:
        return M**-2.3
    return M**-2.7

def M_from_T(T):
    L = (T/5700)**4
    return L**(1/3.5)

def T_from_M(M):
    if M >= 55:
        L = 32000*M
    if M < 55:
        L = 1.4*M**3.5
    if M < 2:
        L = M**4
    if M < 0.43:
        L = 0.23*M**2.3
    
    R = M
    if M >= 1.7:
        R = M**(1/2)
    return 5700*L**(1/4)/R**(1/2)

def T_from_M_simple(M):
    T = 5700*M**(2.5/4)
    return T

def Balmer(T, f):
    if T<=10000:
        l = const.c.to(u.nm/u.s).value/f
        if l < 364.5:
            return 0.0001
    return 1

def IMF_T(T):
    M = M_from_T(T)
    return Kroupa01_IMF(M)

def IMF(M):
    return Kroupa01_IMF(M)

def dM(T):
    return 4/3.5 * T**(0.5/3.5)

def dt(M):
    return -10**10/3.5 * M**(-4.5/3.5)

def Mmax_from_time(t):
    return (10**10/t)**(1/2.5)

def gal_blackbody_T(f, Mmax=120):
    M = np.geomspace(0.05,Mmax)
    T = [T_from_M(m) for m in M]
    BB = [Balmer(t,f)*IMF_T(t)*dM(t)*sp.blackbody_intensity(t,f) for t in T]
    return np.trapz(BB, x=T)

def gal_blackbody(f, Mmax=120):
    M = np.geomspace(0.05,Mmax)
    BB = [Balmer(T_from_M(m),f)*IMF(m)*dM(T_from_M(m))*sp.blackbody_intensity(T_from_M(m),f) for m in M]
    return np.trapz(BB, x=M)

def gal_blackbody_no_balmer(f, Mmax=120):
    M = np.geomspace(0.05,Mmax)
    BB = [IMF(m)*dM(T_from_M(m))*sp.blackbody_intensity(T_from_M(m),f) for m in M]
    return np.trapz(BB, x=M)   

def exp_SFR(t, tau=None):
    if not tau:
        tau = 1e8
    return np.exp(-t/tau)

def star_formation_blackbody(f, t1, t2, SFR=None, tau=None, balmer=True):
    tstart = np.max([t1,t2])
    tstop = np.min([t1,t2])
    time = np.linspace(tstart, tstop, 20)
    Mmax = [Mmax_from_time(t) for t in time]
    
    # SFR is the star formation rate function
    # If not given, assume constant star formation rate
    if balmer:
        bbfunc = gal_blackbody
    else:
        bbfunc = gal_blackbody_no_balmer
        
    if SFR:
        BB = [-1*SFR(t,tau)*bbfunc(f, M)*dt(M) for (t,M) in zip(time, Mmax)]
    else:
        BB = [-1*bbfunc(f, M)*dt(M) for (t,M) in zip(time, Mmax)]
    return np.trapz(BB,x=Mmax)

def generate_blackbody(farray, t1, t2, SFR=None, tau=None, balmer=True):
    res = []
    for f in farray:
        B = star_formation_blackbody(f, t1, t2, SFR=SFR, tau=tau, balmer=balmer)
        if B == np.nan:
            B = 0
        res.append(B)
    return res
