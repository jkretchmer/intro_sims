#!/usr/bin/python

import numpy as np
from scipy.special import erf

#####################################################################

def make_ham_diatomic_sto3g(typ,R):
    #subroutine to generate the overlap matrix, S, core hamiltonian, Hcore, 
    #and a 4-index matrix of the 2-e integrals, g, for the diatomics H2 or HeH+
    #in the STO-3G basis

    ng = 3  #number of gaussians, hard-coded to 3 for sto-3g

    zeta_h_sq = 1.24**2
    zeta_he_sq = 2.0925**2

    #define first atom as either hydrogen or he
    if typ=="h2":
        z1 = 1.0
        zeta_1_sq = zeta_h_sq
    elif typ=="hehp":
        z1 = 2.0
        zeta_1_sq = zeta_he_sq
    else:
        print("Eror in make_ham_diatomic_sto3g(): typ must be either 'h2' or 'hehp'")
        sys.exit(1)

    #define second atom always as hydrogen
    z2 = 1.0
    zeta_2_sq = zeta_h_sq

    #calculate nuclear repulsion
    Enuc = z1*z2/R

    #define vectors for the position of the two atoms
    Rvec = np.zeros(2)
    zvec = np.zeros(2)
    Rvec[0] = 0.0
    Rvec[1] = R
    zvec[0] = z1
    zvec[1] = z2

    #define the contraction coefficients and exponents for sto-3g for each atom
    d_coef = np.zeros( (ng,2) )
    alpha = np.zeros( (ng,2) )

    d_coef[0,0] = d_coef[0,1] = 0.444635
    d_coef[1,0] = d_coef[1,1] = 0.535328
    d_coef[2,0] = d_coef[2,1] = 0.154329

    alpha[0,0] = zeta_1_sq*0.109818
    alpha[0,1] = zeta_2_sq*0.109818
    alpha[1,0] = zeta_1_sq*0.405771
    alpha[1,1] = zeta_2_sq*0.405771
    alpha[2,0] = zeta_1_sq*2.227660
    alpha[2,1] = zeta_2_sq*2.227660

    #multiply d_coef by normalization constant from gaussian basis fcns
    for i in np.arange(0,ng):
        for j in np.arange(0,2):
            d_coef[i,j] = d_coef[i,j]*(2*alpha[i,j]/np.pi)**(3./4.)

    #initialize matrices
    Nb = 2 #number of basis functions
    S = np.zeros( (Nb,Nb) ) #overlap matrix
    T = np.zeros( (Nb,Nb) ) #kinetic energy
    Vnuc = np.zeros( (Nb,Nb) ) #e- nuclei attraction
    Hcore = np.zeros( (Nb,Nb) ) #core Hamiltonian
    g = np.zeros( (Nb,Nb,Nb,Nb) ) #4D tensor for 2e integrals

    for p in np.arange(0,Nb):
        for q in np.arange(0,Nb):
            S[p,q] = calc_S(p,q,Rvec,ng,alpha,d_coef)
            T[p,q] = calc_T(p,q,Rvec,ng,alpha,d_coef)
            Vnuc[p,q] = calc_Vnuc(p,q,Rvec,ng,alpha,d_coef,zvec)
            for r in np.arange(0,Nb):
                for s in np.arange(0,Nb):
                    g[p,q,r,s] = calc_g(p,q,r,s,Rvec,ng,alpha,d_coef)

    Hcore = T+Vnuc

    return S,Hcore,g,Enuc

#####################################################################

def calc_S(mu,nu,R,ng,alpha,d_coef):
    #subroutine to calculate the mu,nu element of the overlap matrix S
    #note that d_coef takes care of the normalization constants

    Rmunu=R[mu]-R[nu]

    smunu=0.0
    for i in np.arange(0,ng):
        for j in np.arange(0,ng):
            if mu == nu:
                smunu = 1
            else:
                a=alpha[i,mu]
                b=alpha[j,nu]

                spq = (np.pi/(a+b))**1.5*np.exp(-a*b/(a+b)*Rmunu**2)
                smunu += d_coef[i,mu]*d_coef[j,nu]*spq

    return smunu

#####################################################################

def calc_T(mu,nu,R,ng,alpha,d_coef):
    #subroutine to calculate the mu,nu element of the kinetic energy matrix T
    #note that d_coef takes care of the normalization constants

    Rmunu=R[mu]-R[nu]

    Tmunu=0.0
    for i in np.arange(0,ng):
        for j in np.arange(0,ng):
            a=alpha[i,mu]
            b=alpha[j,nu]

            Tpq = a*b/(a+b)*(3-2*a*b/(a+b)*Rmunu**2)*(np.pi/(a+b))**1.5*np.exp(-a*b/(a+b)*Rmunu**2)
            Tmunu += d_coef[i,mu]*d_coef[j,nu]*Tpq

    return Tmunu


#####################################################################

def calc_Vnuc(mu,nu,R,ng,alpha,d_coef,zvec):
    #subroutine to calculate the mu,nu element of the e-nuclei attraction matrix
    #note that d_coef takes care of the normalization constants

    natms = len(R)

    Rmunu=R[mu]-R[nu]

    Vmunu=0.0
    for i in np.arange(0,ng):
        for j in np.arange(0,ng):
            a=alpha[i,mu]
            b=alpha[j,nu]

            Rp=(a*R[mu]+b*R[nu])/(a+b)

            for k in np.arange(0,natms):

                #position and charge of the interacting nuclei
                Ratm = R[k]
                Zatm = zvec[k]

                Vpq = -2*np.pi/(a+b)*Zatm*np.exp(-a*b/(a+b)*Rmunu**2)*F0((a+b)*(Rp-Ratm)**2)
                Vmunu += d_coef[i,mu]*d_coef[j,nu]*Vpq

    return Vmunu


#####################################################################

def calc_g(p,q,r,s,R,ng,alpha,d_coef):
    #subroutine to calculate the p,q,r,s element of the two-electron tensor
    #i.e. g[p,q,r,s] = (pq|rs) in spatial chemist's notation
    #note that d_coef takes care of the normalization constants
    #note change in index notation from previous subroutines
    #now p,q,r,s represent basis function indices and i,j,k,l represent primitive gaussian indices

    Rp = R[p]
    Rq = R[q]
    Rr = R[r]
    Rs = R[s]

    gpqrs = 0.0
    for i in np.arange(0,ng): #loop over primitives, i, in basis fcn p
        a = alpha[i,p]
        da = d_coef[i,p]
        for j in np.arange(0,ng): #loop over primitives, j, in basis fcn q
            b = alpha[j,q]
            db = d_coef[j,q]

            Rppp = (a*Rp+b*Rq)/(a+b)

            for k in np.arange(0,ng): #loop over primitives, k, in basis fcn r
                c = alpha[k,r]
                dc = d_coef[k,r]
                for l in np.arange(0,ng): #loop over primitives, l, in basis fcn s
                    d = alpha[l,s]
                    dd = d_coef[l,s]

                    Rqqq = (c*Rr+d*Rs)/(c+d)

                    gijkl = 2*np.pi**(5./2.)/( (a+b)*(c+d)*np.sqrt(a+b+c+d) )*(
                            np.exp(-a*b/(a+b)*(Rp-Rq)**2-c*d/(c+d)*(Rr-Rs)**2)
                            *F0( (a+b)*(c+d)/(a+b+c+d)*(Rppp-Rqqq)**2 ) )
                    gpqrs += da*db*dc*dd*gijkl

    return gpqrs


#####################################################################


def F0(x):

    #function to calculate necessary error function calculations
    if x<1e-3:
        return 1.0
    else:
        return 0.5*np.sqrt(np.pi/x)*erf(np.sqrt(x))


#####################################################################

