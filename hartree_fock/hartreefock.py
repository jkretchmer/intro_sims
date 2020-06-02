#!/usr/bin/python

import numpy as np
import scipy.linalg as la
import sys
import os
sys.path.append('/Users/joshkretchmer/Documents/Kretchmer_Group/intro_sims')
from utils import *

#####################################################################

def hf_calc(typ,Ns,Na,Nb,S,Hcore,g):

    #subroutine to perform a hartree-fock calculation
    #param str typ: type of calculation, rhf or uhf
    #param int Ns: number of spatial atomic orbitals ("sites")
    #param int Na: number of up electrons
    #param int Nb: number of down electrons
    #param numpy.ndarray g: four-index tensor of 2-el integrals in AO (site) basis


    #generate initial guess of the density matrix assuming fock operator is given by Hcore
    evals,orbs = diagonalize(Hcore,S)
    Pa = rdm_1el(orbs,Na) #density matrix associated with up electrons
    if typ == "rhf":
        #choose Pb the same as Pa for rhf
        Pb = rdm_1el(orbs,Nb) #density matrix associated with down electrons
    else:
        #choose Pb proportional to identity matrix for uhf
        Pb = (Nb*1.0/Ns)*np.identity(Ns) 

    P=Pa+Pb #total density matrix


    itrmax = 900
    Enew = 9999.9
    Ediff = 10.0
    itr = 0
    while Ediff > 1e-8 and itr < itrmax:
        #SCF Loop:

        #calculate 2e- contribution to fock operators
        h2el_a = make_h2el(g,P,Pa)
        h2el_b = make_h2el(g,P,Pb)

        #form fock operators
        focka = Hcore+h2el_a
        fockb = Hcore+h2el_b

        #solve the fock equations
        evalsa,orbsa = diagonalize(focka,S)
        evalsb,orbsb = diagonalize(fockb,S)

        #form the density matrices
        Pa = rdm_1el(orbsa,Na)
        Pb = rdm_1el(orbsb,Nb)
        P = Pa+Pb

        #calculate the new HF-energy and check convergence
        Eold = Enew
        Enew = Ehf(Hcore,focka,fockb,Pa,Pb,P)
        Ediff = np.fabs(Enew-Eold)

        #accumulate iteration
        itr += 1


    return Enew,evalsa,orbsa,evalsb,orbsb


#####################################################################


def rdm_1el(C,Ne):
    #subroutine that calculates and returns the one-electron density matrix

    Cocc = C[:,:Ne]
    P = np.dot( Cocc,np.transpose(np.conjugate(Cocc)) )

    return P

#####################################################################

def make_h2el(g,P,Pa):
    #subroutine that calculates the two-electron contribution to the fock matrix

    h2el = ( np.tensordot(P,g,axes=([0,1],[3,2])) -
             np.tensordot(Pa,g,axes=([0,1],[1,2])) )

    return h2el

#####################################################################

def Ehf(Hcore,focka,fockb,Pa,Pb,P):

    #subroutine that calculates the HF-energy

    Ehf = 0.5*np.trace( np.dot(P,Hcore)+np.dot(Pa,focka)+np.dot(Pb,fockb) )

    return Ehf

#####################################################################

