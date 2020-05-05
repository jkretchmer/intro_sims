import numpy as np
import sys
import os
sys.path.append('/Users/joshkretchmer/Documents/Kretchmer_Group/intro_sims')
import utils

###############################################################

class gr():

    #class to calculate radial distribution function

    def __init__( self, boxL ):

        self.Nhis = 100 #number of bins in histogram for gr
        self.ngr  = 0 #counts the number of times the system is sampled for gr calculation
        self.delg = boxL / ( 2.0 * self.Nhis ) #bin size, divided by 2 b/c of boundary conditions

        self.gr_array = np.zeros([self.Nhis,2]) #initialize gr array

    def sample( self, N, boxL, xxx ):

        #subroutine to update histogram

        self.ngr += 1

        for i in range( N-1 ):
            for j in range( i+1, N ):

                #distance between particle i and j in each dimension
                xij = xxx[i] - xxx[j]

                #periodic boundary conditions in each dimension
                xij = xij - boxL * np.rint( xij/boxL )

                #absolute distance between particles
                r = np.sqrt( np.sum( xij**2 ) )

                #update histogram for particles within half box length
                if( r < boxL/2.0 ):
                    ig = int( np.floor( r / self.delg ) )
                    self.gr_array[ig,1] += 2

    def calc_gr( self, N, density ):

        #subroutine to calculate final radial distribution function

        for i in range(self.Nhis):

            #distance for bin i
            self.gr_array[i,0] = self.delg * (i+0.5)

            #volume between bin i+1 and i            
            vb = (4.0/3.0) * np.pi * ( (i+1)**3 - i**3 ) * self.delg**3

            #number of ideal gas particles in vb
            nid = vb * density

            #normalize g(r)
            self.gr_array[i,1] = self.gr_array[i,1] / ( self.ngr * N * nid )

        #Print g(r)
        utils.printarray( self.gr_array, 'radial_dist_fcn.dat', True )

