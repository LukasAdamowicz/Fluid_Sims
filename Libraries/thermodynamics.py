""" Object name: Fluid"""
import numpy as np
import scipy
import scipy.optimize
from scipy.constants.constants import C2K
from scipy.constants.constants import K2C
from scipy.constants.constants import F2K
from scipy.constants.constants import K2F
import scipy.constants as sc
import os

def interpolate_table(target,index,xquantity,yquantity):
    return yquantity[index] + \
                (yquantity[index+1]-yquantity[index])* \
                (target-xquantity[index])/(xquantity[index+1]-xquantity[index])
        
class FluidArray(np.ndarray):
    """ How to:
        from Libraries import thermodynamics as thermo
        
        variable = thermo.FluidArray((y_size,x_size),dtype='object')
        variable.Fluid(material) material can be air, water, argon and krypton (see below for ranges)
        variable.get_properties(T) to get thermodynamics of the fluid at temperature T in Kelvin.  
            T is an array, must be same size as specified in initial FluidArray call
        
        Compute thermodynamics properties of air between 100 K and 3000 K, 
        water between 274K and 373K, argon between 100 and 700K and
        krypton between 150 and 700 K under 1 atm. Argon, krypton and water were obtained 
        through http://webbook.nist.gov/chemistry/fluid/
        More fluids to be added in the future
        
        Outputs:
        air_inlet.beta  thermal expansion coefficient
        air_inlet.rho   density
        air_inlet.Cp    specific heat
        air_inlet.mu    dynamic viscosity
        air_inlet.k     thermal conductivity
        air_inlet.nu    kinematic viscosity
        air_inlet.alpha thermal diffusivity
        air_inlet.Pr    Prandtl number
        
        """
        
    def Fluid(self,name):
        self.name = name
        self.y,self.x = self.shape
        
        self.T_water_low = 274
        self.T_water_high = 373
        self.T_argon_low = 100
        self.T_argon_high = 700
        self.T_krypton_low = 150
        self.T_krypton_high = 740
        self.T_air_low = 100
        self.T_air_high = 3000
        
    def get_properties(self,T_o):
        self.T_o = T_o

        if T_o.shape != self.shape:
            print("Insufficient temperature data")
            return
        elif self.name == 'water':
            self.temp_test_low = np.where(np.less(self.T_o,self.T_water_low)==True)
            self.temp_test_high = np.where(np.greater(self.T_o,self.T_water_high)==True)
            
            if self.temp_test_low[0].size > 0:            
                print('Temperature in index (%i,%i) is lower than table limits'%\
                (self.temp_test_low[0][0],self.temp_test_low[1][0]))
                return
            if self.temp_test_high[0].size > 0:
                print('Temperature in index (%i,%i) is higher than table limits'%\
                (self.temp_test_high[0][0],self.temp_test_high[1][0]))
                return
            Ttab,ptab,rhotab,Cptab,mutab,ktab = \
            np.genfromtxt('Libraries/Tables/water1atm.csv',delimiter=',',skip_header = 1,unpack = True,dtype = float)
            Ntab = len(Ttab)
            Cptab *= 1e3
            nutab = mutab/rhotab
            alphatab = ktab/(rhotab*Cptab)
            Prtab = nutab/alphatab
            dTtab = Ttab[1]-Ttab[0]
            #compute beta from -rho(d rho/dT)
            betatab = -(1./rhotab)*np.gradient(rhotab)/dTtab
            #i = np.empty((self.y,self.x),dtype='int')
            i = (T_o-Ttab[0])/dTtab
            i=i.astype('int')
            i_test = np.where(np.equal(i,Ntab-1)==True)
            if i_test[0].size>0:
                i[i_test[0][0],i_test[1][0]] = Ntab-2
        elif self.name == 'argon':
            self.temp_test_low = np.where(np.less(self.T_o,self.T_argon_low)==True)
            self.temp_test_high = np.where(np.greater(self.T_o,self.T_argon_high)==True)
            
            if self.temp_test_low[0].size > 0:            
                print('Temperature in index (%i,%i) is lower than table limits'%\
                (self.temp_test_low[0][0],self.temp_test_low[1][0]))
                return
            if self.temp_test_high[0].size > 0:
                print('Temperature in index (%i,%i) is higher than table limits'%\
                (self.temp_test_high[0][0],self.temp_test_high[1][0]))
                return
            Ttab,ptab,rhotab,Cptab,mutab,ktab = \
            np.genfromtxt('Libraries/Tables/Argon1atm.csv',delimiter=',',skip_header=1, unpack=True,dtype=float)
            Ntab = len(Ttab)
            Cptab *= 1e3
            nutab = mutab/rhotab
            alphatab = ktab/(rhotab*Cptab)
            Prtab = nutab/alphatab
            dTtab = Ttab[1] - Ttab[0]
            betatab = -(1./rhotab)*np.gradient(rhotab)/dTtab
            i = (T_o-Ttab[0])/dTtab
            i=i.astype('int')
            i_test = np.where(np.equal(i,Ntab-1)==True)
            if i_test[0].size>0:
                i[i_test[0][0],i_test[1][0]] = Ntab-2
        elif self.name == 'krypton':
            self.temp_test_low = np.where(np.less(self.T_o,self.T_krypton_low)==True)
            self.temp_test_high = np.where(np.greater(self.T_o,self.T_krypton_high)==True)
            
            if self.temp_test_low[0].size > 0:            
                print('Temperature in index (%i,%i) is lower than table limits'%\
                (self.temp_test_low[0][0],self.temp_test_low[1][0]))
                return
            if self.temp_test_high[0].size > 0:
                print('Temperature in index (%i,%i) is higher than table limits'%\
                (self.temp_test_high[0][0],self.temp_test_high[1][0]))
                return
            Ttab,ptab,rhotab,Cptab,mutab,ktab = \
            np.genfromtxt('Libraries/Tables/Krypton1atm.csv',delimiter=',',skip_header=1, unpack=True,dtype=float)
            Ntab = len(Ttab)
            Cptab *= 1e3
            nutab = mutab/rhotab
            alphatab = ktab/(rhotab*Cptab)
            Prtab = nutab/alphatab
            dTtab = Ttab[1] - Ttab[0]
            betatab = -(1./rhotab)*np.gradient(rhotab)/dTtab
            i = (T_o-Ttab[0])/dTtab
            i=i.astype('int')
            i_test = np.where(np.equal(i,Ntab-1)==True)
            if i_test[0].size>0:
                i[i_test[0][0],i_test[1][0]] = Ntab-2
        elif self.name == 'air':
            self.temp_test_low = np.where(np.less(self.T_o,self.T_air_low)==True)
            self.temp_test_high = np.where(np.greater(self.T_o,self.T_air_high)==True)
            
            if self.temp_test_low[0].size > 0:            
                print('Temperature in index (%i,%i) is lower than table limits'%\
                (self.temp_test_low[0][0],self.temp_test_low[1][0]))
                return
            if self.temp_test_high[0].size > 0:
                print('Temperature in index (%i,%i) is higher than table limits'%\
                (self.temp_test_high[0][0],self.temp_test_high[1][0]))
                return
            Ttab,rhotab,Cptab,ktab,nutab,betatab,Prtab = \
            np.genfromtxt('Libraries/Tables/Air1atm.csv',delimiter=',',skip_header=1, unpack=True,dtype=float)
            Ntab = len(Ttab)
            Cptab *= 1e3
            nutab *= 1e-6
            ktab *= 1e-3
            mutab = rhotab*nutab
            alphatab = ktab/(rhotab*Cptab)
            Prtab = nutab/alphatab
            
            i = np.zeros((self.y,self.x))
            count = 0
            while count<Ntab:
                loc = np.where(np.less(self.T_o,Ttab[count])==False)
                if loc[0].size>0:
                    for q in range(loc[0].size):
                        i[loc[0][q],loc[1][q]] +=1
                count+=1
            i -=1
            i=i.astype('int')
            i_test = np.where(np.equal(i,Ntab-1)==True)
            if i_test[0].size>0:
                i[i_test[0][0],i_test[1][0]] = Ntab-2
        else:
            print('No table available for', self.name)
            return
        
        self.rho = interpolate_table(T_o,i,Ttab,rhotab)
        self.Cp = interpolate_table(T_o,i,Ttab,Cptab)
        self.mu = interpolate_table(T_o,i,Ttab,mutab)
        self.k = interpolate_table(T_o,i,Ttab,ktab)
        self.nu = interpolate_table(T_o,i,Ttab,nutab)
        self.alpha = interpolate_table(T_o,i,Ttab,alphatab)
        self.Pr = interpolate_table(T_o,i,Ttab,Prtab)
        if (self.name == 'air'):
            self.beta = 1./T_o
        else:
            self.beta = interpolate_table(T_o,i,Ttab,betatab)