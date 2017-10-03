# -*- coding: utf-8 -*-
"""
Created on Fri Oct 30 11:27:52 2015

@author: andika
"""
J = []

import numpy as np

bal_qso = [2159, 2979, 3839, 5813, 5887, 6698, 7143, 8780, 12, \
134, 170, 1118, 1957, 2022, 2904, 3685, 3822, 4010, 5011, 8303]

#bal_qso = [3839, 5887, 6698, 7143]

#bal_qso = [3839, 5887, 6698, 7143, 12, \
#134, 1118, 1957, 2022, 2904, 3685, 3822, 5011, 8303]

bal_qso = [3839, 6698, 8780, 12, \
134, 170, 1118, 1957, 2022, 2904, 3685, 3822, 8303]

bal_qso = np.array(bal_qso)

FWHM_bHbeta = []

Q = []
q = 0

#from astropy.cosmology import WMAP5 as cosmo

#from astropy.modeling import models, fitting

#import warnings
from astropy.cosmology import FlatLambdaCDM
import matplotlib.pyplot as plt
from scipy.stats import linregress, spearmanr

#radio loud > np.log10(frec_20cm*10**31.6)


def Ka03(x):#x = log(NII/Halpha), y = log(OIII/Hbeta)
    return 0.61/(x-0.05) + 1.3

def Ke01_a(x):#x = log(NII/Halpha), y = log(OIII/Hbeta)
    return 0.61/(x-0.47) + 1.19

def Ke01_b(x):#x = log(SII/Halpha), y = log(OIII/Hbeta)
    return 0.72/(x-0.32) + 1.3

def Ke06(x):#x = log(SII/Halpha), y = log(OIII/Hbeta)
    return 1.89*x + 0.76

cosmo = FlatLambdaCDM(H0=70, Om0=0.3)


data = np.loadtxt('MAIN_SAMPLE_irhamta_e.csv', skiprows=1, delimiter = ',', dtype=str)

z = [] #redshift
d = [] #distance
e_bv = []

c = 299792458.*10**2 #cm/s
c2 = 299792458. * 10**-3 #km/s
L_sun = 3.846*10**33 #erg/s

#e_bv = [] #sfd

'''========== SDSS Magnitude =========='''

L_u = []
L_g = []
L_r = []
L_i = []
L_z = []


lambda_u = 3543. * 10**-8 #cm
lambda_g = 4770. * 10**-8 #cm
lambda_r = 6231. * 10**-8 #cm
lambda_i = 7625. * 10**-8 #cm
lambda_z = 9134. * 10**-8 #cm

frec_u = c/lambda_u
frec_g = c/lambda_g
frec_r = c/lambda_r
frec_i = c/lambda_i
frec_z = c/lambda_z


mag_u = []
mag_g = []
mag_r = []
mag_i = []
mag_z = []

flux_u = []
flux_g = []
flux_r = []
flux_i = []
flux_z = []

b_u = 1.4 * 10**-10 #maggies
b_g = 0.9 * 10**-10
b_r = 1.2 * 10**-10
b_i = 1.8 * 10**-10
b_z = 7.4 * 10**-10

'''========== 2MASS, WISE, and GALEX Magnitude =========='''

mag_j = []
mag_h = []
mag_k = []
mag_nuv = []
mag_fuv = []
mag_w1 = []
mag_w2 = []
mag_w3 = []
mag_w4 = []

lambda_j = 1.235 #micron
lambda_h = 1.662 #micron
lambda_k = 2.159 #micron
lambda_nuv = 2267 #angstrom
lambda_fuv = 1516 #angstrom
lambda_w1 = 3.4 #micron
lambda_w2 = 4.6 #micron
lambda_w3 = 12 #micron
lambda_w4 = 22 #micron

frec_nuv = c/lambda_nuv/10**-8
frec_fuv = c/lambda_fuv/10**-8

flux_j = []
flux_h = []
flux_k = []
flux_nuv = []
flux_fuv = []

L_j = []
L_h = []
L_k = []
L_j = []
L_nuv = []
L_fuv = []


'''========== ROSAT and FIRST Luminosity =========='''
L_x = []

flux_20cm = []
L_20cm = []
frec_20cm = c/20.
frec_6cm = c/6.

'''========== Emission Line Luminosity and FWHM =========='''
flux_5100   = []
L_5100      = []

flux_OII_3727   = []
flux_nHbeta     = []
flux_bHbeta     = []
flux_OIII_4959  = []
flux_OIII_5007  = []
flux_nHalpha    = []
flux_bHalpha    = []
flux_NII_6548   = []
flux_NII_6584   = []
flux_SII_6717   = []
flux_SII_6731   = []
flux_FeII       = []

L_OII_3727   = []
L_nHbeta     = []
L_bHbeta     = []
L_OIII_4959  = []
L_OIII_5007  = []
L_nHalpha    = []
L_bHalpha    = []
L_NII_6548   = []
L_NII_6584   = []
L_SII_6717   = []
L_SII_6731   = []
L_FeII       = []

L_nHalpha_stern = []
L_bHalpha_stern = []
L_OIII_stern    = []

fwhm_OII_3727   = []
fwhm_nHbeta     = []
fwhm_bHbeta     = []
fwhm_bHbeta_2   = []
fwhm_OIII_4959  = []
fwhm_OIII_5007  = []
fwhm_nHalpha    = []
fwhm_bHalpha    = []
fwhm_bHalpha_2  = []
fwhm_NII_6548   = []
fwhm_NII_6584   = []
fwhm_SII_6717   = []
fwhm_SII_6731   = []
fwhm_FeII       = []

shift_OIII = []
shiftOIII = []

'''========== Emission Lines Correlation =========='''

LO32 = []
LO3 = []
LO2 = []

LnHbeta = []
LbHbeta = []

LnHalpha = []
LbHalpha = []

LNII = []
LSII = []
LFeII = []

LSII_total = []

fwhm_O32 = []
z_O32 = []


M_BH_vir_Hbeta = [] #massa virial BH dalam satuan massa Matahari
M_BH_vir_Halpha = []

L_L_Edd = [] #rasio luminositas terhadap luminositas eddington
L_bol = []

EWO32 = []
EWO3 = []
EWO2 = []

EWnHbeta = []
EWbHbeta = []

EWnHalpha = []
EWbHalpha = []

EWNII = []
EWSII = []
EWFeII = []


flux_bCIV = []
L_bCIV = []


shift_bCIV = []
fwhm_bCIV = []
LbCIV = []
EWbCIV = []

gamma = []
c_12 = []


'''========== Correlation =========='''

LO32_x      = []; LO3_x     = []; LO2_x     = []; Lx    = []; fwhm_O32_x    = []

LO32_fuv    = []; LO3_fuv   = []; LO2_fuv   = []; Lfuv  = []; fwhm_O32_fuv  = []
LO32_nuv    = []; LO3_nuv   = []; LO2_nuv   = []; Lnuv  = []; fwhm_O32_nuv  = []

LO32_u      = []; LO3_u     = []; LO2_u     = []; Lu    = []; fwhm_O32_u    = []
LO32_g      = []; LO3_g     = []; LO2_g     = []; Lg    = []; fwhm_O32_g    = []
LO32_r      = []; LO3_r     = []; LO2_r     = []; Lr    = []; fwhm_O32_r    = []
LO32_i      = []; LO3_i     = []; LO2_i     = []; Li    = []; fwhm_O32_i    = []
LO32_z      = []; LO3_z     = []; LO2_z     = []; Lz    = []; fwhm_O32_z    = []

LO32_cont   = []; LO3_cont  = []; LO2_cont  = []; Lcont = []; fwhm_O32_cont = []

LO32_j      = []; LO3_j     = []; LO2_j     = []; Lj    = []; fwhm_O32_j    = []
LO32_h      = []; LO3_h     = []; LO2_h     = []; Lh    = []; fwhm_O32_h    = []
LO32_k      = []; LO3_k     = []; LO2_k     = []; Lk    = []; fwhm_O32_k    = []

LO32_20cm   = []; LO3_20cm  = []; LO2_20cm  = []; L20cm = []; fwhm_O32_20cm = []
'''
def L_int(x, alpha, beta, lam):
    lam = lam/10**4
    x = 10**x    
    E_bv = 1.97*np.log10(10**(alpha - beta)/2.86)
    
    if lam <= 0.63 and lam <= 2.20:
        k = 2.659 * (-1.857 + 1.040/lam) + 4.05
    
    elif lam <= 0.12 and lam < 0.63:
        k = 2.659 * (-2.156 + 1.509/lam - 0.198/lam**2 + 0.011/lam**3) + 4.05
    else:
        k = 0     
        
    A = k * E_bv
    return np.log10(x * 10**(0.4*A))
'''
from scipy.interpolate import interp1d
ex = np.loadtxt('standard_extinction_curve.csv', skiprows=1, dtype=float, delimiter=',')
itp = interp1d(ex[:, 0], ex[:, 1], kind='cubic')
itp2 = interp1d(ex[:, 0], ex[:, 2], kind='cubic')


#plt.plot(ex[:, 0], ex[:, 1], 'bo')
#plt.plot(ex[:, 0], itp(ex[:, 0]), 'k-')

def L_int(x, alpha, beta, lam): 
    E_bv = 1.99*np.log10(10**(alpha - beta)/2.86)
    if E_bv < 0:
        E_bv = 0 
    c = E_bv/0.77
    return np.log10(10**(x-beta) * 10**(c*itp(lam)) * 10**beta)
    
    



LO32_x_int      = []; LO3_x_int     = []; LO2_x_int     = []; Lx_int = []
LO32_fuv_int    = []; LO3_fuv_int   = []; LO2_fuv_int   = []; Lfuv_int = []
LO32_nuv_int    = []; LO3_nuv_int   = []; LO2_nuv_int   = []; Lnuv_int = []
LO32_u_int      = []; LO3_u_int     = []; LO2_u_int     = []; Lu_int = []
LO32_g_int      = []; LO3_g_int     = []; LO2_g_int     = []; Lg_int = []
LO32_r_int      = []; LO3_r_int     = []; LO2_r_int     = []; Lr_int = []
LO32_i_int      = []; LO3_i_int     = []; LO2_i_int     = []; Li_int = []
LO32_z_int      = []; LO3_z_int     = []; LO2_z_int     = []; Lz_int = []
LO32_cont_int   = []; LO3_cont_int  = []; LO2_cont_int  = []; Lcont_int = []
LO32_j_int      = []; LO3_j_int     = []; LO2_j_int     = []; Lj_int = []
LO32_h_int      = []; LO3_h_int     = []; LO2_h_int     = []; Lh_int = []
LO32_k_int      = []; LO3_k_int     = []; LO2_k_int     = []; Lk_int = []
LO32_20cm_int   = []; LO3_20cm_int  = []; LO2_20cm_int  = []; L20cm_int = []

Lradio = [] 


mag_2MASS = [[], [], []]
mag_SDSS = [[], [], [], [], []]
mag_WISE = [[], [], [], [], []]


low_ion = []
pecu = []
unknown = []
R = []

f_20cm = []
f_6cm = []
f_opt = []
B = []


for i in range(len(data)):
    #print 'row ', i+1
    z.append(float(data[i][7]))    
    d.append(cosmo.luminosity_distance(z[i]).value*3.08567758*10**24) #cm
    e_bv.append(float(data[i][23]))
    
    mag_u.append(float(data[i][24]))
    mag_g.append(float(data[i][25]))
    mag_r.append(float(data[i][26]))
    mag_i.append(float(data[i][27]))
    mag_z.append(float(data[i][28]))
    
    mag_j.append(float(data[i][8]))
    mag_h.append(float(data[i][9]))
    mag_k.append(float(data[i][10]))
    mag_nuv.append(float(data[i][11]))
    mag_fuv.append(float(data[i][12]))
    mag_w1.append(float(data[i][95]))    
    mag_w2.append(float(data[i][96]))
    mag_w3.append(float(data[i][97]))
    mag_w4.append(float(data[i][98]))
    
    
    shift_OIII.append(float(data[i][99]))    
    
    flux_u.append(3631 * 10**-23 *frec_u * \
    np.sinh((mag_u[i]/-2.5) * np.log(10) - np.log(b_u)) * 2.*b_u)
    
    flux_g.append(3631 * 10**-23 *frec_g * \
    np.sinh((mag_g[i]/-2.5) * np.log(10) - np.log(b_g)) * 2.*b_g)
    
    flux_r.append(3631 * 10**-23 *frec_r * \
    np.sinh((mag_r[i]/-2.5) * np.log(10) - np.log(b_r)) * 2.*b_r)
    
    flux_i.append(3631 * 10**-23 *frec_i * \
    np.sinh((mag_i[i]/-2.5) * np.log(10) - np.log(b_i)) * 2.*b_i)
    
    flux_z.append(3631 * 10**-23 *frec_z * \
    np.sinh((mag_z[i]/-2.5) * np.log(10) - np.log(b_z)) * 2.*b_z)
    
    if mag_j[i] == -999 or mag_h[i] == -999 or mag_k[i] == -999:
        flux_j.append(0)
        flux_h.append(0)
        flux_k.append(0)
        
        L_j.append(0) #erg/s    
        L_h.append(0) 
        L_k.append(0)
    else:
        flux_j.append(10**((mag_j[i] - 0.723*e_bv[i])/-2.5)*(3.129e-13) * lambda_j * 10**7)     
        flux_h.append(10**((mag_h[i] - 0.460*e_bv[i])/-2.5)*(1.133e-13) * lambda_h * 10**7)
        flux_k.append(10**((mag_k[i] - 0.310*e_bv[i])/-2.5)*(4.283e-14) * lambda_k * 10**7)
        
        L_j.append(np.log10(flux_j[i]*4*np.pi*d[i]**2)) #erg/s    
        L_h.append(np.log10(flux_h[i]*4*np.pi*d[i]**2)) 
        L_k.append(np.log10(flux_k[i]*4*np.pi*d[i]**2))
    
    if mag_nuv[i] == -999 or mag_fuv[i] == -999:
        flux_nuv.append(0)    
        flux_fuv.append(0)
    
        L_nuv.append(0)
        L_fuv.append(0)
    else:
        flux_nuv.append(10**((mag_nuv[i] + 48.6 - 8.2*e_bv[i])/-2.5) * frec_nuv)    
        flux_fuv.append(10**((mag_fuv[i] + 48.6 - 8.24*e_bv[i])/-2.5) * frec_fuv)
        
        L_nuv.append(np.log10(flux_nuv[i]*4*np.pi*d[i]**2))
        L_fuv.append(np.log10(flux_fuv[i]*4*np.pi*d[i]**2))

    
    L_u.append(np.log10(flux_u[i]*4*np.pi*d[i]**2)) #erg/s     # / L_sun)
    L_g.append(np.log10(flux_g[i]*4*np.pi*d[i]**2)) # / L_sun) #Lsun
    L_r.append(np.log10(flux_r[i]*4*np.pi*d[i]**2)) # / L_sun) #Lsun
    L_i.append(np.log10(flux_i[i]*4*np.pi*d[i]**2)) # / L_sun) #Lsun
    L_z.append(np.log10(flux_z[i]*4*np.pi*d[i]**2)) # / L_sun)
    
    
    flux_20cm.append(float(data[i][16])  * 10**-26 * frec_20cm)
    
    
    if flux_20cm[i] == 0:
        L_20cm.append(0)
    else:
        L_20cm.append(float(np.log10(flux_20cm[i]*4*np.pi*d[i]**2)))
    
    L_x.append(float(data[i][13]))
    
    
    flux_5100.append(float(data[i][29]))
    flux_OII_3727.append(float(data[i][30]))
    flux_nHbeta.append(float(data[i][31]))
    flux_bHbeta.append(float(data[i][32]))
    flux_OIII_4959.append(float(data[i][33]))
    flux_OIII_5007.append(float(data[i][34]))
    flux_nHalpha.append(float(data[i][35]))
    flux_bHalpha.append(float(data[i][36]))
    flux_NII_6548.append(float(data[i][37]))
    flux_NII_6584.append(float(data[i][38]))
    flux_SII_6717.append(float(data[i][39]))
    flux_SII_6731.append(float(data[i][40]))
    flux_FeII.append(float(data[i][54]) + float(data[i][55]) + float(data[i][56])\
    + float(data[i][57]) + float(data[i][58]))    
    
    flux_bCIV.append(float(data[i][89]))    
    
    
    L_5100.append(np.log10(flux_5100[i]*4*np.pi*d[i]**2 * 10**-17*5100.))# check again this one!
    L_OII_3727.append(np.log10(flux_OII_3727[i]*4*np.pi*d[i]**2 * 10**-17))
    L_nHbeta.append(np.log10(flux_nHbeta[i]*4*np.pi*d[i]**2 * 10**-17))
    L_bHbeta.append(np.log10(flux_bHbeta[i]*4*np.pi*d[i]**2 * 10**-17))
    L_OIII_4959.append(np.log10(flux_OIII_4959[i]*4*np.pi*d[i]**2 * 10**-17))
    L_OIII_5007.append(np.log10(flux_OIII_5007[i]*4*np.pi*d[i]**2 * 10**-17))
    L_nHalpha.append(np.log10(flux_nHalpha[i]*4*np.pi*d[i]**2 * 10**-17))
    L_bHalpha.append(np.log10(flux_bHalpha[i]*4*np.pi*d[i]**2 * 10**-17))
    L_NII_6548.append(np.log10(flux_NII_6548[i]*4*np.pi*d[i]**2 * 10**-17))
    L_NII_6584.append(np.log10(flux_NII_6584[i]*4*np.pi*d[i]**2 * 10**-17))
    L_SII_6717.append(np.log10(flux_SII_6717[i]*4*np.pi*d[i]**2 * 10**-17))
    L_SII_6731.append(np.log10(flux_SII_6731[i]*4*np.pi*d[i]**2 * 10**-17))
    L_FeII.append(np.log10(flux_FeII[i]*4*np.pi*d[i]**2 * 10**-17))    
    
    L_bCIV.append(np.log10(flux_bCIV[i]*4*np.pi*d[i]**2))    
    
    L_nHalpha_stern.append(float(data[i][19]))
    L_bHalpha_stern.append(float(data[i][20]))
    L_OIII_stern.append(float(data[i][18]))
    
    if abs(float(data[i][44])) == 217.886197561:
        data[i][44] = data[i][43]    
    FWHM_bHbeta.append(np.mean([abs(float(data[i][43])), abs(float(data[i][44]))]))
    
    #Type 2 Criteria
       
    
    #finding correlation
    if L_OIII_5007[i] > 0 and L_OII_3727[i] > 0 and L_OIII_4959[i] > 0\
    and float(data[i][46])/float(data[i][41]) >= 1/1.95 and float(data[i][46])/float(data[i][41]) <= 1.95\
    and L_OIII_5007[i] - L_OIII_4959[i] >= np.log10(2.7) and L_OIII_5007[i] - L_OIII_4959[i] <= np.log10(3.3)\
    and abs(float(data[i][41])) < 1200.\
    and abs(float(data[i][42])) < 1200.\
    and abs(float(data[i][47])) < 1200.\
    and abs(float(data[i][43])) < 25000.\
    and abs(float(data[i][44])) < 25000.\
    and abs(float(data[i][48])) < 25000.\
    and abs(float(data[i][49])) < 25000.\
    and float(data[i][47])/float(data[i][42]) >= 1/1.95 and float(data[i][47])/float(data[i][42]) <= 1.95\
    and L_nHalpha[i] >= 38. and L_nHbeta[i] >= 38.\
    and L_bHalpha[i] >= 38. and L_bHbeta[i] >= 38.\
    and L_OIII_5007[i] - L_OII_3727[i] >= -1.0 and 10**(L_OIII_5007[i] - L_OII_3727[i]) <= 20.\
    and L_NII_6584[i] >= 38. and np.log10(10**L_SII_6717[i] + 10**L_SII_6731[i]) >= 38.\
    and z[i] < 0.35:#due to low ion
        
        
#        if L_FeII[i] == -np.inf\
#        or str(L_FeII[i]) == 'nan'\
#        or L_FeII[i] < 34.\
#        or 10**(L_FeII[i]-L_bHbeta[i]) > 3.:
#            continue
        if L_FeII[i] < 38.\
        or 10**(L_FeII[i]-L_bHbeta[i]) < 0. or str(10**(L_FeII[i]-L_bHbeta[i])) == 'nan'\
        or 10**(L_FeII[i]-L_bHbeta[i]) > 12:
            continue

#        if abs(float(data[i][43])) < 1000.\
#        or abs(float(data[i][44])) < 1000.\
#        or abs(float(data[i][48])) < 1000.\
#        or abs(float(data[i][49])) < 1000.:
#            continue

        #BLR detected
        if abs(float(data[i][43])) == 217.886197561 and abs(float(data[i][44])) == 217.886197561\
        or abs(float(data[i][48])) == 129.07839953 and abs(float(data[i][49])) == 129.07839953:
            continue
            
        if abs(float(data[i][44])) == 217.886197561:
            data[i][44] = data[i][43]
        if abs(float(data[i][49])) == 129.07839953:
            data[i][49] = data[i][48]
        
        if np.mean([abs(float(data[i][43])), abs(float(data[i][44]))]) < 1000\
        or np.mean([abs(float(data[i][48])), abs(float(data[i][49]))]) < 1000:
            continue

        if L_OIII_5007[i] - L_nHbeta[i] >= 1.5\
        or L_NII_6584[i]-L_nHalpha[i] <= -1.5\
        or L_NII_6584[i]-L_nHalpha[i] >= 0.5\
        or np.log10(10**L_SII_6717[i] + 10**L_SII_6731[i]) - L_nHalpha[i] <= -1.3\
        or np.log10(10**L_SII_6717[i] + 10**L_SII_6731[i]) - L_nHalpha[i] >= 0.5:
            continue
            
#        if np.mean([abs(float(data[i][43])), abs(float(data[i][44]))]) < 1000.\
#        or np.mean([abs(float(data[i][48])), abs(float(data[i][49]))]) < 1000.:
#            continue
        
        
        #Ka03(x):#x = log(NII/Halpha), y = log(OIII/Hbeta)
        if (L_OIII_5007[i]-L_nHbeta[i]) < Ka03(L_NII_6584[i]-L_nHalpha[i]):
            continue

        if (L_OIII_5007[i]-L_nHbeta[i]) < Ke01_a(L_NII_6584[i]-L_nHalpha[i]):
            continue
        
#        if (L_OIII_5007[i]-L_nHbeta[i]) < Ke01_b(np.log10(10**L_SII_6717[i] + 10**L_SII_6731[i])-L_nHalpha[i]):
#            continue        

#        if (L_OIII_5007[i]-L_nHbeta[i]) < Ke06(np.log10(10**L_SII_6717[i] + 10**L_SII_6731[i])-L_nHalpha[i]):
#            continue

#        if abs(float(data[i][43])) < abs(float(data[i][46]))\
#        or abs(float(data[i][44])) < abs(float(data[i][46]))\
#        or abs(float(data[i][48])) < abs(float(data[i][47]))\
#        or abs(float(data[i][49])) < abs(float(data[i][47]))\
#        :
#            continue


        if L_OIII_5007[i] - L_OII_3727[i] < 0.:
            low_ion.append(i)
            
                
            if L_OIII_5007[i] - L_nHbeta[i] > 1.5\
            and L_OIII_5007[i] - L_nHbeta[i] < 1.9:
                unknown.append(i)                

        if (L_OIII_5007[i]-L_nHbeta[i]) < Ke06(np.log10(10**L_SII_6717[i] + 10**L_SII_6731[i])-L_nHalpha[i])\
        and L_OIII_5007[i] - L_nHbeta[i] <= 1.7 and L_OIII_5007[i] - L_nHbeta[i] >= 0.8\
        and L_OIII_5007[i] - L_OII_3727[i] >= 0.5:
            pecu.append(i)

        
        
        f_20cm.append(float(data[i][16])  * 10**-26)
        #f_6cm.append(float(data[i][16])  * 10**-26 * (frec_6cm)**-0.5)
        B.append(mag_g[i] + 0.17*(mag_u[i]-mag_g[i]) + 0.11)        
        
        J.append(i)

        shiftOIII.append(shift_OIII[i])        
        
        gamma.append(float(data[i][14]))
        Lradio.append(L_20cm[i])
        LO32.append(10**(L_OIII_5007[i] - L_OII_3727[i]))
        LO3.append(L_OIII_5007[i])
        LO2.append(L_OII_3727[i])
        z_O32.append(z[i])        
        
        LnHbeta.append(L_nHbeta[i])
        LbHbeta.append(L_bHbeta[i])
        
        LnHalpha.append(L_nHalpha[i])
        LbHalpha.append(L_bHalpha[i])        
        
        LNII.append(L_NII_6584[i])
        LSII.append(np.log10(10**L_SII_6717[i] + 10**L_SII_6731[i]))        
        
        LSII_total.append(np.log10(10**L_SII_6717[i] + 10**L_SII_6731[i]))        
        
        LFeII.append(L_FeII[i])
        LbCIV.append(L_bCIV[i])        
        
        EWO32.append(np.log10(float(data[i][64])/float(data[i][60])))
        EWO3.append(np.log10(float(data[i][64])))
        EWO2.append(np.log10(float(data[i][60])))
        
        EWnHbeta.append(np.log10(float(data[i][61])))
        EWbHbeta.append(np.log10(float(data[i][62])))
        
        EWnHalpha.append(np.log10(float(data[i][65])))
        EWbHalpha.append(np.log10(float(data[i][66])))        
        
        EWNII.append(np.log10(float(data[i][68])))
        EWSII.append(np.log10(float(data[i][69])+float(data[i][70])))        
        
        #EWSII_total.append(np.log10(10**L_SII_6717[i] + 10**L_SII_6731[i]))        
        
        EWFeII.append(np.log10(float(data[i][84])))        
        EWbCIV.append(np.log10(float(data[i][91])))
        
        fwhm_OII_3727.append(abs(float(data[i][41])))
        fwhm_nHbeta.append(abs(float(data[i][42])))
      
        fwhm_bHbeta.append(np.mean([abs(float(data[i][43])), abs(float(data[i][44]))]))

        #abs(float(data[i][43])))
        fwhm_bHbeta_2.append(abs(float(data[i][44])))        
        
        fwhm_OIII_4959.append(abs(float(data[i][45])))
        fwhm_OIII_5007.append(abs(float(data[i][46])))
        
        fwhm_nHalpha.append(abs(float(data[i][47])))
        fwhm_bHalpha.append(np.mean([abs(float(data[i][48])), abs(float(data[i][49]))]))
        #abs(float(data[i][48])))
        fwhm_bHalpha_2.append(abs(float(data[i][49])))
        
        fwhm_NII_6548.append(abs(float(data[i][50])))        
        fwhm_NII_6584.append(abs(float(data[i][51])))
        fwhm_SII_6717.append(abs(float(data[i][52])))
        fwhm_SII_6731.append(abs(float(data[i][53])))
        
        fwhm_FeII.append(abs(float(data[i][59])* 2.3548 * c2/(4570.)))
        fwhm_bCIV.append(abs(float(data[i][90])))
    
        shift_bCIV.append(float(data[i][93])/1549.*c2)        
        
        fwhm_O32.append(float(data[i][46])/float(data[i][41]))
        
        
        c_12.append(float(data[i][15]))        
        
        if L_5100[i] != 0:
            Lcont.append(L_5100[i])
            LO32_cont.append(L_OIII_5007[i]-L_OII_3727[i])
            LO3_cont.append(L_OIII_5007[i])
            LO2_cont.append(L_OII_3727[i])        
            fwhm_O32_cont.append(np.log10(float(data[i][46])/float(data[i][41])))
            #####
            Lcont_int.append(L_int(L_5100[i], L_nHalpha[i], L_nHbeta[i], 5100.))
            LO32_cont_int.append(L_int(L_OIII_5007[i], L_nHalpha[i], L_nHbeta[i], 5007.) - L_int(L_OII_3727[i], L_nHalpha[i], L_nHbeta[i], 3727.))
            LO3_cont_int.append(L_int(L_OIII_5007[i], L_nHalpha[i], L_nHbeta[i], 5007.))
            LO2_cont_int.append(L_int(L_OII_3727[i], L_nHalpha[i], L_nHbeta[i], 3727.))
            #--here
        
        if L_x[i] != 0:
            Lx.append(L_x[i])
            LO32_x.append(L_OIII_5007[i]-L_OII_3727[i])
            LO3_x.append(L_OIII_5007[i])
            LO2_x.append(L_OII_3727[i])        
            fwhm_O32_x.append(np.log10(float(data[i][46])/float(data[i][41])))
            #####
            #Lx_int.append(L_x[i])#L_int(L_x[i], L_nHalpha[i], L_nHbeta[i], 5100.))
            LO32_x_int.append(L_int(L_OIII_5007[i], L_nHalpha[i], L_nHbeta[i], 5007.) - L_int(L_OII_3727[i], L_nHalpha[i], L_nHbeta[i], 3727.))
            LO3_x_int.append(L_int(L_OIII_5007[i], L_nHalpha[i], L_nHbeta[i], 5007.))
            LO2_x_int.append(L_int(L_OII_3727[i], L_nHalpha[i], L_nHbeta[i], 3727.))
            
            R.append(i)
            
            
        if L_nuv[i] > 41. and L_fuv[i] > 41.:
            Lnuv.append(L_nuv[i])
            LO32_nuv.append(L_OIII_5007[i]-L_OII_3727[i])
            LO3_nuv.append(L_OIII_5007[i])
            LO2_nuv.append(L_OII_3727[i])        
            fwhm_O32_nuv.append(np.log10(float(data[i][46])/float(data[i][41])))
            
            Lfuv.append(L_fuv[i])
            LO32_fuv.append(L_OIII_5007[i]-L_OII_3727[i])
            LO3_fuv.append(L_OIII_5007[i])
            LO2_fuv.append(L_OII_3727[i])        
            fwhm_O32_fuv.append(np.log10(float(data[i][46])/float(data[i][41])))
            #####
            #Lnuv_int.append(L_int(L_nuv[i], L_nHalpha[i], L_nHbeta[i], 2267.))
            LO32_nuv_int.append(L_int(L_OIII_5007[i], L_nHalpha[i], L_nHbeta[i], 5007.) - L_int(L_OII_3727[i], L_nHalpha[i], L_nHbeta[i], 3727.))
            LO3_nuv_int.append(L_int(L_OIII_5007[i], L_nHalpha[i], L_nHbeta[i], 5007.))
            LO2_nuv_int.append(L_int(L_OII_3727[i], L_nHalpha[i], L_nHbeta[i], 3727.))

            #Lfuv_int.append(L_int(L_fuv[i], L_nHalpha[i], L_nHbeta[i], 1516.))
            LO32_fuv_int.append(L_int(L_OIII_5007[i], L_nHalpha[i], L_nHbeta[i], 5007.) - L_int(L_OII_3727[i], L_nHalpha[i], L_nHbeta[i], 3727.))
            LO3_fuv_int.append(L_int(L_OIII_5007[i], L_nHalpha[i], L_nHbeta[i], 5007.))
            LO2_fuv_int.append(L_int(L_OII_3727[i], L_nHalpha[i], L_nHbeta[i], 3727.))
            
        
        if L_u[i] > 42.0 and L_g[i] > 42.0 and L_r[i] > 42.0 and L_i[i] != 0 and L_z[i] > 42.0:
            Lu.append(L_u[i])
            LO32_u.append(L_OIII_5007[i]-L_OII_3727[i])
            LO3_u.append(L_OIII_5007[i])
            LO2_u.append(L_OII_3727[i])        
            fwhm_O32_u.append(np.log10(float(data[i][46])/float(data[i][41])))
            
            Lg.append(L_g[i])
            LO32_g.append(L_OIII_5007[i]-L_OII_3727[i])
            LO3_g.append(L_OIII_5007[i])
            LO2_g.append(L_OII_3727[i])        
            fwhm_O32_g.append(np.log10(float(data[i][46])/float(data[i][41])))
            
            Lr.append(L_r[i])
            LO32_r.append(L_OIII_5007[i]-L_OII_3727[i])
            LO3_r.append(L_OIII_5007[i])
            LO2_r.append(L_OII_3727[i])        
            fwhm_O32_r.append(np.log10(float(data[i][46])/float(data[i][41])))
            
            Li.append(L_i[i])
            LO32_i.append(L_OIII_5007[i]-L_OII_3727[i])
            LO3_i.append(L_OIII_5007[i])
            LO2_i.append(L_OII_3727[i])        
            fwhm_O32_i.append(np.log10(float(data[i][46])/float(data[i][41])))
            
            Lz.append(L_z[i])
            LO32_z.append(L_OIII_5007[i]-L_OII_3727[i])
            LO3_z.append(L_OIII_5007[i])
            LO2_z.append(L_OII_3727[i])        
            fwhm_O32_z.append(np.log10(float(data[i][46])/float(data[i][41])))
            #####
            #Lu_int.append(L_int(L_u[i], L_nHalpha[i], L_nHbeta[i], 3543.))
            LO32_u_int.append(L_int(L_OIII_5007[i], L_nHalpha[i], L_nHbeta[i], 5007.) - L_int(L_OII_3727[i], L_nHalpha[i], L_nHbeta[i], 3727.))
            LO3_u_int.append(L_int(L_OIII_5007[i], L_nHalpha[i], L_nHbeta[i], 5007.))
            LO2_u_int.append(L_int(L_OII_3727[i], L_nHalpha[i], L_nHbeta[i], 3727.))

            #Lg_int.append(L_int(L_g[i], L_nHalpha[i], L_nHbeta[i], 4770.))
            LO32_g_int.append(L_int(L_OIII_5007[i], L_nHalpha[i], L_nHbeta[i], 5007.) - L_int(L_OII_3727[i], L_nHalpha[i], L_nHbeta[i], 3727.))
            LO3_g_int.append(L_int(L_OIII_5007[i], L_nHalpha[i], L_nHbeta[i], 5007.))
            LO2_g_int.append(L_int(L_OII_3727[i], L_nHalpha[i], L_nHbeta[i], 3727.))

            #Lr_int.append(L_int(L_r[i], L_nHalpha[i], L_nHbeta[i], 6231.))
            LO32_r_int.append(L_int(L_OIII_5007[i], L_nHalpha[i], L_nHbeta[i], 5007.) - L_int(L_OII_3727[i], L_nHalpha[i], L_nHbeta[i], 3727.))
            LO3_r_int.append(L_int(L_OIII_5007[i], L_nHalpha[i], L_nHbeta[i], 5007.))
            LO2_r_int.append(L_int(L_OII_3727[i], L_nHalpha[i], L_nHbeta[i], 3727.))

            #Li_int.append(L_int(L_i[i], L_nHalpha[i], L_nHbeta[i], 7625.))
            LO32_i_int.append(L_int(L_OIII_5007[i], L_nHalpha[i], L_nHbeta[i], 5007.) - L_int(L_OII_3727[i], L_nHalpha[i], L_nHbeta[i], 3727.))
            LO3_i_int.append(L_int(L_OIII_5007[i], L_nHalpha[i], L_nHbeta[i], 5007.))
            LO2_i_int.append(L_int(L_OII_3727[i], L_nHalpha[i], L_nHbeta[i], 3727.))

            #Lz_int.append(L_int(L_z[i], L_nHalpha[i], L_nHbeta[i], 9134.))
            LO32_z_int.append(L_int(L_OIII_5007[i], L_nHalpha[i], L_nHbeta[i], 5007.) - L_int(L_OII_3727[i], L_nHalpha[i], L_nHbeta[i], 3727.))
            LO3_z_int.append(L_int(L_OIII_5007[i], L_nHalpha[i], L_nHbeta[i], 5007.))
            LO2_z_int.append(L_int(L_OII_3727[i], L_nHalpha[i], L_nHbeta[i], 3727.))

            
        if L_j[i] > 42.0 and L_h[i] > 42.0 and L_k[i] > 42.0:
            Lj.append(L_j[i])
            LO32_j.append(L_OIII_5007[i]-L_OII_3727[i])
            LO3_j.append(L_OIII_5007[i])
            LO2_j.append(L_OII_3727[i])        
            fwhm_O32_j.append(np.log10(float(data[i][46])/float(data[i][41])))
            
            Lh.append(L_h[i])
            LO32_h.append(L_OIII_5007[i]-L_OII_3727[i])
            LO3_h.append(L_OIII_5007[i])
            LO2_h.append(L_OII_3727[i])        
            fwhm_O32_h.append(np.log10(float(data[i][46])/float(data[i][41])))
            
            Lk.append(L_k[i])
            LO32_k.append(L_OIII_5007[i]-L_OII_3727[i])
            LO3_k.append(L_OIII_5007[i])
            LO2_k.append(L_OII_3727[i])        
            fwhm_O32_k.append(np.log10(float(data[i][46])/float(data[i][41])))
            #####
            #Lj_int.append(L_int(L_j[i], L_nHalpha[i], L_nHbeta[i], 12350.))
            LO32_j_int.append(L_int(L_OIII_5007[i], L_nHalpha[i], L_nHbeta[i], 5007.) - L_int(L_OII_3727[i], L_nHalpha[i], L_nHbeta[i], 3727.))
            LO3_j_int.append(L_int(L_OIII_5007[i], L_nHalpha[i], L_nHbeta[i], 5007.))
            LO2_j_int.append(L_int(L_OII_3727[i], L_nHalpha[i], L_nHbeta[i], 3727.))
            
            Lh_int.append(L_int(L_h[i], L_nHalpha[i], L_nHbeta[i], 16620.))
            LO32_h_int.append(L_int(L_OIII_5007[i], L_nHalpha[i], L_nHbeta[i], 5007.) - L_int(L_OII_3727[i], L_nHalpha[i], L_nHbeta[i], 3727.))
            LO3_h_int.append(L_int(L_OIII_5007[i], L_nHalpha[i], L_nHbeta[i], 5007.))
            LO2_h_int.append(L_int(L_OII_3727[i], L_nHalpha[i], L_nHbeta[i], 3727.))
            
            Lk_int.append(L_int(L_k[i], L_nHalpha[i], L_nHbeta[i], 21590.))
            LO32_k_int.append(L_int(L_OIII_5007[i], L_nHalpha[i], L_nHbeta[i], 5007.) - L_int(L_OII_3727[i], L_nHalpha[i], L_nHbeta[i], 3727.))
            LO3_k_int.append(L_int(L_OIII_5007[i], L_nHalpha[i], L_nHbeta[i], 5007.))
            LO2_k_int.append(L_int(L_OII_3727[i], L_nHalpha[i], L_nHbeta[i], 3727.))
            
        if L_20cm[i] != 0:
            L20cm.append(L_20cm[i])
            LO32_20cm.append(L_OIII_5007[i]-L_OII_3727[i])
            LO3_20cm.append(L_OIII_5007[i])
            LO2_20cm.append(L_OII_3727[i])        
            fwhm_O32_20cm.append(np.log10(float(data[i][46])/float(data[i][41])))
            #####
            #L20cm_int.append(L_20cm[i])#L_int(L_20cm[i], L_nHalpha[i], L_nHbeta[i], 21590.))
            LO32_20cm_int.append(L_int(L_OIII_5007[i], L_nHalpha[i], L_nHbeta[i], 5007.) - L_int(L_OII_3727[i], L_nHalpha[i], L_nHbeta[i], 3727.))
            LO3_20cm_int.append(L_int(L_OIII_5007[i], L_nHalpha[i], L_nHbeta[i], 5007.))
            LO2_20cm_int.append(L_int(L_OII_3727[i], L_nHalpha[i], L_nHbeta[i], 3727.))

        if mag_j[i] > 0 and mag_h[i] > 0 and mag_k[i] > 0:
            mag_2MASS[0].append(mag_j[i] - 0.723*e_bv[i])
            mag_2MASS[1].append(mag_h[i] - 0.460*e_bv[i])
            mag_2MASS[2].append(mag_k[i] - 0.310*e_bv[i])
        else:
            mag_2MASS[0].append(16.1 - 0.723*e_bv[i])
            mag_2MASS[1].append(15.3 - 0.460*e_bv[i])
            mag_2MASS[2].append(14.6 - 0.310*e_bv[i])            
            
        mag_SDSS[0].append(mag_u[i])
        mag_SDSS[1].append(mag_g[i])
        mag_SDSS[2].append(mag_r[i])
        mag_SDSS[3].append(mag_i[i])
        mag_SDSS[4].append(mag_z[i])

        if mag_w1[i] > 0 and mag_w2[i] > 0 and mag_w3[i] > 0 and mag_w4[i] > 0 :
            mag_WISE[0].append(mag_w1[i] - 0.189*e_bv[i])
            mag_WISE[1].append(mag_w2[i] - 0.146*e_bv[i])
            mag_WISE[2].append(mag_w3[i] - 0.*e_bv[i])
            mag_WISE[3].append(mag_w3[i] - 0.*e_bv[i])
            Q.append(q)
            q+=1
        
        else:
            mag_WISE[0].append(14.15 - 0.189*e_bv[i])
            mag_WISE[1].append(13.19 - 0.146*e_bv[i])
            mag_WISE[2].append(10.44 - 0.*e_bv[i])
            mag_WISE[3].append(8.04 - 0.*e_bv[i])
            q+=1
       
'''========== Eddington Luminosity and BH Virial Mass =========='''
a = []
shift_OIII = np.array(shift_OIII)
shiftOIII = np.array(shiftOIII)

for i in range(len(LbHbeta)):    
    M_BH_vir_Hbeta.append(0.91 + 0.5*10**LbHbeta[i]/10.**44\
    + 2*np.log10(fwhm_bHbeta[i]))

    M_BH_vir_Halpha.append(7.4 + 0.545*np.log10(10**LbHalpha[i]/10**44)\
    + 2.06*np.log10(fwhm_bHalpha[i]/1000.))
    
    a.append(abs(M_BH_vir_Hbeta[i] - M_BH_vir_Halpha[i]))
    
#    L_L_Edd.append(0.6 + 0.455*np.log10(10**LbHalpha[i]/10**44)\
#    - 2.06*np.log10(np.mean(fwhm_bHalpha[i])/1000.))
    
#    L_bol.append(np.log10(130.) * LbHalpha[i])
    
#    L_bol.append(np.log10(9)*Lcont[i])
    L_bol.append(np.log10(9*10**Lcont[i]))
    
    L_L_Edd.append(L_bol[i]-np.log10(1.5*10**38 * 10**M_BH_vir_Hbeta[i]))   
    
m = np.array(M_BH_vir_Hbeta)
l = np.array(L_L_Edd)
Lbol = np.array(L_bol)

J = np.array(J)

print('')


diff = []

for i in range(len(L_OIII_stern)):
#    if (abs(L_OIII_stern[i] - L_OIII_5007[i])) > 0.1:
        #print ' ==> j removed ', i
        
#        continue

#    elif L_OIII_stern[i] ==0:
#        continue
    
    if L_OIII_stern[i] != 0 and L_OIII_5007[i] > 38.:#else:
        diff.append((L_OIII_5007[i]) - L_OIII_stern[i])
        continue


print '\nMean differences = ', 10**np.mean(diff)
print 'Sigma differences = ', 10**np.std(diff)
print '\n=============================================\n'


plt.figure(0)
plt.hist(Lcont, bins=20)
plt.title('Luminosity Distribution', fontsize='x-large')
plt.xlabel('$\\log \ L_{5100} \ \\rm (erg \ s^{-1})$', fontsize='x-large')
plt.ylabel('Number of AGN', fontsize='x-large')
plt.savefig('figures/fig_0_luminosity_distribution')
#plt.show()

plt.figure(1)
plt.hist(z_O32, bins=20)
plt.title('Redshift Distribution', fontsize='x-large')
plt.xlabel('Redshift ($z$)', fontsize='x-large')
plt.ylabel('Number of AGN', fontsize='x-large')

plt.savefig('figures/fig_1_redshift_distribution')

plt.figure(2)
plt.plot(z_O32, LO32, 'bo')
plt.xlabel('Redshift ($z$)', fontsize='x-large')
plt.ylabel('$L_{\\rm [O \ III]}/L_{\\rm [O \ II]}$', fontsize='x-large')
plt.title('Ionization Distribution', fontsize='x-large')
plt.savefig('figures/fig_2_ionization_distribution')
plt.close('all')

plt.figure(3)
plt.plot(z_O32, LO2, 'bo')
plt.xlabel('Redshift ($z$)', fontsize='x-large')
plt.ylabel('$L_{\\rm [O \ II]}$', fontsize='x-large')
#plt.title('Ionization Distribution')
plt.savefig('figures/fig_3_O2_distribution')
plt.close('all')


plt.figure(6)
plt.plot(z_O32, Lbol, 'b.')
plt.xlabel('Redshift ($z$)', fontsize='x-large')
plt.ylabel('$\\log \ L_{\\rm bol} \ (\\rm erg \ s^{-1})$', fontsize='x-large')
#plt.title('Ionization Distribution')
plt.savefig('figures/fig_6_Lbol_distribution')
plt.close('all')


'''========== Important Functions =========='''

def calc_kcor(filter_name, redshift, colour_name, colour_value):
    """
    K-corrections calculator in Python. See http://kcor.sai.msu.ru for the 
    reference. Available filter-colour combinations must be present in the 
    `coeff` dictionary keys.

    @type   filter_name: string    
    @param  filter_name: Name of the filter to calculate K-correction for, e.g. 
                         'u', 'g', 'r' for some of the SDSS filters, or 'J2', 
                         'H2', 'Ks2' for 2MASS filters (must be present in 
                         `coeff` dictionary)
    @type      redshift: float    
    @param     redshift: Redshift of a galaxy, should be between 0.0 and 0.5 (no
                         check is made, however)
    @type   colour_name: string    
    @param  colour_name: Human name of the colour, e.g. 'u - g', 'g - r', 
                         'V - Rc', 'J2 - Ks2' (must be present in `coeff` dictionary)
    @type  colour_value: float    
    @param colour_value: Value of the galaxy's colour, specified in colour_name    
    @rtype:              float
    @return:             K-correction in specified filter for given redshift and 
                         colour
    @version:            2012
    @author:             Chilingarian, I., Melchior. A.-L., and Zolotukhin, I.
    @license:            Simplified BSD license, see http://kcor.sai.msu.ru/license.txt

    Usage example:
    
        >>> calc_kcor('g', 0.2, 'g - r', 1.1)
        0.5209713975999992
        >>> calc_kcor('Ic', 0.4, 'V - Ic', 2.0)
        0.310069919999993
        >>> calc_kcor('H', 0.5, 'H - K', 0.1)
        -0.14983142499999502
        
    """
    coeff = {

        'B_BRc': [
            [0,0,0,0],
            [-1.99412,3.45377,0.818214,-0.630543],
            [15.9592,-3.99873,6.44175,0.828667],
            [-101.876,-44.4243,-12.6224,0],
            [299.29,86.789,0,0],
            [-304.526,0,0,0],
        ],
        
        'B_BIc': [
            [0,0,0,0],
            [2.11655,-5.28948,4.5095,-0.8891],
            [24.0499,-4.76477,-1.55617,1.85361],
            [-121.96,7.73146,-17.1605,0],
            [236.222,76.5863,0,0],
            [-281.824,0,0,0],
        ],

        'H2_H2Ks2': [
            [0,0,0,0],
            [-1.88351,1.19742,10.0062,-18.0133],
            [11.1068,20.6816,-16.6483,139.907],
            [-79.1256,-406.065,-48.6619,-430.432],
            [551.385,1453.82,354.176,473.859],
            [-1728.49,-1785.33,-705.044,0],
            [2027.48,950.465,0,0],
            [-741.198,0,0,0],
        ],

        'H2_J2H2': [
            [0,0,0,0],
            [-4.99539,5.79815,4.19097,-7.36237],
            [70.4664,-202.698,244.798,-65.7179],
            [-142.831,553.379,-1247.8,574.124],
            [-414.164,1206.23,467.602,-799.626],
            [763.857,-2270.69,1845.38,0],
            [-563.812,-1227.82,0,0],
            [1392.67,0,0,0],
        ],

        'Ic_VIc': [
            [0,0,0,0],
            [-7.92467,17.6389,-15.2414,5.12562],
            [15.7555,-1.99263,10.663,-10.8329],
            [-88.0145,-42.9575,46.7401,0],
            [266.377,-67.5785,0,0],
            [-164.217,0,0,0],
        ],

        'J2_J2Ks2': [
            [0,0,0,0],
            [-2.85079,1.7402,0.754404,-0.41967],
            [24.1679,-34.9114,11.6095,0.691538],
            [-32.3501,59.9733,-29.6886,0],
            [-30.2249,43.3261,0,0],
            [-36.8587,0,0,0],
        ],

        'J2_J2H2': [
            [0,0,0,0],
            [-0.905709,-4.17058,11.5452,-7.7345],
            [5.38206,-6.73039,-5.94359,20.5753],
            [-5.99575,32.9624,-72.08,0],
            [-19.9099,92.1681,0,0],
            [-45.7148,0,0,0],
        ],

        'Ks2_J2Ks2': [
            [0,0,0,0],
            [-5.08065,-0.15919,4.15442,-0.794224],
            [62.8862,-61.9293,-2.11406,1.56637],
            [-191.117,212.626,-15.1137,0],
            [116.797,-151.833,0,0],
            [41.4071,0,0,0],
        ],

        'Ks2_H2Ks2': [
            [0,0,0,0],
            [-3.90879,5.05938,10.5434,-10.9614],
            [23.6036,-97.0952,14.0686,28.994],
            [-44.4514,266.242,-108.639,0],
            [-15.8337,-117.61,0,0],
            [28.3737,0,0,0],
        ],

        'Rc_BRc': [
            [0,0,0,0],
            [-2.83216,4.64989,-2.86494,0.90422],
            [4.97464,5.34587,0.408024,-2.47204],
            [-57.3361,-30.3302,18.4741,0],
            [224.219,-19.3575,0,0],
            [-194.829,0,0,0],
        ],

        'Rc_VRc': [
            [0,0,0,0],
            [-3.39312,16.7423,-29.0396,25.7662],
            [5.88415,6.02901,-5.07557,-66.1624],
            [-50.654,-13.1229,188.091,0],
            [131.682,-191.427,0,0],
            [-36.9821,0,0,0],
        ],

        'U_URc': [
            [0,0,0,0],
            [2.84791,2.31564,-0.411492,-0.0362256],
            [-18.8238,13.2852,6.74212,-2.16222],
            [-307.885,-124.303,-9.92117,12.7453],
            [3040.57,428.811,-124.492,-14.3232],
            [-10677.7,-39.2842,197.445,0],
            [16022.4,-641.309,0,0],
            [-8586.18,0,0,0],
        ],

        'V_VIc': [
            [0,0,0,0],
            [-1.37734,-1.3982,4.76093,-1.59598],
            [19.0533,-17.9194,8.32856,0.622176],
            [-86.9899,-13.6809,-9.25747,0],
            [305.09,39.4246,0,0],
            [-324.357,0,0,0],
        ],

        'V_VRc': [
            [0,0,0,0],
            [-2.21628,8.32648,-7.8023,9.53426],
            [13.136,-1.18745,3.66083,-41.3694],
            [-117.152,-28.1502,116.992,0],
            [365.049,-93.68,0,0],
            [-298.582,0,0,0],
        ],

        'FUV_FUVNUV': [
            [0,0,0,0],
            [-0.866758,0.2405,0.155007,0.0807314],
            [-1.17598,6.90712,3.72288,-4.25468],
            [135.006,-56.4344,-1.19312,25.8617],
            [-1294.67,245.759,-84.6163,-40.8712],
            [4992.29,-477.139,174.281,0],
            [-8606.6,316.571,0,0],
            [5504.2,0,0,0],
        ],

        'FUV_FUVu': [
            [0,0,0,0],
            [-1.67589,0.447786,0.369919,-0.0954247],
            [2.10419,6.49129,-2.54751,0.177888],
            [15.6521,-32.2339,4.4459,0],
            [-48.3912,37.1325,0,0],
            [37.0269,0,0,0],
        ],

        'g_gi': [
            [0,0,0,0],
            [1.59269,-2.97991,7.31089,-3.46913],
            [-27.5631,-9.89034,15.4693,6.53131],
            [161.969,-76.171,-56.1923,0],
            [-204.457,217.977,0,0],
            [-50.6269,0,0,0],
        ],

        'g_gz': [
            [0,0,0,0],
            [2.37454,-4.39943,7.29383,-2.90691],
            [-28.7217,-20.7783,18.3055,5.04468],
            [220.097,-81.883,-55.8349,0],
            [-290.86,253.677,0,0],
            [-73.5316,0,0,0],
        ],

        'g_gr': [
            [0,0,0,0],
            [-2.45204,4.10188,10.5258,-13.5889],
            [56.7969,-140.913,144.572,57.2155],
            [-466.949,222.789,-917.46,-78.0591],
            [2906.77,1500.8,1689.97,30.889],
            [-10453.7,-4419.56,-1011.01,0],
            [17568,3236.68,0,0],
            [-10820.7,0,0,0],
        ],

        'H_JH': [
            [0,0,0,0],
            [-1.6196,3.55254,1.01414,-1.88023],
            [38.4753,-8.9772,-139.021,15.4588],
            [-417.861,89.1454,808.928,-18.9682],
            [2127.81,-405.755,-1710.95,-14.4226],
            [-5719,731.135,1284.35,0],
            [7813.57,-500.95,0,0],
            [-4248.19,0,0,0],
        ],

        'H_HK': [
            [0,0,0,0],
            [0.812404,7.74956,1.43107,-10.3853],
            [-23.6812,-235.584,-147.582,188.064],
            [283.702,2065.89,721.859,-713.536],
            [-1697.78,-7454.39,-1100.02,753.04],
            [5076.66,11997.5,460.328,0],
            [-7352.86,-7166.83,0,0],
            [4125.88,0,0,0],
        ],

        'i_gi': [
            [0,0,0,0],
            [-2.21853,3.94007,0.678402,-1.24751],
            [-15.7929,-19.3587,15.0137,2.27779],
            [118.791,-40.0709,-30.6727,0],
            [-134.571,125.799,0,0],
            [-55.4483,0,0,0],
        ],

        'i_ui': [
            [0,0,0,0],
            [-3.91949,3.20431,-0.431124,-0.000912813],
            [-14.776,-6.56405,1.15975,0.0429679],
            [135.273,-1.30583,-1.81687,0],
            [-264.69,15.2846,0,0],
            [142.624,0,0,0],
        ],

        'J_JH': [
            [0,0,0,0],
            [0.129195,1.57243,-2.79362,-0.177462],
            [-15.9071,-2.22557,-12.3799,-2.14159],
            [89.1236,65.4377,36.9197,0],
            [-209.27,-123.252,0,0],
            [180.138,0,0,0],
        ],

        'J_JK': [
            [0,0,0,0],
            [0.0772766,2.17962,-4.23473,-0.175053],
            [-13.9606,-19.998,22.5939,-3.99985],
            [97.1195,90.4465,-21.6729,0],
            [-283.153,-106.138,0,0],
            [272.291,0,0,0],
        ],

        'K_HK': [
            [0,0,0,0],
            [-2.83918,-2.60467,-8.80285,-1.62272],
            [14.0271,17.5133,42.3171,4.8453],
            [-77.5591,-28.7242,-54.0153,0],
            [186.489,10.6493,0,0],
            [-146.186,0,0,0],
        ],

        'K_JK': [
            [0,0,0,0],
            [-2.58706,1.27843,-5.17966,2.08137],
            [9.63191,-4.8383,19.1588,-5.97411],
            [-55.0642,13.0179,-14.3262,0],
            [131.866,-13.6557,0,0],
            [-101.445,0,0,0],
        ],

        'NUV_NUVr': [
            [0,0,0,0],
            [2.2112,-1.2776,0.219084,0.0181984],
            [-25.0673,5.02341,-0.759049,-0.0652431],
            [115.613,-5.18613,1.78492,0],
            [-278.442,-5.48893,0,0],
            [261.478,0,0,0],
        ],

        'NUV_NUVg': [
            [0,0,0,0],
            [2.60443,-2.04106,0.52215,0.00028771],
            [-24.6891,5.70907,-0.552946,-0.131456],
            [95.908,-0.524918,1.28406,0],
            [-208.296,-10.2545,0,0],
            [186.442,0,0,0],
        ],

        'r_gr': [
            [0,0,0,0],
            [1.83285,-2.71446,4.97336,-3.66864],
            [-19.7595,10.5033,18.8196,6.07785],
            [33.6059,-120.713,-49.299,0],
            [144.371,216.453,0,0],
            [-295.39,0,0,0],
        ],

        'r_ur': [
            [0,0,0,0],
            [3.03458,-1.50775,0.576228,-0.0754155],
            [-47.8362,19.0053,-3.15116,0.286009],
            [154.986,-35.6633,1.09562,0],
            [-188.094,28.1876,0,0],
            [68.9867,0,0,0],
        ],

        'u_ur': [
            [0,0,0,0],
            [10.3686,-6.12658,2.58748,-0.299322],
            [-138.069,45.0511,-10.8074,0.95854],
            [540.494,-43.7644,3.84259,0],
            [-1005.28,10.9763,0,0],
            [710.482,0,0,0],
        ],

        'u_ui': [
            [0,0,0,0],
            [11.0679,-6.43368,2.4874,-0.276358],
            [-134.36,36.0764,-8.06881,0.788515],
            [528.447,-26.7358,0.324884,0],
            [-1023.1,13.8118,0,0],
            [721.096,0,0,0],
        ],

        'u_uz': [
            [0,0,0,0],
            [11.9853,-6.71644,2.31366,-0.234388],
            [-137.024,35.7475,-7.48653,0.655665],
            [519.365,-20.9797,0.670477,0],
            [-1028.36,2.79717,0,0],
            [767.552,0,0,0],
        ],

        'Y_YH': [
            [0,0,0,0],
            [-2.81404,10.7397,-0.869515,-11.7591],
            [10.0424,-58.4924,49.2106,23.6013],
            [-0.311944,84.2151,-100.625,0],
            [-45.306,3.77161,0,0],
            [41.1134,0,0,0],
        ],

        'Y_YK': [
            [0,0,0,0],
            [-0.516651,6.86141,-9.80894,-0.410825],
            [-3.90566,-4.42593,51.4649,-2.86695],
            [-5.38413,-68.218,-50.5315,0],
            [57.4445,97.2834,0,0],
            [-64.6172,0,0,0],
        ],

        'z_gz': [
            [0,0,0,0],
            [0.30146,-0.623614,1.40008,-0.534053],
            [-10.9584,-4.515,2.17456,0.913877],
            [66.0541,4.18323,-8.42098,0],
            [-169.494,14.5628,0,0],
            [144.021,0,0,0],
        ],

        'z_rz': [
            [0,0,0,0],
            [0.669031,-3.08016,9.87081,-7.07135],
            [-18.6165,8.24314,-14.2716,13.8663],
            [94.1113,11.2971,-11.9588,0],
            [-225.428,-17.8509,0,0],
            [197.505,0,0,0],
        ],

        'z_uz': [
            [0,0,0,0],
            [0.623441,-0.293199,0.16293,-0.0134639],
            [-21.567,5.93194,-1.41235,0.0714143],
            [82.8481,-0.245694,0.849976,0],
            [-185.812,-7.9729,0,0],
            [168.691,0,0,0],
        ],

    }

    c = coeff[filter_name + '_' + colour_name.replace(' - ', '')]
    kcor = 0.0

    for x, a in enumerate(c):
	    for y, b in enumerate(c[x]):
		    kcor += c[x][y] * redshift**x * colour_value**y
				
    return kcor
    
if __name__ == "__main__":
    import doctest
    doctest.testmod()



o = open('output_parameters.csv', 'w')
o.write('number\tm\tc\tr\tp\tstd\trho\tP\n')
o.close()
def linear(x, m, c):
    a = []
    for i in range(len(x)):
        a.append(x[i]*m + c)
    return a

def unlog(x):
    a = []
    for i in range(len(x)):
        a.append(10**x[i])
    return a
    
def plot(number, x, y, labelx, labely):
    m, c, r, p, std = linregress(x, y)
    rho, P = spearmanr(x, y)
    o = open('output_parameters.csv', 'a')    
    o.write(str(number) + '\t' + str(m) + '\t' + str(c) + '\t' + str(r) + '\t' + str(p) + '\t' + str(std) + '\t' + str(rho) + '\t' + str(P) + '\n')
    o.close()
    plt.figure(number)
    plt.plot(x, y, 'bo', alpha=1.)
#    plt.plot(x, linear(x, m, c), 'k-', label = ('$\\rho_s = %f$\n$ P_s = %f$' %(rho, P)))
    plt.legend(loc='best')
    plt.xlabel(labelx, fontsize='x-large')
    plt.ylabel(labely, fontsize='x-large')
    plt.savefig('figures/fig_%i' %number)

def plot_2(number, x, y, labelx, labely):
    #m, c, r, p, std = linregress(x, y)
    #o = open('output_parameters.csv', 'a')    
    #o.write(str(number) + '\t' + str(m) + '\t' + str(c) + '\t' + str(r) + '\t' + str(p) + '\n')
    #o.close()
    plt.figure(number)
    plt.plot(x, y, 'bo', alpha=1.)
    #plt.plot(x, linear(x, m, c), 'k-', label = ('$r = %f$\n$ P = %f$' %(r, p)))
    #plt.legend(loc='best')
    plt.xlabel(labelx, fontsize='x-large')
    plt.ylabel(labely, fontsize='x-large')
    plt.savefig('figures/fig_%i' %number)

def moving_average(group, x, y, number, labelx='x', labely='y'):
    binned = []    
    mean = []
    stdx = []
    stdy = []
    count = []
    
    for i in range(len(group)):
        point = []
        axis = []
        for j in range(len(x)):
            #if x[j] >= group[i] and x[j] < group[i+1]:
            if abs(x[j]-group[i]) <= abs((group[0]-group[1])/2.): 
                axis.append(x[j])
                point.append(y[j])
        
        if len(point) >= 15:
            mean.append(np.mean(point))
            stdy.append(np.std(point)*1.)
            stdx.append(np.std(axis)*1.)            
            count.append(len(point))
            binned.append(group[i])
        
    plt.figure(number)
    plt.plot(x, y, 'k.', alpha=0.2)
    plt.errorbar(binned, mean, yerr=stdy, xerr=0, fmt='bo')
    plt.xlabel(labelx, fontsize='x-large')
    plt.ylabel(labely, fontsize='x-large')
    plt.xlim(min(binned)-binned[1]-binned[0], max(binned)+binned[1]-binned[0])
    plt.savefig('figures/fig_%i' %number)
    #plt.show()
    #return binned, mean, std

def plot_average(number, x, y, group, labelx, labely):
    
    binned = []    
    mean = []
    stdx = []
    stdy = []
    count = []
    
    for i in range(len(group)):
        point = []
        axis = []
        for j in range(len(x)):
            #if x[j] >= group[i] and x[j] < group[i+1]:
            if abs(x[j]-group[i]) <= abs((group[0]-group[1])/2.): 
                axis.append(x[j])
                point.append(y[j])
        
        if len(point) >= 15:
            mean.append(np.mean(point))
            stdy.append(np.std(point)*1.)
            stdx.append(np.std(axis)*1.)            
            count.append(len(point))
            binned.append(group[i])
    
    m, c, r, p, std = linregress(x, y)
    rho, P = spearmanr(x, y)
    o = open('output_parameters.csv', 'a')    
    o.write(str(number) + '\t' + str(m) + '\t' + str(c) + '\t' + str(r) + '\t' + str(p) + '\t' + str(std) + '\t' + str(rho) + '\t' + str(P) + '\n')
    o.close()
    plt.figure(number)
    plt.plot(binned, linear(binned, m, c), 'r--', label = ('$\\rho_s = %f$\n$ P_s = %f$' %(rho, P)))
    plt.legend(loc='best')
    plt.plot(x, y, 'k.', alpha=0.15)
    plt.errorbar(binned, mean, yerr=stdy, xerr=0, fmt='bo')
    plt.xlabel(labelx, fontsize='x-large')
    plt.ylabel(labely, fontsize='x-large')
    plt.xlim(min(binned)-0.05, max(binned)+0.05)
    plt.savefig('figures/fig_%i' %number)



z_O32 = np.array(z_O32)

LO3 = np.array(LO3)
LO2 = np.array(LO2)
LnHbeta = np.array(LnHbeta)
LbHbeta = np.array(LbHbeta)
LnHalpha = np.array(LnHalpha)
LbHalpha = np.array(LbHalpha)
LNII = np.array(LNII)
LSII = np.array(LSII)
LFeII = np.array(LFeII)

EWO3 = np.array(EWO3)
EWO2 = np.array(EWO2)
EWnHbeta = np.array(EWnHbeta)
EWbHbeta = np.array(EWbHbeta)
EWnHalpha = np.array(EWnHalpha)
EWbHalpha = np.array(EWbHalpha)
EWNII = np.array(EWNII)
EWSII = np.array(EWSII)
EWFeII = np.array(EWFeII)


LSII_total = np.array(LSII_total)

Lx = np.array(Lx)
Lfuv = np.array(Lfuv)
Lnuv = np.array(Lnuv)
Lu = np.array(Lu)
Lg = np.array(Lg)
Lr = np.array(Lr)
Li = np.array(Li)
Lz = np.array(Lz)
Lj = np.array(Lj)
Lh = np.array(Lh)
Lk = np.array(Lk)
L20cm = np.array(L20cm)
Lcont = np.array(Lcont)

SFR = np.array(np.log10(8.4*10**-41*(10**LO2 - 10**(LO3*np.log10(0.1)))))
log_z_1 = np.array(np.log10(z_O32+1))


plt.figure(4)
plt.plot(LNII-LnHalpha, LO3-LnHbeta, 'k.', alpha=0.5)
plt.plot(np.linspace(-3, 0.2, 100), Ke01_a(np.linspace(-3, 0.2, 100)), 'r-')
plt.plot(np.linspace(-3, -0.2, 100), Ka03(np.linspace(-3, -0.2, 100)), 'b-')
plt.xlabel('$\\log \ \\rm [N \ II]/H\\alpha $', fontsize='x-large')
plt.ylabel('$\\log \ \\rm [O \ III]/H\\beta $', fontsize='x-large')
plt.ylim(min(LO3-LnHbeta)-0.2, max(LO3-LnHbeta)+0.2)
plt.xlim(min(LNII-LnHalpha)-0.2, max(LNII-LnHalpha)+0.2)
plt.title('Diagnostic Diagram', fontsize='x-large')
plt.savefig('figures/fig_4_bpt1')

plt.figure(5)
plt.plot(LSII - LnHalpha, LO3-LnHbeta, 'k.', alpha=0.5)
plt.plot(np.linspace(-4, 0.2, 100), Ke01_b(np.linspace(-4, 0.2, 100)), 'r-')
plt.plot(np.linspace(-0.315, 1., 100), Ke06(np.linspace(-0.315, 1., 100)), 'g-')
plt.xlabel('$\\log \ \\rm [S \ II]/H\\alpha $', fontsize='x-large')
plt.ylabel('$\\log \ \\rm [O \ III]/H\\beta $', fontsize='x-large')
plt.ylim(min(LO3-LnHbeta)-0.2, max(LO3-LnHbeta)+0.2)
plt.xlim(min(LSII-LnHalpha-0.2), max(LSII-LnHalpha+0.2))
plt.title('Diagnostic Diagram', fontsize='x-large')
plt.savefig('figures/fig_5_bpt2')

plt.close('all')


'''
#Correlation among lines luminosity
plot(101, LO3_x, Lx, '$\\log \ L_{\\rm [O \ III]} \ (erg \ s^{-1})$', '$\\log \ L_{\\rm X} \ (erg \ s^{-1})$') 
plot(102, LO3_nuv, Lnuv, '$\\log \ L_{\\rm [O \ III]} \ (erg \ s^{-1})$', '$\\log \ L_{\\rm NUV} \ (erg \ s^{-1})$')
plot(103, LO3_fuv, Lfuv, '$\\log \ L_{\\rm [O \ III]} \ (erg \ s^{-1})$', '$\\log \ L_{\\rm FUV} \ (erg \ s^{-1})$')
plot(104, LO3_u, Lu, '$\\log \ L_{\\rm [O \ III]} \ (erg \ s^{-1})$', '$\\log \ L_u \ (erg \ s^{-1})$') 
plot(105, LO3_g, Lg, '$\\log \ L_{\\rm [O \ III]} \ (erg \ s^{-1})$', '$\\log \ L_g \ (erg \ s^{-1})$') 
plot(106, LO3_r, Lr, '$\\log \ L_{\\rm [O \ III]} \ (erg \ s^{-1})$', '$\\log \ L_r \ (erg \ s^{-1})$') 
plot(107, LO3_i, Li, '$\\log \ L_{\\rm [O \ III]} \ (erg \ s^{-1})$', '$\\log \ L_i \ (erg \ s^{-1})$') 
plot(108, LO3_z, Lz, '$\\log \ L_{\\rm [O \ III]} \ (erg \ s^{-1})$', '$\\log \ L_z \ (erg \ s^{-1})$')
plot(109, LO3_j, Lj, '$\\log \ L_{\\rm [O \ III]} \ (erg \ s^{-1})$', '$\\log \ L_J \ (erg \ s^{-1})$') 
plot(110, LO3_h, Lh, '$\\log \ L_{\\rm [O \ III]} \ (erg \ s^{-1})$', '$\\log \ L_H \ (erg \ s^{-1})$') 
plot(111, LO3_k, Lk, '$\\log \ L_{\\rm [O \ III]} \ (erg \ s^{-1})$', '$\\log \ L_K \ (erg \ s^{-1})$')
plot(112, LO3_20cm, L20cm, '$\\log \ L_{\\rm [O \ III]} \ (erg \ s^{-1})$', '$\\log \ L_{Radio} \ (erg \ s^{-1})$')
plot(113, LO3_cont, Lcont, '$\\log \ L_{\\rm [O \ III]} \ (erg \ s^{-1})$', '$\\log \ L_{5100} \ (erg \ s^{-1})$')
plt.close('all')

plot(201, LO2_x, Lx, '$\\log \ L_{[O \ II]} \ (erg \ s^{-1})$', '$\\log \ L_X \ (erg \ s^{-1})$') 
plot(202, LO2_nuv, Lnuv, '$\\log \ L_{[O \ II]} \ (erg \ s^{-1})$', '$\\log \ L_{NUV} \ (erg \ s^{-1})$')
plot(203, LO2_fuv, Lfuv, '$\\log \ L_{[O \ II]} \ (erg \ s^{-1})$', '$\\log \ L_{FUV} \ (erg \ s^{-1})$')
plot(204, LO2_u, Lu, '$\\log \ L_{[O \ II]} \ (erg \ s^{-1})$', '$\\log \ L_u \ (erg \ s^{-1})$') 
plot(205, LO2_g, Lg, '$\\log \ L_{[O \ II]} \ (erg \ s^{-1})$', '$\\log \ L_g \ (erg \ s^{-1})$') 
plot(206, LO2_r, Lr, '$\\log \ L_{[O \ II]} \ (erg \ s^{-1})$', '$\\log \ L_r \ (erg \ s^{-1})$') 
plot(207, LO2_i, Li, '$\\log \ L_{[O \ II]} \ (erg \ s^{-1})$', '$\\log \ L_i \ (erg \ s^{-1})$') 
plot(208, LO2_z, Lz, '$\\log \ L_{[O \ II]} \ (erg \ s^{-1})$', '$\\log \ L_z \ (erg \ s^{-1})$')
plot(209, LO2_j, Lj, '$\\log \ L_{[O \ II]} \ (erg \ s^{-1})$', '$\\log \ L_J \ (erg \ s^{-1})$') 
plot(210, LO2_h, Lh, '$\\log \ L_{[O \ II]} \ (erg \ s^{-1})$', '$\\log \ L_H \ (erg \ s^{-1})$')
plot(211, LO2_k, Lk, '$\\log \ L_{[O \ II]} \ (erg \ s^{-1})$', '$\\log \ L_K \ (erg \ s^{-1})$')
plot(212, LO2_20cm, L20cm, '$\\log \ L_{[O \ II]} \ (erg \ s^{-1})$', '$\\log \ L_{Radio} \ (erg \ s^{-1})$')
plot(213, LO2_cont, Lcont, '$\\log \ L_{[O \ II]} \ (erg \ s^{-1})$', '$\\log \ L_{5100} \ (erg \ s^{-1})$')
plt.close('all')

plot(301, LO32_x, Lx, '$\\log \ L_{[O \ III]}/L_{[O \ II]}$', '$\\log \ L_X \ (erg \ s^{-1})$') 
plot(302, LO32_nuv, Lnuv, '$\\log \ L_{[O \ III]}/L_{[O \ II]}$', '$\\log \ L_{NUV} \ (erg \ s^{-1})$')
plot(303, LO32_fuv, Lfuv, '$\\log \ L_{[O \ III]}/L_{[O \ II]}$', '$\\log \ L_{FUV} \ (erg \ s^{-1})$')
plot(304, LO32_u, Lu, '$\\log \ L_{[O \ III]}/L_{[O \ II]}$', '$\\log \ L_u \ (erg \ s^{-1})$') 
plot(305, LO32_g, Lg, '$\\log \ L_{[O \ III]}/L_{[O \ II]}$', '$\\log \ L_g \ (erg \ s^{-1})$') 
plot(306, LO32_r, Lr, '$\\log \ L_{[O \ III]}/L_{[O \ II]}$', '$\\log \ L_r \ (erg \ s^{-1})$') 
plot(307, LO32_i, Li, '$\\log \ L_{[O \ III]}/L_{[O \ II]}$', '$\\log \ L_i \ (erg \ s^{-1})$') 
plot(308, LO32_z, Lz, '$\\log \ L_{[O \ III]}/L_{[O \ II]}$', '$\\log \ L_z \ (erg \ s^{-1})$')
plot(309, LO32_j, Lj, '$\\log \ L_{[O \ III]}/L_{[O \ II]}$', '$\\log \ L_J \ (erg \ s^{-1})$') 
plot(310, LO32_h, Lh, '$\\log \ L_{[O \ III]}/L_{[O \ II]}$', '$\\log \ L_H \ (erg \ s^{-1})$') 
plot(311, LO32_k, Lk, '$\\log \ L_{[O \ III]}/L_{[O \ II]}$', '$\\log \ L_K \ (erg \ s^{-1})$')
plot(312, LO32_20cm, L20cm, '$\\log \ L_{[O \ III]}/L_{[O \ II]}$', '$\\log \ L_{Radio} \ (erg \ s^{-1})$')
plot(313, LO32_cont, Lcont, '$\\log \ L_{[O \ III]}/L_{[O \ II]}$', '$\\log \ L_{5100} \ (erg \ s^{-1})$')
plt.close('all')

plot(401, LbHbeta, LO3-LbHbeta,'$\\log \ L_{bH\\beta} \ (erg \ s^{-1})$',  '$\\log \ L_{[O \ III]}/L_{bH\\beta}$')
plot(402, LbHbeta, LO2-LbHbeta, '$\\log \ L_{bH\\beta} \ (erg \ s^{-1})$', '$\\log \ L_{[O \ II]}/L_{bH\\beta}$')
plot(403, LbHbeta, LnHbeta-LbHbeta, '$\\log \ L_{bH\\beta} \ (erg \ s^{-1})$', '$\\log \ L_{nH\\beta}/L_{bH\\beta}$')
plot(404, LbHbeta, LnHalpha-LbHbeta,'$\\log \ L_{bH\\beta} \ (erg \ s^{-1})$',   '$\\log \ L_{nH\\alpha}/L_{bH\\beta}$')
plot(405, LbHbeta, LNII-LbHbeta,'$\\log \ L_{bH\\beta} \ (erg \ s^{-1})$',   '$\\log \ L_{[N \ II]}/L_{bH\\beta}$')
plot(406, LbHbeta, LSII-LbHbeta,'$\\log \ L_{bH\\beta} \ (erg \ s^{-1})$',   '$\\log \ L_{[S \ II]}/L_{bH\\beta}$')

plot(407, LbHalpha, LO3-LbHalpha, '$\\log \ L_{bH\\alpha} \ (erg \ s^{-1})$',   '$\\log \ L_{[O \ III]}/L_{bH\\alpha}$')
plot(408, LbHalpha, LO2-LbHalpha, '$\\log \ L_{bH\\alpha} \ (erg \ s^{-1})$',   '$\\log \ L_{[O \ II]}/L_{bH\\alpha}$')
plot(409, LbHalpha, LnHbeta-LbHalpha,'$\\log \ L_{bH\\alpha} \ (erg \ s^{-1})$',   '$\\log \ L_{nH\\beta}/L_{bH\\alpha}$')
plot(410, LbHalpha, LnHalpha-LbHalpha, '$\\log \ L_{bH\\alpha} \ (erg \ s^{-1})$',   '$\\log \ L_{nH\\alpha}/L_{bH\\alpha}$')
plot(411, LbHalpha, LNII-LbHalpha, '$\\log \ L_{bH\\alpha} \ (erg \ s^{-1})$',   '$\\log \ L_{[N \ II]}/L_{bH\\alpha}$')
plot(412, LbHalpha, LSII-LbHalpha, '$\\log \ L_{bH\\alpha} \ (erg \ s^{-1})$',   '$\\log \ L_{[S \ II]}/L_{bH\\alpha}$')
plt.close('all')

plot(501, LO3-LO2, LO2, '$\\log \ L_{[O \ III]}/L_{[O \ II]}$', '$\\log \ L_{[O \ II]} \ (erg \ s^{-1})$') 
plot(502, LO3-LO2, LO3, '$\\log \ L_{[O \ III]}/L_{[O \ II]}$', '$\\log \ L_{[O \ III]} \ (erg \ s^{-1})$')
plot(503, LO3-LO2, LnHbeta, '$\\log \ L_{[O \ III]}/L_{[O \ II]}$', '$\\log \ L_{nH\\beta} \ (erg \ s^{-1})$')
plot(504, LO3-LO2, LbHbeta, '$\\log \ L_{[O \ III]}/L_{[O \ II]}$', '$\\log \ L_{bH\\beta} \ (erg \ s^{-1})$')
plot(505, LO3-LO2, LnHalpha, '$\\log \ L_{[O \ III]}/L_{[O \ II]}$', '$\\log \ L_{nH\\alpha} \ (erg \ s^{-1})$')
plot(506, LO3-LO2, LbHbeta, '$\\log \ L_{[O \ III]}/L_{[O \ II]}$', '$\\log \ L_{bH\\alpha} \ (erg \ s^{-1})$')
plot(507, LO3-LO2, LNII, '$\\log \ L_{[O \ III]}/L_{[O \ II]}$', '$\\log \ L_{[N \ II]} \ (erg \ s^{-1})$')
plot(508, LO3-LO2, LSII, '$\\log \ L_{[O \ III]}/L_{[O \ II]}$', '$\\log \ L_{[S \ II]} \ (erg \ s^{-1})$')
plot(509, LO3-LO2, LFeII, '$\\log \ L_{[O \ III]}/L_{[O \ II]}$', '$\\log \ L_{[Fe \ II]} \ (erg \ s^{-1})$')
plt.close('all')

plot(601, LO3-LO2, m, '$\\log \ L_{[O \ III]}/L_{[O \ II]}$', '$\\log \ M/M_{Sun}$')
plot(602, LO3-LO2, l, '$\\log \ L_{[O \ III]}/L_{[O \ II]}$', '$\\log \ L/L_{Edd}$')
plot(603, LO3-LO2, Lbol, '$\\log \ L_{[O \ III]}/L_{[O \ II]}$', '$\\log \ L_{bol} \ (erg \ s^{-1})$')
plt.close('all')

plot(701, LO3-LO2, LnHbeta-LbHbeta, '$\\log \ L_{[O \ III]}/L_{[O \ II]}$', '$\\log \ L_{nH\\beta}/L_{bH\\beta} \ (erg \ s^{-1})$') 
plot(702, LO3-LO2, LnHalpha-LbHalpha, '$\\log \ L_{[O \ III]}/L_{[O \ II]}$', '$\\log \ L_{nH\\alpha}/L_{bH\\alpha} \ (erg \ s^{-1})$') 
plt.close('all')

#Correlation among lines fwhm
plot(801, 10**(LO3-LO2), fwhm_OII_3727, '$ L_{[O \ III]}/L_{[O \ II]}$', '$FWHM \ [O \ II] \ (km/s)$')
plot(802, 10**(LO3-LO2), fwhm_OIII_5007, '$L_{[O \ III]}/L_{[O \ II]}$', '$FWHM \ [O \ III] \ (km/s)$') 
plot(803, 10**(LO3-LO2), fwhm_nHbeta, '$L_{[O \ III]}/L_{[O \ II]}$', '$FWHM \ nH\\beta \ (km/s)$')
plot(804, 10**(LO3-LO2), fwhm_bHbeta, '$L_{[O \ III]}/L_{[O \ II]}$', '$FWHM \ bH\\beta \ (km/s)$') 
plot(805, 10**(LO3-LO2), fwhm_nHalpha, '$L_{[O \ III]}/L_{[O \ II]}$', '$FWHM \ nH\\alpha \ (km/s)$')
plot(806, 10**(LO3-LO2), fwhm_bHalpha, '$L_{[O \ III]}/L_{[O \ II]}$', '$FWHM \ bH\\alpha \ (km/s)$') 
plot(807, 10**(LO3-LO2), fwhm_NII_6584, '$L_{[O \ III]}/L_{[O \ II]}$', '$FWHM \ [N \ II] \ (km/s)$')
plot(808, 10**(LO3-LO2), fwhm_SII_6731, '$L_{[O \ III]}/L_{[O \ II]}$', '$FWHM \ [S \ II] \ (km/s)$') 
plot(809, 10**(LO3-LO2), fwhm_FeII, '$L_{[O \ III]}/L_{[O \ II]}$', '$FWHM \ [Fe \ II] \ (km/s)$') 
plot(810, 10**(LO3-LO2), fwhm_O32, '$L_{[O \ III]}/L_{[O \ II]}$', '$FWHM \ [O \ III]/[O \ II] \ (km/s)$') 
plt.close('all')


plot(901, 10**(LFeII-LbHbeta), fwhm_bHbeta,'$L_{bH\\beta}/L_{[Fe \ II]} \ (erg \ s^{-1})$',   '$FWHM \ bH\\beta \ (km/s)$')
plt.close('all')
'''


bin_cont = np.arange(min(Lcont), max(Lcont), 0.25/2.)
bin_bHbeta = np.arange(min(LbHbeta), max(LbHbeta), 0.25/2.)

'''
plot_average(1001, Lcont, LO2-Lcont, bin_cont, '$\\log \ L_{5100} \ \\rm  (erg \ s^{-1})$', '$\\log \ L_{\\rm [O \ II]}/L_{5100}$')
plot_average(1002, Lcont, LO3-Lcont, bin_cont, '$\\log \ L_{5100} \ \\rm  (erg \ s^{-1})$',  '$\\log \ L_{\\rm [O \ III]}/L_{5100}$')
plot_average(1003, Lcont, LnHbeta-Lcont, bin_cont, '$\\log \ L_{5100} \ \\rm  (erg \ s^{-1})$', '$\\log \ L_{\\rm nH\\beta}/L_{5100}$')
plot_average(1004, Lcont, LbHbeta-Lcont, bin_cont, '$\\log \ L_{5100} \ \\rm  (erg \ s^{-1})$', '$\\log \ L_{\\rm bH\\beta}/L_{5100}$')
plot_average(1005, Lcont, LnHalpha-Lcont, bin_cont,'$\\log \ L_{5100} \ \\rm  (erg \ s^{-1})$',   '$\\log \ L_{\\rm nH\\alpha}/L_{5100}$')
plot_average(1006, Lcont, LbHalpha-Lcont, bin_cont,'$\\log \ L_{5100} \ \\rm  (erg \ s^{-1})$',   '$\\log \ L_{\\rm bH\\alpha}/L_{5100}$')
plot_average(1007, Lcont, LNII-Lcont, bin_cont,'$\\log \ L_{5100} \ \\rm  (erg \ s^{-1})$',   '$\\log \ L_{\\rm [N \ II]}/L_{5100}$')
plot_average(1008, Lcont, LSII-Lcont, bin_cont,'$\\log \ L_{5100} \ \\rm  (erg \ s^{-1})$',   '$\\log \ L_{\\rm [S \ II]}/L_{5100}$')
plot_average(1009, Lcont, LFeII-Lcont, bin_cont,'$\\log \ L_{5100} \ \\rm  (erg \ s^{-1})$',   '$\\log \ L_{\\rm Fe \ II}/L_{5100}$')
plt.close('all')
'''

'''
Lcont_2 = []

def hostcor(x):
    y = (0.8052 - 1.5502*(x-44.) + 0.9121*(x-44.)**2 - 0.1577*(x-44.)**3)    
        
    return y#np.log10(y)
'''

'''
plot_average(1011, Lcont, EWO2, bin_cont, '$\\log \ L_{5100} \ \\rm  (erg \ s^{-1})$', '$\\log \ \\rm  EW_{[O \ II]} \ (\\AA)$')
plot_average(1012, Lcont, EWO3, bin_cont, '$\\log \ L_{5100} \ \\rm  (erg \ s^{-1})$',  '$\\log \ \\rm  EW_{[O \ III]} \ (\\AA)$')

corr = []
for i in range(len(EWnHbeta)):
    if EWnHbeta[i] > -0.5:
        corr.append(i)
plot_average(1013, Lcont[corr], EWnHbeta[corr], bin_cont, '$\\log \ L_{5100} \ \\rm  (erg \ s^{-1})$', '$\\log \ \\rm  EW_{nH\\beta} \ (\\AA)$')

corr = []
for i in range(len(EWbHbeta)):
    if EWbHbeta[i] > 0.5:
        corr.append(i)
plot_average(1014, Lcont[corr], EWbHbeta[corr], bin_cont, '$\\log \ L_{5100} \ \\rm  (erg \ s^{-1})$', '$\\log \ \\rm  EW_{bH\\beta} \ (\\AA)$')

corr = []
for i in range(len(EWnHalpha)):
    if EWnHalpha[i] > 0:
        corr.append(i)        
plot_average(1015, Lcont[corr], EWnHalpha[corr], bin_cont,'$\\log \ L_{5100} \ \\rm  (erg \ s^{-1})$',   '$\\log \ \\rm  EW_{nH\\alpha} \ (\\AA)$')

corr = []
for i in range(len(EWbHalpha)):
    if EWbHalpha[i] > 1.2:
        corr.append(i)        
plot_average(1016, Lcont[corr], EWbHalpha[corr], bin_cont,'$\\log \ L_{5100} \ \\rm  (erg \ s^{-1})$',   '$\\log \ \\rm  EW_{bH\\alpha} \ (\\AA)$')

corr = []
for i in range(len(EWNII)):
    if EWNII[i] > 0:
        corr.append(i)        
plot_average(1017, Lcont[corr], EWNII[corr], bin_cont,'$\\log \ L_{5100} \ \\rm  (erg \ s^{-1})$',   '$\\log \ \\rm  EW_{[N \ II]} \ (\\AA)$')

corr = []
for i in range(len(EWSII)):
    if EWSII[i] > 0:
        corr.append(i)        
plot_average(1018, Lcont[corr], EWSII[corr], bin_cont,'$\\log \ L_{5100} \ \\rm  (erg \ s^{-1})$',   '$\\log \ \\rm  EW_{[S \ II]} \ (\\AA)$')

#plot_average(1019, Lcont, EWFeII, bin_cont,'$\\log \ L_{5100} \ \\rm  (erg \ s^{-1})$',   '$\\log \ \\rm  EW_{Fe \ II} \ (\\AA)$')
plt.close('all')
'''


#The Baldwin Effect
plot_average(1021, LbHbeta, LO2-LbHbeta, bin_bHbeta, '$\\log \ L_{\\rm bH\\beta} \ \\rm (erg \ s^{-1})$', '$\\log \ L_{\\rm [O \ II]}/L_{\\rm bH\\beta}$')
plot_average(1022, LbHbeta, LO3-LbHbeta, bin_bHbeta, '$\\log \ L_{\\rm bH\\beta} \ \\rm (erg \ s^{-1})$',  '$\\log \ L_{\\rm [O \ III]}/L_{\\rm bH\\beta}$')
plot_average(1023, LbHbeta, LnHbeta-LbHbeta, bin_bHbeta, '$\\log \ L_{\\rm bH\\beta} \ \\rm (erg \ s^{-1})$', '$\\log \ L_{\\rm nH\\beta}/L_{\\rm bH\\beta}$')
plot_average(1024, LbHbeta, LbHbeta-LbHbeta, bin_bHbeta, '$\\log \ L_{\\rm bH\\beta} \ \\rm (erg \ s^{-1})$', '$\\log \ L_{\\rm bH\\beta}/L_{\\rm bH\\beta}$')
plot_average(1025, LbHbeta, LnHalpha-LbHbeta, bin_bHbeta,'$\\log \ L_{\\rm bH\\beta} \ \\rm (erg \ s^{-1})$',   '$\\log \ L_{\\rm nH\\alpha}/L_{\\rm bH\\beta}$')
plot_average(1026, LbHbeta, LbHalpha-LbHbeta, bin_bHbeta,'$\\log \ L_{\\rm bH\\beta} \ \\rm (erg \ s^{-1})$',   '$\\log \ L_{\\rm bH\\alpha}/L_{\\rm bH\\beta}$')
plot_average(1027, LbHbeta, LNII-LbHbeta, bin_bHbeta,'$\\log \ L_{\\rm bH\\beta} \ \\rm (erg \ s^{-1})$',   '$\\log \ L_{\\rm [N \ II]}/L_{\\rm bH\\beta}$')
plot_average(1028, LbHbeta, LSII-LbHbeta, bin_bHbeta,'$\\log \ L_{\\rm bH\\beta} \ \\rm (erg \ s^{-1})$',   '$\\log \ L_{\\rm [S \ II]}/L_{\\rm bH\\beta}$')
plot_average(1029, LbHbeta, LFeII-LbHbeta, bin_bHbeta,'$\\log \ L_{\\rm bH\\beta} \ \\rm (erg \ s^{-1})$',   '$\\log \ L_{\\rm Fe \ II}/L_{\\rm bH\\beta}$')
plt.close('all')

#mine, Zhang, Croom

BE_slope = [-0.49, -0.33, -0.39, -0.40, -0.53, -0.43,\
-0.47, -0.53, -0.42, -0.44, -0.37, -0.32, -0.45, -0.26, -0.36,\
0.86-1, 0.49-1, 0.58-1, 0.74-1]

BE_IP = [13.62, 35.11, 13.60, 13.60, 14.50, 10.36,\
97.12, 13.62, 40.96, 13.6, 35.11, 0, 13.6, 14.5, 10.36,\
35.11, 13.62, 97.12, 40.96]

BE_nc = [3.50, 5.80, np.inf, np.inf, 4.82, 2.30,\
7.3, 3.5, 5.5, np.inf, 5.8, 6.3, np.inf, 4.82, 2.3,\
5.8, 3.5, 7.3, 5.5]

BE_IP = np.array(BE_IP)
BE_nc = np.array(BE_nc)
BE_slope = np.array(BE_slope)

plot(1230, BE_IP, BE_slope, '$\\rm IP \ (eV)$', '$\\rm Slope \ (m)$')
plot(1231, BE_nc, BE_slope, '$\\rm \\log \ n_c (cm^{-3})$', '$\\rm Slope \ (m)$')

plt.figure(11230)
plt.plot(BE_IP[: 6], BE_slope[: 6], 'bo', label='Our Data')
plt.plot(BE_IP[6 : 15], BE_slope[6 : 15], 'ro', label='Zhang et al. (2013)')
plt.plot(BE_IP[15 :], BE_slope[15 :], 'go', label='Croom et al. (2002)')
plt.xlim(-5, 100)
plt.xlabel('$\\rm IP \ (eV)$', fontsize='x-large')
plt.ylabel('$\\rm Slope \ (m)$', fontsize='x-large')
plt.legend(loc='best', fontsize='small')
plt.savefig('figures/fig_11230')
plt.close('all')

plt.figure(11231)
plt.plot(BE_nc[: 6], BE_slope[: 6], 'bo', label='Our Data')
plt.plot(BE_nc[6 : 15], BE_slope[6 : 15], 'ro', label='Zhang et al. (2013)')
plt.plot(BE_nc[15 :], BE_slope[15 :], 'go', label='Croom et al. (2002)')
plt.xlim(2, 8)
plt.xlabel('$\\rm \\log \ n_c (cm^{-3})$', fontsize='x-large')
plt.ylabel('$\\rm Slope \ (m)$', fontsize='x-large')
plt.legend(loc='best', fontsize='small')
plt.savefig('figures/fig_11231')


'''
plot(1101, LO3-LO2, LO3 - np.log10(10**LbHbeta+10**LnHbeta), '$\\log \ L_{[O \ III]}/L_{[O \ II]}$', '$\\log \ L_{[O \ III]}/L_{H\\beta} \ (erg \ s^{-1})$')
plt.close('all')
'''


'''
plot_2(1201, 10**(LO3-LO2), fwhm_bHbeta, '$L_{[O \ III]}/L_{[O \ II]}$', '$FWHM \ bH\\beta \ (km/s)$') 
plot_2(1203, 10**(LFeII-LbHbeta), fwhm_bHbeta,'$L_{Fe \ II}/L_{bH\\beta}$',   '$FWHM \ bH\\beta \ (km/s)$')
plot_2(1205, 10**(LFeII-LbHbeta), 10**(LO3-LO2),'$L_{Fe \ II}/L_{bH\\beta}$',   '$L_{[O \ III]}/L_{[O \ II]}$')
'''



RFeII = 10**(EWFeII - EWbHbeta)
bin_O32 = np.arange(10**min(LO3-LO2), 10**max(LO3-LO2), 1.5)
bin_FeII_bHbeta = np.arange(10**min(LFeII-LbHbeta), 12., 1.)
bin_RFeII = np.arange(min(RFeII), 12.1, 0.2)


bin_fwhm_bHbeta = np.linspace(min(fwhm_bHbeta), max(fwhm_bHbeta), len(bin_FeII_bHbeta))
bin_fwhm_nHbeta = np.linspace(min(fwhm_nHbeta), max(fwhm_nHbeta), len(bin_FeII_bHbeta))


'''
moving_average(bin_O32, 10**(LO3-LO2), fwhm_bHbeta, 1202,\
'$\\log \ L_{[O \ III]}/L_{[O \ II]}$', '$FWHM \ bH\\beta \ (km/s)$')
moving_average(bin_FeII_bHbeta, 10**(LFeII-LbHbeta), fwhm_bHbeta, 1204,\
'$L_{[Fe \ II]}/L_{bH\\beta}$', '$FWHM \ bH\\beta \ (km/s)$')
moving_average(bin_FeII_bHbeta, 10**(LFeII-LbHbeta), 10**(LO3-LO2), 1206,\
'$L_{[Fe \ II]}/L_{bH\\beta}$', '$L_{[O \ III]}/L_{[O \ II]}$')
plt.close('all')
'''


def binning(a, b, mida, midb, number, labelx='x', labely='y'):
  
    xbin = []
    ybin = []
    stdx = []
    stdy = []
    count = []

    for i in range(len(midb)):
        #print i
        for j in range(len(mida)):
            point = []
            axis = []
            
            for n in range(len(a)):
                if abs(a[n]-mida[j]) <= abs((mida[0]-mida[1])/2.)\
                and abs(b[n]-midb[i]) <= abs((midb[0]-midb[1])/2.):
                    axis.append(a[n])
                    point.append(b[n])
                    #print 'yay2'
            
            if len(point) >= 15 and len(axis) >=15 :
                ybin.append(np.mean(point))
                xbin.append(np.mean(axis))
                stdy.append(np.std(point)*1.)
                stdx.append(np.std(axis)*1.)            
                count.append(len(point))
    
    plt.figure(number)
    plt.plot(a, b, 'k.', alpha=0.2)
    plt.errorbar(xbin, ybin, yerr=stdy, xerr=stdx, fmt='bo')
    plt.xlabel(labelx, fontsize='x-large')
    plt.ylabel(labely, fontsize='x-large')
    plt.xlim(-1., 12.)
    plt.savefig('figures/fig_%i' %number)
    #plt.show()
    
    return xbin, ybin
            
'''
binning(10**(LFeII-LbHbeta), fwhm_bHbeta, bin_FeII_bHbeta, bin_fwhm_bHbeta, 1207,\
'$L_{\\rm Fe \ II}/L_{\\rm bH\\beta}$', '$\\rm FWHM \ bH\\beta \ (km/s)$')

binning(10**(LO3-LO2), fwhm_bHbeta, bin_O32, bin_fwhm_bHbeta, 1208,\
'$\\log \ L_{\\rm [O \ III]}/L_{\\rm [O \ II]}$', '$\\rm FWHM \ bH\\beta \ (km/s)$')

binning(10**(LFeII-LbHbeta), 10**(LO3-LO2), bin_FeII_bHbeta, bin_O32, 1209,\
'$L_{\\rm Fe \ II}/L_{bH\\beta}$', '$L_{\\rm [O \ III]}/L_{\\rm [O \ II]}$')

binning(10**(LFeII-LbHbeta), fwhm_nHbeta, bin_FeII_bHbeta, bin_fwhm_nHbeta, 1210,\
'$L_{\\rm Fe \ II}/L_{\\rm bH\\beta}$', '$\\rm FWHM \ nH\\beta \ (km/s)$')


binning(RFeII, fwhm_bHbeta, bin_RFeII, bin_fwhm_bHbeta, 1247,\
'$R_{\\rm Fe II}$', '$\\rm FWHM \ bH\\beta \ (km/s)$')

plt.close('all')
'''

Lradio = np.array(Lradio)
RL = [[], []]#np.zeros((2, len(LO3)), float)
RQ = [[], []]#np.zeros((2, len(LO3)), float)

quiet = []
loud = []

for i in range(len(Lradio)):
    if Lradio[i] >= np.log10(frec_20cm*10**31.5):
        RL[0].append(10**(LFeII-LbHbeta)[i])
        RL[1].append(fwhm_bHbeta[i])
        loud.append(i)
    
    elif Lradio[i] < np.log10(frec_20cm*10**31.5) and Lradio[i] != 0:
        RQ[0].append(10**(LFeII-LbHbeta)[i])
        RQ[1].append(fwhm_bHbeta[i])
        quiet.append(i)
    
    else:
        continue
        
RL = np.array(RL)
RQ = np.array(RQ)

plt.figure(1257)

plt.plot(10**(LFeII-LbHbeta), fwhm_bHbeta, 'k.', alpha=0.2)
plt.plot(RQ[0], RQ[1], 'b.', label='Radio Quiet')
plt.plot(RL[0], RL[1], 'r.', label='Radio Loud')
plt.xlim(-1, 12)
plt.xlabel('$L_{\\rm Fe \ II}/L_{\\rm bH\\beta}$', fontsize='x-large')
plt.ylabel('$\\rm FWHM \ bH\\beta \ (km/s)$', fontsize='x-large')
plt.legend(loc='best')
plt.savefig('figures/fig_1257_EV1')
plt.close('all')



from matplotlib import cm

fig = plt.figure(1258)
ax = fig.add_subplot(111)
X = 10**(LFeII-LbHbeta)
Y = fwhm_bHbeta
Z = LO3
sc = ax.scatter(X, Y, s=20, c=Z, marker='o', cmap=cm.jet)
col = plt.colorbar(sc )
col.set_label('$\\log \ L_{\\rm [O \ III]} \ \\rm  (erg \ s^{-1})$', fontsize='x-large')
plt.xlim(-1, 12)
plt.ylim(0, 20000)
plt.xlabel('$L_{\\rm Fe \ II}/L_{\\rm bH\\beta}$', fontsize='x-large')
plt.ylabel('$\\rm FWHM \ bH\\beta \ (km/s)$', fontsize='x-large')
#, cmap= cm.jet)
#ax.scatter(X, Y, Z)
plt.savefig('figures/fig_1258_EV1')
#plt.show()
plt.close('all')

fig = plt.figure(1259)
ax = fig.add_subplot(111)
X = 10**(LFeII-LbHbeta)
Y = fwhm_bHbeta
Z = 10**(LO3-LO2)
sc = ax.scatter(X, Y, s=20, c=Z, marker='o', cmap=cm.jet)
col = plt.colorbar(sc )
col.set_label('$L_{\\rm [O \ III]}/L_{\\rm [O \ II]}$', fontsize='x-large')
plt.xlim(-1, 12)
plt.ylim(0, 20000)
plt.xlabel('$L_{\\rm Fe \ II}/L_{\\rm bH\\beta}$', fontsize='x-large')
plt.ylabel('$\\rm FWHM \ bH\\beta \ (km/s)$', fontsize='x-large')
#, cmap= cm.jet)
#ax.scatter(X, Y, Z)
plt.savefig('figures/fig_1259_EV1')
#plt.show()
plt.close('all')

fig = plt.figure(1260)
ax = fig.add_subplot(111)
X = 10**(LFeII-LbHbeta)
Y = fwhm_bHbeta
Z = (LnHalpha-LbHalpha)
sc = ax.scatter(X, Y, s=20, c=Z, marker='o', cmap=cm.jet)
col = plt.colorbar(sc )
col.set_label('$\\log \ L_{\\rm nH\\alpha}/L_{\\rm bH\\alpha}$', fontsize='x-large')
plt.xlim(-1, 12)
plt.ylim(0, 20000)
plt.xlabel('$L_{\\rm Fe \ II}/L_{\\rm bH\\beta}$', fontsize='x-large')
plt.ylabel('$\\rm FWHM \ bH\\beta \ (km/s)$', fontsize='x-large')
#, cmap= cm.jet)
#ax.scatter(X, Y, Z)
plt.savefig('figures/fig_1260_EV1')
#plt.show()
plt.close('all')
######################################
'''
from scipy.interpolate import griddata
X = 10**(LFeII-LbHbeta)
Y = fwhm_bHbeta
Z = LO3

xi = np.arange(10**min(LFeII-LbHbeta), 12., 0.1)
yi = np.linspace(min(fwhm_bHbeta), max(fwhm_bHbeta), len(bin_FeII_bHbeta))
zi = griddata((X, Y), Z, (xi[None,:], yi[:,None]), method='linear')

fig = plt.figure(1279)
CS = plt.contourf(xi, yi, zi)
col = plt.colorbar(CS )
plt.xlabel('$L_{Fe \ II}/L_{bH\\beta}$', fontsize='large')
plt.ylabel('$FWHM \ bH\\beta \ (km/s)$', fontsize='large')
col.set_label('$\\log \ L_{nH\\alpha}/L_{bH\\alpha}$', fontsize='large')
plt.show()
'''



L_OII_3727 = np.array(L_OII_3727)
L_OIII_5007 = np.array(L_OIII_5007)
L_FeII = np.array(L_FeII)
L_bHbeta = np.array(L_bHbeta)
L_nHbeta = np.array(L_nHbeta)
FWHM_bHbeta = np.array(FWHM_bHbeta)

def binning2(c, a, b, mida, midb, number, labelx='x', labely='y', labelz='z'):
    xbin = []
    ybin = []
    zbin = []
    stdx = []
    stdy = []
    count = []
    
    if number == 2514:
        deleted = []
    

    for i in range(len(midb)):
        #print i
        for j in range(len(mida)):
            point = []
            axis = []
            value = []            
            
            if number ==2514:
                outlier = []
                
            
            for n in range(len(a)):
                if abs(a[n]-mida[j]) <= abs((mida[0]-mida[1])/2.)\
                and abs(b[n]-midb[i]) <= abs((midb[0]-midb[1])/2.):
                    axis.append(a[n])
                    point.append(b[n])
                    value.append(c[n])
                    #print 'yay2'
                    
                    if number == 2514:
                        outlier.append(n)
                    
                        
            if number ==2514 and len(point) < 3 and len(axis) < 3:
                for dt in outlier:                    
                    deleted.append(dt)
                
            if len(point) >= 3 and len(axis) >= 3 :
                ybin.append(np.median(point))
                xbin.append(np.median(axis))
                zbin.append(np.median(value))
                stdy.append(np.std(point)*1.)
                stdx.append(np.std(axis)*1.)            
                count.append(len(point))
                
                
    
#    plt.figure(number)
#    plt.plot(a, b, 'k.', alpha=0.2)
#    plt.errorbar(xbin, ybin, yerr=stdy, xerr=stdx, fmt='bo')
#    plt.xlabel(labelx, fontsize='large')
#    plt.ylabel(labely, fontsize='large')
#    plt.xlim(-1., 12.)
#    plt.savefig('figures/fig_%i' %number)
    #plt.show()
    
    fig = plt.figure(number)
    ax = fig.add_subplot(111)
    X = xbin#10**(LFeII-LbHbeta)
    Y = ybin#fwhm_bHbeta
    Z = zbin#LO3

    
    
    if number == 1279:
        sc = ax.scatter(np.concatenate((X, 10**(L_FeII[bal_qso]-L_bHbeta[bal_qso]))), \
        np.concatenate((Y, FWHM_bHbeta[bal_qso])), s=20, \
        c=np.concatenate((Z, 10**(L_OIII_5007[bal_qso]-L_OII_3727[bal_qso]))), marker='o', cmap=cm.jet)
        
        ax.scatter(10**(L_FeII[bal_qso]-L_bHbeta[bal_qso]), \
        FWHM_bHbeta[bal_qso], marker='s',s=60, facecolors='none', edgecolors='k')
    
    elif number == 1278:
        sc = ax.scatter(np.concatenate((X, 10**(L_FeII[bal_qso]-L_bHbeta[bal_qso]))), \
        np.concatenate((Y, FWHM_bHbeta[bal_qso])), s=20, \
        c=np.concatenate((Z, L_OIII_5007[bal_qso])), marker='o', cmap=cm.jet)
        
        ax.scatter(10**(L_FeII[bal_qso]-L_bHbeta[bal_qso]), \
        FWHM_bHbeta[bal_qso], marker='s',s=60, facecolors='none', edgecolors='k')    
    
    elif number == 1285:
        plt.plot(X, Y, 'ko', alpha=0.1, markersize=5)
        sc = ax.scatter(10**(LFeII[O]-LbHbeta[O]), fwhm_bHbeta[O], s=25, c=gamma[O], marker='s', cmap=cm.jet, vmin=min(gamma[O]), vmax=max(gamma[O]))

    elif(number == 2514):
        sc = ax.scatter(np.concatenate((X, a[deleted])), np.concatenate((Y, b[deleted])), s=15, c=np.concatenate((Z, c[deleted])), marker='o', cmap=cm.jet)
   
        ax.scatter(a[deleted], \
        b[deleted], marker='s',s=40, facecolors='none', edgecolors='k', alpha=0.25)           
   
    elif(number == 2526):
        sc = ax.scatter(np.concatenate((X, a[low_ion2])), np.concatenate((Y, b[low_ion2])), s=10, c=np.concatenate((Z, c[low_ion2])), marker='o', cmap=cm.jet)
   
        ax.scatter(a[liner], \
        b[liner], marker='s',s=50, facecolors='none', edgecolors='k', label ='LINER')           

        ax.scatter(a[composite_sii], \
        b[composite_sii], marker='^',s=40, facecolors='none', edgecolors='k', label ='SF')           

    elif(number == 2528):
        sc = ax.scatter(np.concatenate((X, a[low_ion2])), np.concatenate((Y, b[low_ion2])), s=10, c=np.concatenate((Z, c[low_ion2])), marker='o', cmap=cm.jet)
                   

    else:
        sc = ax.scatter(X, Y, s=20, c=Z, marker='o', cmap=cm.jet)
#    ax.scatter(10**(L_FeII[bal_qso]-L_bHbeta[bal_qso]), FWHM_bHbeta[bal_qso],\
#    s=60, c=10**(L_OIII_5007[bal_qso]-L_OII_3727[bal_qso]), marker='s', cmap=cm.jet)
    
    col = plt.colorbar(sc )
    col.set_label(labelz, fontsize='x-large')
    plt.xlim(-1, 12)
    if number < 1300:
        plt.ylim(0, 20000)
    
    if number > 2500 and number < 2600:
        plt.xlim(-0.3, 1.2)
    if number == 2514:
        plt.ylim(-1, 5)
        plt.xlim(-0.5, 1.5)
    if number == 2526:
#        plt.ylim(-1, 5)
        plt.xlim(-0.5, 1.5)
        plt.legend(loc='best', fontsize='small')

    if number == 2528:
        plt.ylim(0, 1.6)
        plt.xlim(-1, 1.5)
#        plt.legend(loc='best', fontsize='small')


        
    plt.xlabel(labelx, fontsize='x-large')
    plt.ylabel(labely, fontsize='x-large')
    plt.savefig('figures/fig_%i' %number)
    #plt.show()
    #plt.close('all')    
    
    return np.array([xbin, ybin, zbin])


mag_SDSS = np.array(mag_SDSS)
mag_2MASS = np.array(mag_2MASS)
mag_WISE = np.array(mag_WISE)

bin_FeII_bHbeta_2 = np.arange(10**min(LFeII-LbHbeta), 12.1, 0.2)
#bin_fwhm_bHbeta = np.linspace(min(fwhm_bHbeta), max(fwhm_bHbeta), len(bin_FeII_bHbeta))
bin_fwhm_bHbeta_2 = np.arange(0., 21000., 1000.)
bin_LO3_2 = np.arange(min(LO3), max(LO3)+0.25, 0.25)
bin_LO32_2 = np.arange(min(10**(LO3-LO2)), max(10**(LO3-LO2)), 0.2)
bin_sdss_wise_2 = np.arange(min(mag_SDSS[2]-mag_WISE[0]), max(mag_SDSS[2]-mag_WISE[0]), 0.2)

'''
fig_1278 = binning2(LO3, 10**(LFeII-LbHbeta), fwhm_bHbeta, bin_FeII_bHbeta_2, bin_fwhm_bHbeta_2, 1278,\
'$L_{\\rm Fe \ II}/L_{\\rm bH\\beta}$', '$\\rm FWHM \ bH\\beta \ (km/s)$', '$\\log \ L_{\\rm [O \ III]} \ \\rm  (erg \ s^{-1})$')

fig_1378 = binning2(fwhm_bHbeta, 10**(LFeII-LbHbeta), LO3, bin_FeII_bHbeta_2, bin_LO3_2, 1378,\
'$L_{\\rm Fe \ II}/L_{\\rm bH\\beta}$', '$\\log \ L_{\\rm [O \ III]} \ \\rm  (erg \ s^{-1})$', '$\\rm FWHM \ bH\\beta \ (km/s)$')

#fig_1478 = binning2(10**(LFeII-LbHbeta), LO3, fwhm_bHbeta, bin_LO3_2, bin_fwhm_bHbeta_2, 1478,\
#'$L_{\\rm Fe \ II}/L_{\\rm bH\\beta}$', '$\\log \ L_{\\rm [O \ III]} \ \\rm  (erg \ s^{-1})$', '$\\rm FWHM \ bH\\beta \ (km/s)$')


fig_1279 = binning2(10**(LO3-LO2), 10**(LFeII-LbHbeta), fwhm_bHbeta, bin_FeII_bHbeta_2, bin_fwhm_bHbeta_2, 1279,\
'$L_{\\rm Fe \ II}/L_{\\rm bH\\beta}$', '$\\rm FWHM \ bH\\beta \ (km/s)$', '$L_{\\rm [O \ III]}/L_{\\rm [O \ II]}$')

fig_1379 = binning2(fwhm_bHbeta, 10**(LFeII-LbHbeta), 10**(LO3-LO2), bin_FeII_bHbeta_2, bin_LO32_2, 1379,\
'$L_{\\rm Fe \ II}/L_{\\rm bH\\beta}$', '$L_{\\rm [O \ III]}/L_{\\rm [O \ II]}$', '$\\rm FWHM \ bH\\beta \ (km/s)$')



fig_1280 = binning2((LnHalpha-LbHalpha), 10**(LFeII-LbHbeta), fwhm_bHbeta, bin_FeII_bHbeta_2, bin_fwhm_bHbeta_2, 1280,\
'$L_{\\rm Fe \ II}/L_{\\rm bH\\beta}$', '$\\rm FWHM \ bH\\beta \ (km/s)$', '$\\log \ L_{\\rm nH\\alpha}/L_{bH\\alpha}$')



fig_1281 = binning2(mag_SDSS[2]-mag_2MASS[2], 10**(LFeII-LbHbeta), fwhm_bHbeta, bin_FeII_bHbeta_2, bin_fwhm_bHbeta_2, 1281,\
'$L_{\\rm Fe \ II}/L_{\\rm bH\\beta}$', '$\\rm FWHM \ bH\\beta \ (km/s)$', '$r-K_s$')

fig_1286 = binning2(mag_SDSS[2]-mag_WISE[0], 10**(LFeII-LbHbeta), fwhm_bHbeta, bin_FeII_bHbeta_2, bin_fwhm_bHbeta_2, 1286,\
'$L_{\\rm Fe \ II}/L_{\\rm bH\\beta}$', '$\\rm FWHM \ bH\\beta \ (km/s)$', '$r-W1$')

fig_1386 = binning2(fwhm_bHbeta, 10**(LFeII-LbHbeta), mag_SDSS[2]-mag_WISE[0], bin_FeII_bHbeta_2, bin_sdss_wise_2, 1386,\
'$L_{\\rm Fe \ II}/L_{\\rm bH\\beta}$', '$r-W1$', '$\\rm FWHM \ bH\\beta \ (km/s)$')

L_L_Edd = np.array(L_L_Edd)
m = np.array(m)
Lbol = np.array(Lbol)

fig_1282 = binning2(L_L_Edd, 10**(LFeII-LbHbeta), fwhm_bHbeta, bin_FeII_bHbeta_2, bin_fwhm_bHbeta_2, 1282,\
'$L_{\\rm Fe \ II}/L_{\\rm bH\\beta}$', '$\\rm FWHM \ bH\\beta \ (km/s)$', '$\\log \ L/L_{\\rm Edd}$')

fig_1283 = binning2(m, 10**(LFeII-LbHbeta), fwhm_bHbeta, bin_FeII_bHbeta_2, bin_fwhm_bHbeta_2, 1283,\
'$L_{\\rm Fe \ II}/L_{\\rm bH\\beta}$', '$\\rm FWHM \ bH\\beta \ (km/s)$', '$\\log \ \\rm M_{BH}/M_{\\odot}$')

fig_1284 = binning2(Lbol, 10**(LFeII-LbHbeta), fwhm_bHbeta, bin_FeII_bHbeta_2, bin_fwhm_bHbeta_2, 1284,\
'$L_{\\rm Fe \ II}/L_{\\rm bH\\beta}$', '$\\rm FWHM \ bH\\beta \ (km/s)$', '$\\log \ L_{\\rm bol} \ \\rm  (erg \ s^{-1})$')
'''


O = []
gamma = np.array(gamma)
fwhm_bHbeta = np.array(fwhm_bHbeta)
for i in range(len(gamma)):
    if gamma[i] != 0. :#and gamma[i] < 10.:
#    and 10.**(LFeII[i]-LbHbeta[i]) < 5.:
        
#        if 10.**(LFeII[i]-LbHbeta[i]) < 2.\
#        and gamma[i] > 2.:
#            continue
        
        O.append(i)
        

#    and FWHM_bHbeta[i] > 1000.:
        
'''
fig_1285 = binning2(gamma, 10**(LFeII-LbHbeta), fwhm_bHbeta, bin_FeII_bHbeta_2, bin_fwhm_bHbeta_2, 1285,\
'$L_{\\rm Fe \ II}/L_{\\rm bH\\beta}$', '$\\rm FWHM \ bH\\beta \ (km/s)$', '$\\Gamma_s$')


fig_1287 = binning2(mag_SDSS[2]-mag_WISE[0], RFeII, fwhm_bHbeta, bin_RFeII, bin_fwhm_bHbeta_2, 1287,\
'$R_{\\rm Fe II}$', '$\\rm FWHM \ bH\\beta \ (km/s)$', '$r-W1$')

fig_1288 = binning2(EWO3, RFeII, fwhm_bHbeta, bin_RFeII, bin_fwhm_bHbeta_2, 1288,\
'$R_{\\rm Fe II}$', '$\\rm FWHM \ bH\\beta \ (km/s)$', '$\\rm EW_{[O III]}$')

fig_1289 = binning2(L_L_Edd, RFeII, fwhm_bHbeta, bin_RFeII, bin_fwhm_bHbeta_2, 1289,\
'$R_{\\rm Fe II}$', '$\\rm FWHM \ bH\\beta \ (km/s)$', '$\\log \ L/L_{\\rm Edd}$')

###########
'''


########################################3


def bin_test(c, a, b, mida, midb, number, labelx='x', labely='y', labelz='z'):
    xbin = []
    ybin = []
    zbin = []
    stdx = []
    stdy = []
    count = []
    
    for i in range(len(midb)):
        #print i
        for j in range(len(mida)):
            point = []
            axis = []
            value = []            
                            
            
            for n in range(len(a)):
                if abs(a[n]-mida[j]) <= abs((mida[0]-mida[1])/2.)\
                and abs(b[n]-midb[i]) <= abs((midb[0]-midb[1])/2.):
                    axis.append(a[n])
                    point.append(b[n])
                    value.append(c[n])
                    #print 'yay2'
                                        
                                        
            if len(point) >= 3 and len(axis) >= 3 :
                
                if number == 2530:
                    if np.median(axis) < 0:
                        continue
                    
                ybin.append((point))
                xbin.append((axis))
                zbin.append([np.mean(value)]*len(point))
                stdy.append(np.std(point)*1.)
                stdx.append(np.std(axis)*1.)            
                count.append(len(point))
     
#    xbin = np.array(xbin).ravel()
#    ybin = np.array(ybin).ravel()
#    zbin = np.array(zbin).ravel()

    X = []#10**(LFeII-LbHbeta)
    Y = []#fwhm_bHbeta
    Z = []#LO3

    for i in range(len(xbin)):
        for j in range(len(xbin[i])):
            X.append(xbin[i][j])
            Y.append(ybin[i][j])
            Z.append(zbin[i][j])


    
    fig = plt.figure(number)
    ax = fig.add_subplot(111)


    if number == 11278:
        
        import scipy.stats as st
        #xx, yy = np.mgrid[min(X):max(X):30j, min(Y):max(Y):30j]
        xx, yy = np.mgrid[-0.1:13:30j, 0:21000:30j]
        positions = np.vstack([xx.ravel(), yy.ravel()])
        values = np.vstack([X, Y])
        kernel = st.gaussian_kde(values)
        f = np.reshape(kernel(positions).T, xx.shape)
        ax.contour(xx, yy, f, colors='k')
#        ax.clabel(cset, inline=1, fontsize=10)
        
        sc = ax.scatter(np.concatenate((X, 10**(L_FeII[bal_qso]-L_bHbeta[bal_qso]))), \
        np.concatenate((Y, FWHM_bHbeta[bal_qso])), s=20, \
        c=np.concatenate((Z, L_OIII_5007[bal_qso])), marker='*', edgecolors='none', cmap=cm.jet)
        
        ax.scatter(10**(L_FeII[bal_qso]-L_bHbeta[bal_qso]), \
        FWHM_bHbeta[bal_qso], marker='s',s=60, facecolors='none', edgecolors='k', label='BAL QSO')
        
#        plt.xlim(-0.1, 2.1)
        plt.ylim(0., 20000)

    elif number ==11279:

        import scipy.stats as st
        #xx, yy = np.mgrid[min(X):max(X):30j, min(Y):max(Y):30j]
        xx, yy = np.mgrid[-0.1:13:30j, 0:21000:30j]
        positions = np.vstack([xx.ravel(), yy.ravel()])
        values = np.vstack([X, Y])
        kernel = st.gaussian_kde(values)
        f = np.reshape(kernel(positions).T, xx.shape)
        ax.contour(xx, yy, f, colors='k')
#        ax.clabel(cset, inline=1, fontsize=10)

        sc = ax.scatter(np.concatenate((X, 10**(L_FeII[bal_qso]-L_bHbeta[bal_qso]))), \
        np.concatenate((Y, FWHM_bHbeta[bal_qso])), s=20, \
        c=np.concatenate((Z, 10**(L_OIII_5007[bal_qso]-L_OII_3727[bal_qso]))), marker='*', edgecolors='none', cmap=cm.jet, vmax=6.5)
        
        ax.scatter(10**(L_FeII[bal_qso]-L_bHbeta[bal_qso]), \
        FWHM_bHbeta[bal_qso], marker='s',s=60, facecolors='none', edgecolors='k', label='BAL QSO')
        
        
        plt.ylim(0., 20000)

    elif number ==11290:

        import scipy.stats as st
        #xx, yy = np.mgrid[min(X):max(X):30j, min(Y):max(Y):30j]
        xx, yy = np.mgrid[-0.1:13:30j, 0:21000:30j]
        positions = np.vstack([xx.ravel(), yy.ravel()])
        values = np.vstack([X, Y])
        kernel = st.gaussian_kde(values)
        f = np.reshape(kernel(positions).T, xx.shape)
        ax.contour(xx, yy, f, colors='k')
#        ax.clabel(cset, inline=1, fontsize=10)

        sc = ax.scatter(np.concatenate((X, 10**(L_FeII[bal_qso]-L_bHbeta[bal_qso]))), \
        np.concatenate((Y, FWHM_bHbeta[bal_qso])), s=20, \
        c=np.concatenate((Z, 10**(L_SII_6731[bal_qso]-L_SII_6717[bal_qso]))), marker='*', edgecolors='none', cmap=cm.jet, vmin=0.6, vmax = 1.)
        
        ax.scatter(10**(L_FeII[bal_qso]-L_bHbeta[bal_qso]), \
        FWHM_bHbeta[bal_qso], marker='s',s=60, facecolors='none', edgecolors='k', label='BAL QSO')
        
        
        plt.ylim(0., 20000)


    elif number == 11291:
        
        import scipy.stats as st
        #xx, yy = np.mgrid[min(X):max(X):30j, min(Y):max(Y):30j]
        xx, yy = np.mgrid[-0.1:13:30j, 0:21000:30j]
        positions = np.vstack([xx.ravel(), yy.ravel()])
        values = np.vstack([X, Y])
        kernel = st.gaussian_kde(values)
        f = np.reshape(kernel(positions).T, xx.shape)
        ax.contour(xx, yy, f, colors='k')
#        ax.clabel(cset, inline=1, fontsize=10)

        sc = ax.scatter((X), \
        (Y), s=20, \
        c=(Z), marker='*', edgecolors='none', cmap=cm.jet)
        
#        sc = ax.scatter(np.concatenate((X, 10**(L_FeII[bal_qso]-L_bHbeta[bal_qso]))), \
#        np.concatenate((Y, FWHM_bHbeta[bal_qso])), s=20, \
#        c=np.concatenate((Z, L_OIII_5007[bal_qso])), marker='*', edgecolors='none', cmap=cm.jet)
        
#        ax.scatter(10**(L_FeII[bal_qso]-L_bHbeta[bal_qso]), \
#        FWHM_bHbeta[bal_qso], marker='s',s=60, facecolors='none', edgecolors='k', label='BAL QSO')
        
#        plt.xlim(-0.1, 2.1)
        plt.ylim(0., 20000)

    elif number == 11292:
        
        import scipy.stats as st
        #xx, yy = np.mgrid[min(X):max(X):30j, min(Y):max(Y):30j]
        xx, yy = np.mgrid[-0.1:13:30j, 0:21000:30j]
        positions = np.vstack([xx.ravel(), yy.ravel()])
        values = np.vstack([X, Y])
        kernel = st.gaussian_kde(values)
        f = np.reshape(kernel(positions).T, xx.shape)
        ax.contour(xx, yy, f, colors='k')
#        ax.clabel(cset, inline=1, fontsize=10)

        sc = ax.scatter((X), \
        (Y), s=20, \
        c=(Z), marker='*', edgecolors='none', cmap=cm.jet)
        
#        sc = ax.scatter(np.concatenate((X, 10**(L_FeII[bal_qso]-L_bHbeta[bal_qso]))), \
#        np.concatenate((Y, FWHM_bHbeta[bal_qso])), s=20, \
#        c=np.concatenate((Z, L_OIII_5007[bal_qso])), marker='*', edgecolors='none', cmap=cm.jet)
        
#        ax.scatter(10**(L_FeII[bal_qso]-L_bHbeta[bal_qso]), \
#        FWHM_bHbeta[bal_qso], marker='s',s=60, facecolors='none', edgecolors='k', label='BAL QSO')
        
#        plt.xlim(-0.1, 2.1)
        plt.ylim(0., 20000)

    elif number == 11293:
        
        import scipy.stats as st
        #xx, yy = np.mgrid[min(X):max(X):30j, min(Y):max(Y):30j]
        xx, yy = np.mgrid[-0.1:13:30j, 0:21000:30j]
        positions = np.vstack([xx.ravel(), yy.ravel()])
        values = np.vstack([X, Y])
        kernel = st.gaussian_kde(values)
        f = np.reshape(kernel(positions).T, xx.shape)
        ax.contour(xx, yy, f, colors='k')
#        ax.clabel(cset, inline=1, fontsize=10)

        sc = ax.scatter((X), \
        (Y), s=20, \
        c=(Z), marker='*', edgecolors='none', cmap=cm.jet)
        
#        sc = ax.scatter(np.concatenate((X, 10**(L_FeII[bal_qso]-L_bHbeta[bal_qso]))), \
#        np.concatenate((Y, FWHM_bHbeta[bal_qso])), s=20, \
#        c=np.concatenate((Z, L_OIII_5007[bal_qso])), marker='*', edgecolors='none', cmap=cm.jet)
        
#        ax.scatter(10**(L_FeII[bal_qso]-L_bHbeta[bal_qso]), \
#        FWHM_bHbeta[bal_qso], marker='s',s=60, facecolors='none', edgecolors='k', label='BAL QSO')
        
#        plt.xlim(-0.1, 2.1)
        plt.ylim(0., 20000)

    elif number == 11286:

        import scipy.stats as st
        #xx, yy = np.mgrid[min(X):max(X):30j, min(Y):max(Y):30j]
        xx, yy = np.mgrid[-0.1:13:30j, 0:21000:30j]
        positions = np.vstack([xx.ravel(), yy.ravel()])
        values = np.vstack([X, Y])
        kernel = st.gaussian_kde(values)
        f = np.reshape(kernel(positions).T, xx.shape)
        ax.contour(xx, yy, f, colors='k')
#        ax.clabel(cset, inline=1, fontsize=10)

        sc = ax.scatter((X), \
        (Y), s=20, \
        c=(Z), marker='*', edgecolors='none', cmap=cm.jet)
        
#        ax.scatter(10**(L_FeII[bal_qso]-L_bHbeta[bal_qso]), \
#        FWHM_bHbeta[bal_qso], marker='s',s=60, facecolors='none', edgecolors='k', label='BAL QSO')
        
        
        plt.ylim(0., 20000)

    elif number == 11287:

        import scipy.stats as st
        #xx, yy = np.mgrid[min(X):max(X):30j, min(Y):max(Y):30j]
        xx, yy = np.mgrid[-0.1:13:30j, 0:21000:30j]
        positions = np.vstack([xx.ravel(), yy.ravel()])
        values = np.vstack([X, Y])
        kernel = st.gaussian_kde(values)
        f = np.reshape(kernel(positions).T, xx.shape)
        ax.contour(xx, yy, f, colors='k')
#        ax.clabel(cset, inline=1, fontsize=10)

        sc = ax.scatter((X), \
        (Y), s=20, \
        c=(Z), marker='*', edgecolors='none', cmap=cm.jet)
        
#        ax.scatter(10**(L_FeII[bal_qso]-L_bHbeta[bal_qso]), \
#        FWHM_bHbeta[bal_qso], marker='s',s=60, facecolors='none', edgecolors='k', label='BAL QSO')
        
        
        plt.ylim(0., 20000)
        
    
    elif number == 2529 or number == 2533 or number == 2534:
        
        low_ion2 = []
        liner = []
        composite_sii = []

        for i in range(len(LO3)):
        
            if LO3[i]-LO2[i] < 0. and LO3[i]-LO2[i] > -0.5:
                low_ion2.append(i)

                if (LO3[i]-LnHbeta[i]) < Ke01_b(LSII[i]-LnHalpha[i]):
                    composite_sii.append(i)        

            if (LO3[i]-LnHbeta[i]) < Ke06(LSII[i]-LnHalpha[i]) and LSII[i]-LnHalpha[i] < 0.25 and LO3[i]-LO2[i] < 0.5\
            and LO3[i]-LO2[i] < 0.\
            or (LO3[i]-LnHbeta[i]) < Ke06(LSII[i]-LnHalpha[i]) and LSII[i]-LnHalpha[i] < 0.25 and LO3[i]-LO2[i] < 0.5\
            and LO3[i]-LO2[i] > 0.065 and LO3[i]-LnHbeta[i] > 0.3:
                liner.append(i)
        
        bal = []
        for i in bal_qso:
            if L_OIII_5007[i] - L_nHbeta[i] > 0.4\
            and L_OIII_5007[i] - L_OII_3727[i] < 0.7:
                bal.append(i)
                
        if number == 2533:
            sc = ax.scatter(np.concatenate((X, a[low_ion2])), np.concatenate((Y, b[low_ion2])),\
            s=15, c=np.concatenate((Z, c[low_ion2])), marker='*', edgecolor='none', cmap=cm.jet, vmax=4.8)

        elif number == 2534:
            sc = ax.scatter(np.concatenate((X, a[low_ion2])), np.concatenate((Y, b[low_ion2])),\
            s=15, c=np.concatenate((Z, c[low_ion2])), marker='*', edgecolor='none', cmap=cm.jet, vmax=2.25)

        else:
            sc = ax.scatter(np.concatenate((X, a[low_ion2])), np.concatenate((Y, b[low_ion2])),\
            s=15, c=np.concatenate((Z, c[low_ion2])), marker='*', edgecolor='none', cmap=cm.jet)


        ax.scatter(a[liner], \
        b[liner], marker='s',s=50, facecolors='none', edgecolors='k', label ='LINER')           

        ax.scatter(a[composite_sii], \
        b[composite_sii], marker='^',s=40, facecolors='none', edgecolors='k', label ='SF')           

#        ax.scatter((L_OIII_5007-L_OII_3727)[bal], \
#        (L_OIII_5007-L_nHbeta)[bal], marker='D',s=60, facecolors='none', edgecolors='r', label ='BAL QSO')           

        plt.ylim(0, 1.6)
        plt.xlim(-1, 1.5)
        plt.plot([0]*50, np.linspace(0, 3.5, 50), 'k--')


    elif number == 2530:
        
        low_ion2 = []
        liner = []
        composite_sii = []

        for i in range(len(LO3)):
        
            if LO3_int[i]-LO2_int[i] < 0. and LO3_int[i]-LO2_int[i] > -0.5\
            and LO3_int[i] -LnHbeta_int[i] < 0.8:
                low_ion2.append(i)

                if (LO3_int[i]-LnHbeta_int[i]) < Ke01_b(LSII_int[i]-LnHalpha_int[i]):
                    composite_sii.append(i)        

            if (LO3_int[i]-LnHbeta_int[i]) < Ke06(LSII_int[i]-LnHalpha_int[i]) and LSII_int[i]-LnHalpha_int[i] < 0.25 and LO3_int[i]-LO2_int[i] < 0.5\
            and LO3_int[i]-LO2_int[i] < 0.\
            or (LO3_int[i]-LnHbeta_int[i]) < Ke06(LSII_int[i]-LnHalpha_int[i]) and LSII_int[i]-LnHalpha_int[i] < 0.25 and LO3_int[i]-LO2_int[i] < 0.5\
            and LO3_int[i]-LO2_int[i] > 0.065 and LO3_int[i]-LnHbeta_int[i] > 0.32:
                liner.append(i)
        
#        bal = []
#        for i in bal_qso:
#            if L_OIII_5007[i] - L_nHbeta[i] > 0.4\
#            and L_OIII_5007[i] - L_OII_3727[i] < 0.7:
#                bal.append(i)
        
        sc = ax.scatter(np.concatenate((X, a[low_ion2])), np.concatenate((Y, b[low_ion2])),\
        s=15, c=np.concatenate((Z, c[low_ion2])), marker='*', edgecolor='none', cmap=cm.jet, vmax=11.)

        ax.scatter(a[liner], \
        b[liner], marker='s',s=50, facecolors='none', edgecolors='k', label ='LINER')           

        ax.scatter(a[composite_sii], \
        b[composite_sii], marker='^',s=40, facecolors='none', edgecolors='k', label ='SF')           

#        ax.scatter((L_OIII_5007-L_OII_3727)[bal], \
#        (L_OIII_5007-L_nHbeta)[bal], marker='D',s=60, facecolors='none', edgecolors='r', label ='BAL QSO')           

        plt.ylim(0, 1.6)
        plt.xlim(-1, 1.5)
        plt.plot([0]*50, np.linspace(0, 3.5, 50), 'k--')

    elif number == 2531:

        O = []
        for i in range(len(gamma)):
            if gamma[i] != 0. :#and gamma[i] < 10.:
                if LO3[i]-LO2[i] > 0.8\
                or LO3[i]-LO2[i] > 0. and  LO3[i]-LnHbeta[i] < 0.5:
                    continue

        
                O.append(i)

        low_ion2 = []
        liner = []
        composite_sii = []

        for i in range(len(LO3)):
        
            if LO3[i]-LO2[i] < 0. and LO3[i]-LO2[i] > -0.5:
                low_ion2.append(i)

                if (LO3[i]-LnHbeta[i]) < Ke01_b(LSII[i]-LnHalpha[i]):
                    composite_sii.append(i)        

            if (LO3[i]-LnHbeta[i]) < Ke06(LSII[i]-LnHalpha[i]) and LSII[i]-LnHalpha[i] < 0.25 and LO3[i]-LO2[i] < 0.5\
            and LO3[i]-LO2[i] < 0.\
            or (LO3[i]-LnHbeta[i]) < Ke06(LSII[i]-LnHalpha[i]) and LSII[i]-LnHalpha[i] < 0.25 and LO3[i]-LO2[i] < 0.5\
            and LO3[i]-LO2[i] > 0.065 and LO3[i]-LnHbeta[i] > 0.3:
                liner.append(i)

        


        
        ax.scatter(np.concatenate((X, a[low_ion2])), np.concatenate((Y, b[low_ion2])),\
        s=20, marker='o', edgecolor='none', facecolors = 'k', alpha=0.05)

        ax.scatter(a[liner], \
        b[liner], marker='s',s=50, facecolors='none', edgecolors='k', label ='LINER')           

        ax.scatter(a[composite_sii], \
        b[composite_sii], marker='^',s=40, facecolors='none', edgecolors='k', label ='SF')           

        sc = ax.scatter(a[O], b[O], s=25, c=gamma[O], marker='s', cmap=cm.jet, vmin=min(gamma[O]), vmax=max(gamma[O]))

#        ax.scatter((L_OIII_5007-L_OII_3727)[bal], \
#        (L_OIII_5007-L_nHbeta)[bal], marker='D',s=60, facecolors='none', edgecolors='r', label ='BAL QSO')           

        plt.ylim(0, 1.6)
        plt.xlim(-1, 1.5)
        plt.plot([0]*50, np.linspace(0, 3.5, 50), 'k--')


    elif number == 11257:
        lobe = [3140, 4346, 6698, 7440, 7879, 9105, 1984, 2045, 4431, 4482, 4501, 4886, 5045, 5877, 6246, 6807, 7356, 7807, 7952, 8232, 8269, 8977, 9014]
        lobe = [3140, 4346, 6698, 7440, 7879, 9105]
        core_jet = [1984, 2045, 4431, 4482, 4501, 4886, 5045, 5877, 6246, 6807, 7356, 7807, 7952, 8232, 8269, 8977, 9014]        
        #lobe = [3140, 4346, 6698, 7440, 7879, 9105, 1984, 2045, 4431, 4482, 4501, 4886, 5045, 5877, 6246, 6807, 7356, 7807]        
        LD = []
        CD = []
        for i in range(len(J)):
            if J[i] in lobe:
                LD.append(i)
            if J[i] in core_jet and 10**(LFeII-LbHbeta)[i] < 6:
                CD.append(i)
        #print CD
        quiet = []
        loud = []
        non_detection = []

        for i in range(len(Lradio)):

            if str(10**(LFeII-LbHbeta)[i] in X) == 'False'\
            and str(fwhm_bHbeta[i] in Y) == 'False':
                continue

            if Lradio[i] >= np.log10(frec_20cm*10**31.5)  and 10**(LFeII-LbHbeta)[i] < 6:
                loud.append(i)
    
            elif Lradio[i] < np.log10(frec_20cm*10**31.5) and Lradio[i] != 0:
                quiet.append(i)
    
            else:
                non_detection.append(i)
                continue

        
#        sc = ax.scatter(X, Y, s=20, marker='o', facecolors='k', alpha=0.2)
        ax.scatter(X, Y, marker='o',s=20, facecolors='k', edgecolors='k', alpha=0.05)           
        ax.scatter(10**(LFeII-LbHbeta)[quiet], np.array(fwhm_bHbeta)[quiet], marker='o',s=30, facecolors='b', edgecolors='k', label = 'Radio Quiet')        
        ax.scatter(10**(LFeII-LbHbeta)[loud], np.array(fwhm_bHbeta)[loud], marker='o',s=30, facecolors='r', edgecolors='k', label ='Radio Loud CD')
        ax.scatter(10**(LFeII-LbHbeta)[CD], np.array(fwhm_bHbeta)[CD], marker='o',s=30, facecolors='r', edgecolors='k')
        ax.scatter(10**(LFeII-LbHbeta)[LD], np.array(fwhm_bHbeta)[LD], marker='s',s=40, facecolors='r', edgecolors='k', label ='Radio Loud LD')
        
        plt.plot(np.linspace(-1, 14), [4000.]*len(np.linspace(0, 13)), 'k--')
        plt.ylim(0., 20000)
        plt.xlim(-0.7, 13.6)


    elif number == 11258:
        lobe = [3140, 4346, 6698, 7440, 7879, 9105]        
        lobe = [7807, 5877, 4346, 3140, 6246, 7879, 6698, 9105, 7440, 7356, 9014, 5045, 6807, 4482]
        core_jet = [1984, 2045, 4431, 4482, 4501, 4886, 5045, 5877, 6246, 6807, 7356, 7807, 7952, 8232, 8269, 8977, 9014]        
        #lobe = [3140, 4346, 6698, 7440, 7879, 9105, 1984, 2045, 4431, 4482, 4501, 4886, 5045, 5877, 6246, 6807, 7356, 7807]        
        LD = []
        CD = []
        for i in range(len(J)):
            if J[i] in lobe and 10**(LFeII-LbHbeta)[i] < 6:
                LD.append(i)
            if J[i] in core_jet and 10**(LFeII-LbHbeta)[i] < 6:
                CD.append(i)



        tanda = []
        quiet = []
        loud = []
        non_detection = []
        intermediate = []

        for i in range(len(Lradio)):

            if str(10**(LFeII-LbHbeta)[i] in X) == 'False'\
            and str(fwhm_bHbeta[i] in Y) == 'False':
                continue
            
            if Rk[i] >= 70.  :#and 10**(LFeII-LbHbeta)[i] < 6:
                loud.append(i)
    
            elif Rk[i] < 70. and Lradio[i] != 0:
                quiet.append(i)
    
            else:
                non_detection.append(i)
                continue

            if Rk[i] < 70. and Rk[i] > 10.:
                intermediate.append(i)
            
            tanda.append(J[i])

        
#        sc = ax.scatter(X, Y, s=20, marker='o', facecolors='k', alpha=0.2)
        ax.scatter(X, Y, marker='o',s=20, facecolors='k', edgecolors='k', alpha=0.05)           
        ax.scatter(10**(LFeII-LbHbeta)[quiet], np.array(fwhm_bHbeta)[quiet], marker='o',s=30, facecolors='b', edgecolors='k', label = 'Radio Quiet')        
        ax.scatter(10**(LFeII-LbHbeta)[loud], np.array(fwhm_bHbeta)[loud], marker='o',s=30, facecolors='r', edgecolors='k', label ='Radio Loud CD')

        #ax.scatter(10**(LFeII-LbHbeta)[CD], np.array(fwhm_bHbeta)[CD], marker='o',s=30, facecolors='r', edgecolors='k')
        ax.scatter(10**(LFeII-LbHbeta)[LD], np.array(fwhm_bHbeta)[LD], marker='s',s=40, facecolors='r', edgecolors='k', label ='Radio Loud LD')

        plt.plot(np.linspace(-1, 14), [4000.]*len(np.linspace(0, 13)), 'k--')
        plt.ylim(0., 20000)
        plt.xlim(-0.7, 13.6)

    elif number == 11259:

        quiet = []
        loud = []
        non_detection = []

        for i in range(len(Lradio)):

            if str(10**(LFeII-LbHbeta)[i] in X) == 'False'\
            and str(fwhm_bHbeta[i] in Y) == 'False':
                continue
            
            if J[i] == 4687:
                print 'Rk = ', Rk[i]

            if Rk[i] >= 10  :#and 10**(LFeII-LbHbeta)[i] < 6:
                loud.append(i)
    
            elif Rk[i] < 10 and Lradio[i] != 0:
                quiet.append(i)
    
            else:
                non_detection.append(i)
                continue

        
#        sc = ax.scatter(X, Y, s=20, marker='o', facecolors='k', alpha=0.2)
        ax.scatter(X, Y, marker='o',s=20, facecolors='k', edgecolors='k', alpha=0.05)           
        ax.scatter(10**(LFeII-LbHbeta)[quiet], np.array(fwhm_bHbeta)[quiet], marker='o',s=20, facecolors='b', edgecolors='k', label = 'Radio Quiet')        
        ax.scatter(10**(LFeII-LbHbeta)[loud], np.array(fwhm_bHbeta)[loud], marker='o',s=20, facecolors='r', edgecolors='k', label ='Radio Loud')
        plt.plot(np.linspace(-1, 14), [4000.]*len(np.linspace(0, 13)), 'k--')
        plt.ylim(0., 20000)
        plt.xlim(-0.7, 13.6)

    elif number == 2532:
#        ax.scatter(X, Y, marker='o',s=20, facecolors='k', edgecolors='k', alpha=0.05)           
        sc = ax.scatter(X, Y, s=15, c=Z, marker='o', cmap=cm.jet, vmin=min(Z))        
        plt.ylim(0, 1.6)
        plt.xlim(-1, 1.5)
        plt.plot([0]*50, np.linspace(0, 3.5, 50), 'k--')
    else:
        sc = ax.scatter(X, Y, s=20, c=Z, marker='o', cmap=cm.jet, vmin=37.)

    
    if number != 11257 and number != 11258 and number != 11259:
        col = plt.colorbar(sc)
        col.set_label(labelz, fontsize='x-large')
        
    plt.xlabel(labelx, fontsize='x-large')
    plt.ylabel(labely, fontsize='x-large')
    plt.legend(loc='best', fontsize='small')
    plt.savefig('figures/fig_%i' %number)
    
    #plt.show()
    #plt.close('all')    
    if number == 11257  or number == 11259:
        return np.array((quiet, loud, non_detection))
    elif number == 11258:
        return np.array((quiet, loud, non_detection, tanda, intermediate))
    else:
        return np.array((X, Y, Z))

#######################
bin_FeII_bHbeta_3 = np.arange(0, 2.1, 0.2)
bin_fwhm_bHbeta_3 = np.arange(0., 21000., 1000.)

fig_11278 = bin_test(LO3, 10**(LFeII-LbHbeta), fwhm_bHbeta, bin_FeII_bHbeta_2, bin_fwhm_bHbeta_2, 11278,\
'$L_{\\rm Fe \ II}/L_{\\rm bH\\beta}$', '$\\rm FWHM \ bH\\beta \ (km/s)$', '$\\log \ L_{\\rm [O \ III]} \ \\rm  (erg \ s^{-1})$')

fig_11279 = bin_test(10**(LO3-LO2), 10**(LFeII-LbHbeta), fwhm_bHbeta, bin_FeII_bHbeta_2, bin_fwhm_bHbeta_2, 11279,\
'$L_{\\rm Fe \ II}/L_{\\rm bH\\beta}$', '$\\rm FWHM \ bH\\beta \ (km/s)$', '$L_{\\rm [O \ III]}/L_{\\rm [O \ II]}$')

fig_11286 = bin_test(mag_SDSS[2]-mag_WISE[0], 10**(LFeII-LbHbeta), fwhm_bHbeta, bin_FeII_bHbeta_2, bin_fwhm_bHbeta_2, 11286,\
'$L_{\\rm Fe \ II}/L_{\\rm bH\\beta}$', '$\\rm FWHM \ bH\\beta \ (km/s)$', '$r-W1$')

r_mag_cor = []
J_mag_cor = []

for i in range(len(mag_SDSS[2])):
    r_mag_cor.append(mag_SDSS[2][i]-calc_kcor('r', z_O32[i], 'g - r', mag_SDSS[1][i]-mag_SDSS[2][i]))
    J_mag_cor.append(mag_2MASS[0][i]-calc_kcor('J2', z_O32[i], 'J2 - H2', mag_2MASS[0][i]-mag_2MASS[1][i]))

r_mag_cor = np.array(r_mag_cor)
J_mag_cor = np.array(J_mag_cor)

fig_11287 = bin_test(r_mag_cor-J_mag_cor, 10**(LFeII-LbHbeta), fwhm_bHbeta, bin_FeII_bHbeta_2, bin_fwhm_bHbeta_2, 11287,\
'$L_{\\rm Fe \ II}/L_{\\rm bH\\beta}$', '$\\rm FWHM \ bH\\beta \ (km/s)$', '$r-J$')


L_SII_6717 = np.array(L_SII_6717)
L_SII_6731 = np.array(L_SII_6731)

es = []
for i in J:
    if 10**(L_SII_6731[i] - L_SII_6717[i]) >= 0.6 and 10**(L_SII_6731[i] - L_SII_6717[i]) <= 1.2:
       es.append(i) 


fig_11290 = bin_test(10**(L_SII_6731[es]-L_SII_6717[es]), 10**(L_FeII[es]-L_bHbeta[es]), FWHM_bHbeta[es], bin_FeII_bHbeta_2, bin_fwhm_bHbeta_2, 11290,\
'$L_{\\rm Fe \ II}/L_{\\rm bH\\beta}$', '$\\rm FWHM \ bH\\beta \ (km/s)$', '$L_{\\rm [S \ II] \ \\lambda 6731}/L_{\\rm [S \ II] \ \\lambda6717}$')

fig_11291 = bin_test(z_O32, 10**(LFeII-LbHbeta), fwhm_bHbeta, bin_FeII_bHbeta_2, bin_fwhm_bHbeta_2, 11291,\
'$L_{\\rm Fe \ II}/L_{\\rm bH\\beta}$', '$\\rm FWHM \ bH\\beta \ (km/s)$', '$z$')

fig_11292 = bin_test(shiftOIII, 10**(LFeII-LbHbeta), fwhm_bHbeta, bin_FeII_bHbeta_2, bin_fwhm_bHbeta_2, 11292,\
'$L_{\\rm Fe \ II}/L_{\\rm bH\\beta}$', '$\\rm FWHM \ bH\\beta \ (km/s)$', '$\\rm Shift \ [O \ III] \ (km/s)$')

fig_11293 = bin_test(fwhm_OIII_5007, 10**(LFeII-LbHbeta), fwhm_bHbeta, bin_FeII_bHbeta_2, bin_fwhm_bHbeta_2, 11293,\
'$L_{\\rm Fe \ II}/L_{\\rm bH\\beta}$', '$\\rm FWHM \ bH\\beta \ (km/s)$', '$\\rm FWHM \ [O \ III] \ (km/s)$')

fig_11257 = bin_test(Lradio, 10**(LFeII-LbHbeta), fwhm_bHbeta, bin_FeII_bHbeta_2, bin_fwhm_bHbeta_2, 11257,\
'$L_{\\rm Fe \ II}/L_{\\rm bH\\beta}$', '$\\rm FWHM \ bH\\beta \ (km/s)$', '$\\log \ L_{\\rm Radio} \ \\rm  (erg \ s^{-1})$')


f_20cm = np.array(f_20cm)
B = np.array(B)
f_6cm = f_20cm * (6./20.)**-0.5
f_opt = 10.**((B + 48.36)/-2.5)
Rk = f_6cm/f_opt


fig_11258 = bin_test(Lradio, 10**(LFeII-LbHbeta), fwhm_bHbeta, bin_FeII_bHbeta_2, bin_fwhm_bHbeta_2, 11258,\
'$L_{\\rm Fe \ II}/L_{\\rm bH\\beta}$', '$\\rm FWHM \ bH\\beta \ (km/s)$', '$\\log \ L_{\\rm 20 \, cm} \ \\rm  (erg \ s^{-1})$')

fig_11259 = bin_test(Lradio, 10**(LFeII-LbHbeta), fwhm_bHbeta, bin_FeII_bHbeta_2, bin_fwhm_bHbeta_2, 11259,\
'$L_{\\rm Fe \ II}/L_{\\rm bH\\beta}$', '$\\rm FWHM \ bH\\beta \ (km/s)$', '$\\log \ L_{\\rm 20 \, cm} \ \\rm  (erg \ s^{-1})$')

plt.close('all')




plt.figure(11258)
plt.plot(z_O32[fig_11257[0]], Lradio[fig_11257[0]], 'bo', label = 'Radio Quiet')
plt.plot(z_O32[fig_11257[1]], Lradio[fig_11257[1]], 'ro', label = 'Radio Loud')
plt.xlabel('$z$', fontsize='x-large')
plt.ylabel('$\\log \ L_{\\rm Radio} \ \\rm  (erg \ s^{-1})$', fontsize='x-large')
plt.legend(loc='best', fontsize='small')
plt.savefig('figures/fig_11257_sulentic')
plt.close('all')

plt.figure(11259)
plt.plot(z_O32[fig_11257[0]], Lradio[fig_11257[0]]/np.max(Lradio), 'ro', label = 'Radio Loud')
plt.plot(z_O32[fig_11257[1]], Lradio[fig_11257[1]]/np.max(Lradio), 'ro')
plt.plot(z_O32[fig_11257[2]], Lradio[fig_11257[2]], 'bo', label = 'Radio Quiet')
plt.xlabel('$z$', fontsize='x-large')
plt.ylabel('Normalized $\\log \ L_{\\rm Radio}$')
plt.ylim(-0.1, 1)
plt.legend(loc='best', fontsize='small')
plt.savefig('figures/fig_11257_shen')
plt.close('all')






def L_int(x, alpha, beta, lam): 
    E_bv = 1.99*np.log10(10**(alpha - beta)/2.86)
    for i in range(len(E_bv)):
        if E_bv[i] < 0:
            E_bv[i] = 0.
    c = E_bv/0.77
    return np.log10(10**(x-beta) * 10**(c*itp(lam)) * 10**beta)


plt.figure(11260)
plt.plot(10**(LO3-LO2)[fig_11257[0]], Lradio[fig_11257[0]], 'bo', label = 'Radio Quiet')
plt.plot(10**(LO3-LO2)[fig_11257[1]], Lradio[fig_11257[1]], 'ro', label = 'Radio Loud')
plt.xlabel('$L_{\\rm [O \ III]}/L_{\\rm [O \ II]}$', fontsize='x-large')
plt.ylabel('$\\log \ L_{\\rm Radio} \ \\rm  (erg \ s^{-1})$', fontsize='x-large')
plt.legend(loc='best', fontsize='small')
plt.savefig('figures/fig_11260')

LnHbeta_int = L_int(LnHbeta, LnHalpha, LnHbeta, 4860.)
LO2_int = L_int(LO2, LnHalpha, LnHbeta, 3727.)
LO3_int = L_int(LO3, LnHalpha, LnHbeta, 5007.)

plt.figure(11261)
plt.plot(10**(LO3_int-LO2_int)[fig_11257[0]], Lradio[fig_11257[0]], 'bo', label = 'Radio Quiet')
plt.plot(10**(LO3_int-LO2_int)[fig_11257[1]], Lradio[fig_11257[1]], 'ro', label = 'Radio Loud')
plt.xlabel('$L_{\\rm [O \ III]}^{cor}/L_{\\rm [O \ II]}^{cor}$', fontsize='x-large')
plt.ylabel('$\\log \ L_{\\rm Radio} \ \\rm  (erg \ s^{-1})$', fontsize='x-large')
plt.legend(loc='best', fontsize='small')
plt.xlim(0, 18)
plt.savefig('figures/fig_11261')

plt.figure(11262)
plt.plot((LO3)[fig_11257[0]], Lradio[fig_11257[0]], 'bo', label = 'Radio Quiet')
plt.plot((LO3)[fig_11257[1]], Lradio[fig_11257[1]], 'ro', label = 'Radio Loud')
plt.xlabel('$\\log \ L_{\\rm [O \ III]} \ \\rm  (erg \ s^{-1})$', fontsize='x-large')
plt.ylabel('$\\log \ L_{\\rm Radio} \ \\rm  (erg \ s^{-1})$', fontsize='x-large')
plt.legend(loc='best', fontsize='small')
plt.savefig('figures/fig_11262')

plt.figure(11263)
plt.plot((LO3_int)[fig_11257[0]], Lradio[fig_11257[0]], 'bo', label = 'Radio Quiet')
plt.plot((LO3_int)[fig_11257[1]], Lradio[fig_11257[1]], 'ro', label = 'Radio Loud')
plt.xlabel('$\\log \ L_{\\rm [O \ III]}^{cor} \ \\rm  (erg \ s^{-1})$', fontsize='x-large')
plt.ylabel('$\\log \ L_{\\rm Radio} \ \\rm  (erg \ s^{-1})$', fontsize='x-large')
plt.legend(loc='best', fontsize='small')
plt.savefig('figures/fig_11263')

plt.figure(11264)
plt.plot((LO2)[fig_11257[0]], Lradio[fig_11257[0]], 'bo', label = 'Radio Quiet')
plt.plot((LO2)[fig_11257[1]], Lradio[fig_11257[1]], 'ro', label = 'Radio Loud')
plt.xlabel('$\\log \ L_{\\rm [O \ II]} \ \\rm  (erg \ s^{-1})$', fontsize='x-large')
plt.ylabel('$\\log \ L_{\\rm Radio} \ \\rm  (erg \ s^{-1})$', fontsize='x-large')
plt.legend(loc='best', fontsize='small')
plt.savefig('figures/fig_11264')

plt.figure(11265)
plt.plot((LO2_int)[fig_11257[0]], Lradio[fig_11257[0]], 'bo', label = 'Radio Quiet')
plt.plot((LO2_int)[fig_11257[1]], Lradio[fig_11257[1]], 'ro', label = 'Radio Loud')
plt.xlabel('$\\log \ L_{\\rm [O \ II]}^{cor} \ \\rm  (erg \ s^{-1})$', fontsize='x-large')
plt.ylabel('$\\log \ L_{\\rm Radio} \ \\rm  (erg \ s^{-1})$', fontsize='x-large')
plt.legend(loc='best', fontsize='small')
plt.xlim(39.5, 43.)
plt.savefig('figures/fig_11265')
####################################################################################

lobe = [3140, 4346, 6698, 7440, 7879, 9105, 1984, 2045, 4431, 4482, 4501, 4886, 5045, 5877, 6246, 6807, 7356, 7807, 7952, 8232, 8269, 8977, 9014]
lobe = [3140, 4346, 6698, 7440, 7879, 9105]
lobe = [7807, 5877, 4346, 3140, 6246, 7879, 6698, 9105, 7440, 7356, 9014, 5045, 6807, 4482]
LD = []
CD = [694, 713, 1529, 1555, 1567, 1702, 1760, 2053, 2170, 2583, 2745, 2788, 2887, 2899, 3127, 3139]
for i in range(len(J)):
    if J[i] in lobe:
       LD.append(i) 



plt.figure(11270)
plt.plot(10**(LO3-LO2)[fig_11258[0]], Lradio[fig_11258[0]], 'bo', label = 'Radio Quiet')
#plt.plot(10**(LO3-LO2)[fig_11258[1]], Lradio[fig_11258[1]], 'ro', label = 'Radio Loud')
plt.plot(10**(LO3-LO2)[fig_11258[1]], Lradio[fig_11258[1]], 'ro', label = 'Radio Loud CD')
#plt.plot(10**(LO3-LO2)[CD], Lradio[CD], 'ro')
plt.plot(10**(LO3-LO2)[LD], Lradio[LD], 'rs', label = 'Radio Loud LD')
plt.xlabel('$L_{\\rm [O \ III]}/L_{\\rm [O \ II]}$', fontsize='x-large')
plt.ylabel('$\\log \ L_{\\rm Radio} \ \\rm  (erg \ s^{-1})$', fontsize='x-large')
plt.legend(loc='best', fontsize='small')
plt.savefig('figures/fig_11270')

LnHbeta_int = L_int(LnHbeta, LnHalpha, LnHbeta, 4860.)
LO2_int = L_int(LO2, LnHalpha, LnHbeta, 3727.)
LO3_int = L_int(LO3, LnHalpha, LnHbeta, 5007.)

plt.figure(11271)
plt.plot(10**(LO3_int-LO2_int)[fig_11258[0]], Lradio[fig_11258[0]], 'bo', label = 'Radio Quiet')
plt.plot(10**(LO3_int-LO2_int)[fig_11258[1]], Lradio[fig_11258[1]], 'ro', label = 'Radio Loud')
plt.xlabel('$L_{\\rm [O \ III]}^{cor}/L_{\\rm [O \ II]}^{cor}$', fontsize='x-large')
plt.ylabel('$\\log \ L_{\\rm Radio} \ \\rm  (erg \ s^{-1})$', fontsize='x-large')
plt.legend(loc='best', fontsize='small')
plt.xlim(0, 18)
plt.savefig('figures/fig_11271')

plt.figure(11272)
plt.plot((LO3)[fig_11258[0]], Lradio[fig_11258[0]], 'bo', label = 'Radio Quiet')
plt.plot((LO3)[fig_11258[1]], Lradio[fig_11258[1]], 'ro', label = 'Radio Loud CD')
print 'This is for [O III]'
print linregress((LO3)[fig_11258[0]], Lradio[fig_11258[0]])
#plt.plot((LO3)[CD], Lradio[CD], 'ro')
plt.plot((LO3)[LD], Lradio[LD], 'rs', label = 'Radio Loud LD')
plt.xlabel('$\\log \ L_{\\rm [O \ III]} \ \\rm  (erg \ s^{-1})$', fontsize='x-large')
plt.ylabel('$\\log \ L_{\\rm Radio} \ \\rm  (erg \ s^{-1})$', fontsize='x-large')
plt.legend(loc='best', fontsize='small')
plt.savefig('figures/fig_11272')


plt.figure(11282)

plt.plot((LO3)[fig_11258[0]], Lradio[fig_11258[0]], 'bo', label = 'Radio Quiet')
plt.plot((LO3)[fig_11258[1]], Lradio[fig_11258[1]], 'ro', label = 'Radio Loud CD')
plt.plot((LO3)[fig_11258[-1]], Lradio[fig_11258[-1]], 'wo', label = 'Radio Intermediate')
#plt.plot((LO3)[CD], Lradio[CD], 'ro')
plt.plot((LO3)[LD], Lradio[LD], 'rs', label = 'Radio Loud LD')
plt.xlabel('$\\log \ L_{\\rm [O \ III]} \ \\rm  (erg \ s^{-1})$', fontsize='x-large')
plt.ylabel('$\\log \ L_{\\rm Radio} \ \\rm  (erg \ s^{-1})$', fontsize='x-large')
plt.legend(loc='best', fontsize='small')
plt.savefig('figures/fig_11282')


plt.figure(11273)
plt.plot((LO3_int)[fig_11258[0]], Lradio[fig_11258[0]], 'bo', label = 'Radio Quiet')
plt.plot((LO3_int)[fig_11258[1]], Lradio[fig_11258[1]], 'ro', label = 'Radio Loud')
plt.xlabel('$\\log \ L_{\\rm [O \ III]}^{cor} \ \\rm  (erg \ s^{-1})$', fontsize='x-large')
plt.ylabel('$\\log \ L_{\\rm Radio} \ \\rm  (erg \ s^{-1})$', fontsize='x-large')
plt.legend(loc='best', fontsize='small')
plt.savefig('figures/fig_11273')

plt.figure(11274)
plt.plot((LO2)[fig_11258[0]], Lradio[fig_11258[0]], 'bo', label = 'Radio Quiet')
plt.plot((LO2)[fig_11258[1]], Lradio[fig_11258[1]], 'ro', label = 'Radio Loud CD')
print 'This is for [O II]'
print linregress((LO2)[fig_11258[0]], Lradio[fig_11258[0]])
#plt.plot((LO2)[CD], Lradio[CD], 'ro')
plt.plot((LO2)[LD], Lradio[LD], 'rs', label = 'Radio Loud LD')
plt.xlabel('$\\log \ L_{\\rm [O \ II]} \ \\rm  (erg \ s^{-1})$', fontsize='x-large')
plt.ylabel('$\\log \ L_{\\rm Radio} \ \\rm  (erg \ s^{-1})$', fontsize='x-large')
plt.legend(loc='best', fontsize='small')
plt.savefig('figures/fig_11274')

plt.figure(11284)
plt.plot((LO2)[fig_11258[0]], Lradio[fig_11258[0]], 'bo', label = 'Radio Quiet')
plt.plot((LO2)[fig_11258[1]], Lradio[fig_11258[1]], 'ro', label = 'Radio Loud CD')
#plt.plot((LO2)[CD], Lradio[CD], 'ro')
plt.plot((LO2)[LD], Lradio[LD], 'rs', label = 'Radio Loud LD')
plt.plot((LO2)[fig_11258[-1]], Lradio[fig_11258[-1]], 'wo', label = 'Radio Intermediate')

plt.xlabel('$\\log \ L_{\\rm [O \ II]} \ \\rm  (erg \ s^{-1})$', fontsize='x-large')
plt.ylabel('$\\log \ L_{\\rm Radio} \ \\rm  (erg \ s^{-1})$', fontsize='x-large')
plt.legend(loc='best', fontsize='small')
plt.savefig('figures/fig_11284')

plt.figure(11275)
plt.plot((LO2_int)[fig_11258[0]], Lradio[fig_11258[0]], 'bo', label = 'Radio Quiet')
plt.plot((LO2_int)[fig_11258[1]], Lradio[fig_11258[1]], 'ro', label = 'Radio Loud')
plt.xlabel('$\\log \ L_{\\rm [O \ II]}^{cor} \ \\rm  (erg \ s^{-1})$', fontsize='x-large')
plt.ylabel('$\\log \ L_{\\rm Radio} \ \\rm  (erg \ s^{-1})$', fontsize='x-large')
plt.legend(loc='best', fontsize='small')
plt.xlim(39.5, 43.)
plt.savefig('figures/fig_11275')

plt.close('all')

'''
plt.figure(1278)
plt.plot(10**(L_FeII[bal_qso]-L_bHbeta[bal_qso]), FWHM_bHbeta[bal_qso], 'ko')
plt.savefig('figures/fig_%i' %1278)
plt.show()
'''

'''
shen = np.loadtxt('shen_data.tsv', delimiter=';', dtype=str)

shen_Rfe = []
shen_fwhm_bHbeta = []
shen_Lbol = []
shen_L_L_Edd = []
shen_W1 = []
shen_r = []
shen_LO3 = []
shen_EWO3 = []


for i in range(len(shen)):
    if float(shen[i][0]) < 0.35:
        shen_Lbol.append(float(shen[i][1]))
        shen_fwhm_bHbeta.append(float(shen[i][2]))
        shen_Rfe.append(float(shen[i][4])/float(shen[i][3]))
        shen_L_L_Edd.append(float(shen[i][5]))        
        shen_W1.append(float(shen[i][6]))        
        shen_r.append(float(shen[i][7]))
        shen_LO3.append(float(shen[i][8]))
        shen_EWO3.append(float(shen[i][9]))
        
shen_Rfe = np.array(shen_Rfe)
shen_fwhm_bHbeta = np.array(shen_fwhm_bHbeta)
shen_Lbol = np.array(shen_Lbol)
shen_L_L_Edd = np.array(shen_L_L_Edd)
shen_W1 = np.array(shen_W1)
shen_r = np.array(shen_r)
shen_LO3 = np.array(shen_LO3)
shen_EWO3 = np.array(shen_EWO3)


fig_1290 = binning2(shen_L_L_Edd, shen_Rfe, shen_fwhm_bHbeta, bin_RFeII, bin_fwhm_bHbeta_2, 1290,\
'$R_{\\rm Fe II}$', '$\\rm FWHM \ bH\\beta \ (km/s)$', '$\\log \ L/L_{\\rm Edd}$')

fig_1291 = binning2(shen_r-shen_W1, shen_Rfe, shen_fwhm_bHbeta, bin_RFeII, bin_fwhm_bHbeta_2, 1291,\
'$R_{\\rm Fe II}$', '$\\rm FWHM \ bH\\beta \ (km/s)$', '$r-W1$')

fig_1292 = binning2(shen_Lbol, shen_Rfe, shen_fwhm_bHbeta, bin_RFeII, bin_fwhm_bHbeta_2, 1292,\
'$R_{\\rm Fe II}$', '$\\rm FWHM \ bH\\beta \ (km/s)$', '$\\log \ L_{\\rm bol} \ \\rm  (erg \ s^{-1})$')

fig_1293 = binning2(shen_LO3, shen_Rfe, shen_fwhm_bHbeta, bin_RFeII, bin_fwhm_bHbeta_2, 1293,\
'$R_{\\rm Fe II}$', '$\\rm FWHM \ bH\\beta \ (km/s)$', '$\\log \ L_{\\rm [O III]} \ \\rm  (erg \ s^{-1})$')

fig_1294 = binning2(shen_EWO3, shen_Rfe, shen_fwhm_bHbeta, bin_RFeII, bin_fwhm_bHbeta_2, 1294,\
'$R_{\\rm Fe II}$', '$\\rm FWHM \ bH\\beta \ (km/s)$', '$\\rm EW_{[O III]}$')

#RFeII = np.array(RFeII)
#fig_1292 = binning2(mag_SDSS[2][Q]-np.array(mag_WISE[0])[Q], RFeII[Q], fwhm_bHbeta[Q], bin_RFeII, bin_fwhm_bHbeta_2, 1292,\
#'$R_{\\rm Fe II}$', '$\\rm FWHM \ bH\\beta \ (km/s)$', '$r-W1$')
'''



plt.close('all')


######################################


LO2_int = L_int(LO2, LnHalpha, LnHbeta, 3727.)
LO3_int = L_int(LO3, LnHalpha, LnHbeta, 5007.)
LnHbeta_int = L_int(LnHbeta, LnHalpha, LnHbeta, 4860.)
LnHalpha_int = L_int(LnHalpha, LnHalpha, LnHbeta, 6563.)
LNII_int = L_int(LNII, LnHalpha, LnHbeta, 6584.)
LSII_int = L_int(LSII, LnHalpha, LnHbeta, 6731.)
 
LbHbeta_int = L_int(LbHbeta, LbHalpha, LbHbeta, 4860.)
LbHalpha_int = L_int(LbHalpha, LbHalpha, LbHbeta, 6563.)
LFeII_int = L_int(LFeII, LbHalpha, LbHbeta, 4570.)
 
bin_FeII_bHbeta_int = np.arange(10**min(LFeII_int-LbHbeta_int), 12., 1.)
bin_bHbeta_FeII_int = np.arange(min(LbHbeta_int-LFeII_int), max(LbHbeta_int-LFeII_int), 0.5/4.)
bin_O32_int = np.arange(10**min(LO3_int-LO2_int), 10**max(LO3_int-LO2_int), 1.5)
 
 

'''
binning(10**(LFeII_int-LbHbeta_int), fwhm_bHbeta, bin_FeII_bHbeta_int, bin_fwhm_bHbeta, 1217,\
'$L_{\\rm Fe \ II}^{\\rm cor}/L_{\\rm bH\\beta}^{\\rm cor}$', '$\\rm FWHM \ bH\\beta \ (km/s)$')
 
binning(10**(LO3_int-LO2_int), fwhm_bHbeta, bin_O32_int, bin_fwhm_bHbeta, 1218,\
'$\\log \ L_{\\rm [O \ III]}^{\\rm cor}/L_{\\rm [O \ II]}^{\\rm cor}$', '$\\rm FWHM \ bH\\beta \ (km/s)$')
 
binning(10**(LFeII_int-LbHbeta_int), 10**(LO3_int-LO2_int), bin_FeII_bHbeta_int, bin_O32_int, 1219,\
'$L_{\\rm Fe \ II}^{\\rm cor}/L_{\\rm bH\\beta}^{\\rm cor}$', '$L_{\\rm [O \ III]}^{\\rm cor}/L_{\\rm [O \ II]}^{\\rm cor}$')
 
binning(10**(LFeII_int-LbHbeta_int), fwhm_nHbeta, bin_FeII_bHbeta_int, bin_fwhm_nHbeta, 1220,\
'$L_{\\rm Fe \ II}^{\\rm cor}/L_{\\rm bH\\beta}^{\\rm cor}$', '$\\rm FWHM \ nH\\beta \ (km/s)$')
'''


###################

AB_1 = binning(10**(LFeII-LbHbeta), fwhm_bHbeta, bin_FeII_bHbeta, bin_fwhm_bHbeta, 1207,\
'$L_{Fe \ II}/L_{bH\\beta}$', '$FWHM \ bH\\beta \ (km/s)$')
 
plt.close('all') 
 
AB_1 = np.array(AB_1)
 
fwhm_bHbeta_A = []
fwhm_bHbeta_B = []
 
RFeII_A = []
RFeII_B = []
 
for i in range(len(AB_1[1, :])):
    if AB_1[1][i] >= 4000.:
        fwhm_bHbeta_B.append(AB_1[1][i])
        RFeII_B.append(AB_1[0][i])
    else:
        fwhm_bHbeta_A.append(AB_1[1][i])
        RFeII_A.append(AB_1[0][i])

plt.close('all')

sule = np.loadtxt('sulentic.tsv', delimiter=',', dtype=str)

fwhm_bHbeta_sule = np.array(sule[:, 2], dtype=float)
RFeII_sule = np.array(sule[:, 3], dtype=float)
gamma_sule = np.array(sule[:, 4], dtype=float)
c_12_sule = np.array(sule[:, 10], dtype=float)
fwhm_bCIV_sule = np.array(sule[:, 13], dtype=float)

sule2 = np.loadtxt('c_iv3.csv', delimiter=',', dtype=float, skiprows=1)

O32 = np.array((sule2[:, 0]), dtype=float)
CIV_shift = np.array((sule2[:, 1]), dtype=float) *c2/1549.
Gamma = np.array((sule2[:, 2]), dtype=float)
W_bHbeta = np.array((sule2[:, 3]), dtype=float)
Rfe = np.array((sule2[:, 4]), dtype=float)

u_gamma = []
v_shift = []
u_gamma_a = []
v_shift_a = []
u_gamma_b = []
v_shift_b = []


for i in range(len(Gamma)):
    if Gamma[i] > 1 and Gamma[i] < 5:
        u_gamma.append(Gamma[i])
        v_shift.append(CIV_shift[i])
        
    if i < 4 and Gamma[i] > 1 and Gamma[i] < 5:
        u_gamma_a.append(Gamma[i])
        v_shift_a.append(CIV_shift[i])

    if i > 4 and Gamma[i] > 1 and Gamma[i] < 5:
        u_gamma_b.append(Gamma[i])
        v_shift_b.append(CIV_shift[i])        

for i in range(len(gamma_sule)):
    if gamma_sule[i] > 1 and gamma_sule[i] < 5:
        u_gamma.append(gamma_sule[i])
        v_shift.append(c_12_sule[i])
        
    if i < 51 and gamma_sule[i] > 1 and gamma_sule[i] < 5:
        u_gamma_a.append(gamma_sule[i])
        v_shift_a.append(c_12_sule[i])

    if i > 51 and gamma_sule[i] > 1 and gamma_sule[i] < 5:
        u_gamma_b.append(gamma_sule[i])
        v_shift_b.append(c_12_sule[i])        



plt.figure(1231)
plt.plot(W_bHbeta[: 5], CIV_shift[: 5], 'bs')
plt.plot(W_bHbeta[5 :], CIV_shift[5 :], 'rs')
plt.plot(fwhm_bHbeta_sule[: 51], c_12_sule[: 51], 'bo', label='Population A')
plt.plot(fwhm_bHbeta_sule[51 :], c_12_sule[51 :], 'ro', label='Population B')
plt.plot( [4000]*30, np.linspace(-5000, 2000, 30),'k-.')
plt.plot( np.linspace(0, 20000, 30), [0]*30, 'k-.')
plt.xlabel('$\\rm FWHM \ bH\\beta \ (km/s)$')
plt.ylabel('$\\rm c(1/2) \ CIV \ \\lambda1549 \ (km/s)$')
plt.legend(loc='best')
plt.savefig('figures/fig_%i' %1231)




plt.figure(1232)
plt.plot(Rfe[: 5], CIV_shift[: 5], 'bs')
plt.plot(Rfe[5 :], CIV_shift[5 :], 'rs')
plt.plot(RFeII_sule[: 51], c_12_sule[: 51], 'bo', label='Population A')
plt.plot(RFeII_sule[51 :], c_12_sule[51 :], 'ro', label='Population B')
#plt.plot( [4000]*30, np.linspace(-5000, 2000, 30),'k-.')
plt.plot( np.linspace(-0.5, 2.5, 30), [0]*30, 'k-.')
plt.xlim(-0.1, 2.5)
plt.xlabel('$R_{\\rm Fe \ II}$')
plt.ylabel('$\\rm c(1/2) \ CIV \ \\lambda1549 \ (km/s)$')
plt.legend(loc='best')
plt.savefig('figures/fig_%i' %1232)

plt.figure(1233)

plt.plot(Gamma[: 5], CIV_shift[: 5], 'bs')
plt.plot(Gamma[5 :], CIV_shift[5 :], 'rs')
plt.plot(gamma_sule[: 51], c_12_sule[: 51], 'bo', label='Population A')
plt.plot(gamma_sule[51 :], c_12_sule[51 :], 'ro', label='Population B')

m_sule, c_sule, r_sule, p_sule, std_sule = linregress(u_gamma, v_shift)
rho_sule, P_sule = spearmanr(u_gamma, v_shift)
#plt.plot(np.linspace(0, 6), linear(np.linspace(0, 6), m_sule, c_sule), 'k--', label = ('$\\rho_s = %f$\n$ P_s = %f$' %(rho_sule, P_sule)))

m_sule_a, c_sule_a, r_sule_a, p_sule_a, std_sule_a = linregress(u_gamma_a, v_shift_a)
rho_sule_a, P_sule_a = spearmanr(u_gamma_a, v_shift_a)
#plt.plot(np.linspace(0, 6), linear(np.linspace(0, 6), m_sule_a, c_sule_a), 'b--', label = ('$\\rho_s = %f$\n$ P_s = %f$' %(rho_sule_a, P_sule_a)))

m_sule_b, c_sule_b, r_sule_b, p_sule_b, std_sule_b = linregress(u_gamma_b, v_shift_b)
rho_sule_b, P_sule_b = spearmanr(u_gamma_b, v_shift_b)
#plt.plot(np.linspace(0, 6), linear(np.linspace(0, 6), m_sule_b, c_sule_b), 'r--', label = ('$\\rho_s = %f$\n$ P_s = %f$' %(rho_sule_b, P_sule_b)))

#plt.plot( [4000]*30, np.linspace(-5000, 2000, 30),'k-.')
plt.plot( np.linspace(1, 5, 30), [0]*30, 'k-.')
plt.xlim(1, 5)
plt.xlabel('$\\Gamma_s$')
plt.ylabel('$\\rm c(1/2) \ CIV \ \\lambda1549 \ (km/s)$')
plt.legend(loc='best', fontsize='x-small')
plt.savefig('figures/fig_%i' %1233)
'''
plt.figure(1234)
plt.plot(c_12_sule[: 51], fwhm_bCIV_sule[: 51], 'bo', label='Population A')
plt.plot(c_12_sule[51 :], fwhm_bCIV_sule[51 :], 'ro', label='Population B')
plt.plot([0]*30, np.linspace(0, 12000, 30), 'k-.')
plt.ylabel('$\\rm FWHM \ bC \ IV \ (km/s)$')
plt.xlabel('$\\rm c(1/2) \ CIV \ \\lambda1549 \ (km/s)$')
plt.legend(loc='best')
plt.savefig('figures/fig_%i' %1234)


plt.figure(1235)
plt.plot(O32[: 4], CIV_shift[: 4], 'bo')
plt.plot(O32[4 :], CIV_shift[4 :], 'bo')
#plt.plot(gamma_sule[: 51], c_12_sule[: 51], 'bo', label='Population A')
#plt.plot(gamma_sule[51 :], c_12_sule[51 :], 'ro', label='Population B')
#plt.plot( [4000]*30, np.linspace(-5000, 2000, 30),'k-.')
plt.plot( np.linspace(0, 21, 50), [0]*50, 'k-.')
plt.xlim(1, 20)
plt.xlabel('$L_{\\rm [O III]}/L_{\\rm [O II]}$')
plt.ylabel('$\\rm c(1/2) \ CIV \ \\lambda1549 \ (km/s)$')
plt.legend(loc='best')
plt.savefig('figures/fig_%i' %1235)

plt.figure(1236)
plt.plot(O32[: 4], Gamma[: 4], 'bo')
plt.plot(O32[4 :], Gamma[4 :], 'bo')
#plt.plot(gamma_sule[: 51], c_12_sule[: 51], 'bo', label='Population A')
#plt.plot(gamma_sule[51 :], c_12_sule[51 :], 'ro', label='Population B')
#plt.plot( [4000]*30, np.linspace(-5000, 2000, 30),'k-.')
#plt.plot( np.linspace(0, 21, 50), [0]*50, 'k-.')
plt.ylim(1, 5)
plt.xlabel('$L_{\\rm [O III]}/L_{\\rm [O II]}$')
plt.ylabel('$\\Gamma_s$')
plt.legend(loc='best')
plt.savefig('figures/fig_%i' %1236)
plt.close('all')
'''


'''
################## Part #########################

bin_z = np.arange(min(z), max(z), 0.025)
bin_log_z_1 = np.arange(min(log_z_1), max(log_z_1), 0.01)

#plot_average(1301, z_O32, (LO2), bin_z, 'Redshift $(z)$', '$\\log \ L_{[O \ II]}$')
#plot_average(1302, z_O32, (LO3), bin_z, 'Redshift $(z)$', '$\\log \ L_{[O \ III]}$')
#plot_average(1303, z_O32, (LNII), bin_z, 'Redshift $(z)$', '$\\log \ L_{[N \ II]}$')
#plot_average(1304, z_O32, (LSII), bin_z, 'Redshift $(z)$', '$\\log \ L_{[S \ II]}$')
#plot_average(1305, log_z_1, (SFR), bin_log_z_1, '$\\log \ (z+1)$', '$\\log \ SFR \ M_{\\odot} \ yr^{-1}$')
#plt.close('all')

#plot(1306, LFeII, fwhm_bHbeta, '$L_{[Fe \ II]}$', '$FWHM \ bH\\beta \ (km/s)$') 
'''

bin_O32 = np.arange(min(LO3-LO2), max(LO3-LO2), 0.25/2.)
bin_O3 = np.arange(min(LO3), max(LO3), 0.5/2.)
bin_O2 = np.arange(min(LO2), max(LO2), 0.5/2.)

'''
plot_average(2101, LO3_x, Lx, bin_O3, '$\\log \ L_{\\rm [O \ III]} \ \\rm (erg \ s^{-1})$', '$\\log \ L_{\\rm X} \ \\rm (erg \ s^{-1})$') 
plot_average(2102, LO3_nuv, Lnuv, bin_O3, '$\\log \ L_{\\rm [O \ III]} \ \\rm (erg \ s^{-1})$', '$\\log \ L_{\\rm NUV} \ \\rm (erg \ s^{-1})$')
plot_average(2103, LO3_fuv, Lfuv, bin_O3, '$\\log \ L_{\\rm [O \ III]} \ \\rm (erg \ s^{-1})$', '$\\log \ L_{\\rm FUV} \ \\rm (erg \ s^{-1})$')
plot_average(2104, LO3_u, Lu, bin_O3, '$\\log \ L_{\\rm [O \ III]} \ \\rm (erg \ s^{-1})$', '$\\log \ L_u \ \\rm (erg \ s^{-1})$') 
plot_average(2105, LO3_g, Lg, bin_O3, '$\\log \ L_{\\rm [O \ III]} \ \\rm (erg \ s^{-1})$', '$\\log \ L_g \ \\rm (erg \ s^{-1})$') 
plot_average(2106, LO3_r, Lr, bin_O3, '$\\log \ L_{\\rm [O \ III]} \ \\rm (erg \ s^{-1})$', '$\\log \ L_r \ \\rm (erg \ s^{-1})$') 
plot_average(2107, LO3_i, Li, bin_O3, '$\\log \ L_{\\rm [O \ III]} \ \\rm (erg \ s^{-1})$', '$\\log \ L_i \ \\rm (erg \ s^{-1})$') 
plot_average(2108, LO3_z, Lz, bin_O3, '$\\log \ L_{\\rm [O \ III]} \ \\rm (erg \ s^{-1})$', '$\\log \ L_z \ \\rm (erg \ s^{-1})$')
plot_average(2109, LO3_j, Lj, bin_O3, '$\\log \ L_{\\rm [O \ III]} \ \\rm (erg \ s^{-1})$', '$\\log \ L_J \ \\rm (erg \ s^{-1})$') 
plot_average(2110, LO3_h, Lh, bin_O3, '$\\log \ L_{\\rm [O \ III]} \ \\rm (erg \ s^{-1})$', '$\\log \ L_H \ \\rm (erg \ s^{-1})$') 
plot_average(2111, LO3_k, Lk, bin_O3, '$\\log \ L_{\\rm [O \ III]} \ \\rm (erg \ s^{-1})$', '$\\log \ L_K \ \\rm (erg \ s^{-1})$')
plot_average(2112, LO3_20cm, L20cm, bin_O3, '$\\log \ L_{\\rm [O \ III]} \ \\rm (erg \ s^{-1})$', '$\\log \ L_{\\rm Radio} \ \\rm (erg \ s^{-1})$')
plot_average(2113, LO3_cont, Lcont, bin_O3, '$\\log \ L_{\\rm [O \ III]} \ \\rm (erg \ s^{-1})$', '$\\log \ L_{5100} \ \\rm (erg \ s^{-1})$')
plt.close('all')



plot_average(2201, LO2_x, Lx, bin_O2, '$\\log \ L_{\\rm [O \ II]} \ \\rm (erg \ s^{-1})$', '$\\log \ L_{\\rm X} \ \\rm (erg \ s^{-1})$') 
plot_average(2202, LO2_nuv, Lnuv, bin_O2, '$\\log \ L_{\\rm [O \ II]} \ \\rm (erg \ s^{-1})$', '$\\log \ L_{\\rm NUV} \ \\rm (erg \ s^{-1})$')
plot_average(2203, LO2_fuv, Lfuv, bin_O2, '$\\log \ L_{\\rm [O \ II]} \ \\rm (erg \ s^{-1})$', '$\\log \ L_{\\rm FUV} \ \\rm (erg \ s^{-1})$')
plot_average(2204, LO2_u, Lu, bin_O2, '$\\log \ L_{\\rm [O \ II]} \ \\rm (erg \ s^{-1})$', '$\\log \ L_u \ \\rm (erg \ s^{-1})$') 
plot_average(2205, LO2_g, Lg, bin_O2, '$\\log \ L_{\\rm [O \ II]} \ \\rm (erg \ s^{-1})$', '$\\log \ L_g \ \\rm (erg \ s^{-1})$') 
plot_average(2206, LO2_r, Lr, bin_O2, '$\\log \ L_{\\rm [O \ II]} \ \\rm (erg \ s^{-1})$', '$\\log \ L_r \ \\rm (erg \ s^{-1})$') 
plot_average(2207, LO2_i, Li, bin_O2, '$\\log \ L_{\\rm [O \ II]} \ \\rm (erg \ s^{-1})$', '$\\log \ L_i \ \\rm (erg \ s^{-1})$') 
plot_average(2208, LO2_z, Lz, bin_O2, '$\\log \ L_{\\rm [O \ II]} \ \\rm (erg \ s^{-1})$', '$\\log \ L_z \ \\rm (erg \ s^{-1})$')
plot_average(2209, LO2_j, Lj, bin_O2, '$\\log \ L_{\\rm [O \ II]} \ \\rm (erg \ s^{-1})$', '$\\log \ L_J \ \\rm (erg \ s^{-1})$') 
plot_average(2210, LO2_h, Lh, bin_O2, '$\\log \ L_{\\rm [O \ II]} \ \\rm (erg \ s^{-1})$', '$\\log \ L_H \ \\rm (erg \ s^{-1})$')
plot_average(2211, LO2_k, Lk, bin_O2, '$\\log \ L_{\\rm [O \ II]} \ \\rm (erg \ s^{-1})$', '$\\log \ L_K \ \\rm (erg \ s^{-1})$')
plot_average(2212, LO2_20cm, L20cm, bin_O2, '$\\log \ L_{\\rm [O \ II]} \ \\rm (erg \ s^{-1})$', '$\\log \ L_{\\rm Radio} \ \\rm (erg \ s^{-1})$')
plot_average(2213, LO2_cont, Lcont, bin_O2, '$\\log \ L_{\\rm [O \ II]} \ \\rm (erg \ s^{-1})$', '$\\log \ L_{5100} \ \\rm (erg \ s^{-1})$')
plt.close('all')

plot_average(2301, LO32_x, Lx, bin_O32, '$\\log \ L_{\\rm [O \ III]}/L_{[O \ II]}$', '$\\log \ L_{\\rm X} \ \\rm (erg \ s^{-1})$') 
plot_average(2302, LO32_nuv, Lnuv, bin_O32, '$\\log \ L_{\\rm [O \ III]}/L_{[O \ II]}$', '$\\log \ L_{\\rm NUV} \ \\rm (erg \ s^{-1})$')
plot_average(2303, LO32_fuv, Lfuv, bin_O32, '$\\log \ L_{\\rm [O \ III]}/L_{[O \ II]}$', '$\\log \ L_{\\rm FUV} \ \\rm (erg \ s^{-1})$')
plot_average(2304, LO32_u, Lu, bin_O32, '$\\log \ L_{\\rm [O \ III]}/L_{\\rm [O \ II]}$', '$\\log \ L_u \ \\rm (erg \ s^{-1})$') 
plot_average(2305, LO32_g, Lg, bin_O32, '$\\log \ L_{\\rm [O \ III]}/L_{\\rm [O \ II]}$', '$\\log \ L_g \ \\rm (erg \ s^{-1})$') 
plot_average(2306, LO32_r, Lr, bin_O32, '$\\log \ L_{\\rm [O \ III]}/L_{\\rm [O \ II]}$', '$\\log \ L_r \ \\rm (erg \ s^{-1})$') 
plot_average(2307, LO32_i, Li, bin_O32, '$\\log \ L_{\\rm [O \ III]}/L_{\\rm [O \ II]}$', '$\\log \ L_i \ \\rm (erg \ s^{-1})$') 
plot_average(2308, LO32_z, Lz, bin_O32, '$\\log \ L_{\\rm [O \ III]}/L_{\\rm [O \ II]}$', '$\\log \ L_z \ \\rm (erg \ s^{-1})$')
plot_average(2309, LO32_j, Lj, bin_O32, '$\\log \ L_{\\rm [O \ III]}/L_{\\rm [O \ II]}$', '$\\log \ L_J \ \\rm (erg \ s^{-1})$') 
plot_average(2310, LO32_h, Lh, bin_O32, '$\\log \ L_{\\rm [O \ III]}/L_{\\rm [O \ II]}$', '$\\log \ L_H \ \\rm (erg \ s^{-1})$') 
plot_average(2311, LO32_k, Lk, bin_O32, '$\\log \ L_{\\rm [O \ III]}/L_{\\rm [O \ II]}$', '$\\log \ L_K \ \\rm (erg \ s^{-1})$')
plot_average(2312, LO32_20cm, L20cm, bin_O32, '$\\log \ L_{\\rm [O \ III]}/L_{\\rm [O \ II]}$', '$\\log \ L_{\\rm Radio} \ \\rm (erg \ s^{-1})$')
plot_average(2313, LO32_cont, Lcont, bin_O32, '$\\log \ L_{\\rm [O \ III]}/L_{\\rm [O \ II]}$', '$\\log \ L_{5100} \ \\rm (erg \ s^{-1})$')



plot_average(2502, LO3, LO2-LO3, bin_O3, '$\\log \ L_{\\rm [O \ III]} \ \\rm (erg \ s^{-1})$', '$\\log \ L_{\\rm [O \ II]}/L_{\\rm [O \ III]}$')
plt.close('all')

plot_average(2503, LO3, LbHalpha, bin_O3, '$\\log \ L_{\\rm [O \ III]} \ \\rm (erg \ s^{-1})$', '$\\log \ L_{\\rm bH\\alpha} \ \\rm (erg \ s^{-1})$')

'''

bin_O32_int = np.arange(min(LO3_int-LO2_int), max(LO3_int-LO2_int), 0.25/2.)

kappa_1 = []
kappa_2 = []
for i in range(len(LO3)):
    if LO3[i]-LnHbeta[i] < 1.5:
        kappa_1.append(i)
    if LO3[i]-LnHalpha[i] > -0.25:
        kappa_2.append(i)
plot_average(2504, LO3-LO2, LO3-LnHbeta, bin_O32,'$\\log \ L_{\\rm [O \ III]}/L_{\\rm [O \ II]}$', '$\\log \ L_{\\rm [O \ III]}/L_{\\rm nH\\beta}$')
plot_average(2505, LO3_int-LO2_int, LO3_int-LnHbeta_int, bin_O32_int,'$\\log \ L_{\\rm [O \ III]}^{\\rm cor}/L_{\\rm [O \ II]}^{\\rm cor}$', '$\\log \ L_{\\rm [O \ III]}^{\\rm cor}/L_{\\rm nH\\beta}^{\\rm cor}$')
plot_average(2506, LO3[kappa_1]-LO2[kappa_1], LO3[kappa_1]-LnHbeta[kappa_1], bin_O32,'$\\log \ L_{\\rm [O \ III]}/L_{\\rm [O \ II]}$', '$\\log \ L_{\\rm [O \ III]}/L_{\\rm nH\\beta}$')

plot_average(2507, LO3-LO2, LO3-LnHalpha, bin_O32,'$\\log \ L_{\\rm [O \ III]}/L_{\\rm [O \ II]}$', '$\\log \ L_{\\rm [O \ III]}/L_{\\rm nH\\alpha}$')
plot_average(2508, LO3_int-LO2_int, LO3_int-LnHalpha_int, bin_O32_int,'$\\log \ L_{\\rm [O \ III]}^{\\rm cor}/L_{\\rm [O \ II]}^{\\rm cor}$', '$\\log \ L_{\\rm [O \ III]}^{\\rm cor}/L_{\\rm nH\\alpha}^{\\rm cor}$')
plot_average(2509, LO3[kappa_2]-LO2[kappa_2], LO3[kappa_2]-LnHalpha[kappa_2], bin_O32,'$\\log \ L_{\\rm [O \ III]}/L_{\\rm [O \ II]}$', '$\\log \ L_{\\rm [O \ III]}/L_{\\rm nH\\alpha}$')

plt.close('all')


bin_O32_2 = np.arange(-0.5, 1.5, 0.05)
bin_O3nHbeta = np.arange(0, 4.5, 0.05)
#E1 continued
fig_2514 = binning2(LO3, LO3-LO2, LO3-LnHbeta, bin_O32_2, bin_O3nHbeta, 2514,\
'$\\log \ L_{\\rm [O \ III]}/L_{\\rm [O \ II]}$', '$\\log \ L_{\\rm [O \ III]}/L_{\\rm nH\\beta}$', '$\\log \ L_{\\rm [O III]} \ \\rm (erg \ s^{-1})$')


normal_ion = []
low_ion2 = []
liner = []
composite_sii = []

for i in range(len(LO3)):
    if LO3[i]-LO2[i] >= 0.:
        normal_ion.append(i)
        
    if LO3[i]-LO2[i] < 0.:
        low_ion2.append(i)

    if (LO3[i]-LnHbeta[i]) < Ke01_b(LSII[i]-LnHalpha[i]):
        composite_sii.append(i)        

    if (LO3[i]-LnHbeta[i]) < Ke06(LSII[i]-LnHalpha[i]):
        liner.append(i)

fig = plt.figure(2525)
ax = fig.add_subplot(111)
X = LSII[low_ion2]-LnHalpha[low_ion2]
Y = LO3[low_ion2]-LnHbeta[low_ion2]
Z = LO3[low_ion2]-LO2[low_ion2]
sc = ax.scatter(X, Y, s=20, c=Z, marker='o', cmap=cm.jet, vmin=-0.56)
col = plt.colorbar(sc )
col.set_label('$\\log \ L_{\\rm [O \ III]}/L_{\\rm [O \ II]}$', fontsize='x-large')
plt.xlabel('$\\log \ L_{\\rm [S \ II]}/L_{\\rm nH\\alpha}$', fontsize='x-large')
plt.ylabel('$\\log \ L_{\\rm [O \ III]}/L_{\\rm nH\\beta}$', fontsize='x-large')
plt.plot(np.linspace(-4, 0.2, 100), Ke01_b(np.linspace(-4, 0.2, 100)), 'r-', label='Kewley et al. 2001')
plt.plot(np.linspace(-0.315, 1., 100), Ke06(np.linspace(-0.315, 1., 100)), 'b-', label='Kewley et al. 2006')
plt.ylim(-1.0, max(LO3-LnHbeta))
plt.xlim(-2, 0.5)
plt.legend(loc='best', fontsize='small')
plt.savefig('figures/fig_2525')



plt.close('all')


fig = plt.figure(2524)
ax = fig.add_subplot(111)
X = LO3-LO2
Y = LO3-LnHbeta
Z = LO3
sc = ax.scatter(X, Y, s=10, c=Z, marker='o', cmap=cm.jet)
col = plt.colorbar(sc )
col.set_label('$\\log \ L_{\\rm [O \ III]} \ \\rm  (erg \ s^{-1})$', fontsize='x-large')
#plt.xlim(-1, 12)
#plt.ylim(0, 20000)
plt.plot([0]*50, np.linspace(0, 3.5, 50), 'k--')
ax.scatter(X[liner], Y[liner], marker='s',s=50, facecolors='none', edgecolors='k', label ='LINER')           
ax.scatter(X[composite_sii], Y[composite_sii], marker='^',s=50, facecolors='none', edgecolors='k', label ='SF?')           
plt.xlabel('$\\log \ L_{\\rm [O \ III]}/L_{\\rm [O \ II]}$', fontsize='x-large')
plt.ylabel('$\\log \ L_{\\rm [O \ III]}/L_{\\rm nH\\beta}$', fontsize='x-large')
plt.ylim(0, 1.6)
plt.legend(loc='best', fontsize='small')
m, c, r, p, std = linregress(X, Y)
rho, P = spearmanr(X, Y)
#plt.plot(X, linear(X, m, c), 'k--')#, label = ('$\\rho_s = %f$\n$ P_s = %f$' %(rho, P)))
plt.savefig('figures/fig_2524')


fig = plt.figure(2527)
ax = fig.add_subplot(111)
X = LO3-LO2
Y = LO3-LnHbeta
Z = 10**(LFeII-LbHbeta)
sc = ax.scatter(X, Y, s=20, c=Z, marker='o', cmap=cm.jet)
col = plt.colorbar(sc )
col.set_label('$L_{\\rm Fe \ II}/L_{\\rm bH\\beta}$', fontsize='x-large')
#plt.xlim(-1, 12)
#plt.ylim(0, 20000)
plt.plot([0]*50, np.linspace(0, 3.5, 50), 'k--')
ax.scatter(X[liner], Y[liner], marker='s',s=50, facecolors='none', edgecolors='k', label ='LINER')           
ax.scatter(X[composite_sii], Y[composite_sii], marker='^',s=50, facecolors='none', edgecolors='k', label ='SF?')           
plt.xlabel('$\\log \ L_{\\rm [O \ III]}/L_{\\rm [O \ II]}$', fontsize='x-large')
plt.ylabel('$\\log \ L_{\\rm [O \ III]}/L_{\\rm nH\\beta}$', fontsize='x-large')
plt.ylim(0, 1.6)
plt.legend(loc='best', fontsize='small')
m, c, r, p, std = linregress(X, Y)
rho, P = spearmanr(X, Y)
#plt.plot(X, linear(X, m, c), 'k--')#, label = ('$\\rho_s = %f$\n$ P_s = %f$' %(rho, P)))
plt.savefig('figures/fig_2527')

plt.close('all')








outlier = []

fig_2526 = binning2(LO3, LO3-LO2, LO3-LnHbeta, bin_O32_2, bin_O3nHbeta, 2526,\
'$\\log \ L_{\\rm [O \ III]}/L_{\\rm [O \ II]}$', '$\\log \ L_{\\rm [O \ III]}/L_{\\rm nH\\beta}$', '$\\log \ L_{\\rm [O III]} \ \\rm (erg \ s^{-1})$')

fig_2528 = binning2(10**(LFeII-LbHbeta), LO3-LO2, LO3-LnHbeta, bin_O32_2, bin_O3nHbeta, 2528,\
'$\\log \ L_{\\rm [O \ III]}/L_{\\rm [O \ II]}$', '$\\log \ L_{\\rm [O \ III]}/L_{\\rm nH\\beta}$', '$\\log \ L_{\\rm Fe II}/L_{\\rm bH\\beta}$')


fig_2529 = bin_test(10**(LFeII-LbHbeta), LO3-LO2, LO3-LnHbeta, bin_O32_2, bin_O3nHbeta, 2529,\
'$\\log \ L_{\\rm [O \ III]}/L_{\\rm [O \ II]}$', '$\\log \ L_{\\rm [O \ III]}/L_{\\rm nH\\beta}$', '$L_{\\rm Fe \ II}/L_{\\rm bH\\beta}$')


fig_2530 = bin_test(10**(LFeII_int-LbHbeta_int), LO3_int-LO2_int, LO3_int-LnHbeta_int, bin_O32_2, bin_O3nHbeta, 2530,\
'$\\log \ L_{\\rm [O \ III]}^{cor}/L_{\\rm [O \ II]}^{cor}$', '$\\log \ L_{\\rm [O \ III]}^{cor}/L_{\\rm nH\\beta}^{cor}$', '$L_{\\rm Fe \ II}^{cor}/L_{\\rm bH\\beta}^{cor}$')

fig_2531 = bin_test(10**(LFeII-LbHbeta), LO3-LO2, LO3-LnHbeta, bin_O32_2, bin_O3nHbeta, 2531,\
'$\\log \ L_{\\rm [O \ III]}/L_{\\rm [O \ II]}$', '$\\log \ L_{\\rm [O \ III]}/L_{\\rm nH\\beta}$', '$\\Gamma_s$')

fig_2532 = bin_test(np.array(L_x)[R], (np.array(L_OIII_5007)-np.array(L_OII_3727))[R], (np.array(L_OIII_5007)-np.array(L_nHbeta))[R], bin_O32_2, bin_O3nHbeta, 2532,\
'$\\log \ L_{\\rm [O \ III]}/L_{\\rm [O \ II]}$', '$\\log \ L_{\\rm [O \ III]}/L_{\\rm nH\\beta}$', '$\\log \ L_{\\rm X} \ (erg \ s^{-1})$')
########## SFR Rework ##########


fig_2533 = bin_test((mag_SDSS[2]-mag_WISE[0]), LO3-LO2, LO3-LnHbeta, bin_O32_2, bin_O3nHbeta, 2533,\
'$\\log \ L_{\\rm [O \ III]}/L_{\\rm [O \ II]}$', '$\\log \ L_{\\rm [O \ III]}/L_{\\rm nH\\beta}$', '$r-W1$')

fig_2534 = bin_test((mag_SDSS[2]-mag_2MASS[0]), LO3-LO2, LO3-LnHbeta, bin_O32_2, bin_O3nHbeta, 2534,\
'$\\log \ L_{\\rm [O \ III]}/L_{\\rm [O \ II]}$', '$\\log \ L_{\\rm [O \ III]}/L_{\\rm nH\\beta}$', '$r-J$')


plt.close('all')










Lcont_bin2 = [[], [], [], [], [], [], [], [], [], []]
LO3_bin = [[], [], [], [], [], [], [], [], [], []]
LO2_bin = [[], [], [], [], [], [], [], [], [], []]
LNII_bin = [[], [], [], [], [], [], [], [], [], []]
LSII_bin = [[], [], [], [], [], [], [], [], [], []]
zO3_bin = [[], [], [], [], [], [], [], [], [], []]
log_z_1 = [[], [], [], [], [], [], [], [], [], []]

Lcont_bin = np.histogram(Lcont)    
print Lcont_bin

SFR = np.array(np.log10(8.4*10**-41*(10**LO2 - 10**(LO3*np.log10(0.1)))))

SFR_bin = [[], [], [], [], [], [], [], [], [], []]


k = 0

f = open('note/spectral_combine.txt', 'w')
f.close()

for i in range(len(Lcont_bin[1]) - 1):
    
    for j in range(len(Lcont)):
        if Lcont[j] > Lcont_bin[1][i] and Lcont[j] <= Lcont_bin[1][i+1]:
            LO3_bin[i].append(LO3[j])
            LO2_bin[i].append(LO2[j])
            LNII_bin[i].append(LNII[j])
            LSII_bin[i].append(LSII[j])
            
            zO3_bin[i].append(z_O32[j])            
            log_z_1[i].append(z_O32[j])
            Lcont_bin2[i].append(Lcont[j])
            
            SFR_bin[i].append(np.log10(8.4*10**-41*(10**LO2[j] - 10**(LO3[j]*np.log10(0.1)))))
            f = open('note/spectral_combine.txt', 'a')
            f.write(str(J[j]) + ', ')            
            f.close()
            
            
        else:
            continue
    
    f = open('note/spectral_combine.txt', 'a')
    f.write('\n##########\n')            
    f.close()



LO3_bin = np.array(LO3_bin)
LO2_bin = np.array(LO2_bin)
LNII_bin = np.array(LNII_bin)
LSII_bin = np.array(LSII_bin)

zO3_bin = np.array(zO3_bin)
Lcont_bin2 = np.array(Lcont_bin2)

SFR_bin = np.array(SFR_bin)
log_z_1 = np.array(log_z_1)

for i in range(len(zO3_bin)):
    for j in range(len(zO3_bin[i])):
        log_z_1[i][j] = np.log10(log_z_1[i][j]+1)






bin_z = np.arange(0., 0.35 + 0.05/1.5, 0.05/1.5)#min(z), max(z), 0.07)#0.025*1.5)
#bin_z = np.array([0., 0.1, 0.2, 0.25, 0.3, 0.35])
bin_log_z_1 = np.arange(np.min(log_z_1[0]), np.max(log_z_1[-1]), 0.01)

def plot_average2(lab, col, number, x, y, group, labelx, labely):
    
    binned = []    
    mean = []
    stdx = []
    stdy = []
    count = []
    
    for i in range(len(group)):
        point = []
        axis = []
        for j in range(len(x)):
            #if x[j] >= group[i] and x[j] < group[i+1]:
            if abs(x[j]-group[i]) <= abs((group[0]-group[1])/2.): 
                axis.append(x[j])
                point.append(y[j])
        
        if len(point) >= 5:
            if number >= 3000 and number <= 3005:
                mean.append(np.mean(point))
            else:
                mean.append(np.median(point))
                
            if number == 3000:
                stdy.append(np.std(point)*1.)
            else:
                stdy.append(np.std(point)*0.25)
                
            stdx.append(np.std(axis)*1.)            
            count.append(len(point))
            binned.append(group[i])
    
    m, c, r, p, std = linregress(binned, mean)
    rho, P = spearmanr(binned, mean)
    o = open('output_parameters.csv', 'a')    
    o.write(str(number) + '\t' + str(m) + '\t' + str(c) + '\t' + str(r) + '\t' + str(p) + '\t' + str(std) + '\t' + str(rho) + '\t' + str(P) + '\n')
    o.close()
    plt.figure(number)
    #plt.plot(binned, linear(binned, m, c), 'r--', label = ('$\\rho_s = %f$\n$ P_s = %f$' %(rho, P)))
    #plt.legend(loc='best')
    #plt.plot(x, y, 'k.', alpha=0.15)
    plt.errorbar(binned, mean, yerr=stdy, xerr=0, fmt=str(col), label=str(lab))
    if number <= 3000:
        plt.plot(binned, mean, str(col)[0]+'-')
        
    #plt.plot(binned, mean, 'b-')
    plt.xlabel(labelx, fontsize='x-large')
    plt.ylabel(labely, fontsize='x-large')
    if number <= 3005:
        plt.xlim(min(binned)-0.15, max(binned)+0.05)
    if number == 3006:
        plt.xlim(-0.5, 10.)
    if number == 3007:
        plt.xlim(0, 20000)
        
    plt.legend(loc='lower right', fontsize='x-small')
    plt.savefig('figures/fig_%i' %number)



#plot_average2('L3', 'go', 3000, zO3_bin[2], (Lcont_bin2[2]), bin_z, 'Redshift $(z)$', '$\\log \ L_{5100}$')
plot_average2('L4', 'bo', 3000, zO3_bin[3], (Lcont_bin2[3]), bin_z, 'Redshift $(z)$', '$\\log \ L_{5100}$')
plot_average2('L5', 'ko', 3000, zO3_bin[4], (Lcont_bin2[4]), bin_z, 'Redshift $(z)$', '$\\log \ L_{5100}$')
plot_average2('L6', 'co', 3000, zO3_bin[5], (Lcont_bin2[5]), bin_z, 'Redshift $(z)$', '$\\log \ L_{5100}$')
plot_average2('L7', 'yo', 3000, zO3_bin[6], (Lcont_bin2[6]), bin_z, 'Redshift $(z)$', '$\\log \ L_{5100}$')
#plot_average2('L8', 'mo', 3000, zO3_bin[7], (Lcont_bin2[7]), bin_z, 'Redshift $(z)$', '$\\log \ L_{5100}$')
#plot_average2('L9', 'ro', 3000, zO3_bin[8], (Lcont_bin2[8]), bin_z, 'Redshift $(z)$', '$\\log \ L_{5100}$')


#plot_average2('L2', 'ro', 3001, zO3_bin[0], (LO3_bin[0]), bin_z, 'Redshift $(z)$', '$\\log \ L_{[O \ III]}$')
#plot_average2('L2', 'ro', 3001, zO3_bin[1], (LO3_bin[1]), bin_z, 'Redshift $(z)$', '$\\log \ L_{[O \ III]}$')
#plot_average2('L3', 'go', 3001, zO3_bin[2], (LO3_bin[2]), bin_z, 'Redshift $(z)$', '$\\log \ L_{\\rm [O \ III]}$')
plot_average2('L4', 'bo', 3001, zO3_bin[3], (LO3_bin[3]), bin_z, 'Redshift $(z)$', '$\\log \ L_{\\rm [O \ III]}$')
plot_average2('L5', 'ko', 3001, zO3_bin[4], (LO3_bin[4]), bin_z, 'Redshift $(z)$', '$\\log \ L_{\\rm [O \ III]}$')
plot_average2('L6', 'co', 3001, zO3_bin[5], (LO3_bin[5]), bin_z, 'Redshift $(z)$', '$\\log \ L_{\\rm [O \ III]}$')
plot_average2('L7', 'yo', 3001, zO3_bin[6], (LO3_bin[6]), bin_z, 'Redshift $(z)$', '$\\log \ L_{\\rm [O \ III]}$')
#plot_average2('L8', 'mo', 3001, zO3_bin[7], (LO3_bin[7]), bin_z, 'Redshift $(z)$', '$\\log \ L_{\\rm [O \ III]}$')
#plot_average2('L9', 'ro', 3001, zO3_bin[8], (LO3_bin[8]), bin_z, 'Redshift $(z)$', '$\\log \ L_{[O \ III]}$')
#plot_average2('L10', 'ro', 3001, zO3_bin[9], (LO3_bin[9]), bin_z, 'Redshift $(z)$', '$\\log \ L_{[O \ III]}$')


#plot_average2('L3', 'go', 3002, zO3_bin[2], (LO2_bin[2]), bin_z, 'Redshift $(z)$', '$\\log \ L_{\\rm [O \ II]}$')
plot_average2('L4', 'bo', 3002, zO3_bin[3], (LO2_bin[3]), bin_z, 'Redshift $(z)$', '$\\log \ L_{\\rm [O \ II]}$')
plot_average2('L5', 'ko', 3002, zO3_bin[4], (LO2_bin[4]), bin_z, 'Redshift $(z)$', '$\\log \ L_{\\rm [O \ II]}$')
plot_average2('L6', 'co', 3002, zO3_bin[5], (LO2_bin[5]), bin_z, 'Redshift $(z)$', '$\\log \ L_{\\rm [O \ II]}$')
plot_average2('L7', 'yo', 3002, zO3_bin[6], (LO2_bin[6]), bin_z, 'Redshift $(z)$', '$\\log \ L_{\\rm [O \ II]}$')
#plot_average2('L8', 'mo', 3002, zO3_bin[7], (LO2_bin[7]), bin_z, 'Redshift $(z)$', '$\\log \ L_{\\rm [O \ II]}$')

#plot_average2('L3', 'go', 3003, zO3_bin[2], (LNII_bin[2]), bin_z, 'Redshift $(z)$', '$\\log \ L_{\\rm [N \ II]}$')
plot_average2('L4', 'bo', 3003, zO3_bin[3], (LNII_bin[3]), bin_z, 'Redshift $(z)$', '$\\log \ L_{\\rm [N \ II]}$')
plot_average2('L5', 'ko', 3003, zO3_bin[4], (LNII_bin[4]), bin_z, 'Redshift $(z)$', '$\\log \ L_{\\rm [N \ II]}$')
plot_average2('L6', 'co', 3003, zO3_bin[5], (LNII_bin[5]), bin_z, 'Redshift $(z)$', '$\\log \ L_{\\rm [N \ II]}$')
plot_average2('L7', 'yo', 3003, zO3_bin[6], (LNII_bin[6]), bin_z, 'Redshift $(z)$', '$\\log \ L_{\\rm [N \ II]}$')
#plot_average2('L8', 'mo', 3003, zO3_bin[7], (LNII_bin[7]), bin_z, 'Redshift $(z)$', '$\\log \ L_{\\rm [N \ II]}$')

#plot_average2('L3', 'go', 3004, zO3_bin[2], (LSII_bin[2]), bin_z, 'Redshift $(z)$', '$\\log \ L_{\\rm [S \ II]}$')
plot_average2('L4', 'bo', 3004, zO3_bin[3], (LSII_bin[3]), bin_z, 'Redshift $(z)$', '$\\log \ L_{\\rm [S \ II]}$')
plot_average2('L5', 'ko', 3004, zO3_bin[4], (LSII_bin[4]), bin_z, 'Redshift $(z)$', '$\\log \ L_{\\rm [S \ II]}$')
plot_average2('L6', 'co', 3004, zO3_bin[5], (LSII_bin[5]), bin_z, 'Redshift $(z)$', '$\\log \ L_{\\rm [S \ II]}$')
plot_average2('L7', 'yo', 3004, zO3_bin[6], (LSII_bin[6]), bin_z, 'Redshift $(z)$', '$\\log \ L_{\\rm [S \ II]}$')
#plot_average2('L8', 'mo', 3004, zO3_bin[7], (LSII_bin[7]), bin_z, 'Redshift $(z)$', '$\\log \ L_{\\rm [S \ II]}$')

plt.close('all')

#plot_average2('L3', 'go', 3005, log_z_1[2], (SFR_bin[2]), bin_log_z_1, '$\\log \ (z+1)$', '$\\log \ \\rm  SFR \ M_{\\odot} \ yr^{-1}$')
plot_average2('L4', 'bo', 3005, log_z_1[3], (SFR_bin[3]), bin_log_z_1, '$\\log \ (z+1)$', '$\\log \ \\rm SFR \ M_{\\odot} \ yr^{-1}$')
plot_average2('L5', 'ko', 3005, log_z_1[4], (SFR_bin[4]), bin_log_z_1, '$\\log \ (z+1)$', '$\\log \ \\rm SFR \ M_{\\odot} \ yr^{-1}$')
plot_average2('L6', 'co', 3005, log_z_1[5], (SFR_bin[5]), bin_log_z_1, '$\\log \ (z+1)$', '$\\log \ \\rm SFR \ M_{\\odot} \ yr^{-1}$')
plot_average2('L7', 'yo', 3005, log_z_1[6], (SFR_bin[6]), bin_log_z_1, '$\\log \ (z+1)$', '$\\log \ \\rm SFR \ M_{\\odot} \ yr^{-1}$')
#plot_average2('L8', 'mo', 3005, log_z_1[7], (SFR_bin[7]), bin_log_z_1, '$\\log \ (z+1)$', '$\\log \ \\rm SFR \ M_{\\odot} \ yr^{-1}$')

plt.figure(3005)
#plt.plot(bin_log_z_1, linear(bin_log_z_1, 2.02, 0.68), 'r-', label='$0.68 + 2.02 \ \\log(z+1)$')
#plt.plot(bin_log_z_1, linear(bin_log_z_1, 4.50, 0.55), 'k-', label='$0.55 + 4.50 \ \\log(z+1)$')
plt.xlim(min(bin_log_z_1)-0.01, max(bin_log_z_1)+0.01)
#plt.legend(loc='lower right', fontsize='x-small')
plt.savefig('figures/fig_3005')


rfeii = [[], [], [], [], [], []]
LO3_bin2 = [[], [], [], [], [], []]
fwii = [[], [], [], [], [], []]


for i in range(len(np.histogram(Lcont, bins=6)[1]) - 1):
    
    for j in range(len(Lcont)):
        if Lcont[j] > np.histogram(Lcont, bins=6)[1][i] and Lcont[j] <= np.histogram(Lcont, bins=6)[1][i+1]:
            LO3_bin2[i].append(LO3[j])
        
            rfeii[i].append(10**(LFeII[j]-LbHbeta[j]))
            fwii[i].append(fwhm_bHbeta[j])            
            
        else:
            continue



#plot_average2('L1', 'go', 3006, rfeii[0], (LO3_bin2[0]), bin_FeII_bHbeta_2, '$L_{\\rm Fe \ II}/L_{\\rm bH\\beta}$', '$\\log \ L_{\\rm [O \ III]} \ \\rm  (erg \ s^{-1})$')
plot_average2('L2', 'bo', 3006, rfeii[1], (LO3_bin2[1]), bin_FeII_bHbeta_2, '$L_{\\rm Fe \ II}/L_{\\rm bH\\beta}$', '$\\log \ L_{\\rm [O \ III]} \ \\rm  (erg \ s^{-1})$')
plot_average2('L3', 'ko', 3006, rfeii[2], (LO3_bin2[2]), bin_FeII_bHbeta_2, '$L_{\\rm Fe \ II}/L_{\\rm bH\\beta}$', '$\\log \ L_{\\rm [O \ III]} \ \\rm  (erg \ s^{-1})$')
plot_average2('L4', 'co', 3006, rfeii[3], (LO3_bin2[3]), bin_FeII_bHbeta_2, '$L_{\\rm Fe \ II}/L_{\\rm bH\\beta}$', '$\\log \ L_{\\rm [O \ III]} \ \\rm  (erg \ s^{-1})$')
plot_average2('L5', 'yo', 3006, rfeii[4], (LO3_bin2[4]), bin_FeII_bHbeta_2, '$L_{\\rm Fe \ II}/L_{\\rm bH\\beta}$', '$\\log \ L_{\\rm [O \ III]} \ \\rm  (erg \ s^{-1})$')
#plot_average2('L8', 'mo', 3006, rfeii[5], (LO3_bin2[5]), bin_FeII_bHbeta_2, '$L_{\\rm Fe \ II}/L_{\\rm bH\\beta}$', '$\\log \ L_{\\rm [O \ III]} \ \\rm  (erg \ s^{-1})$')


#plot_average2('L1', 'go', 3007, rfeii[0], (LO3_bin2[0]), bin_FeII_bHbeta_2, '$\\rm FWHM \ bH\\beta \ (km/s)$', '$\\log \ L_{\\rm [O \ III]} \ \\rm  (erg \ s^{-1})$')
plot_average2('L2', 'bo', 3007, fwii[1], (LO3_bin2[1]), bin_fwhm_bHbeta_2, '$\\rm FWHM \ bH\\beta \ (km/s)$', '$\\log \ L_{\\rm [O \ III]} \ \\rm  (erg \ s^{-1})$')
plot_average2('L3', 'ko', 3007, fwii[2], (LO3_bin2[2]), bin_fwhm_bHbeta_2, '$\\rm FWHM \ bH\\beta \ (km/s)$', '$\\log \ L_{\\rm [O \ III]} \ \\rm  (erg \ s^{-1})$')
plot_average2('L4', 'co', 3007, fwii[3], (LO3_bin2[3]), bin_fwhm_bHbeta_2, '$\\rm FWHM \ bH\\beta \ (km/s)$', '$\\log \ L_{\\rm [O \ III]} \ \\rm  (erg \ s^{-1})$')
plot_average2('L5', 'yo', 3007, fwii[4], (LO3_bin2[4]), bin_fwhm_bHbeta_2, '$\\rm FWHM \ bH\\beta \ (km/s)$', '$\\log \ L_{\\rm [O \ III]} \ \\rm  (erg \ s^{-1})$')
#plot_average2('L8', 'mo', 3007, rfeii[5], (LO3_bin2[5]), bin_FeII_bHbeta_2, '$\\rm FWHM \ bH\\beta \ (km/s)$', '$\\log \ L_{\\rm [O \ III]} \ \\rm  (erg \ s^{-1})$')



plt.close('all')
















###########################################################

'''

def plot_average3(number, x, y, group, labelx, labely):
    
    binned = []    
    mean = []
    stdx = []
    stdy = []
    count = []
    
    for i in range(len(group)):
        point = []
        axis = []
        for j in range(len(x)):
            #if x[j] >= group[i] and x[j] < group[i+1]:
            if abs(x[j]-group[i]) <= abs((group[0]-group[1])/2.): 
                axis.append(x[j])
                point.append(y[j])
        
        if len(point) >= 15:
            mean.append(np.mean(point))
            stdy.append(np.std(point)*1.)
            stdx.append(np.std(axis)*1.)            
            count.append(len(point))
            binned.append(group[i])
    
    m, c, r, p, std = linregress(x, y)
    rho, P = spearmanr(x, y)
    o = open('output_parameters.csv', 'a')    
    o.write(str(number) + '\t' + str(m) + '\t' + str(c) + '\t' + str(r) + '\t' + str(p) + '\t' + str(std) + '\t' + str(rho) + '\t' + str(P) + '\n')
    o.close()
    plt.figure(number)
    plt.plot(binned, linear(binned, m, c), 'r--',  alpha=0, label = ('$\\rho_s = %f$\n$ P_s = %f$' %(rho, P)))
    plt.legend(loc='best')
    plt.plot(x, y, 'k.', alpha=0.15)
    plt.errorbar(binned, mean, yerr=stdy, xerr=0, fmt='bo')
    plt.plot(binned, mean, 'b-')
    plt.xlabel(labelx, fontsize='large')
    plt.ylabel(labely, fontsize='large')
    plt.xlim(min(binned)-0.05, max(binned)+0.05)
    plt.savefig('figures/fig_%i' %number)

bin_FeII_bHbeta = np.arange(min(LFeII-LbHbeta), max(LFeII-LbHbeta), 0.5/4.)
bin_bHbeta_FeII = np.arange(min(LbHbeta-LFeII), max(LbHbeta-LFeII), 0.5/4.)


plot_average3(3006, LFeII-LbHbeta, LO2, bin_FeII_bHbeta, '$\\log \ L_{Fe \ II}/L_{bH\\beta}$', '$\\log \ L_{[O \ II]}$') 
plot_average3(3007, LFeII-LbHbeta, LO3, bin_FeII_bHbeta, '$\\log \ L_{Fe \ II}/L_{bH\\beta}$', '$\\log \ L_{[O \ III]}$') 
plot_average3(3008, LFeII-LbHbeta, LnHbeta, bin_FeII_bHbeta, '$\\log \ L_{Fe \ II}/L_{bH\\beta}$', '$\\log \ L_{nH\\beta}$') 
plot_average3(3009, LFeII-LbHbeta, LnHalpha, bin_FeII_bHbeta, '$\\log \ L_{Fe \ II}/L_{bH\\beta}$', '$\\log \ L_{nH\\alpha}$')
plot_average3(3010, LFeII-LbHbeta, LNII, bin_FeII_bHbeta, '$\\log \ L_{Fe \ II}/L_{bH\\beta}$', '$\\log \ L_{[N \ II]}$')
plot_average3(3011, LFeII-LbHbeta, LSII, bin_FeII_bHbeta, '$\\log \ L_{Fe \ II}/L_{bH\\beta}$', '$\\log \ L_{[S \ II]}$')
plot_average3(3012, LFeII-LbHbeta, LbHbeta, bin_FeII_bHbeta, '$\\log \ L_{Fe \ II}/L_{bH\\beta}$', '$\\log \ L_{bH\\beta}$') 
plot_average3(3013, LFeII-LbHbeta, LbHalpha, bin_FeII_bHbeta, '$\\log \ L_{Fe \ II}/L_{bH\\beta}$', '$\\log \ L_{bH\\alpha}$')
plot_average3(3014, LFeII-LbHbeta, Lcont, bin_FeII_bHbeta, '$\\log \ L_{Fe \ II}/L_{bH\\beta}$', '$\\log \ L_{5100}$')
plt.close('all')


LO2_int = L_int(LO2, LnHalpha, LnHbeta, 3727.)
LO3_int = L_int(LO3, LnHalpha, LnHbeta, 5007.)
LnHbeta_int = L_int(LnHbeta, LnHalpha, LnHbeta, 4860.)
LnHalpha_int = L_int(LnHalpha, LnHalpha, LnHbeta, 6563.)
LNII_int = L_int(LNII, LnHalpha, LnHbeta, 6584.)
LSII_int = L_int(LSII, LnHalpha, LnHbeta, 6731.)

LbHbeta_int = L_int(LbHbeta, LbHalpha, LbHbeta, 4860.)
LbHalpha_int = L_int(LbHalpha, LbHalpha, LbHbeta, 6563.)
LFeII_int = L_int(LFeII, LbHalpha, LbHbeta, 4570.)

bin_FeII_bHbeta_int = np.arange(min(LFeII_int-LbHbeta_int), max(LFeII_int-LbHbeta_int), 0.5/4.)
bin_bHbeta_FeII_int = np.arange(min(LbHbeta_int-LFeII_int), max(LbHbeta_int-LFeII_int), 0.5/4.)

plot_average3(3106, LFeII_int-LbHbeta_int, LO2_int, bin_FeII_bHbeta_int, '$\\log \ L_{Fe \ II}^{cor}/L_{bH\\beta}^{cor}$', '$\\log \ L_{[O \ II]}^{cor}$') 
plot_average3(3107, LFeII_int-LbHbeta_int, LO3_int, bin_FeII_bHbeta_int, '$\\log \ L_{Fe \ II}^{cor}/L_{bH\\beta}^{cor}$', '$\\log \ L_{[O \ III]}^{cor}$') 
plot_average3(3108, LFeII_int-LbHbeta_int, LnHbeta_int, bin_FeII_bHbeta_int, '$\\log \ L_{Fe \ II}^{cor}/L_{bH\\beta}^{cor}$', '$\\log \ L_{nH\\beta}^{cor}$') 
plot_average3(3109, LFeII_int-LbHbeta_int, LnHalpha_int, bin_FeII_bHbeta_int, '$\\log \ L_{Fe \ II}^{cor}/L_{bH\\beta}^{cor}$', '$\\log \ L_{nH\\alpha}^{cor}$')
plot_average3(3110, LFeII_int-LbHbeta_int, LNII_int, bin_FeII_bHbeta_int, '$\\log \ L_{Fe \ II}^{cor}/L_{bH\\beta}^{cor}$', '$\\log \ L_{[N \ II]}^{cor}$')
plot_average3(3111, LFeII_int-LbHbeta_int, LSII_int, bin_FeII_bHbeta_int, '$\\log \ L_{Fe \ II}^{cor}/L_{bH\\beta}^{cor}$', '$\\log \ L_{[S \ II]}^{cor}$')
plot_average3(3112, LFeII_int-LbHbeta_int, LbHbeta_int, bin_FeII_bHbeta_int, '$\\log \ L_{Fe \ II}^{cor}/L_{bH\\beta}^{cor}$', '$\\log \ L_{bH\\beta}^{cor}$') 
plot_average3(3113, LFeII_int-LbHbeta_int, LbHalpha_int, bin_FeII_bHbeta_int, '$\\log \ L_{Fe \ II}^{cor}/L_{bH\\beta}^{cor}$', '$\\log \ L_{bH\\alpha}^{cor}$')
plot_average3(3114, LFeII_int-LbHbeta_int, Lcont_int, bin_FeII_bHbeta_int, '$\\log \ L_{Fe \ II}^{cor}/L_{bH\\beta}^{cor}$', '$\\log \ L_{5100}^{cor}$')

bin_O3_int = np.arange(min(LO3_int), max(LO3_int), 0.5/2.)
plot_average(3503, LO3_int, LbHalpha_int, bin_O3_int, '$\\log \ L_{[O \ III]}^{cor} \ (erg \ s^{-1})$', '$\\log \ L_{bH\\alpha}^{cor} \ (erg \ s^{-1})$')
plot_average(3504, LO3_int, LFeII_int, bin_O3_int, '$\\log \ L_{[O \ III]}^{cor} \ (erg \ s^{-1})$', '$\\log \ L_{Fe II}^{cor} \ (erg \ s^{-1})$')
plt.close('all')

'''






'''

########## Dereddening by Balmer decrement ##########

LO3_x = np.array(LO3_x)     ; LO2_x = np.array(LO2_x)
LO3_fuv = np.array(LO3_fuv) ; LO2_fuv = np.array(LO2_fuv)
LO3_nuv = np.array(LO3_nuv) ; LO2_nuv = np.array(LO2_nuv)
LO3_u = np.array(LO3_u)     ; LO2_u = np.array(LO2_u)
LO3_g = np.array(LO3_g)     ; LO2_g = np.array(LO2_g)
LO3_r = np.array(LO3_r)     ; LO2_r = np.array(LO2_r)
LO3_i = np.array(LO3_i)     ; LO2_i = np.array(LO2_i)
LO3_z = np.array(LO3_z)     ; LO2_z = np.array(LO2_z)
LO3_j = np.array(LO3_j)     ; LO2_j = np.array(LO2_j)
LO3_h = np.array(LO3_h)     ; LO2_h = np.array(LO2_h)
LO3_k = np.array(LO3_k)     ; LO2_k = np.array(LO2_k)
LO3_20cm = np.array(LO3_20cm)     ; LO2_20cm = np.array(LO2_20cm)
LO3_cont = np.array(LO3_cont)   ; LO2_cont = np.array(LO2_cont)

 
LO3_int = L_int(LO3, LnHalpha, LnHbeta, 5007.)
LO2_int = L_int(LO2, LnHalpha, LnHbeta, 3727.)

Lcont_int = np.array(Lcont_int)

bin_O32_int = np.arange(min(LO3_int-LO2_int), max(LO3_int-LO2_int), 0.25/2.)
bin_O3_int = np.arange(min(LO3_int), max(LO3_int), 0.5/2.)
bin_O2_int = np.arange(min(LO2_int), max(LO2_int), 0.5/2.)

#LO3_x_int = []

plot_average(4101, LO3_x_int, Lx, bin_O3_int, '$\\log \ L_{[O \ III]}^{cor} \ (erg \ s^{-1})$', '$\\log \ L_X \ (erg \ s^{-1})$') 
plot_average(4102, LO3_nuv_int, Lnuv, bin_O3_int, '$\\log \ L_{[O \ III]}^{cor} \ (erg \ s^{-1})$', '$\\log \ L_{NUV} \ (erg \ s^{-1})$')
plot_average(4103, LO3_fuv_int, Lfuv, bin_O3_int, '$\\log \ L_{[O \ III]}^{cor} \ (erg \ s^{-1})$', '$\\log \ L_{FUV} \ (erg \ s^{-1})$')
plot_average(4104, LO3_u_int, Lu, bin_O3_int, '$\\log \ L_{[O \ III]}^{cor} \ (erg \ s^{-1})$', '$\\log \ L_u \ (erg \ s^{-1})$') 
plot_average(4105, LO3_g_int, Lg, bin_O3_int, '$\\log \ L_{[O \ III]}^{cor} \ (erg \ s^{-1})$', '$\\log \ L_g \ (erg \ s^{-1})$') 
plot_average(4106, LO3_r_int, Lr, bin_O3_int, '$\\log \ L_{[O \ III]}^{cor} \ (erg \ s^{-1})$', '$\\log \ L_r \ (erg \ s^{-1})$') 
plot_average(4107, LO3_i_int, Li, bin_O3_int, '$\\log \ L_{[O \ III]}^{cor} \ (erg \ s^{-1})$', '$\\log \ L_i \ (erg \ s^{-1})$') 
plot_average(4108, LO3_z_int, Lz, bin_O3_int, '$\\log \ L_{[O \ III]}^{cor} \ (erg \ s^{-1})$', '$\\log \ L_z \ (erg \ s^{-1})$')
plot_average(4109, LO3_j_int, Lj, bin_O3_int, '$\\log \ L_{[O \ III]}^{cor} \ (erg \ s^{-1})$', '$\\log \ L_J \ (erg \ s^{-1})$') 
plot_average(4110, LO3_h_int, Lh, bin_O3_int, '$\\log \ L_{[O \ III]}^{cor} \ (erg \ s^{-1})$', '$\\log \ L_H \ (erg \ s^{-1})$') 
plot_average(4111, LO3_k_int, Lk, bin_O3_int, '$\\log \ L_{[O \ III]}^{cor} \ (erg \ s^{-1})$', '$\\log \ L_K \ (erg \ s^{-1})$')
plot_average(4112, LO3_20cm_int, L20cm, bin_O3_int, '$\\log \ L_{[O \ III]}^{cor} \ (erg \ s^{-1})$', '$\\log \ L_{Radio} \ (erg \ s^{-1})$')
plot_average(4113, LO3_cont_int, Lcont_int, bin_O3_int, '$\\log \ L_{[O \ III]}^{cor} \ (erg \ s^{-1})$', '$\\log \ L_{5100} \ (erg \ s^{-1})$')
plt.close('all')



plot_average(4201, LO2_x_int, Lx, bin_O2_int, '$\\log \ L_{[O \ II]}^{cor} \ (erg \ s^{-1})$', '$\\log \ L_X \ (erg \ s^{-1})$') 
plot_average(4202, LO2_nuv_int, Lnuv, bin_O2_int, '$\\log \ L_{[O \ II]}^{cor} \ (erg \ s^{-1})$', '$\\log \ L_{NUV} \ (erg \ s^{-1})$')
plot_average(4203, LO2_fuv_int, Lfuv, bin_O2_int, '$\\log \ L_{[O \ II]}^{cor} \ (erg \ s^{-1})$', '$\\log \ L_{FUV} \ (erg \ s^{-1})$')
plot_average(4204, LO2_u_int, Lu, bin_O2_int, '$\\log \ L_{[O \ II]}^{cor} \ (erg \ s^{-1})$', '$\\log \ L_u \ (erg \ s^{-1})$') 
plot_average(4205, LO2_g_int, Lg, bin_O2_int, '$\\log \ L_{[O \ II]}^{cor} \ (erg \ s^{-1})$', '$\\log \ L_g \ (erg \ s^{-1})$') 
plot_average(4206, LO2_r_int, Lr, bin_O2_int, '$\\log \ L_{[O \ II]}^{cor} \ (erg \ s^{-1})$', '$\\log \ L_r \ (erg \ s^{-1})$') 
plot_average(4207, LO2_i_int, Li, bin_O2_int, '$\\log \ L_{[O \ II]}^{cor} \ (erg \ s^{-1})$', '$\\log \ L_i \ (erg \ s^{-1})$') 
plot_average(4208, LO2_z_int, Lz, bin_O2_int, '$\\log \ L_{[O \ II]}^{cor} \ (erg \ s^{-1})$', '$\\log \ L_z \ (erg \ s^{-1})$')
plot_average(4209, LO2_j_int, Lj, bin_O2_int, '$\\log \ L_{[O \ II]}^{cor} \ (erg \ s^{-1})$', '$\\log \ L_J \ (erg \ s^{-1})$') 
plot_average(4210, LO2_h_int, Lh, bin_O2_int, '$\\log \ L_{[O \ II]}^{cor} \ (erg \ s^{-1})$', '$\\log \ L_H \ (erg \ s^{-1})$')
plot_average(4211, LO2_k_int, Lk, bin_O2_int, '$\\log \ L_{[O \ II]}^{cor} \ (erg \ s^{-1})$', '$\\log \ L_K \ (erg \ s^{-1})$')
plot_average(4212, LO2_20cm_int, L20cm, bin_O2_int, '$\\log \ L_{[O \ II]}^{cor} \ (erg \ s^{-1})$', '$\\log \ L_{Radio} \ (erg \ s^{-1})$')
plot_average(4213, LO2_cont_int, Lcont_int, bin_O2_int, '$\\log \ L_{[O \ II]}^{cor} \ (erg \ s^{-1})$', '$\\log \ L_{5100} \ (erg \ s^{-1})$')
plt.close('all')

plot_average(4301, LO32_x_int, Lx, bin_O32_int, '$\\log \ L_{[O \ III]}^{cor}/L_{[O \ II]}^{cor}$', '$\\log \ L_X \ (erg \ s^{-1})$')
plot_average(4302, LO32_nuv_int, Lnuv, bin_O32_int, '$\\log \ L_{[O \ III]}^{cor}/L_{[O \ II]}^{cor}$', '$\\log \ L_{NUV} \ (erg \ s^{-1})$')
plot_average(4303, LO32_fuv_int, Lfuv, bin_O32_int, '$\\log \ L_{[O \ III]}^{cor}/L_{[O \ II]}^{cor}$', '$\\log \ L_{FUV} \ (erg \ s^{-1})$')
plot_average(4304, LO32_u_int, Lu, bin_O32_int, '$\\log \ L_{[O \ III]}^{cor}/L_{[O \ II]}^{cor}$', '$\\log \ L_u \ (erg \ s^{-1})$') 
plot_average(4305, LO32_g_int, Lg, bin_O32_int, '$\\log \ L_{[O \ III]}^{cor}/L_{[O \ II]}^{cor}$', '$\\log \ L_g \ (erg \ s^{-1})$') 
plot_average(4306, LO32_r_int, Lr, bin_O32_int, '$\\log \ L_{[O \ III]}^{cor}/L_{[O \ II]}^{cor}$', '$\\log \ L_r \ (erg \ s^{-1})$') 
plot_average(4307, LO32_i_int, Li, bin_O32_int, '$\\log \ L_{[O \ III]}^{cor}/L_{[O \ II]}^{cor}$', '$\\log \ L_i \ (erg \ s^{-1})$') 
plot_average(4308, LO32_z_int, Lz, bin_O32_int, '$\\log \ L_{[O \ III]}^{cor}/L_{[O \ II]}^{cor}$', '$\\log \ L_z \ (erg \ s^{-1})$')
plot_average(4309, LO32_j_int, Lj, bin_O32_int, '$\\log \ L_{[O \ III]}^{cor}/L_{[O \ II]}^{cor}$', '$\\log \ L_J \ (erg \ s^{-1})$') 
plot_average(4310, LO32_h_int, Lh, bin_O32_int, '$\\log \ L_{[O \ III]}^{cor}/L_{[O \ II]}^{cor}$', '$\\log \ L_H \ (erg \ s^{-1})$') 
plot_average(4311, LO32_k_int, Lk, bin_O32_int, '$\\log \ L_{[O \ III]}^{cor}/L_{[O \ II]}^{cor}$', '$\\log \ L_K \ (erg \ s^{-1})$')
plot_average(4312, LO32_20cm_int, L20cm, bin_O32_int, '$\\log \ L_{[O \ III]}^{cor}/L_{[O \ II]}^{cor}$', '$\\log \ L_{Radio} \ (erg \ s^{-1})$')
plot_average(4313, LO32_cont_int, Lcont_int, bin_O32_int, '$\\log \ L_{[O \ III]}^{cor}/L_{[O \ II]}^{cor}$', '$\\log \ L_{5100} \ (erg \ s^{-1})$') 
plt.close('all')

plot_average(4502, LO3_int, LO2_int-LO3_int, bin_O3_int, '$\\log \ L_{[O \ III]}^{cor} \ (erg \ s^{-1})$', '$\\log \ L_{[O \ II]}^{cor}/L_{[O \ III]}^{cor}$')
plt.close('all')
'''


#from scipy import interpolate

q=0
for i in range(10):
    composite = np.loadtxt('note/L'+str(i), dtype=float, delimiter='  ')
    plt.figure(10000)
    plt.plot(composite[:, 0], composite[:, 1]/np.median(composite[:, 1])+q)#, label='L'+str(i+1))
    #plt.plot(composite[:, 0], composite[:, 1]/np.median(interpolate.spline(composite[:, 0], composite[:, 1], np.linspace(3700, 3710), order=3)))
    
    q+=1

#plt.title()
plt.xlabel('Wavelength ($\AA$)')
plt.ylabel('Arbitrary Unit')
#plt.legend(loc='best', fontsize='xx-small')
plt.savefig('figures/fig_10000')
#plt.show()
plt.close('all')

########## Save Object To Table ##########
from astropy import units as u
from astropy.coordinates import SkyCoord

c = SkyCoord(ra=10.625*u.degree, dec=41.2*u.degree, frame='icrs')
c.to_string('dms')



def name(x, y):
    c = SkyCoord(ra=x*u.degree, dec=y*u.degree, frame='icrs')
    if y >= 0:
        return str('SDSS J%02d%02d%05.2f+%02d%02d%04.1f' %(c.ra.hms[0], c.ra.hms[1], c.ra.hms[2], c.dec.dms[0], c.dec.dms[1], c.dec.dms[2]))
    else:
        return str('SDSS J%02d%02d%05.2f-%02d%02d%04.1f' %(c.ra.hms[0], c.ra.hms[1], c.ra.hms[2], abs(c.dec.dms[0]), abs(c.dec.dms[1]), abs(c.dec.dms[2])))

name(10.625, 41.2)




o = open('note/result.csv', 'w')
o.write('name\tplate\tmjd\tfiberid\tz\tLO2\tLO3\tLnHbeta\tLbHbeta\tLnHalpha\tLbHalpha\tLNII\tLSII\tLFeII\tL5100\
\tfwhm_bHbeta\t')
o.write('L_X\tL_FUV\tL_NUV\tL_u\tL_g\tL_r\tL_i\tL_z\tL_J\tL_H\tL_K\tL_Radio\tGamma\tShift_CIV\n')

for i in range(len(LO3)):
    o.write(name(float(data[J[i]][5]), float(data[J[i]][6])) + '\t' + str((data[J[i]][2])) + '\t' + str((data[J[i]][3])) + '\t' + str((data[J[i]][4])) + '\t' + str(z_O32[i]) + '\t'\
    + str(LO2[i]) + '\t' + str(LO3[i]) + '\t' + str(LnHbeta[i]) + '\t' + str(LbHbeta[i]) + '\t'\
    + str(LnHalpha[i]) + '\t' + str(LbHalpha[i]) + '\t' + str(LNII[i]) + '\t' + str(LSII[i]) + '\t'\
    + str(LFeII[i]) + '\t' + str(Lcont[i]) + '\t' + str(fwhm_bHbeta[i]) + '\t')
    
    o.write(str(L_x[J[i]]) + '\t' + str(L_fuv[J[i]]) + '\t' + str(L_nuv[J[i]]) + '\t' + str(L_u[J[i]]) + '\t'\
    + str(L_g[J[i]]) + '\t' + str(L_r[J[i]]) + '\t' + str(L_i[J[i]]) + '\t' + str(L_z[J[i]]) + '\t'\
    + str(L_j[J[i]]) + '\t' + str(L_h[J[i]]) + '\t' + str(L_k[J[i]]) + '\t' + str(L_20cm[J[i]]) + '\t')
    
    o.write(str(gamma[i]) + '\t' + str(shift_bCIV[i]) + '\n')

o.close()


sign = np.concatenate((fig_11258[0], fig_11258[1], fig_11258[2]))

o = open('note/result_for_pca_.csv', 'w')
o.write('name\tplate\tmjd\tfiberid\tz\tLO2\tLO3\tLnHbeta\tLbHbeta\tLnHalpha\tLbHalpha\tLNII\tLSII\tLFeII\tL5100\
\tfwhm_bHbeta\t')
o.write('L_X\tL_FUV\tL_NUV\tL_u\tL_g\tL_r\tL_i\tL_z\tL_J\tL_H\tL_K\tL_Radio\tGamma\tShift_CIV\tfwhm_OIII\tL_L_Edd\tM_BH\tL_bol\n')

for i in sign:
    o.write(name(float(data[J[i]][5]), float(data[J[i]][6])) + '\t' + str((data[J[i]][2])) + '\t' + str((data[J[i]][3])) + '\t' + str((data[J[i]][4])) + '\t' + str(z_O32[i]) + '\t'\
    + str(LO2[i]) + '\t' + str(LO3[i]) + '\t' + str(LnHbeta[i]) + '\t' + str(LbHbeta[i]) + '\t'\
    + str(LnHalpha[i]) + '\t' + str(LbHalpha[i]) + '\t' + str(LNII[i]) + '\t' + str(LSII[i]) + '\t'\
    + str(LFeII[i]) + '\t' + str(Lcont[i]) + '\t' + str(fwhm_bHbeta[i]) + '\t')
    
    o.write(str(L_x[J[i]]) + '\t' + str(L_fuv[J[i]]) + '\t' + str(L_nuv[J[i]]) + '\t' + str(L_u[J[i]]) + '\t'\
    + str(L_g[J[i]]) + '\t' + str(L_r[J[i]]) + '\t' + str(L_i[J[i]]) + '\t' + str(L_z[J[i]]) + '\t'\
    + str(L_j[J[i]]) + '\t' + str(L_h[J[i]]) + '\t' + str(L_k[J[i]]) + '\t' + str(L_20cm[J[i]]) + '\t')
    
    o.write(str(gamma[i]) + '\t' + str(shift_bCIV[i]) + '\t' + str(fwhm_OIII_5007[i]) + '\t' + str(L_L_Edd[i]) + '\t' + str(M_BH_vir_Hbeta[i]) + '\t' + str(L_bol[i]) + '\n')

o.close()


o = open('note/data_2.csv', 'w')

o.write('plate\tmjd\tfiberid\tLO2\tLO3\tLO3/LO2\tLradio\tj\ttype\n')

for i in fig_11258[3]:
    o.write(str((data[i][2])) + '\t' + str((data[i][3])) + '\t' + str((data[i][4])) + '\t'\
    + str(L_OII_3727[i]) + '\t' + str(L_OIII_5007[i]) + '\t' + str(10**(L_OIII_5007[i]-L_OII_3727[i])) + '\t'\
    + str(L_20cm[i]) + '\t'+ str(i) + '\t')
    
    if i in J[fig_11258[0]]:
        o.write('RQ' + '\n')
    elif i in J[fig_11258[1]]:
        o.write('RL' + '\n')
    else:
        None

o.close()


o = open('note/radcoor.csv', 'w')

o.write('RA\tDec\n')

for i in fig_11258[3]:

    o.write(str(float(data[i][5])/360.*24.) + '\t')
    
    if float(data[i][6]) > 0.:
        o.write('\'+' + str(data[i][6]) + '\n')
    else:
        o.write(str(data[i][6]) + '\n')

o.close()




######### Experiment #########

