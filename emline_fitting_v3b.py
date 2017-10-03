# -*- coding: utf-8 -*-
"""
Created on Mon Nov  2 16:15:56 2015

@author: andika
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Oct  6 18:08:15 2015

@author: andika
"""

import numpy as np
import pylab as plt
from astropy.modeling import fitting
from astropy.modeling.models import custom_model
from scipy import integrate
from scipy.stats import chisquare


q = 0

o = open('calculation/line_flux.csv', 'w')
o.write('flux_[OII]_3727\tflux_n_H_beta\tflux_b_H_beta\tflux_[OIII]_4959\tflux_[OIII]_5007\
\tflux_n_H_alpha\tflux_b_H_alpha\tflux_[NII]_6548\tflux_[NII]_6584\tflux_[SII]_6717\tflux_[SII]_6731\n')

f = open('calculation/line_fwhm.csv', 'w')
f.write('fwhm_[OII]_3727\tfwhm_n_H_beta\tfwhm_b_H_beta\tfwhm_b_H_beta2\tfwhm_[OIII]_4959\tfwhm_[OIII]_5007\
\tfwhm_n_H_alpha\tfwhm_b_H_alpha\tfwhm_b_H_alpha2\tfwhm_[NII]_6548\tfwhm_[NII]_6584\tfwhm_[SII]_6717\tfwhm_[SII]_6731\tshift_[OIII]_5007a\tshift_[OIII]_5007b\n')

####################
v = open('calculation/line_ew.csv', 'w')
v.write('EW_[OII]_3727\tEW_n_H_beta\tEW_b_H_beta\tEW_[OIII]_4959\tEW_[OIII]_5007\
\tEW_n_H_alpha\tEW_b_H_alpha\tEW_[NII]_6548\tEW_[NII]_6584\tEW_[SII]_6717\tEW_[SII]_6731\n')

h = open('calculation/line_norm_fwhm.csv', 'w')
h.write('fwhm_[OII]_3727\tfwhm_n_H_beta\tfwhm_b_H_beta\tfwhm_b_H_beta2\tfwhm_[OIII]_4959\tfwhm_[OIII]_5007\
\tfwhm_n_H_alpha\tfwhm_b_H_alpha\tfwhm_b_H_alpha2\tfwhm_[NII]_6548\tfwhm_[NII]_6584\tfwhm_[SII]_6717\tfwhm_[SII]_6731\n')


o.close()
f.close()
v.close()
h.close()

#J = [3, 69, 97, 106, 117, 126, 134, 175, 202, 232, 236, 300, 360, 754, 1189, 1220, 1521, 1749, 2417, 2528]

c = 299792458 * 10**-3 #km/s

flux_OII_3727 = []
flux_nHbeta = []
flux_bHbeta = []
flux_OIII_4959 = []
flux_OIII_5007 = []
flux_nHalpha = []
flux_bHalpha = []
flux_NII_6548 = []
flux_NII_6584 = []
flux_SII_6717 = []
flux_SII_6731 = []

EW_OII_3727 = []
EW_nHbeta = []
EW_bHbeta = []
EW_OIII_4959 = []
EW_OIII_5007 = []
EW_nHalpha = []
EW_bHalpha = []
EW_NII_6548 = []
EW_NII_6584 = []
EW_SII_6717 = []
EW_SII_6731 = []

fwhm_OII_3727 = []

fwhm_nHbeta = []
fwhm_bHbeta = []
fwhm_bHbeta2 = []

fwhm_OIII_4959 = []
fwhm_OIII_5007 = []
shift_OIII_a = []
shift_OIII_b = []

fwhm_nHalpha = []
fwhm_bHalpha = []
fwhm_bHalpha2 = []

fwhm_NII_6548 = []
fwhm_NII_6584 = []
fwhm_SII_6717 = []
fwhm_SII_6731 = []
###########################
fwhm_OII_3727_norm = []

fwhm_nHbeta_norm = []
fwhm_bHbeta_norm = []
fwhm_bHbeta2_norm = []

fwhm_OIII_4959_norm = []
fwhm_OIII_5007_norm = []

fwhm_nHalpha_norm = []
fwhm_bHalpha_norm = []
fwhm_bHalpha2_norm = []

fwhm_NII_6548_norm = []
fwhm_NII_6584_norm = []
fwhm_SII_6717_norm = []
fwhm_SII_6731_norm = []


#agn = np.loadtxt('cont_sub/sub_spec-0542-51993-0560.fits', skiprows=1, dtype = float)


sample = np.loadtxt('MAIN_SAMPLE_irhamta.csv', skiprows=1, dtype=str, delimiter=',')

plate = []
mjd = []
fiberid = []
z = []
e_bv = []

skip = np.loadtxt('notes/skip.csv', delimiter=',', skiprows=1)


for j in range (10):#9184):#len(sample)):
    
    if len(sample[j][2]) < 4:
        plate.append(str(0)+sample[j][2])
    else:
        plate.append(sample[j][2])
    
    mjd.append(sample[j][3])

    if len(sample[j][4]) == 4:
        fiberid.append(sample[j][4])    
    elif len(sample[j][4]) == 3:
        fiberid.append(str(0) + sample[j][4])
    elif len(sample[j][4]) == 2:
        fiberid.append(str(0) + str(0) + sample[j][4])
    else:
        fiberid.append(str(0) + str(0) + str(0) + sample[j][4])

    
    if j < 1:
        print 'skipped, j = ', j
        continue    
    
    
    if j in skip[:, 1] or j == skip[0, 4]: #or j < 3455:#for until redshift 3.5
        o = open('calculation/line_flux.csv', 'a')
        f = open('calculation/line_fwhm.csv', 'a')

        #for i in range(len(flux_nHbeta)):
        o.write(str(0) + '\t' + str(0) + '\t' + str(0)\
        + '\t' + str(0) + '\t' + str(0)\
        + '\t' + str(0) + '\t' + str(0)\
        + '\t' + str(0) + '\t' + str(0)\
        + '\t' + str(0) + '\t' + str(0) + '\n')
    
        #for i in range(len(fwhm_nHbeta)):
        f.write(str(0) + '\t' + str(0) + '\t' + str(0) + '\t' + str(0)\
        + '\t' + str(0) + '\t' + str(0)\
        + '\t' + str(0) + '\t' + str(0) + '\t' + str(0)\
        + '\t' + str(0) + '\t' + str(0)\
        + '\t' + str(0) + '\t' + str(0) + '\t' + str(0) + '\t' + str(0) + '\n')
     
        o.close()
        f.close()
    
        v = open('calculation/line_ew.csv', 'a')
        h = open('calculation/line_norm_fwhm.csv', 'a')
    
        v.write(str(0) + '\t' + str(0) + '\t' + str(0)\
        + '\t' + str(0) + '\t' + str(0)\
        + '\t' + str(0) + '\t' + str(0)\
        + '\t' + str(0) + '\t' + str(0)\
        + '\t' + str(0) + '\t' + str(0) + '\n')
    
        #for i in range(len(fwhm_nHbeta)):
        h.write(str(0) + '\t' + str(0) + '\t' + str(0) + '\t' + str(0)\
        + '\t' + str(0) + '\t' + str(0)\
        + '\t' + str(0) + '\t' + str(0) + '\t' + str(0)\
        + '\t' + str(0) + '\t' + str(0)\
        + '\t' + str(0) + '\t' + str(0) + '\n')
    
        v.close()
        h.close()
        
        print 'skipped, j = ', j
        continue

    
    agn = np.loadtxt('cont_sub/sub_spec-' + plate[j] + '-' + mjd[j] + '-' + fiberid[j] + '.fits')
    agn2 = np.loadtxt('cont_norm/norm_spec-' + plate[j] + '-' + mjd[j] + '-' + fiberid[j] + '.fits')
   





    wl_OII_3727 = []
    fl_OII_3727 = []

    wl_hbeta = []
    fl_hbeta = []

    wl_halpha = []
    fl_halpha = []

    wl_OIII_4959 = []
    fl_OIII_4959 = []

    wl_OIII_5007 = []
    fl_OIII_5007 = []
    
    
    wl_OII_3727_norm = []
    fl_OII_3727_norm = []

    wl_hbeta_norm = []
    fl_hbeta_norm = []

    wl_halpha_norm = []
    fl_halpha_norm = []

    wl_OIII_4959_norm = []
    fl_OIII_4959_norm = []

    wl_OIII_5007_norm = []
    fl_OIII_5007_norm = []    
    
    
    

    for i in range(len(agn)):#Hbeta window
        if agn[i][0] >= 3717 and agn[i][0] <=3740:
            wl_OII_3727.append(agn[i][0])
            fl_OII_3727.append(agn[i][1])
            
            wl_OII_3727_norm.append(agn2[i][0])
            fl_OII_3727_norm.append(agn2[i][1])
        
        elif agn[i][0] >= 6400 and agn[i][0] <=6800:
            wl_halpha.append(agn[i][0])
            fl_halpha.append(agn[i][1])
            
            wl_halpha_norm.append(agn2[i][0])
            fl_halpha_norm.append(agn2[i][1])
    
        elif agn[i][0] >= 4700 and agn[i][0] <=5100:#3747:
            wl_hbeta.append(agn[i][0])
            fl_hbeta.append(agn[i][1])   
            
            wl_hbeta_norm.append(agn2[i][0])
            fl_hbeta_norm.append(agn2[i][1])   
        else:
            continue
    
    fl_OII_3727_norm = np.array(fl_OII_3727_norm)
    fl_halpha_norm = np.array(fl_halpha_norm)
    fl_hbeta_norm = np.array(fl_hbeta_norm)


    @custom_model
    def broad_hbeta(x, amplitude_b1=np.mean(fl_hbeta), amplitude_b2=np.mean(fl_hbeta), amplitude_b3=np.mean(fl_hbeta), sigma_b1=1.5, sigma_b2=1.5, sigma_b3=1, shift_b=0, shift_b2=0):
        return ((abs(amplitude_b1)* 1./(abs(sigma_b1)*np.sqrt(2.*np.pi)) * np.exp(-0.5 * ((x - 4860 + shift_b) / abs(sigma_b1))**2.))\
    + (abs(amplitude_b2)* 1./(abs(sigma_b2)*np.sqrt(2.*np.pi)) * np.exp(-0.5 * ((x - 4860 + shift_b2) / abs(sigma_b2))**2.)))
    #+ (abs(amplitude_b3)* 1./(abs(sigma_b3)*np.sqrt(2.*np.pi)) * np.exp(-0.5 * ((x - 4860 + shift_b) / abs(sigma_b3))**2.)))


    @custom_model
    def narrow_line_1(x, amplitude_n1=np.mean(fl_hbeta), amplitude_n2=np.mean(fl_hbeta), amplitude_n3 = np.mean(fl_hbeta), shift_n=0, shift_n2 = 0, sigma_n=1, amplitude_n4 = np.mean(fl_hbeta), amplitude_n5 = np.mean(fl_hbeta)):
        return (abs(amplitude_n1)* 1./(abs(sigma_n)*np.sqrt(2.*np.pi)) * np.exp(-0.5 * ((x - 4959 + shift_n) / abs(sigma_n))**2.)\
        + abs(3.*amplitude_n1)* 1./(abs(sigma_n)*np.sqrt(2.*np.pi)) * np.exp(-0.5 * ((x - 5007 + shift_n) / abs(sigma_n))**2.)\
        + abs(amplitude_n3)* 1./(abs(sigma_n)*np.sqrt(2.*np.pi)) * np.exp(-0.5 * ((x - 4860 + shift_n) / abs(sigma_n))**2.)\
    
        + abs(amplitude_n4)* 1./(abs(sigma_n)*np.sqrt(2.*np.pi)) * np.exp(-0.5 * ((x - 4959 + shift_n2) / abs(sigma_n))**2.)\
        + abs(3.*amplitude_n4)* 1./(abs(sigma_n)*np.sqrt(2.*np.pi)) * np.exp(-0.5 * ((x - 5007 + shift_n2) / abs(sigma_n))**2.))


    @custom_model
    def n_oii(x, amplitude=np.mean(fl_OII_3727), sigma=1., shift=0.):
        return (abs(amplitude)* 1./(abs(sigma)*np.sqrt(2.*np.pi)) * np.exp(-0.5 * ((x - 3727 + shift) / abs(sigma))**2.))

     
    #m_init.sigma_n_0.bounds = (0, 1200./c*5007./2.3458)
    #m_init.sigma_n2_0.bounds = (0, 1200./c*5007./2.3458)
    #m_init.shift_n_0.bounds = (-5, 5)
    #m_init.shift_n2_0.bounds = (-10, 10)     
    #m_init.shift_n2_0.bounds = (-10, 10)
    
    #m_init.sigma_b1_1.bounds = ((1200./c*4860./2.3458, 25000./c*4860./2.3458))   
    #m_init.sigma_b2_1.bounds = ((1200./c*4860./2.3458, 25000./c*4860./2.3458))
    
    ############### Hbeta region###################
    m_init = narrow_line_1() + broad_hbeta()    
    fit = fitting.LevMarLSQFitter()
    m = fit(m_init, wl_hbeta, fl_hbeta, maxiter=10**3)

    if m.sigma_b1_1.value < m.sigma_n_0.value\
    or m.sigma_b2_1.value < m.sigma_n_0.value:
        m_init = narrow_line_1() + broad_hbeta(amplitude_b2=0)
        m_init.amplitude_b2_1.fixed = True
        m = 0
        m = fit(m_init, wl_hbeta, fl_hbeta, maxiter=10**3)
        
    if 4860 - m.shift_b_1.value > 4950\
    or 4860 - m.shift_b2_1.value > 4950:
        m_init = narrow_line_1() + broad_hbeta(amplitude_b1=0, amplitude_b2=0)
        m_init.amplitude_b1_1.fixed = True
        m_init.amplitude_b2_1.fixed = True
        m = 0
        m = fit(m_init, wl_hbeta, fl_hbeta, maxiter=10**3)

    ########
#    m_init = narrow_line_1(amplitude_n1=np.mean(fl_hbeta_norm), amplitude_n2=np.mean(fl_hbeta_norm), amplitude_n3 = np.mean(fl_hbeta_norm))\
#    + broad_hbeta(amplitude_b1=np.mean(fl_hbeta_norm), amplitude_b2=np.mean(fl_hbeta_norm), amplitude_b3=np.mean(fl_hbeta_norm))
#    p = fit(m_init, wl_hbeta_norm, fl_hbeta_norm-1, maxiter=10**7)
    
#    if p.sigma_b1_1.value < p.sigma_n_0.value\
#    or p.sigma_b2_1.value < p.sigma_n_0.value:
#        m_init = narrow_line_1(amplitude_n1=np.mean(fl_hbeta_norm), amplitude_n2=np.mean(fl_hbeta_norm), amplitude_n3 = np.mean(fl_hbeta_norm))\
#        + broad_hbeta(amplitude_b1=np.mean(fl_hbeta_norm), amplitude_b2=0, amplitude_b3=np.mean(fl_hbeta_norm))
#        m_init.amplitude_b2_1.fixed = True
#        p = 0
#        p = fit(m_init, wl_hbeta_norm, fl_hbeta_norm-1, maxiter=10**7)
        
#    if 4860 - p.shift_b_1.value > 4950\
#    or 4860 - p.shift_b2_1.value > 4950:
#        m_init = narrow_line_1(amplitude_n1=np.mean(fl_hbeta_norm), amplitude_n2=np.mean(fl_hbeta_norm), amplitude_n3 = np.mean(fl_hbeta_norm))\
#        + broad_hbeta(amplitude_b1=0, amplitude_b2=0, amplitude_b3=np.mean(fl_hbeta_norm))
#        m_init.amplitude_b1_1.fixed = True
#        m_init.amplitude_b2_1.fixed = True
#        p = 0
#        p = fit(m_init, wl_hbeta_norm, fl_hbeta_norm-1, maxiter=10**7)

    ############### O2 region###################

    m_init2 = n_oii()
    m2 = fit(m_init2, wl_OII_3727, fl_OII_3727, maxiter=10**3)
    
#    m_init2 = n_oii(amplitude=np.mean(fl_OII_3727_norm))
#    p2 = fit(m_init2, wl_OII_3727_norm, fl_OII_3727_norm-1, maxiter=10**7)

    
    #m_init3.sigma_n_0.bounds = (0, 1000./c*6563./2.3458) #constraint parameters
    #m_init3.sigma_n2_0.bounds = (0, 1000./c*6717./2.3458)
    
    #m_init3.shift_n_0.bounds = (-5, 5)
    #m_init3.shift_n2_0.bounds = (-5, 5)

#m_init3.sigma_b1_1.bounds = (1200./c*6563./2.3458, np.inf)
#m_init3.sigma_b2_1.bounds = (1200./c*6563./2.3458, np.inf)

    ############### Halpha region###################

    @custom_model
    def narrow_line_2(x, amplitude_n1=np.mean(fl_halpha), amplitude_n2=np.mean(fl_halpha), amplitude_n3 = np.mean(fl_halpha), amplitude_n4 = np.mean(fl_halpha), amplitude_n5 = np.mean(fl_halpha), shift_n=0, sigma_n=1, shift_n2=0, sigma_n2=m.sigma_n_0.value, shift_n3=0, sigma_n3=1):        
        return (abs(amplitude_n1)* 1./(abs(sigma_n2)*np.sqrt(2.*np.pi)) * np.exp(-0.5 * ((x - 6563 + shift_n2) / abs(sigma_n2))**2.)\
    
        + abs(amplitude_n2)* 1./(abs(sigma_n2)*np.sqrt(2.*np.pi)) * np.exp(-0.5 * ((x - 6548 + shift_n2) / abs(sigma_n2))**2.)\
        + abs((2.96)*amplitude_n2)* 1./(abs(sigma_n2)*np.sqrt(2.*np.pi)) * np.exp(-0.5 * ((x - 6584 + shift_n2) / abs(sigma_n2))**2.)\
        
        + abs(amplitude_n4)* 1./(abs(sigma_n2)*np.sqrt(2.*np.pi)) * np.exp(-0.5 * ((x - 6717 + shift_n2) / abs(sigma_n2))**2.)\
        + abs(amplitude_n5)* 1./(abs(sigma_n2)*np.sqrt(2.*np.pi)) * np.exp(-0.5 * ((x - 6731 + shift_n2) / abs(sigma_n2))**2.))



    @custom_model
    def broad_halpha(x, amplitude_b1=np.mean(fl_halpha), sigma_b1=1.2, shift_b=0, amplitude_b2=np.mean(fl_halpha), sigma_b2=1.2, shift_b2=0):
        return ((abs(amplitude_b1)* 1./(abs(sigma_b1)*np.sqrt(2.*np.pi)) * np.exp(-0.5 * ((x - 6563 + shift_b) / abs(sigma_b1))**2.))\
        + (abs(amplitude_b2)* 1./(abs(sigma_b2)*np.sqrt(2.*np.pi)) * np.exp(-0.5 * ((x - 6563 + shift_b2) / abs(sigma_b2))**2.)))


    m_init3 = narrow_line_2() + broad_halpha()
    m3 = fit(m_init3, wl_halpha, fl_halpha, maxiter=10**3)


#    if m3.sigma_b1_1.value < m3.sigma_n2_0.value \
#    or m3.sigma_b2_1.value < m3.sigma_n2_0.value :
#        m_init3 = narrow_line_2() + broad_halpha(amplitude_b2 = 0)
#        m_init3.amplitude_b2_1.fixed = True
#        m3 = 0
#        m3 = fit(m_init3, wl_halpha, fl_halpha, maxiter=10**7)
        
#    if 6563. - m3.shift_b_1.value > 6650.\
#    or 6563. - m3.shift_b2_1.value > 6650.:
#        m_init3 = narrow_line_2() + broad_halpha(amplitude_b1 = 0, amplitude_b2 = 0)
#        m_init3.amplitude_b1_1.fixed = True        
#        m_init3.amplitude_b2_1.fixed = True
#        m3 = 0
 #       m3 = fit(m_init3, wl_halpha, fl_halpha, maxiter=10**7)

    if m3.sigma_b1_1.value < m3.sigma_n2_0.value\
    or m3.sigma_b2_1.value < m3.sigma_n2_0.value:
#    or 6563 - m3.shift_b_1.value > 6660\
#    or 6563 - m3.shift_b2_1.value > 6660:
        m_init3 = narrow_line_2(sigma_n2 = m.sigma_n_0.value, shift_n2=m.shift_n_0.value) + broad_halpha(amplitude_b2 = 0)
        m_init3.amplitude_b2_1.fixed = True        
        m_init3.sigma_n2_0.fixed = True
        m_init3.shift_n2_0.fixed = True
        m3 = 0
        m3 = fit(m_init3, wl_halpha, fl_halpha, maxiter=10**3)



    
    
#    m_init3 = narrow_line_2(amplitude_n1=np.mean(fl_halpha_norm), amplitude_n2=np.mean(fl_halpha_norm), amplitude_n3 = np.mean(fl_halpha_norm), amplitude_n4 = np.mean(fl_halpha_norm), amplitude_n5 = np.mean(fl_halpha_norm))\
#    + broad_halpha(amplitude_b1=np.mean(fl_halpha_norm), amplitude_b2=np.mean(fl_halpha_norm))    
#    p3 = fit(m_init3, wl_halpha_norm, fl_halpha_norm-1, maxiter=10**7)


    def gauss(x, x0, amplitude, sigma, shift):
        return (abs(amplitude)* 1./(abs(sigma)*np.sqrt(2.*np.pi)) * np.exp(-0.5 * ((x - x0 + shift) / abs(sigma))**2.))


    b_hbeta_1 = []
    b_hbeta_2 = []
    b_hbeta_3 = []

    n_hbeta = []
    n_oiii_4959 = []
    n_oiii_5007 = []


    n_oii_3727 = []

    b_halpha = []
    n_halpha = []
    n_nii_6548 = []
    n_nii_6584 = []
    n_sii_6717 = []
    n_sii_6731 = []
    #################### hati-hati ###################
    b_hbeta_1_norm = []
    b_hbeta_2_norm = []
    b_hbeta_3_norm = []

    n_hbeta_norm = []
    n_oiii_4959_norm = []
    n_oiii_5007_norm = []


    n_oii_3727_norm = []

    b_halpha_norm = []
    n_halpha_norm = []
    n_nii_6548_norm = []
    n_nii_6584_norm = []
    n_sii_6717_norm = []
    n_sii_6731_norm = []


    for i in range(len(wl_halpha)):
        b_halpha.append(gauss(wl_halpha[i], 6563., m3.amplitude_b1_1.value, m3.sigma_b1_1.value, m3.shift_b_1.value)\
        + gauss(wl_halpha[i], 6563., m3.amplitude_b2_1.value, m3.sigma_b2_1.value, m3.shift_b2_1.value))
        
        n_halpha.append(gauss(wl_halpha[i], 6563., m3.amplitude_n1_0.value, m3.sigma_n2_0.value, m3.shift_n2_0.value))

        n_nii_6548.append(gauss(wl_halpha[i], 6548, m3.amplitude_n2_0.value, m3.sigma_n2_0.value, m3.shift_n2_0.value))
        n_nii_6584.append(gauss(wl_halpha[i], 6584., (2.96)*m3.amplitude_n2_0.value, m3.sigma_n2_0.value, m3.shift_n2_0.value))
    
        n_sii_6717.append(gauss(wl_halpha[i], 6717., m3.amplitude_n4_0.value, m3.sigma_n2_0.value, m3.shift_n2_0.value))
        n_sii_6731.append(gauss(wl_halpha[i], 6731., m3.amplitude_n5_0.value, m3.sigma_n2_0.value, m3.shift_n2_0.value))
###################################################### norm here ########################################
#        b_halpha_norm.append(gauss(wl_halpha_norm[i], 6563., p3.amplitude_b1_1.value, p3.sigma_b1_1.value, p3.shift_b_1.value)\
#        + gauss(wl_halpha_norm[i], 6563., p3.amplitude_b2_1.value, p3.sigma_b2_1.value, p3.shift_b2_1.value))
        
#        n_halpha_norm.append(gauss(wl_halpha_norm[i], 6563., p3.amplitude_n1_0.value, p3.sigma_n2_0.value, p3.shift_n2_0.value))

#        n_nii_6548_norm.append(gauss(wl_halpha_norm[i], 6548, p3.amplitude_n2_0.value, p3.sigma_n2_0.value, p3.shift_n2_0.value))
#        n_nii_6584_norm.append(gauss(wl_halpha_norm[i], 6584., (2.96)*p3.amplitude_n2_0.value, p3.sigma_n2_0.value, p3.shift_n2_0.value))
    
#        n_sii_6717_norm.append(gauss(wl_halpha_norm[i], 6717., p3.amplitude_n4_0.value, p3.sigma_n2_0.value, p3.shift_n2_0.value))
#        n_sii_6731_norm.append(gauss(wl_halpha_norm[i], 6731., p3.amplitude_n5_0.value, p3.sigma_n2_0.value, p3.shift_n2_0.value))


#for i in range(len(wl_OII_3727)):
#    n_oii_3727.append(-300+gauss(wl_OII_3727[i], 3727., m2.amplitude.value, m2.sigma.value, m2.shift.value))


    for i in range(len(wl_hbeta)):
        n_oiii_4959.append(gauss(wl_hbeta[i], 4959., m.amplitude_n1_0.value, m.sigma_n_0.value, m.shift_n_0.value)\
        + gauss(wl_hbeta[i], 4959., m.amplitude_n4_0.value, m.sigma_n_0.value, m.shift_n2_0.value))
        
        n_oiii_5007.append(gauss(wl_hbeta[i], 5007., 3.*m.amplitude_n1_0.value, m.sigma_n_0.value, m.shift_n_0.value)\
        + gauss(wl_hbeta[i], 5007., 3.*m.amplitude_n4_0.value, m.sigma_n_0.value, m.shift_n2_0.value))
        
        n_hbeta.append(gauss(wl_hbeta[i], 4860., m.amplitude_n3_0.value, m.sigma_n_0.value, m.shift_n_0.value))
    
        b_hbeta_1.append(gauss(wl_hbeta[i], 4860., m.amplitude_b1_1.value, m.sigma_b1_1.value, m.shift_b_1.value)\
        + gauss(wl_hbeta[i], 4860., m.amplitude_b2_1.value, m.sigma_b2_1.value, m.shift_b2_1.value))
        #b_hbeta_2.append(gauss(wl_hbeta[i], 4860., m.amplitude_b2_1.value, m.sigma_b1_1.value, m.shift_b_1.value))
    #b_hbeta_3.append(-300 + gauss(wl_hbeta[i], 4860., m.amplitude_b3_1.value, m.sigma_b3_1.value, m.shift_b_1.value))
        
        #########################################################

#        n_oiii_4959_norm.append(gauss(wl_hbeta_norm[i], 4959., p.amplitude_n1_0.value, p.sigma_n_0.value, p.shift_n_0.value)\
#        + gauss(wl_hbeta_norm[i], 4959., p.amplitude_n4_0.value, p.sigma_n_0.value, p.shift_n2_0.value))
        
#        n_oiii_5007_norm.append(gauss(wl_hbeta_norm[i], 5007., 3.*p.amplitude_n1_0.value, p.sigma_n_0.value, p.shift_n_0.value)\
#        + gauss(wl_hbeta_norm[i], 5007., 3.*p.amplitude_n4_0.value, p.sigma_n_0.value, p.shift_n2_0.value))
        
#        n_hbeta_norm.append(gauss(wl_hbeta_norm[i], 4860., p.amplitude_n3_0.value, p.sigma_n_0.value, p.shift_n_0.value))
    
#        b_hbeta_1_norm.append(gauss(wl_hbeta_norm[i], 4860., p.amplitude_b1_1.value, p.sigma_b1_1.value, p.shift_b_1.value)\
#        + gauss(wl_hbeta_norm[i], 4860., p.amplitude_b2_1.value, p.sigma_b2_1.value, p.shift_b2_1.value))        
        

    flux_OII_3727.append(integrate.trapz(m2(wl_OII_3727)))
    flux_nHbeta.append(integrate.trapz(n_hbeta))
    flux_bHbeta.append(integrate.trapz(b_hbeta_1))
    flux_OIII_4959.append(integrate.trapz(n_oiii_4959))
    flux_OIII_5007.append(integrate.trapz(n_oiii_5007))    
    flux_nHalpha.append(integrate.trapz(n_halpha))    
    flux_bHalpha.append(integrate.trapz(b_halpha))
    flux_NII_6548.append(integrate.trapz(n_nii_6548))
    flux_NII_6584.append(integrate.trapz(n_nii_6584))    
    flux_SII_6717.append(integrate.trapz(n_sii_6717))
    flux_SII_6731.append(integrate.trapz(n_sii_6731))
    
    ##################################
#    EW_OII_3727.append(integrate.trapz(p2(wl_OII_3727)))# not yet
    EW_nHbeta.append(integrate.trapz(n_hbeta_norm))
    EW_bHbeta.append(integrate.trapz(b_hbeta_1_norm))
    EW_OIII_4959.append(integrate.trapz(n_oiii_4959_norm))
    EW_OIII_5007.append(integrate.trapz(n_oiii_5007_norm))    
    EW_nHalpha.append(integrate.trapz(n_halpha_norm))    
    EW_bHalpha.append(integrate.trapz(b_halpha_norm))
    EW_NII_6548.append(integrate.trapz(n_nii_6548_norm))
    EW_NII_6584.append(integrate.trapz(n_nii_6584_norm))    
    EW_SII_6717.append(integrate.trapz(n_sii_6717_norm))
    EW_SII_6731.append(integrate.trapz(n_sii_6731_norm))

    fwhm_OII_3727.append((abs(m2.sigma.value)) * 2.3548 * c/(3727.+ m2.shift.value))
    
    fwhm_nHbeta.append((abs(m.sigma_n_0.value) ) * 2.3548 * c/(4860.+ m.shift_n_0.value))
    fwhm_bHbeta.append((abs(m.sigma_b1_1.value) ) * 2.3548 * c/(4860.+ m.shift_b_1.value))
    fwhm_bHbeta2.append((abs(m.sigma_b2_1.value) ) * 2.3548 * c/(4860.+ m.shift_b2_1.value))
    
    fwhm_OIII_4959.append((abs(m.sigma_n_0.value) ) * 2.3548 * c/(4959.+ m.shift_n_0.value))
    fwhm_OIII_5007.append((abs(m.sigma_n_0.value) ) * 2.3548 * c/(5007.+ m.shift_n_0.value))

    shift_OIII_a.append(((m.shift_n_0.value) ) * c/(5007.+ m.shift_n_0.value))
    shift_OIII_b.append(((m.shift_n2_0.value) ) * c/(5007.+ m.shift_n2_0.value))

    fwhm_nHalpha.append((abs(m3.sigma_n2_0.value) ) * 2.3548 * c/(6563.+ m3.shift_n2_0.value))
    fwhm_bHalpha.append((abs(m3.sigma_b1_1.value) ) * 2.3548 * c/(6563. + m3.shift_b_1.value))
    fwhm_bHalpha2.append((abs(m3.sigma_b2_1.value) ) * 2.3548 * c/(6563. + m3.shift_b2_1.value))    
    
    fwhm_NII_6548.append((abs(m3.sigma_n2_0.value) ) * 2.3548 * c/(6548.+ m3.shift_n2_0.value))
    fwhm_NII_6584.append((abs(m3.sigma_n2_0.value) ) * 2.3548 * c/(6584.+ m3.shift_n2_0.value))
    
    fwhm_SII_6717.append((abs(m3.sigma_n2_0.value) ) * 2.3548 * c/(6717.+ m3.shift_n2_0.value))
    fwhm_SII_6731.append((abs(m3.sigma_n2_0.value) ) * 2.3548 * c/(6731.+ m3.shift_n2_0.value))
    
    ##############################################
#    fwhm_OII_3727_norm.append((abs(p2.sigma.value)) * 2.3548 * c/(3727.+ p2.shift.value))
    
#    fwhm_nHbeta_norm.append((abs(p.sigma_n_0.value) ) * 2.3548 * c/(4860.+ p.shift_n_0.value))
#    fwhm_bHbeta_norm.append((abs(p.sigma_b1_1.value) ) * 2.3548 * c/(4860.+ p.shift_b_1.value))
#    fwhm_bHbeta2_norm.append((abs(p.sigma_b2_1.value) ) * 2.3548 * c/(4860.+ p.shift_b2_1.value))
    
#    fwhm_OIII_4959_norm.append((abs(p.sigma_n_0.value) ) * 2.3548 * c/(4959.+ p.shift_n_0.value))
#    fwhm_OIII_5007_norm.append((abs(p.sigma_n_0.value) ) * 2.3548 * c/(5007.+ p.shift_n_0.value))
    
#    fwhm_nHalpha_norm.append((abs(p3.sigma_n2_0.value) ) * 2.3548 * c/(6563.+ p3.shift_n2_0.value))
#    fwhm_bHalpha_norm.append((abs(p3.sigma_b1_1.value) ) * 2.3548 * c/(6563. + p3.shift_b_1.value))
#    fwhm_bHalpha2_norm.append((abs(p3.sigma_b2_1.value) ) * 2.3548 * c/(6563. + p3.shift_b2_1.value))    
    
#    fwhm_NII_6548_norm.append((abs(p3.sigma_n2_0.value) ) * 2.3548 * c/(6548.+ p3.shift_n2_0.value))
#    fwhm_NII_6584_norm.append((abs(p3.sigma_n2_0.value) ) * 2.3548 * c/(6584.+ p3.shift_n2_0.value))
    
#    fwhm_SII_6717_norm.append((abs(p3.sigma_n2_0.value) ) * 2.3548 * c/(6717.+ p3.shift_n2_0.value))
#    fwhm_SII_6731_norm.append((abs(p3.sigma_n2_0.value) ) * 2.3548 * c/(6731.+ p3.shift_n2_0.value))
    
    o = open('calculation/line_flux.csv', 'a')
    f = open('calculation/line_fwhm.csv', 'a')

    #for i in range(len(flux_nHbeta)):
    o.write(str(flux_OII_3727[q]) + '\t' + str(flux_nHbeta[q]) + '\t' + str(flux_bHbeta[q])\
     + '\t' + str(flux_OIII_4959[q]) + '\t' + str(flux_OIII_5007[q])\
     + '\t' + str(flux_nHalpha[q]) + '\t' + str(flux_bHalpha[q])\
     + '\t' + str(flux_NII_6548[q]) + '\t' + str(flux_NII_6584[q])\
     + '\t' + str(flux_SII_6717[q]) + '\t' + str(flux_SII_6731[q]) + '\n')
    
    #for i in range(len(fwhm_nHbeta)):
    f.write(str(fwhm_OII_3727[q]) + '\t' + str(fwhm_nHbeta[q]) + '\t' + str(fwhm_bHbeta[q]) + '\t' + str(fwhm_bHbeta2[q])\
     + '\t' + str(fwhm_OIII_4959[q]) + '\t' + str(fwhm_OIII_5007[q])\
     + '\t' + str(fwhm_nHalpha[q]) + '\t' + str(fwhm_bHalpha[q]) + '\t' + str(fwhm_bHalpha2[q])\
     + '\t' + str(fwhm_NII_6548[q]) + '\t' + str(fwhm_NII_6584[q])\
     + '\t' + str(fwhm_SII_6717[q]) + '\t' + str(fwhm_SII_6731[q]) + '\t' + str(shift_OIII_a[q]) + '\t' + str(shift_OIII_b[q]) + '\n')
     
    o.close()
    f.close()
    
    v = open('calculation/line_ew.csv', 'a')
    h = open('calculation/line_norm_fwhm.csv', 'a')
    
#    v.write(str(EW_OII_3727[q]) + '\t' + str(EW_nHbeta[q]) + '\t' + str(EW_bHbeta[q])\
#     + '\t' + str(EW_OIII_4959[q]) + '\t' + str(EW_OIII_5007[q])\
#     + '\t' + str(EW_nHalpha[q]) + '\t' + str(EW_bHalpha[q])\
#     + '\t' + str(EW_NII_6548[q]) + '\t' + str(EW_NII_6584[q])\
#     + '\t' + str(EW_SII_6717[q]) + '\t' + str(EW_SII_6731[q]) + '\n')
    
    #for i in range(len(fwhm_nHbeta)):
#    h.write(str(fwhm_OII_3727_norm[q]) + '\t' + str(fwhm_nHbeta_norm[q]) + '\t' + str(fwhm_bHbeta_norm[q]) + '\t' + str(fwhm_bHbeta2_norm[q])\
#     + '\t' + str(fwhm_OIII_4959_norm[q]) + '\t' + str(fwhm_OIII_5007_norm[q])\
#     + '\t' + str(fwhm_nHalpha_norm[q]) + '\t' + str(fwhm_bHalpha_norm[q]) + '\t' + str(fwhm_bHalpha2_norm[q])\
#     + '\t' + str(fwhm_NII_6548_norm[q]) + '\t' + str(fwhm_NII_6584_norm[q])\
#     + '\t' + str(fwhm_SII_6717_norm[q]) + '\t' + str(fwhm_SII_6731_norm[q]) + '\n')
    
    v.close()
    h.close()
    
    
    

    plt.figure(j)
    plt.plot(wl_hbeta, fl_hbeta, 'b-', label='Observed Spectra')
    plt.plot(wl_hbeta, m(wl_hbeta), 'r-', label='Model')

    plt.plot(wl_hbeta, n_oiii_4959, 'g-')
    plt.plot(wl_hbeta, n_oiii_5007, 'g-')
    plt.plot(wl_hbeta, n_hbeta, 'c-')
    plt.plot(wl_hbeta, b_hbeta_1, 'k-')
#plt.plot(wl_hbeta, b_hbeta_2)
#plt.plot(wl_hbeta, b_hbeta_3)



#    plt.plot(wl_halpha, fl_halpha, 'b-', label='Observed Spectra')
#    plt.plot(wl_halpha, m3(wl_halpha), 'r-', label='Model')
#    plt.plot(wl_halpha, b_halpha, 'k-')
#    plt.plot(wl_halpha, n_halpha, 'c-')

#    plt.plot(wl_halpha, n_nii_6548, 'y')
#    plt.plot(wl_halpha, n_nii_6584, 'y')

#    plt.plot(wl_halpha, n_sii_6717, 'g')
#    plt.plot(wl_halpha, n_sii_6731, 'g')
    

#    plt.plot(wl_OII_3727, fl_OII_3727, 'b-', label='Observed Spectra')
#    plt.plot(wl_OII_3727, m2(wl_OII_3727), 'r-', label='Model')
    #print 'Chi square = ', chisquare(fl_hbeta, m(wl_hbeta))
    #print 'Chi square = ', chisquare(fl_OII_3727, m2(wl_OII_3727))#[0]/(len(fl_OII_3727) - 3)

    plt.xlabel('Wavelength ($\AA$)')
    plt.ylabel('Flux Density ($\\rm \\times 10^{-17} \ erg \ cm^{-2} s^{-1} \AA^{-1}$)')
    plt.title('Emission Lines Fitting\nspec-' + plate[j] + '-' + mjd[j] + '-' + fiberid[j] + '.fits')
    plt.savefig('figure3/fig3_'+ str(j) +'_spec-' + plate[j] + '-' + mjd[j] + '-' + fiberid[j])
    plt.legend(loc='best')
#    plt.show()
    
    
############## Equivalent Width ##########################    
    
    
    
    
    
    
    
    
    
    
    print 'j =', j
#    plt.show()
    plt.close('all')

    q+=1






