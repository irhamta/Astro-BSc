# -*- coding: utf-8 -*-
"""
Created on Mon Oct  5 22:34:32 2015

@author: irham
"""

import numpy as np
import matplotlib.pyplot as plt
from astropy.modeling import models, fitting
from astropy.modeling.models import custom_model
from scipy import interpolate, integrate

#J = np.loadtxt('notes/skipped_j_opt_fix.csv', skiprows=1, dtype=int)
f = open('calculation/continuum_flux.csv', 'w')
f.write('F_5100\n')
f.close()

o = open('calculation/Fe_II_flux_2.csv', 'w')
o.write('flux_F\tflux_G\tflux_P\tflux_S\tflux_Z\tsigma\n')
o.close()


q = 0
'''========== Fe II Template and Spectra Data =========='''

F = np.loadtxt('FeII_template_4000_5500/FeII_rel_int/F_rel_int.txt')
G = np.loadtxt('FeII_template_4000_5500/FeII_rel_int/G_rel_int.txt')
P = np.loadtxt('FeII_template_4000_5500/FeII_rel_int/P_rel_int.txt')
S = np.loadtxt('FeII_template_4000_5500/FeII_rel_int/S_rel_int.txt')
Z = np.loadtxt('FeII_template_4000_5500/FeII_rel_int/IZw1_rel_int.txt')
#agn = np.loadtxt('shifted_redcor/out_spec-0542-51993-0560.fits')

flux_F = []
flux_G = []
flux_P = []
flux_S = []
flux_Z = []

EW_FeII = []

fwhm_F = []
fwhm_G = []
fwhm_P = []
fwhm_S = []
fwhm_Z = []

sigma_fe = []


'''========== Galaxy Template =========='''


gal_template = np.array((\
np.loadtxt('eigen_spectra/bc03/templates/cst_6gyr_z02.spec'),\
np.loadtxt('eigen_spectra/bc03/templates/cst_6gyr_z05.spec'),\
np.loadtxt('eigen_spectra/bc03/templates/cst_6gyr_z008.spec'),\
np.loadtxt('eigen_spectra/bc03/templates/ssp_1.4Gyr_z02.spec'),\
np.loadtxt('eigen_spectra/bc03/templates/ssp_1.4Gyr_z05.spec'),\
np.loadtxt('eigen_spectra/bc03/templates/ssp_1.4Gyr_z008.spec'),\
np.loadtxt('eigen_spectra/bc03/templates/ssp_2.5Gyr_z02.spec'),\
np.loadtxt('eigen_spectra/bc03/templates/ssp_2.5Gyr_z05.spec'),\
np.loadtxt('eigen_spectra/bc03/templates/ssp_2.5Gyr_z008.spec'),\
np.loadtxt('eigen_spectra/bc03/templates/ssp_5Gyr_z02.spec'),\
np.loadtxt('eigen_spectra/bc03/templates/ssp_5Gyr_z05.spec'),\
np.loadtxt('eigen_spectra/bc03/templates/ssp_5Gyr_z008.spec'),\
np.loadtxt('eigen_spectra/bc03/templates/ssp_5Myr_z02.spec'),\
np.loadtxt('eigen_spectra/bc03/templates/ssp_5Myr_z05.spec'),\
np.loadtxt('eigen_spectra/bc03/templates/ssp_5Myr_z008.spec'),\
np.loadtxt('eigen_spectra/bc03/templates/ssp_11Gyr_z02.spec'),\
np.loadtxt('eigen_spectra/bc03/templates/ssp_11Gyr_z05.spec'),\
np.loadtxt('eigen_spectra/bc03/templates/ssp_11Gyr_z008.spec'),\
np.loadtxt('eigen_spectra/bc03/templates/ssp_25Myr_z02.spec'),\
np.loadtxt('eigen_spectra/bc03/templates/ssp_25Myr_z05.spec'),\
np.loadtxt('eigen_spectra/bc03/templates/ssp_25Myr_z008.spec'),\
np.loadtxt('eigen_spectra/bc03/templates/ssp_100Myr_z02.spec'),\
np.loadtxt('eigen_spectra/bc03/templates/ssp_100Myr_z05.spec'),\
np.loadtxt('eigen_spectra/bc03/templates/ssp_100Myr_z008.spec'),\
np.loadtxt('eigen_spectra/bc03/templates/ssp_290Myr_z02.spec'),\
np.loadtxt('eigen_spectra/bc03/templates/ssp_290Myr_z05.spec'),\
np.loadtxt('eigen_spectra/bc03/templates/ssp_290Myr_z008.spec'),\
np.loadtxt('eigen_spectra/bc03/templates/ssp_640Myr_z02.spec'),\
np.loadtxt('eigen_spectra/bc03/templates/ssp_640Myr_z05.spec'),\
np.loadtxt('eigen_spectra/bc03/templates/ssp_640Myr_z008.spec'),\
np.loadtxt('eigen_spectra/bc03/templates/ssp_900Myr_z02.spec'),\
np.loadtxt('eigen_spectra/bc03/templates/ssp_900Myr_z05.spec'),\
np.loadtxt('eigen_spectra/bc03/templates/ssp_900Myr_z008.spec'),\
np.loadtxt('eigen_spectra/bc03/templates/t5e9_12gyr_z02.spec'),\
np.loadtxt('eigen_spectra/bc03/templates/t5e9_12gyr_z05.spec'),\
np.loadtxt('eigen_spectra/bc03/templates/t5e9_12gyr_z008.spec'),\
np.loadtxt('eigen_spectra/bc03/templates/t9e9_12gyr_z02.spec'),\
np.loadtxt('eigen_spectra/bc03/templates/t9e9_12gyr_z05.spec'),\
np.loadtxt('eigen_spectra/bc03/templates/t9e9_12gyr_z008.spec'),\
))

gal_template_2 = np.array((\
np.loadtxt('eigen_spectra/galaxyKL_eigSpec_1.dat'),\
np.loadtxt('eigen_spectra/galaxyKL_eigSpec_2.dat'),\
np.loadtxt('eigen_spectra/galaxyKL_eigSpec_3.dat'),\
))



gal_model = []
for i in range(len(gal_template_2)):
    #print i
    u = []
    v = []
    for j in range(len(gal_template_2[i])):
        if gal_template_2[i][j][0] >= 2000. and gal_template_2[i][j][0] <= 8000.:
            u.append(gal_template_2[i][j][0])
            v.append(gal_template_2[i][j][1])
        else:
            continue            
            
    gal_model.append(interpolate.interp1d(u, v, kind='slinear'))

'''
plt.plot(gal_template[i, :, 0], gal_template[i, :, 1], 'ko')
plt.plot(u, gal_model[i](u), 'r-')
plt.ylim(0, 2)
plt.xlim(min(u), max(u))
plt.show()
'''





xsamp = []
ysamp = []

wl_cont = []
fl_cont = []

sample = np.loadtxt('MAIN_SAMPLE_irhamta.csv', skiprows=1, dtype=str, delimiter=',')

plate = []
mjd = []
fiberid = []
z = []
e_bv = []

skip = np.loadtxt('notes/skip.csv', delimiter=',', skiprows=1)

for j in range(100):#len(sample)):
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
    
    if j < 90:
        print 'skipped, j = ', j
        continue    
    
    
    if j in skip[:, 1] or j == skip[0, 4]: #or j < 3455:#for until redshift 3.5
        f = open('calculation/continuum_flux.csv', 'a')
        #for i in range(len(fl_cont)):
        f.write(str(0) + '\n')

        f.close()
        
        o = open('calculation/Fe_II_flux_2.csv', 'a')
        o.write(str(0) + '\t' + str(0) + '\t' + str(0)\
        + '\t' + str(0) + '\t' + str(0) + '\t' + str(0) + '\n')
    
        o.close()

#        o = open('calculation/Fe_II_ew_2.csv', 'a')
        
#        o.write(str(0) + '\t' + str(0) + '\t' + str(0) + '\n')# + str(2.5066*m2.sigma.value) + '\n')
    
#        o.close()
        
        
        print 'skipped, j = ', j
        continue
    
    agn = np.loadtxt('shifted_redcor/out_spec-' + plate[j] + '-' + mjd[j] + '-' + fiberid[j] + '.fits')
    


#'''========== Fitting Near Hbeta ==========='''
    wl_plot = []
    fl_plot = []

    wl_hbeta_cont = []
    fl_hbeta_cont = []
    wl_hbeta = []
    fl_hbeta = []


    wl_halpha_cont = []
    fl_halpha_cont = []
    wl_halpha = []
    fl_halpha = []


    wl_OII_cont = []
    fl_OII_cont = []
    wl_OII = []
    fl_OII = []

    for i in range(len(agn)):#Hbeta window
        
        if agn[i][0] >= 3500 and agn[i][0] <= 7500:
            wl_plot.append(agn[i][0])         
            fl_plot.append(agn[i][1])
        
        if agn[i][0] >= 5095 and agn[i][0] <=5105:
            wl_cont.append(agn[i][0])
#            fl_cont.append(agn[i][1])

        if agn[i][0] >= 3455 and agn[i][0] <= 3490\
        or agn[i][0] >= 3500 and agn[i][0] <= 3550\
        or agn[i][0] >= 3600 and agn[i][0] <= 3700\
        or agn[i][0] >= 4200 and agn[i][0] <= 4250\
        or agn[i][0] >= 4400 and agn[i][0] <= 4460\
        or agn[i][0] >= 4490 and agn[i][0] <= 4600\
        or agn[i][0] >= 5130 and agn[i][0] <= 5680\
        or agn[i][0] >= 5750 and agn[i][0] <= 5830\
        or agn[i][0] >= 5920 and agn[i][0] <= 6050\
        or agn[i][0] >= 6130 and agn[i][0] <= 6200:    

#        if agn[i][0] >= 3650 and agn[i][0] <= 3717\
#        or agn[i][0] >= 3790 and agn[i][0] <= 3810\
#        or agn[i][0] >= 4435 and agn[i][0] <= 4700\
#        or agn[i][0] >= 5100 and agn[i][0] <= 5535\
#        or agn[i][0] >= 6000 and agn[i][0] <= 6250\
#        or agn[i][0] >= 6800 and agn[i][0] <= 7000\
#        :
            xsamp.append(agn[i][0])
            ysamp.append(agn[i][1])                
    
        if agn[i][0] >= 4435 and agn[i][0] <=5535:
            wl_hbeta.append(agn[i][0])
            fl_hbeta.append(agn[i][1])
        elif agn[i][0] >= 6000 and agn[i][0] <=7000:
            wl_halpha.append(agn[i][0])
            fl_halpha.append(agn[i][1])
        elif agn[i][0] >= 3650 and agn[i][0] <=3810:#3747:
            wl_OII.append(agn[i][0])
            fl_OII.append(agn[i][1])
        else:
            continue

    #if min(wl_OII) > 3717:
    #    J.append(j)
    #    print 'skipped, j = ', j
    #    continue
    wl_fe_cont = []
    fl_fe_cont = []
    


    for i in range(len(agn)):#Hbeta continuum window 

        if agn[i][0] >= 4150 and agn[i][0] <= 4250\
        or agn[i][0] >= 4450 and agn[i][0] <= 4600\
        or agn[i][0] >= 5100 and agn[i][0] <= 5500:
            wl_fe_cont.append(agn[i][0])
            fl_fe_cont.append(agn[i][1])


        if agn[i][0] >= 4435 and agn[i][0] <=4700\
        or agn[i][0] >= 5100 and agn[i][0] <=5535:
            wl_hbeta_cont.append(agn[i][0])
            fl_hbeta_cont.append(agn[i][1])
        
        elif agn[i][0] >= 6000 and agn[i][0] <=6250\
        or agn[i][0] >= 6800 and agn[i][0] <=7000: 
            wl_halpha_cont.append(agn[i][0])
            fl_halpha_cont.append(agn[i][1])
    
        elif agn[i][0] >= 3650 and agn[i][0] <= 3717\
        or agn[i][0] >= 3790 and agn[i][0] <= 3810:
            wl_OII_cont.append(agn[i][0])
            fl_OII_cont.append(agn[i][1])
        
        else:
            continue


    @custom_model
    def gal_mod(x, a0=0*min(fl_hbeta_cont),\
                a1=0*min(fl_hbeta_cont),\
                a2=0*min(fl_hbeta_cont),\
                a3=0*min(fl_hbeta_cont),\
                a4=0*min(fl_hbeta_cont),\
                a5=0*min(fl_hbeta_cont),\
                a6=0*min(fl_hbeta_cont),\
                a7=0*min(fl_hbeta_cont),\
                a8=0*min(fl_hbeta_cont),\
                a9=0*min(fl_hbeta_cont),\
                a10=0*min(fl_hbeta_cont),\
                a11=0*min(fl_hbeta_cont),\
                a12=0*min(fl_hbeta_cont),\
                a13=0*min(fl_hbeta_cont),\
                a14=0*min(fl_hbeta_cont),\
                a15=0*min(fl_hbeta_cont),\
                a16=0*min(fl_hbeta_cont),\
                a17=0*min(fl_hbeta_cont),\
                a18=0*min(fl_hbeta_cont),\
                a19=0*min(fl_hbeta_cont),\
                a20=0*min(fl_hbeta_cont),\
                a21=0*min(fl_hbeta_cont),\
                a22=0*min(fl_hbeta_cont),\
                a23=0*min(fl_hbeta_cont),\
                a24=0*min(fl_hbeta_cont),\
                a25=0*min(fl_hbeta_cont),\
                a26=0*min(fl_hbeta_cont),\
                a27=0*min(fl_hbeta_cont),\
                a28=0*min(fl_hbeta_cont),\
                a29=0*min(fl_hbeta_cont),\
                a30=0*min(fl_hbeta_cont),\
                a31=0*min(fl_hbeta_cont),\
                a32=0*min(fl_hbeta_cont),\
                a33=0*min(fl_hbeta_cont),\
                a34=0*min(fl_hbeta_cont),\
                a35=0*min(fl_hbeta_cont),\
                a36=0*min(fl_hbeta_cont),\
                a37=0*min(fl_hbeta_cont),\
                a38=0*min(fl_hbeta_cont),\
                ):
        
        return (\
        abs(a0)*gal_model[0](x)\
        + abs(a1)*gal_model[1](x)\
        + abs(a2)*gal_model[2](x)\
#        + abs(a3)*gal_model[3](x)\
#        + abs(a4)*gal_model[4](x)\
#        + abs(a5)*gal_model[5](x)\
#        + abs(a6)*gal_model[6](x)\
#        + abs(a7)*gal_model[7](x)\
#        + abs(a8)*gal_model[8](x)\
#        + abs(a9)*gal_model[9](x)\
#        + abs(a10)*gal_model[10](x)\
#        + abs(a11)*gal_model[11](x)\
#        + abs(a12)*gal_model[12](x)\
#        + abs(a13)*gal_model[13](x)\
#        + abs(a14)*gal_model[14](x)\
#        + abs(a15)*gal_model[15](x)\
#        + abs(a16)*gal_model[16](x)\
#        + abs(a17)*gal_model[17](x)\
#        + abs(a18)*gal_model[18](x)\
#        + abs(a19)*gal_model[19](x)\
#        + abs(a20)*gal_model[20](x)\
#        + abs(a21)*gal_model[21](x)\
#        + abs(a22)*gal_model[22](x)\
#        + abs(a23)*gal_model[23](x)\
#        + abs(a24)*gal_model[24](x)\
#        + abs(a25)*gal_model[25](x)\
#        + abs(a26)*gal_model[26](x)\
#        + abs(a27)*gal_model[27](x)\
#        + abs(a28)*gal_model[28](x)\
#        + abs(a29)*gal_model[29](x)\
#        + abs(a30)*gal_model[30](x)\
#        + abs(a31)*gal_model[31](x)\
#        + abs(a32)*gal_model[32](x)\
#        + abs(a33)*gal_model[33](x)\
#        + abs(a34)*gal_model[34](x)\
#        + abs(a35)*gal_model[35](x)\
#        + abs(a36)*gal_model[36](x)\
#        + abs(a37)*gal_model[37](x)\
#        + abs(a38)*gal_model[38](x)\
        )
                    

    @custom_model   
    def power_law_hbeta(x, amplitude = max(fl_hbeta_cont), alpha = 1.5):
        return abs(amplitude) * (x**-alpha)

    @custom_model
    def FeII_template_hbeta(x, mF = np.mean(fl_hbeta_cont), \
    mG = np.mean(fl_hbeta_cont), mP = np.mean(fl_hbeta_cont), \
    mS = np.mean(fl_hbeta_cont), mZ = np.mean(fl_hbeta_cont), \
    shift = 0, sigma = 1):
    
        return (abs(mF) * \
        (F[0][1]* 1./(sigma*np.sqrt(2.*np.pi)) * np.exp(-0.5 * ((x - F[0][0] + shift) / sigma)**2.)\
        + F[1][1]* 1./(sigma*np.sqrt(2.*np.pi)) * np.exp(-0.5 * ((x - F[1][0] + shift) / sigma)**2.)\
        + F[2][1]* 1./(sigma*np.sqrt(2.*np.pi)) * np.exp(-0.5 * ((x - F[2][0] + shift) / sigma)**2.)\
        + F[3][1]* 1./(sigma*np.sqrt(2.*np.pi)) * np.exp(-0.5 * ((x - F[3][0] + shift) / sigma)**2.)\
        + F[4][1]* 1./(sigma*np.sqrt(2.*np.pi)) * np.exp(-0.5 * ((x - F[4][0] + shift) / sigma)**2.)\
        + F[5][1]* 1./(sigma*np.sqrt(2.*np.pi)) * np.exp(-0.5 * ((x - F[5][0] + shift) / sigma)**2.)\
        + F[6][1]* 1./(sigma*np.sqrt(2.*np.pi)) * np.exp(-0.5 * ((x - F[6][0] + shift) / sigma)**2.)\
        + F[7][1]* 1./(sigma*np.sqrt(2.*np.pi)) * np.exp(-0.5 * ((x - F[7][0] + shift) / sigma)**2.)\
        + F[8][1]* 1./(sigma*np.sqrt(2.*np.pi)) * np.exp(-0.5 * ((x - F[8][0] + shift) / sigma)**2.)\
        + F[9][1]* 1./(sigma*np.sqrt(2.*np.pi)) * np.exp(-0.5 * ((x - F[9][0] + shift) / sigma)**2.)\
        + F[10][1]* 1./(sigma*np.sqrt(2.*np.pi)) * np.exp(-0.5 * ((x - F[10][0] + shift) / sigma)**2.)\
        + F[11][1]* 1./(sigma*np.sqrt(2.*np.pi)) * np.exp(-0.5 * ((x - F[11][0] + shift) / sigma)**2.)\
        + F[12][1]* 1./(sigma*np.sqrt(2.*np.pi)) * np.exp(-0.5 * ((x - F[12][0] + shift) / sigma)**2.)\
        + F[13][1]* 1./(sigma*np.sqrt(2.*np.pi)) * np.exp(-0.5 * ((x - F[13][0] + shift) / sigma)**2.)\
        + F[14][1]* 1./(sigma*np.sqrt(2.*np.pi)) * np.exp(-0.5 * ((x - F[14][0] + shift) / sigma)**2.)\
        + F[15][1]* 1./(sigma*np.sqrt(2.*np.pi)) * np.exp(-0.5 * ((x - F[15][0] + shift) / sigma)**2.)\
        + F[16][1]* 1./(sigma*np.sqrt(2.*np.pi)) * np.exp(-0.5 * ((x - F[16][0] + shift) / sigma)**2.)\
        + F[17][1]* 1./(sigma*np.sqrt(2.*np.pi)) * np.exp(-0.5 * ((x - F[17][0] + shift) / sigma)**2.)\
        + F[18][1]* 1./(sigma*np.sqrt(2.*np.pi)) * np.exp(-0.5 * ((x - F[18][0] + shift) / sigma)**2.)\
        + F[19][1]* 1./(sigma*np.sqrt(2.*np.pi)) * np.exp(-0.5 * ((x - F[19][0] + shift) / sigma)**2.)\
        + F[20][1]* 1./(sigma*np.sqrt(2.*np.pi)) * np.exp(-0.5 * ((x - F[20][0] + shift) / sigma)**2.))\
    
        + abs(mG) * \
        (+ G[0][1]* 1./(sigma*np.sqrt(2.*np.pi)) * np.exp(-0.5 * ((x - G[0][0] + shift) / sigma)**2.)\
        + G[1][1]* 1./(sigma*np.sqrt(2.*np.pi)) * np.exp(-0.5 * ((x - G[1][0] + shift) / sigma)**2.)\
        + G[2][1]* 1./(sigma*np.sqrt(2.*np.pi)) * np.exp(-0.5 * ((x - G[2][0] + shift) / sigma)**2.)\
        + G[3][1]* 1./(sigma*np.sqrt(2.*np.pi)) * np.exp(-0.5 * ((x - G[3][0] + shift) / sigma)**2.)\
        + G[4][1]* 1./(sigma*np.sqrt(2.*np.pi)) * np.exp(-0.5 * ((x - G[4][0] + shift) / sigma)**2.)\
        + G[5][1]* 1./(sigma*np.sqrt(2.*np.pi)) * np.exp(-0.5 * ((x - G[5][0] + shift) / sigma)**2.)\
        + G[6][1]* 1./(sigma*np.sqrt(2.*np.pi)) * np.exp(-0.5 * ((x - G[6][0] + shift) / sigma)**2.)\
        + G[7][1]* 1./(sigma*np.sqrt(2.*np.pi)) * np.exp(-0.5 * ((x - G[7][0] + shift) / sigma)**2.)\
        + G[8][1]* 1./(sigma*np.sqrt(2.*np.pi)) * np.exp(-0.5 * ((x - G[8][0] + shift) / sigma)**2.)\
        + G[9][1]* 1./(sigma*np.sqrt(2.*np.pi)) * np.exp(-0.5 * ((x - G[9][0] + shift) / sigma)**2.)\
        + G[10][1]* 1./(sigma*np.sqrt(2.*np.pi)) * np.exp(-0.5 * ((x - G[10][0] + shift) / sigma)**2.))\
    
        + abs(mP) * \
        (+ P[0][1]* 1./(sigma*np.sqrt(2.*np.pi)) * np.exp(-0.5 * ((x - P[0][0] + shift) / sigma)**2.)\
        + P[1][1]* 1./(sigma*np.sqrt(2.*np.pi)) * np.exp(-0.5 * ((x - P[1][0] + shift) / sigma)**2.)\
        + P[2][1]* 1./(sigma*np.sqrt(2.*np.pi)) * np.exp(-0.5 * ((x - P[2][0] + shift) / sigma)**2.)\
        + P[3][1]* 1./(sigma*np.sqrt(2.*np.pi)) * np.exp(-0.5 * ((x - P[3][0] + shift) / sigma)**2.)\
        + P[4][1]* 1./(sigma*np.sqrt(2.*np.pi)) * np.exp(-0.5 * ((x - P[4][0] + shift) / sigma)**2.)\
        + P[5][1]* 1./(sigma*np.sqrt(2.*np.pi)) * np.exp(-0.5 * ((x - P[5][0] + shift) / sigma)**2.)\
        + P[6][1]* 1./(sigma*np.sqrt(2.*np.pi)) * np.exp(-0.5 * ((x - P[6][0] + shift) / sigma)**2.)\
        + P[7][1]* 1./(sigma*np.sqrt(2.*np.pi)) * np.exp(-0.5 * ((x - P[7][0] + shift) / sigma)**2.)\
        + P[8][1]* 1./(sigma*np.sqrt(2.*np.pi)) * np.exp(-0.5 * ((x - P[8][0] + shift) / sigma)**2.)\
        + P[9][1]* 1./(sigma*np.sqrt(2.*np.pi)) * np.exp(-0.5 * ((x - P[9][0] + shift) / sigma)**2.)\
        + P[10][1]* 1./(sigma*np.sqrt(2.*np.pi)) * np.exp(-0.5 * ((x - P[10][0] + shift) / sigma)**2.)\
        + P[11][1]* 1./(sigma*np.sqrt(2.*np.pi)) * np.exp(-0.5 * ((x - P[11][0] + shift) / sigma)**2.)\
        + P[12][1]* 1./(sigma*np.sqrt(2.*np.pi)) * np.exp(-0.5 * ((x - P[12][0] + shift) / sigma)**2.)\
        + P[13][1]* 1./(sigma*np.sqrt(2.*np.pi)) * np.exp(-0.5 * ((x - P[13][0] + shift) / sigma)**2.)\
        + P[14][1]* 1./(sigma*np.sqrt(2.*np.pi)) * np.exp(-0.5 * ((x - P[14][0] + shift) / sigma)**2.))\
        
        + abs(mS) * \
        (+ S[0][1]* 1./(sigma*np.sqrt(2.*np.pi)) * np.exp(-0.5 * ((x - S[0][0] + shift) / sigma)**2.)\
        + S[1][1]* 1./(sigma*np.sqrt(2.*np.pi)) * np.exp(-0.5 * ((x - S[1][0] + shift) / sigma)**2.)\
        + S[2][1]* 1./(sigma*np.sqrt(2.*np.pi)) * np.exp(-0.5 * ((x - S[2][0] + shift) / sigma)**2.)\
        + S[3][1]* 1./(sigma*np.sqrt(2.*np.pi)) * np.exp(-0.5 * ((x - S[3][0] + shift) / sigma)**2.)\
        + S[4][1]* 1./(sigma*np.sqrt(2.*np.pi)) * np.exp(-0.5 * ((x - S[4][0] + shift) / sigma)**2.)\
        + S[5][1]* 1./(sigma*np.sqrt(2.*np.pi)) * np.exp(-0.5 * ((x - S[5][0] + shift) / sigma)**2.)\
        + S[6][1]* 1./(sigma*np.sqrt(2.*np.pi)) * np.exp(-0.5 * ((x - S[6][0] + shift) / sigma)**2.))\
    
        + abs(mZ) * \
        (+ Z[0][1]* 1./(sigma*np.sqrt(2.*np.pi)) * np.exp(-0.5 * ((x - Z[0][0] + shift) / sigma)**2.)\
        + Z[1][1]* 1./(sigma*np.sqrt(2.*np.pi)) * np.exp(-0.5 * ((x - Z[1][0] + shift) / sigma)**2.)\
        + Z[2][1]* 1./(sigma*np.sqrt(2.*np.pi)) * np.exp(-0.5 * ((x - Z[2][0] + shift) / sigma)**2.)\
        + Z[3][1]* 1./(sigma*np.sqrt(2.*np.pi)) * np.exp(-0.5 * ((x - Z[3][0] + shift) / sigma)**2.)\
        + Z[4][1]* 1./(sigma*np.sqrt(2.*np.pi)) * np.exp(-0.5 * ((x - Z[4][0] + shift) / sigma)**2.)\
        + Z[5][1]* 1./(sigma*np.sqrt(2.*np.pi)) * np.exp(-0.5 * ((x - Z[5][0] + shift) / sigma)**2.)\
        + Z[6][1]* 1./(sigma*np.sqrt(2.*np.pi)) * np.exp(-0.5 * ((x - Z[6][0] + shift) / sigma)**2.)\
        + Z[7][1]* 1./(sigma*np.sqrt(2.*np.pi)) * np.exp(-0.5 * ((x - Z[7][0] + shift) / sigma)**2.)\
        + Z[8][1]* 1./(sigma*np.sqrt(2.*np.pi)) * np.exp(-0.5 * ((x - Z[8][0] + shift) / sigma)**2.)\
        + Z[9][1]* 1./(sigma*np.sqrt(2.*np.pi)) * np.exp(-0.5 * ((x - Z[9][0] + shift) / sigma)**2.)\
        + Z[10][1]* 1./(sigma*np.sqrt(2.*np.pi)) * np.exp(-0.5 * ((x - Z[10][0] + shift) / sigma)**2.)\
        + Z[11][1]* 1./(sigma*np.sqrt(2.*np.pi)) * np.exp(-0.5 * ((x - Z[11][0] + shift) / sigma)**2.)\
        + Z[12][1]* 1./(sigma*np.sqrt(2.*np.pi)) * np.exp(-0.5 * ((x - Z[12][0] + shift) / sigma)**2.)\
        + Z[13][1]* 1./(sigma*np.sqrt(2.*np.pi)) * np.exp(-0.5 * ((x - Z[13][0] + shift) / sigma)**2.)\
        + Z[14][1]* 1./(sigma*np.sqrt(2.*np.pi)) * np.exp(-0.5 * ((x - Z[14][0] + shift) / sigma)**2.)))

#    m_init = FeII_template_hbeta() + power_law_hbeta() #For Hbeta
    fit = fitting.LevMarLSQFitter()
#    m = fit(m_init, wl_hbeta_cont, fl_hbeta_cont, maxiter=10**7)


#    m_init2 = FeII_template_hbeta(mF = np.mean(fl_halpha_cont), mG = np.mean(fl_halpha_cont), \
#    mP = np.mean(fl_halpha_cont), mS = np.mean(fl_halpha_cont), \
#    mZ = np.mean(fl_halpha_cont), shift = 0, sigma = 1) \
#    + power_law_hbeta(amplitude = min(fl_halpha_cont), alpha = 1.5)
 
#    fit = fitting.LevMarLSQFitter()
#    m2 = fit(m_init2, wl_halpha_cont, fl_halpha_cont, maxiter=10**7)

    m_init_new = FeII_template_hbeta() + power_law_hbeta() + gal_mod()
    m_init_new.alpha_1.bounds = (1, 2)
    m_new = fit(m_init_new, xsamp, ysamp, maxiter=10**7)

#'''
#m_init3 = FeII_template_hbeta(mF = np.mean(fl_OII_cont), mG = np.mean(fl_OII_cont), \
#mP = np.mean(fl_OII_cont), mS = np.mean(fl_OII_cont), \
#mZ = np.mean(fl_OII_cont), shift = 0, sigma = 1) \
#+ power_law_hbeta(amplitude = min(fl_OII_cont), alpha = 1.5)

#fit = fitting.LevMarLSQFitter()
#m3 = fit(m_init3, wl_OII_cont, fl_OII_cont, maxiter=300000)
#'''



#'''========== Fitting Decompotition =========='''
    def F_fun(x, mF, shift, sigma):
        return (abs(mF) * \
        (F[0][1]* 1./(sigma*np.sqrt(2.*np.pi)) * np.exp(-0.5 * ((x - F[0][0] + shift) / sigma)**2.)\
        + F[1][1]* 1./(sigma*np.sqrt(2.*np.pi)) * np.exp(-0.5 * ((x - F[1][0] + shift) / sigma)**2.)\
        + F[2][1]* 1./(sigma*np.sqrt(2.*np.pi)) * np.exp(-0.5 * ((x - F[2][0] + shift) / sigma)**2.)\
        + F[3][1]* 1./(sigma*np.sqrt(2.*np.pi)) * np.exp(-0.5 * ((x - F[3][0] + shift) / sigma)**2.)\
        + F[4][1]* 1./(sigma*np.sqrt(2.*np.pi)) * np.exp(-0.5 * ((x - F[4][0] + shift) / sigma)**2.)\
        + F[5][1]* 1./(sigma*np.sqrt(2.*np.pi)) * np.exp(-0.5 * ((x - F[5][0] + shift) / sigma)**2.)\
        + F[6][1]* 1./(sigma*np.sqrt(2.*np.pi)) * np.exp(-0.5 * ((x - F[6][0] + shift) / sigma)**2.)\
        + F[7][1]* 1./(sigma*np.sqrt(2.*np.pi)) * np.exp(-0.5 * ((x - F[7][0] + shift) / sigma)**2.)\
        + F[8][1]* 1./(sigma*np.sqrt(2.*np.pi)) * np.exp(-0.5 * ((x - F[8][0] + shift) / sigma)**2.)\
        + F[9][1]* 1./(sigma*np.sqrt(2.*np.pi)) * np.exp(-0.5 * ((x - F[9][0] + shift) / sigma)**2.)\
        + F[10][1]* 1./(sigma*np.sqrt(2.*np.pi)) * np.exp(-0.5 * ((x - F[10][0] + shift) / sigma)**2.)\
        + F[11][1]* 1./(sigma*np.sqrt(2.*np.pi)) * np.exp(-0.5 * ((x - F[11][0] + shift) / sigma)**2.)\
        + F[12][1]* 1./(sigma*np.sqrt(2.*np.pi)) * np.exp(-0.5 * ((x - F[12][0] + shift) / sigma)**2.)\
        + F[13][1]* 1./(sigma*np.sqrt(2.*np.pi)) * np.exp(-0.5 * ((x - F[13][0] + shift) / sigma)**2.)\
        + F[14][1]* 1./(sigma*np.sqrt(2.*np.pi)) * np.exp(-0.5 * ((x - F[14][0] + shift) / sigma)**2.)\
        + F[15][1]* 1./(sigma*np.sqrt(2.*np.pi)) * np.exp(-0.5 * ((x - F[15][0] + shift) / sigma)**2.)\
        + F[16][1]* 1./(sigma*np.sqrt(2.*np.pi)) * np.exp(-0.5 * ((x - F[16][0] + shift) / sigma)**2.)\
        + F[17][1]* 1./(sigma*np.sqrt(2.*np.pi)) * np.exp(-0.5 * ((x - F[17][0] + shift) / sigma)**2.)\
        + F[18][1]* 1./(sigma*np.sqrt(2.*np.pi)) * np.exp(-0.5 * ((x - F[18][0] + shift) / sigma)**2.)\
        + F[19][1]* 1./(sigma*np.sqrt(2.*np.pi)) * np.exp(-0.5 * ((x - F[19][0] + shift) / sigma)**2.)\
        + F[20][1]* 1./(sigma*np.sqrt(2.*np.pi)) * np.exp(-0.5 * ((x - F[20][0] + shift) / sigma)**2.)))

    def G_fun(x, mG, shift, sigma):
        return (abs(mG) * \
        (+ G[0][1]* 1./(sigma*np.sqrt(2.*np.pi)) * np.exp(-0.5 * ((x - G[0][0] + shift) / sigma)**2.)\
        + G[1][1]* 1./(sigma*np.sqrt(2.*np.pi)) * np.exp(-0.5 * ((x - G[1][0] + shift) / sigma)**2.)\
        + G[2][1]* 1./(sigma*np.sqrt(2.*np.pi)) * np.exp(-0.5 * ((x - G[2][0] + shift) / sigma)**2.)\
        + G[3][1]* 1./(sigma*np.sqrt(2.*np.pi)) * np.exp(-0.5 * ((x - G[3][0] + shift) / sigma)**2.)\
        + G[4][1]* 1./(sigma*np.sqrt(2.*np.pi)) * np.exp(-0.5 * ((x - G[4][0] + shift) / sigma)**2.)\
        + G[5][1]* 1./(sigma*np.sqrt(2.*np.pi)) * np.exp(-0.5 * ((x - G[5][0] + shift) / sigma)**2.)\
        + G[6][1]* 1./(sigma*np.sqrt(2.*np.pi)) * np.exp(-0.5 * ((x - G[6][0] + shift) / sigma)**2.)\
        + G[7][1]* 1./(sigma*np.sqrt(2.*np.pi)) * np.exp(-0.5 * ((x - G[7][0] + shift) / sigma)**2.)\
        + G[8][1]* 1./(sigma*np.sqrt(2.*np.pi)) * np.exp(-0.5 * ((x - G[8][0] + shift) / sigma)**2.)\
        + G[9][1]* 1./(sigma*np.sqrt(2.*np.pi)) * np.exp(-0.5 * ((x - G[9][0] + shift) / sigma)**2.)\
        + G[10][1]* 1./(sigma*np.sqrt(2.*np.pi)) * np.exp(-0.5 * ((x - G[10][0] + shift) / sigma)**2.)))


    def P_fun(x, mP, shift, sigma):
        return (abs(mP) * \
        (+ P[0][1]* 1./(sigma*np.sqrt(2.*np.pi)) * np.exp(-0.5 * ((x - P[0][0] + shift) / sigma)**2.)\
        + P[1][1]* 1./(sigma*np.sqrt(2.*np.pi)) * np.exp(-0.5 * ((x - P[1][0] + shift) / sigma)**2.)\
        + P[2][1]* 1./(sigma*np.sqrt(2.*np.pi)) * np.exp(-0.5 * ((x - P[2][0] + shift) / sigma)**2.)\
        + P[3][1]* 1./(sigma*np.sqrt(2.*np.pi)) * np.exp(-0.5 * ((x - P[3][0] + shift) / sigma)**2.)\
        + P[4][1]* 1./(sigma*np.sqrt(2.*np.pi)) * np.exp(-0.5 * ((x - P[4][0] + shift) / sigma)**2.)\
        + P[5][1]* 1./(sigma*np.sqrt(2.*np.pi)) * np.exp(-0.5 * ((x - P[5][0] + shift) / sigma)**2.)\
        + P[6][1]* 1./(sigma*np.sqrt(2.*np.pi)) * np.exp(-0.5 * ((x - P[6][0] + shift) / sigma)**2.)\
        + P[7][1]* 1./(sigma*np.sqrt(2.*np.pi)) * np.exp(-0.5 * ((x - P[7][0] + shift) / sigma)**2.)\
        + P[8][1]* 1./(sigma*np.sqrt(2.*np.pi)) * np.exp(-0.5 * ((x - P[8][0] + shift) / sigma)**2.)\
        + P[9][1]* 1./(sigma*np.sqrt(2.*np.pi)) * np.exp(-0.5 * ((x - P[9][0] + shift) / sigma)**2.)\
        + P[10][1]* 1./(sigma*np.sqrt(2.*np.pi)) * np.exp(-0.5 * ((x - P[10][0] + shift) / sigma)**2.)\
        + P[11][1]* 1./(sigma*np.sqrt(2.*np.pi)) * np.exp(-0.5 * ((x - P[11][0] + shift) / sigma)**2.)\
        + P[12][1]* 1./(sigma*np.sqrt(2.*np.pi)) * np.exp(-0.5 * ((x - P[12][0] + shift) / sigma)**2.)\
        + P[13][1]* 1./(sigma*np.sqrt(2.*np.pi)) * np.exp(-0.5 * ((x - P[13][0] + shift) / sigma)**2.)\
        + P[14][1]* 1./(sigma*np.sqrt(2.*np.pi)) * np.exp(-0.5 * ((x - P[14][0] + shift) / sigma)**2.)))

    def S_fun(x, mS, shift, sigma):
        return (abs(mS) * \
        (+ S[0][1]* 1./(sigma*np.sqrt(2.*np.pi)) * np.exp(-0.5 * ((x - S[0][0] + shift) / sigma)**2.)\
        + S[1][1]* 1./(sigma*np.sqrt(2.*np.pi)) * np.exp(-0.5 * ((x - S[1][0] + shift) / sigma)**2.)\
        + S[2][1]* 1./(sigma*np.sqrt(2.*np.pi)) * np.exp(-0.5 * ((x - S[2][0] + shift) / sigma)**2.)\
        + S[3][1]* 1./(sigma*np.sqrt(2.*np.pi)) * np.exp(-0.5 * ((x - S[3][0] + shift) / sigma)**2.)\
        + S[4][1]* 1./(sigma*np.sqrt(2.*np.pi)) * np.exp(-0.5 * ((x - S[4][0] + shift) / sigma)**2.)\
        + S[5][1]* 1./(sigma*np.sqrt(2.*np.pi)) * np.exp(-0.5 * ((x - S[5][0] + shift) / sigma)**2.)\
        + S[6][1]* 1./(sigma*np.sqrt(2.*np.pi)) * np.exp(-0.5 * ((x - S[6][0] + shift) / sigma)**2.)))
    
    def Z_fun(x, mZ, shift, sigma):
        return (abs(mZ) * \
        (+ Z[0][1]* 1./(sigma*np.sqrt(2.*np.pi)) * np.exp(-0.5 * ((x - Z[0][0] + shift) / sigma)**2.)\
        + Z[1][1]* 1./(sigma*np.sqrt(2.*np.pi)) * np.exp(-0.5 * ((x - Z[1][0] + shift) / sigma)**2.)\
        + Z[2][1]* 1./(sigma*np.sqrt(2.*np.pi)) * np.exp(-0.5 * ((x - Z[2][0] + shift) / sigma)**2.)\
        + Z[3][1]* 1./(sigma*np.sqrt(2.*np.pi)) * np.exp(-0.5 * ((x - Z[3][0] + shift) / sigma)**2.)\
        + Z[4][1]* 1./(sigma*np.sqrt(2.*np.pi)) * np.exp(-0.5 * ((x - Z[4][0] + shift) / sigma)**2.)\
        + Z[5][1]* 1./(sigma*np.sqrt(2.*np.pi)) * np.exp(-0.5 * ((x - Z[5][0] + shift) / sigma)**2.)\
        + Z[6][1]* 1./(sigma*np.sqrt(2.*np.pi)) * np.exp(-0.5 * ((x - Z[6][0] + shift) / sigma)**2.)\
        + Z[7][1]* 1./(sigma*np.sqrt(2.*np.pi)) * np.exp(-0.5 * ((x - Z[7][0] + shift) / sigma)**2.)\
        + Z[8][1]* 1./(sigma*np.sqrt(2.*np.pi)) * np.exp(-0.5 * ((x - Z[8][0] + shift) / sigma)**2.)\
        + Z[9][1]* 1./(sigma*np.sqrt(2.*np.pi)) * np.exp(-0.5 * ((x - Z[9][0] + shift) / sigma)**2.)\
        + Z[10][1]* 1./(sigma*np.sqrt(2.*np.pi)) * np.exp(-0.5 * ((x - Z[10][0] + shift) / sigma)**2.)\
        + Z[11][1]* 1./(sigma*np.sqrt(2.*np.pi)) * np.exp(-0.5 * ((x - Z[11][0] + shift) / sigma)**2.)\
        + Z[12][1]* 1./(sigma*np.sqrt(2.*np.pi)) * np.exp(-0.5 * ((x - Z[12][0] + shift) / sigma)**2.)\
        + Z[13][1]* 1./(sigma*np.sqrt(2.*np.pi)) * np.exp(-0.5 * ((x - Z[13][0] + shift) / sigma)**2.)\
        + Z[14][1]* 1./(sigma*np.sqrt(2.*np.pi)) * np.exp(-0.5 * ((x - Z[14][0] + shift) / sigma)**2.)))

    def power_law2(x, amplitude, alpha):
        return abs(amplitude) * (x**-alpha)
    

#    inter = interpolate.interp1d(wl_OII_cont, fl_OII_cont, kind='slinear')


    plt.figure(j)
    plt.plot(agn[:, 0], agn[:, 1], 'b-', label='Observed Spectra')
    plt.plot(wl_plot, m_new(wl_plot), 'r-', label='Model')
    plt.plot(wl_plot, power_law2(wl_plot, m_new.amplitude_1.value, m_new.alpha_1.value), 'g-', label='Power Law')
    plt.plot(wl_plot, F_fun(wl_plot, m_new.mF_0.value, m_new.shift_0.value, m_new.sigma_0.value), 'c-')#, label='F Fe II Group')
    plt.plot(wl_plot, G_fun(wl_plot, m_new.mG_0.value, m_new.shift_0.value, m_new.sigma_0.value), 'g-')#, label='G Fe II Group')
    plt.plot(wl_plot, P_fun(wl_plot, m_new.mP_0.value, m_new.shift_0.value, m_new.sigma_0.value), 'm-')#, label='P Fe II Group')
    plt.plot(wl_plot, S_fun(wl_plot, m_new.mS_0.value, m_new.shift_0.value, m_new.sigma_0.value), 'y-')#, label='S Fe II Group')
    plt.plot(wl_plot, Z_fun(wl_plot, m_new.mZ_0.value, m_new.shift_0.value, m_new.sigma_0.value), 'k-')#, label='I Zw 1 Fe II Group')



#    plt.plot(wl_hbeta, m(wl_hbeta), 'r-',label='Model')
#    plt.plot(wl_halpha, m2(wl_halpha), 'r-')
    
    #--plt.plot(wl_OII, m3(wl_OII), 'r-')
    #--plt.plot(wl_OII, interpolate.spline(wl_OII_cont, fl_OII_cont, wl_OII, order=1), 'r-')
    
#    plt.plot(wl_OII, inter(wl_OII), 'r-')

#    plt.plot(wl_hbeta, power_law2(wl_hbeta, m_new.amplitude_1.value, m_new.alpha_1.value), 'g-', label='Power Law')
#    plt.plot(wl_halpha, power_law2(wl_halpha, m_new.amplitude_1.value, m_new.alpha_1.value), 'g-')
    #--plt.plot(wl_OII, power_law2(wl_OII, m3.amplitude_1.value, m3.alpha_1.value), 'g-')



#    plt.plot(wl_hbeta, F_fun(wl_hbeta, m_new.mF_0.value, m_new.shift_0.value, m_new.sigma_0.value), 'c-', label='F Fe II Group')
#    plt.plot(wl_hbeta, G_fun(wl_hbeta, m_new.mG_0.value, m_new.shift_0.value, m_new.sigma_0.value), 'g-', label='G Fe II Group')
#    plt.plot(wl_hbeta, P_fun(wl_hbeta, m_new.mP_0.value, m_new.shift_0.value, m_new.sigma_0.value), 'm-', label='P Fe II Group')
#    plt.plot(wl_hbeta, S_fun(wl_hbeta, m_new.mS_0.value, m_new.shift_0.value, m_new.sigma_0.value), 'y-', label='S Fe II Group')
#    plt.plot(wl_hbeta, Z_fun(wl_hbeta, m_new.mZ_0.value, m_new.shift_0.value, m_new.sigma_0.value), 'k-', label='I Zw 1 Fe II Group')
    
#    plt.plot(wl_halpha, F_fun(wl_halpha, m2.mF_0.value, m2.shift_0.value, m2.sigma_0.value), 'c-')
#    plt.plot(wl_halpha, G_fun(wl_halpha, m2.mG_0.value, m2.shift_0.value, m2.sigma_0.value), 'g-')
#    plt.plot(wl_halpha, P_fun(wl_halpha, m2.mP_0.value, m2.shift_0.value, m2.sigma_0.value), 'm-')
#    plt.plot(wl_halpha, S_fun(wl_halpha, m2.mS_0.value, m2.shift_0.value, m2.sigma_0.value), 'y-')
#    plt.plot(wl_halpha, Z_fun(wl_halpha, m2.mZ_0.value, m2.shift_0.value, m2.sigma_0.value), 'k-')

    #--plt.plot(wl_OII, F_fun(wl_OII, m3.mF_0.value, m3.shift_0.value, m3.sigma_0.value), 'c-')
    #--plt.plot(wl_OII, G_fun(wl_OII, m3.mG_0.value, m3.shift_0.value, m3.sigma_0.value), 'g-')
    #--plt.plot(wl_OII, P_fun(wl_OII, m3.mP_0.value, m3.shift_0.value, m3.sigma_0.value), 'm-')
    #--plt.plot(wl_OII, S_fun(wl_OII, m3.mS_0.value, m3.shift_0.value, m3.sigma_0.value), 'y-')
    #--plt.plot(wl_OII, Z_fun(wl_OII, m3.mZ_0.value, m3.shift_0.value, m3.sigma_0.value), 'k-')


    plt.legend(loc='best')
    
    plt.xlabel('Wavelength ($\AA$)')
    plt.ylabel('Flux Density ($ \\times 10^{-17} \ erg \ cm^{-2} s^{-1} \AA^{-1}$)')
    plt.title('Continuum Subtraction\nspec-' + plate[j] + '-' + mjd[j] + '-' + fiberid[j] + '.fits')
    plt.savefig('figure2/fig2_'+'spec-' + plate[j] + '-' + mjd[j] + '-' + fiberid[j])
    plt.close('all')


    flux_F.append(integrate.trapz(F_fun(agn[:, 0], m_new.mF_0.value, m_new.shift_0.value, m_new.sigma_0.value)))
    flux_G.append(integrate.trapz(G_fun(agn[:, 0], m_new.mG_0.value, m_new.shift_0.value, m_new.sigma_0.value)))
    flux_P.append(integrate.trapz(P_fun(agn[:, 0], m_new.mP_0.value, m_new.shift_0.value, m_new.sigma_0.value)))
    flux_S.append(integrate.trapz(S_fun(agn[:, 0], m_new.mS_0.value, m_new.shift_0.value, m_new.sigma_0.value)))
    flux_Z.append(integrate.trapz(Z_fun(agn[:, 0], m_new.mZ_0.value, m_new.shift_0.value, m_new.sigma_0.value)))
    sigma_fe.append(m_new.sigma_0.value)
    
    o = open('calculation/Fe_II_flux_2.csv', 'a')
    #for i in range(len(flux_F)):
    o.write(str(flux_F[q]) + '\t' + str(flux_G[q]) + '\t' + str(flux_P[q])\
     + '\t' + str(flux_S[q]) + '\t' + str(flux_Z[q]) + '\t' + str(sigma_fe[q]) + '\n')
    
    o.close()





#'''========== Continuum Substraction =========='''

    fl_OII_sub = []
    fl_hbeta_sub = []
    fl_halpha_sub = []
    
    fl_OII_norm = []
    fl_hbeta_norm = []
    fl_halpha_norm = []
    
    fl_cont_sub = []


#    for i in range(len(wl_cont)):
#        fl_cont_sub.append(fl_cont[i] - m_new(wl_cont)[i])

    for i in range(len(wl_OII)):
        fl_OII_sub.append(fl_OII[i] - m_new(wl_OII)[i])
        fl_OII_norm.append(fl_OII[i] / m_new(wl_OII)[i])
        
    for i in range(len(wl_hbeta)):
        fl_hbeta_sub.append(fl_hbeta[i] - m_new(wl_hbeta)[i])
        fl_hbeta_norm.append(fl_hbeta[i] / m_new(wl_hbeta)[i])

    for i in range(len(wl_halpha)):
        fl_halpha_sub.append(fl_halpha[i] - m_new(wl_halpha)[i])
        fl_halpha_norm.append(fl_halpha[i] / m_new(wl_halpha)[i])
        
    oii = 0
    hbeta = 0
    halpha = 0
    
    o = open('cont_sub/sub_'+'spec-' + plate[j] + '-' + mjd[j] + '-' + fiberid[j] + '.fits', 'w')
    o.write('#Wavelength (Angstroms)\Flux')
    o.write('\n\n\n')
    
    
    for i in range(len(agn)):
        
        if agn[i][0] >= min(wl_OII) and agn[i][0] <= max(wl_OII):
            o.write(str(agn[i][0]) + '\t' + str(fl_OII_sub[oii]) + '\n')
            oii += 1
            
        elif agn[i][0] >= min(wl_hbeta) and agn[i][0] <= max(wl_hbeta):
            o.write(str(agn[i][0]) + '\t' + str(fl_hbeta_sub[hbeta]) + '\n')
            hbeta += 1
        
        elif agn[i][0] >= min(wl_halpha) and agn[i][0] <= max(wl_halpha):
            o.write(str(agn[i][0]) + '\t' + str(fl_halpha_sub[halpha]) + '\n')
            halpha+=1
    
        else:
            o.write(str(agn[i][0]) + '\t' + str(0) + '\n')
            
    o.close()    
    #################
    oii = 0
    hbeta = 0
    halpha = 0
    
    o = open('cont_norm/norm_'+'spec-' + plate[j] + '-' + mjd[j] + '-' + fiberid[j] + '.fits', 'w')
    o.write('#Wavelength (Angstroms)\Flux')
    o.write('\n\n\n')
    
    
    for i in range(len(agn)):
        
        if agn[i][0] >= min(wl_OII) and agn[i][0] <= max(wl_OII):
            o.write(str(agn[i][0]) + '\t' + str(fl_OII_norm[oii]) + '\n')
            oii += 1
            
        elif agn[i][0] >= min(wl_hbeta) and agn[i][0] <= max(wl_hbeta):
            o.write(str(agn[i][0]) + '\t' + str(fl_hbeta_norm[hbeta]) + '\n')
            hbeta += 1
        
        elif agn[i][0] >= min(wl_halpha) and agn[i][0] <= max(wl_halpha):
            o.write(str(agn[i][0]) + '\t' + str(fl_halpha_norm[halpha]) + '\n')
            halpha+=1
    
        else:
            o.write(str(agn[i][0]) + '\t' + str(1) + '\n')# penting
            
    o.close()    
    
    fl_cont.append(np.mean(power_law2(wl_cont, m_new.amplitude_1.value, m_new.alpha_1.value)))
    
    #np.median(fl_cont_sub)
        
    f = open('calculation/continuum_flux.csv', 'a')
    #for i in range(len(fl_cont)):
    f.write(str(fl_cont[q]) + '\n')

    f.close()
    
    print'j = ', j
    
    q+=1

'''
f = open('skipped_j.csv', 'w')
f.write('j\n')
for i in range(len(J)):
    f.write(str(J[i]) + '\n')

f.close()
'''


#plt.figure(2)
#plt.plot(wl_OII, fl_OII_sub)
#plt.plot(wl_hbeta, fl_hbeta_sub)
#plt.plot(wl_halpha, fl_halpha_sub)