# -*- coding: utf-8 -*-
"""
Created on Wed Sep 30 14:33:03 2015

@author: andika
"""
from astropy.modeling import models, fitting
from astropy.io import fits
import numpy as np
import pylab as pl
from scipy import interpolate

J = []
K = [] # for H alpha
L = [] # for H beta and OII
M = [] # for Mg

sample = np.loadtxt('MAIN_SAMPLE_irhamta.csv', skiprows=1, dtype=str, delimiter=',')

plate = []
mjd = []
fiberid = []
z = []
e_bv = []

for j in range(len(sample)):
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
        
    z.append(float(sample[j][7]))
    e_bv.append(float(sample[j][23]))

    if j == 18390 or j < 13899:#or j == 4815 or j < 5482:
        print 'skipped, j = ', j
        continue
#'''========== Import Data =========='''

    hdulist = fits.open('/media/irham/Chance/spectra/spec-' + plate[j] + '-' + mjd[j] + '-' + fiberid[j] + '.fits')
    header = hdulist[0].header
    tbdata = hdulist[1].data

    
    
    
    flux_unit = header['BUNIT']

    wl = []
    fl = []

    for i in range (len(tbdata)):
        wl.append(10**tbdata[i][1])
        fl.append(tbdata[i][0])


#'''========== Reddening Correction =========='''
    Ebv = e_bv[j]

    RE = np.loadtxt('Relative Extinction for Selected Bandpasses.txt', dtype = str, delimiter=',')
    A_Av = [] # A/Av
    A_Ebv = [] # A/E(B-V)
    A = []
    lam = []
    inlam = []

    for i in range (len(RE)):
        lam.append(float(RE[i][1])) 
        inlam.append(1./float(RE[i][1]))
        A_Av.append(float(RE[i][2]))
        A_Ebv.append(float(RE[i][3]))
        A.append(Ebv*A_Ebv[i])

#t_init3 = models.Polynomial1D(degree=5)
#fit_t3 = fitting.LevMarLSQFitter()
#t3 = fit_t3(t_init3, lam, A)

#A_lambda = t3(wl)

    A_lambda = interpolate.spline(lam, A, wl, order=3)

    fl_0 = []

    for i in range(len(A_lambda)):
        fl_0.append(fl[i]/(10.**(A_lambda[i]/(-2.5))))

    fl_nocor = fl # Flux without reddening correction
    fl = fl_0 # Overwrite flux to do reddening correction




#'''========= Redshift Correction =========='''
    Z = z[j]

    for i in range(len(wl)):
        wl[i] = wl[i]/(Z+1)

    if min(wl) > 3717 or max(wl) < 6800:
        K.append(j)
        #print 'skipped, j for H alpha = ', j
        #continue
        
    if min(wl) > 3717 or max(wl) < 5105:
        L.append(j)
        #print 'skipped, j for H beta = ', j
        #continue
    
    if min(wl) > 2700 or max(wl) < 5105:
        M.append(j)
        #print 'skipped, j for Mg II = ', j
        #continue
        
    emline = hdulist[3].data
    line_name = emline['LINENAME']
    wave = emline['LINEWAVE']

    line_label1 = []
    line_wave = []
    for i in range (len(wave)):
        if wave[i] >= min(wl) and wave[i] <= max(wl):
            line_label1.append(line_name[i])
            line_wave.append(wave[i] - 2)
        else:
            continue


    hdulist.close()

#'''========== Plot =========='''
#'''
#pl.figure(0)

#pl.plot(inlam, A, 'bo')
#pl.plot(inlam, interpolate.spline(lam, A, lam, order=1), 'k--')
#pl.title ('Reddening Curve')
#pl.xlabel ('$1/\\lambda$')
#pl.ylabel ('$A_{\\lambda}$')
#pl.show()
#pl.savefig('Reddening Curve')
#'''

#'''========== Optical Continuum Window =========='''
    
    

#'''========== Continuum Window Spline(Cubic) Interpolation =========='''
#t_init2 = models.Polynomial1D(degree=3)
#fit_t2 = fitting.LevMarLSQFitter()
#t2 = fit_t2(t_init2, xsamp, ysamp)

    pl.figure(j)

#line.plot_line_ids(wl, fl, line_wave, line_label1,box_axes_space=0.1)
    pl.plot(wl, fl, 'b-', linewidth=1, label='with reddening correction') #with reddening correction
    pl.plot(wl, fl_nocor, 'k-', linewidth=1 , label='without reddening correction') #without reddening correction
#pl.plot(wl, interpolate.spline(xsamp, ysamp, wl, order = 1), 'r-')
#pl.plot(wl, t2(wl), 'r-')

    pl.xlabel('Wavelength ($\AA$)')
#pl.ylabel('Normalized Flux')
    pl.ylabel('Flux Density ($ \\times 10^{-17} \ erg \ cm^{-2} s^{-1} \AA^{-1}$)')
    pl.title('Deredshift and Reddening Correction\nspec-' + plate[j] + '-' + mjd[j] + '-' + fiberid[j] + '.fits')
    pl.legend(loc='best')
    pl.savefig('figure/fig_'+'spec-' + plate[j] + '-' + mjd[j] + '-' + fiberid[j])
    #pl.show()
    pl.close('all')
#pl.figure(2)

#'''========== Exporting Normalized Spectrum =========='''

    o = open('shifted_redcor/out_'+'spec-' + plate[j] + '-' + mjd[j] + '-' + fiberid[j] + '.fits', 'w')
    o.write('#Wavelength (Angstroms)\Flux')
    o.write('\n\n\n')
    for i in range(len(wl)):
        o.write(str(wl[i]) + '\t' + str(fl[i]) + '\n')
    o.close()
    
    
    fl_cont = []    
    
        
    #if min(wl) > 4150 or max(wl) < 5500:
    #    J.append(j)
    #    print 'skipped, j = ', j
    #    continue
    
    xsamp = []
    ysamp = []
    
    wl_fe = []
    fl_fe = []

    fl_fe_sub = []
    fl_fe_norm = []        
        
    for i in range(len(wl)):
        if wl[i] >= 4000 and wl[i] <= 5500:
            wl_fe.append(wl[i])
            fl_fe.append(fl[i])
    
    for i in range(len(wl)):
        if wl[i] >= 3010 and wl[i] <= 3040\
        or wl[i] >= 3240 and wl[i] <= 3270\
        or wl[i] >= 3790 and wl[i] <= 3810\
        or wl[i] >= 4210 and wl[i] <= 4230\
        or wl[i] >= 5080 and wl[i] <= 5100\
        or wl[i] >= 5600 and wl[i] <= 5630\
        or wl[i] >= 5970 and wl[i] <= 6000:
        
            xsamp.append(wl[i])
            ysamp.append(fl[i])
    

    if min(xsamp) > min(wl_fe) or max(xsamp) < max(wl_fe):
        J.append(j)
        print 'skipped, j = ', j
        continue

    
    c = interpolate.interp1d(xsamp, ysamp, kind='slinear')
    
    for i in range(len(wl_fe)):
        fl_fe_sub.append(fl_fe[i] - c(wl_fe[i]))
        fl_fe_norm.append(fl_fe[i] / c(wl_fe[i]))
    
    
    
    o = open('for_fe/fe_'+'spec-' + plate[j] + '-' + mjd[j] + '-' + fiberid[j] + '.fits', 'w')
    o.write('#Wavelength (Angstroms)\Flux')
    o.write('\n\n\n')
    for i in range(len(wl_fe)):
        o.write(str(wl_fe[i]) + '\t' + str(fl_fe_sub[i]) + '\n')
    o.close()    
    
    o = open('for_fe_norm/fe_norm_'+'spec-' + plate[j] + '-' + mjd[j] + '-' + fiberid[j] + '.fits', 'w')
    o.write('#Wavelength (Angstroms)\Flux')
    o.write('\n\n\n')
    for i in range(len(wl_fe)):
        o.write(str(wl_fe[i]) + '\t' + str(fl_fe_norm[i]) + '\n')
    o.close()    
    
    print 'j = ', j

f = open('notes/skipped_j_opt_halpha.csv', 'w')
f.write('j\n')
for i in range(len(K)):
    f.write(str(K[i]) + '\n')
f.close()

f = open('notes/skipped_j_opt_hbeta.csv', 'w')
f.write('j\n')
for i in range(len(L)):
    f.write(str(L[i]) + '\n')
f.close()

f = open('notes/skipped_j_opt_mgII.csv', 'w')
f.write('j\n')
for i in range(len(M)):
    f.write(str(M[i]) + '\n')
f.close()

f = open('notes/skipped_j_Fe.csv', 'w')
f.write('j\n')
for i in range(len(J)):
    f.write(str(J[i]) + '\n')
f.close()
