#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 20 11:30:38 2023

@author: easher
"""


import sys
import os
#import cv2
from datetime import datetime as dt
from datetime import timedelta
import time
import pandas as pd
import numpy as np
import glob
import netCDF4 as nc
from matplotlib import pyplot as plt
import matplotlib as mpl
import matplotlib.ticker as ticker
from matplotlib.colors import LogNorm
import matplotlib.colors as colors
import seaborn as seaborn
from scipy import stats
from scipy.optimize import fsolve
import math
import PyMieScatt
from scipy.interpolate import interp1d

    
def CleanData(df1):
    
    #Filter POPS data based on various conditions... 
    df1 = df1[df1.POPSflowCCpS >= 3.0]
    df1[df1.B1 != 99999] #99999 in the place of nans
    df1 = df1[df1.POPSbaselineStdDev <= 10]
    df1 = df1[df1.GPSsat >= 6] # 
    df1 = df1[df1.GPSsat != 99999] 
    
    #could create a function to do this cleaning...            
    df1['POPSflowCCpS'] = pd.to_numeric(df1['POPSflowCCpS'], errors='coerce').fillna(0)    
    #flow corrected for differences in POPS temperature and outside air temperature (in Kelvin)
    df1['POPSTempCorrection'] = (df1['iMetTempCorrC']+273.15)/(df1['POPStempC']+273.15)
    df1['Air Temp (K)'] = df1['iMetTempCorrC']+273.15
    df1['POPSflowCorrCCpS'] = df1['POPSflowCCpS']*df1['POPSTempCorrection']
    df1['ratioWidth2Flow'] = df1['POPSavgWdithUS']/df1['POPSflowCCpS']
    #filter out data at high altitude where flow is too low and jet is rapidly expanding 
    
    
    #create binned altitude and Flight (ascent/descent) for merge
    df1['Altitude (km)'] = pd.cut(df1['GPSaltkm'], bins=np.linspace(0.125, 30.125, 121),labels = AltLabels, include_lowest=True) # 301? bin every 100 m;  if 61 bin every 500m (0.125, 30.125, 121)
    df1['Altitude (km)'] = pd.to_numeric(df1['Altitude (km)'], errors='coerce').fillna(0) 
    df1['Flight'] = df1['iMetASCENTms'].apply(lambda x: 'Ascent' if x >= 0 else 'Descent')
    df1 = df1[df1['Flight'] == 'Ascent'] #using only ascent data (can also average)
    df = df1.groupby(['Launch','Altitude (km)', 'Flight'],as_index=False).mean() #.mean
    
    #calculate number concentration (#/ cm-3), surface area, extinction and mass
    df[['B1', 'B2','B3','B4', 'B5', 'B6', 'B7','B8', 'B9', 'B10', 'B11', 'B12', 'B13', 'B14','B15']]\
    = df[['B1', 'B2','B3','B4', 'B5', 'B6', 'B7','B8', 'B9', 'B10', 'B11', 'B12', 'B13','B14','B15']].div(df['POPSflowCorrCCpS'].values,axis=0)
    df['Aer_Conc'] =  df[['B1', 'B2','B3','B4', 'B5', 'B6', 'B7','B8', 'B9','B10', 'B11', 'B12', 'B13', 'B14', 'B15']].sum(axis=1)
    df['Composition'] = df.apply(lambda x: Plume(x['Launch'], x['Flight'], x['Altitude (km)']), axis=1)
    Wavelength = 532 #Extinction wavelenth - see function below).
    df['Extinction'] = df.apply(lambda x: VPsca(Wavelength, x['Composition'], x['B1'], x['B2'], x['B3'], x['B4'], x['B5'], x['B6'], x['B7'], x['B8'], x['B9'], x['B10'], x['B11'], x['B12'], x['B13'], x['B14'], x['B15']), axis=1)
    df['BinDp'] = df.apply(lambda x: DiaLookUpTable(x['Composition']), axis=1)
    df['Surface Area Density'] =  df.apply(lambda x: CalcSurfArea(x['Composition'], x['B1'], x['B2'], x['B3'], x['B4'], x['B5'], x['B6'], x['B7'], x['B8'], x['B9'], x['B10'], x['B11'], x['B12'], x['B13'], x['B14'], x['B15']), axis = 1) 
    df['Dry Mass'] = df.apply(lambda x: MassCalc(x['Composition'], x['iMetPressMB'], x['iMetTempCorrC'], x['B1'], x['B2'], x['B3'], x['B4'], x['B5'], x['B6'], x['B7'], x['B8'], x['B9'], x['B10'], x['B11'], x['B12'], x['B13'], x['B14'], x['B15']), axis = 1) 


    return df

def plot_errorbars(arg, **kws):
    np.random.seed(sum(map(ord, "error_bars")))
    x = np.random.normal(0, 1, 100)
    f, axs = plt.subplots(2, figsize=(7, 2), sharex=True, layout="tight")
    seaborn.pointplot(x=x, errorbar=arg, **kws, capsize=.3, ax=axs[0])
    seaborn.stripplot(x=x, jitter=.3, ax=axs[1])
    
    
#Use the look-up table to find the particle diameter and dLogDp calculated from POPS signal 
def DiaLookUpTable(Composition):
    file = '/Users/easher/Documents/NOAAweb/POPS_Sizes.csv'
    df1=pd.read_csv(file,sep=',', dtype=None, engine='python')

    if Composition == 'Sulfate20':
        dp_in = df1[(df1.Composition == 'Sulfate') & (df1.Binning == 'Manual20')]
        #if you do not 
        #dpAmb = [0.1463, 0.1594, 0.1737, 0.18515, 0.19325, 0.20175, 0.2106, 0.2198, 0.22945, 0.29785, 0.46415, 0.61235, 0.7103, 0.8239, 0.95565, 1.1085, 1.2858, 1.49145, 1.72995, 2.179]
        dp =  np.array(dp_in.Diameter*1000).tolist()

    else:
        dp_in = df1[(df1.Composition == 'Sulfate') & (df1.Binning == 'Manual')]
        dp = np.array(dp_in.Diameter*1000).tolist()
        
    del dp_in
    
    return dp

    
def DnDlogDp(Launch, Composition, B1, B2, B3, B4, B5, B6, B7, B8, B9, B10, B11, B12, B13, B14, B15, B16, B17, B18, B19, B20):
    file_ior = '/Users/easher/Documents/NOAAweb/POPS_Sizes.csv'
    dfIOR = pd.read_csv(file_ior,sep=',', dtype=None, engine='python')
    #dLogDp = DLogDpLookUpTable(Composition)
    
    if Composition == 'Sulfate20':
        dp_in = dfIOR[(dfIOR.Composition == 'Sulfate') & (dfIOR.Binning == 'Manual20')]
        dLogDp = np.array(dp_in.dLogDp)

        ndp = np.array([B1, B2, B3, B4, B5, B6, B7, B8, B9, B10, B11, B12, B13, B14, B15, B16, B17, B18, B19, B20])
        dNdLogDpA = np.divide(ndp, dLogDp)

        #dNdLogDp = np.divide(np.array([B1, B2, B3, B4, B5, B6, B7, B8, B9, B10, B11, B12, B13, B14, B15, B16, B17, B18, B19, B20]), np.array(dLogDp)).tolist()
    else:
        dp_in = dfIOR[(dfIOR.Composition == 'Sulfate') & (dfIOR.Binning == 'Manual')]
        dLogDp = np.array(dp_in.dLogDp)

        ndp = np.array([B1, B2, B3, B4, B5, B6, B7, B8, B9, B10, B11, B12, B13, B14, B15])
        dNdLogDpA = np.divide(ndp, dLogDp)
        
        for x in range(5):
            dNdLogDpA = np.append(dNdLogDpA, [np.nan])
            #print(x)
        
    dNdLogDp = dNdLogDpA.tolist()   
    return dNdLogDp


#calculates dry surface area
def CalcSurfArea(Composition, B1, B2, B3, B4, B5, B6, B7, B8, B9, B10, B11, B12, B13, B14, B15, B16, B17, B18, B19, B20):
    dp = DiaLookUpTable(Composition)
    if Composition == 'Sulfate20':

            ndp = [B1, B2, B3, B4, B5, B6, B7, B8, B9, B10, B11, B12, B13, B14, B15, B16, B17, B18, B19, B20] 
    else:
            ndp = [B1, B2, B3, B4, B5, B6, B7, B8, B9, B10, B11, B12, B13, B14, B15] 
            
    Conc = pd.Series(ndp)
    Rad =  pd.Series(np.divide(dp,2))*1000
    Area = np.pi*4*np.power(Rad,2)
    DrySurfaceArea = (Conc.mul(Area)).sum()*1E-6
    
    return DrySurfaceArea #units are XXX
    

#functions to calculate ambient aerosol size
def VPsca(Wavelength, LaunchNo, Composition, RH, T, B1, B2, B3, B4, B5, B6, B7, B8, B9, B10, B11, B12, B13, B14, B15, B16, B17, B18, B19, B20):
    dp = DiaLookUpTable(Composition)
    
    dpAmb = []
    m = 1.45+0.00j #possibly should use 1.44 _ 0.00j; what if more like 1.38
    #m = 1.39+0.00j 
    Kchem = 0.87 # for H2SO4
    if Composition == 'Sulfate20':
        ndp = [B1, B2, B3, B4, B5, B6, B7, B8, B9, B10, B11, B12, B13, B14, B15, B16, B17, B18, B19, B20] 
    else:
        ndp = [B1, B2, B3, B4, B5, B6, B7, B8, B9, B10, B11, B12, B13, B14, B15] 
    #dp = [154,190,230,270,330,400, 490,590,720,870,1050,1280,1550, 1870, 2270]
    for i in dp:
        Dd = i
        func = lambda Drh : (Drh**3 - Dd**3)/ (Drh**3 - Dd**3*(1-Kchem))*np.exp(4*0.072*0.018/(8.314*T*1000*Drh))-RH/100
        Drh_initial_guess = Dd # this is true unless RH is very high and the particle is very small... then its size may change a lot
        Drh_solution = fsolve(func, Drh_initial_guess)
        dpAmb.append(Drh_solution[0])
    
    #dpAmbnp = np.array(dpAmb).tolist()
    
    dpAmbnp = dp
    #print(dpAmbnp)
    #print(ndp)
    [Bext, Bsca, Babs, bigG, Bpr, Bback, Bratio] = PyMieScatt.Mie_SD(m, Wavelength, dpAmbnp, ndp, nMedium=1.0, interpolate=True, SMPS = True,  asDict=False) 
    BextF = Bext
    return BextF

def AmbAer(RH, T, Composition):
    
    dp = DiaLookUpTable(Composition)
    dpAmb = []
        
    #define the expression whose roots we want to find (then create a function...)
    #RH = 80.0; T = 270.0; Dd = 275 #dry diameter
    Kchem = 0.87 # for H2SO4
    for i in dp:
        Dd = i
        func = lambda Drh : (Drh**3 - Dd**3)/ (Drh**3 - Dd**3*(1-Kchem))*np.exp(4*0.072*0.018/(8.314*T*1000*Drh))-RH/100
        Drh_initial_guess = Dd # this is true unless RH is very high and the particle is very small... then its size may change a lot
        Drh_solution = fsolve(func, Drh_initial_guess)
        dpAmb.append(Drh_solution[0])
    dpAmbn= np.array(dpAmb).tolist()
    
    #dpAmbn =  dp # looking at dried aerosol only...
    DpDiff = pd.DataFrame({'Dp':dp, 'dpAmbn':dpAmbn})
    DpDiff['Diff'] =  DpDiff['dpAmbn'] - DpDiff['Dp']
    DpDiff['Ratio'] = DpDiff['dpAmbn'] / DpDiff['Dp']
    #DpDiff['Altitutde'] = Altitude
    DpDiff['RH'] = RH
    #DpDiff.to_csv('/Users/easher/Desktop/DpDiff.csv', mode='a')
    #dpAmpnp = dpAmbn.reshape(1, -1)
    return dpAmbn

def SADrh(Composition, B1, B2, B3, B4, B5, B6, B7, B8, B9, B10, B11, B12, B13, B14, B15, B16, B17, B18, B19, B20, BinDp0, BinDp1, BinDp2, BinDp3, BinDp4, BinDp5, BinDp6, BinDp7, BinDp8, BinDp9, BinDp10,
          BinDp11, BinDp12, BinDp13, BinDp14, BinDp15, BinDp16, BinDp17, BinDp18, BinDp19): #was based on date, can do based on composition, no?
    
    if Composition == 'Sulfate20':

            ndp = [B1, B2, B3, B4, B5, B6, B7, B8, B9, B10, B11, B12, B13, B14, B15, B16, B17, B18, B19, B20] 
            dp = [ BinDp0, BinDp1, BinDp2, BinDp3, BinDp4, BinDp5, BinDp6, BinDp7, BinDp8, BinDp9, BinDp10, BinDp11, BinDp12, BinDp13, BinDp14, BinDp15, BinDp16, BinDp17, BinDp18, BinDp19]
    else:
            ndp = [B1, B2, B3, B4, B5, B6, B7, B8, B9, B10, B11, B12, B13, B14, B15] 
            dp = [ BinDp0, BinDp1, BinDp2, BinDp3, BinDp4, BinDp5, BinDp6, BinDp7, BinDp8, BinDp9, BinDp10, BinDp11, BinDp12, BinDp13, BinDp14]
            
    Conc = pd.Series(ndp)
    Rad =  pd.Series(np.divide(dp,2))*1000
    Area = np.pi*4*np.power(Rad,2)
    AmbSurfaceArea = (Conc.mul(Area)).sum()*1E-6
    
    return AmbSurfaceArea


def MassBCalc(LaunchNo, Composition, perwt_pops, LFEtempC, Pressure, AirTemp, B1, B2, B3, B4, B5, B6, B7, B8, B9, B10, B11, B12, B13, B14, B15, B16, B17, B18, B19, B20):
    
    dp = DiaLookUpTable(Composition)
    
    #integration on binned data
    #constants
    Rgas = 287.058 #J K-1 kg-1 or Pa m3 K-1 kg-1
    R_unv = 8.3144 #m3 Pa K-1 mol -1
    convFac = 1E6 #g to ug convert mass of aerosols
    Pressure = Pressure * 100  #convert Pressure hPa to Pa
    
    if Composition == 'Sulfate':
        ndp = [B1, B2, B3, B4, B5, B6, B7, B8, B9, B10, B11, B12, B13, B14, B15] 
        dp_unc = [0.03, 0.03, 0.03, 0.03, 0.03, 0.03, 0.03, 0.03, 0.15, 0.15, 0.15, 0.15, 0.15, 0.15, 0.15]

        #Mass = Vol*1.7 #density of sulftate aerosol...
    else:
        ndp = [B1, B2, B3, B4, B5, B6, B7, B8, B9, B10, B11, B12, B13, B14, B15, B16, B17, B18, B19, B20] 
        dp_unc = [0.03, 0.03, 0.03, 0.03, 0.03, 0.03, 0.03, 0.03, 0.03, 0.03, 0.03, 0.15, 0.15, 0.15, 0.15, 0.15, 0.15, 0.15, 0.15, 0.15]

    Conc = pd.Series(ndp)
    Rad =  pd.Series(np.divide(dp,2))
    dp_Unc = pd.Series(dp_unc)
    
    V = np.pi*4/3*np.power(Rad,3)
    Vol_p = Conc.mul(V)
    Vol = Vol_p.sum()
    #volume uncertainty 
    #propagate per particle volume uncertainty (using dp & power formula)
    Vol_p_u = Vol_p*(dp_Unc*3)
    #propagate total volume uncertainty (using per particle volume uncertainty & addition formula)
    Vol_unc_a = ((Vol_p_u**2).sum())**(0.5)
    #calculate relative uncertainty 
    Vol_unc = Vol_unc_a/Vol # a decimal value (%/100)
    
    if (perwt_pops >= 0.8):
        Density_lb =  DensityCalc(perwt_pops, LFEtempC)
        Density_ub =  DensityCalc(1.0, LFEtempC)
        Density = np.mean([Density_lb, Density_ub])
        Density_unc = (Density_ub - Density_lb)/Density # a percentage...
    else:
        Density =  DensityCalc(perwt_pops, LFEtempC)
        Density_unc = 0.0
    
    #propagate uncertainty (volume and density - multiplication formula)
    Unc = ((Vol_unc)**2 + (Density_unc)**2)**(0.5)
    #print(Vol_unc) #1.83 was assumed to be the density. How does this impact uncertainty?
    Mass = Vol * Density #nm3 -> cm3 aerosol   cm3 -> m3  air 1.8 g cm3 density of sulfate aerosol. final units g/ ambient m3 air 
    Mass_gpkgSTP = Mass / Pressure * Rgas * AirTemp #g m-3 Pa-1 * K * Pa  m3 K-1 kg -1 
    Mass_ugpkgSTP = Mass_gpkgSTP *1E6
    
    mol_per_vol = Vol * Density / 98.08 * 1E6 * 1E-21 #1.8 density of sulfate aerosol. final units mol/ ambient m3
    molpmolair =  mol_per_vol / Pressure * R_unv * AirTemp 
    
    MassSm2 = molpmolair *2.69E25/6.022E23 * Pressure / AirTemp * 273.15 / 100000 * 32.065 *100 #*250 m (vertical resolution)
    #print(MassSm2) # mass S per m2 at stp for binnex data
    #print(mol_per_vol*R_unv*2.69E25/6.022E23 * 273.15 / 100000 * 250 * 32.065)
    #Vol*1E-21*1E6# convert nm3/cm3 to cm3/m3 *1.8 g/cm3  PV = nRT V = n*R T/ P (m3 = J/K/kg * K * Pa)
    return MassSm2 #Mass_gpm3STP # final units g/m3 air at STP 

def MassCalc(GPS_height, Composition, perwt_pops, LFEtempC, Pressure, AirTemp, B1, B2, B3, B4, B5, B6, B7, B8, B9, B10, B11, B12, B13, B14, B15, B16, B17, B18, B19, B20):
    
    dp = DiaLookUpTable(Composition)

    #constants
    Rgas = 287.058 #J K-1 kg-1 or Pa m3 K-1 kg-1
    R_unv = 8.3144 #m3 Pa K-1 mol -1
    convFac = 1E6 #g to ug convert mass of aerosols
    
    #convert Pressure hPa to Pa
    Pressure = Pressure * 100
    GPS_m = GPS_height * 1000 #alt km to m

    
    if Composition == 'Sulfate':
        ndp = [B1, B2, B3, B4, B5, B6, B7, B8, B9, B10, B11, B12, B13, B14, B15] 
        dp_unc = [0.03, 0.03, 0.03, 0.03, 0.03, 0.03, 0.03, 0.03, 0.15, 0.15, 0.15, 0.15, 0.15, 0.15, 0.15]

    else:
        ndp = [B1, B2, B3, B4, B5, B6, B7, B8, B9, B10, B11, B12, B13, B14, B15, B16, B17, B18, B19, B20] 
        dp_unc = [0.03, 0.03, 0.03, 0.03, 0.03, 0.03, 0.03, 0.03, 0.03, 0.03, 0.03, 0.15, 0.15, 0.15, 0.15, 0.15, 0.15, 0.15, 0.15, 0.15]

    Conc = pd.Series(ndp)
    Rad =  pd.Series(np.divide(dp,2))
    dp_Unc = pd.Series(dp_unc)
    
    V = np.pi*4/3*np.power(Rad,3)
    Vol_p = Conc.mul(V)
    Vol = Vol_p.sum()
    #volume uncertainty 
    #propagate per particle volume uncertainty (using dp & power formula)
    Vol_p_u = Vol_p*(dp_Unc*3)
    #propagate total volume uncertainty (using per particle volume uncertainty & addition formula)
    Vol_unc_a = ((Vol_p_u**2).sum())**(0.5)
    #calculate relative uncertainty 
    Vol_unc = Vol_unc_a/Vol # a decimal value (%/100)
    
    if (perwt_pops >= 0.8):
        Density_lb =  DensityCalc(perwt_pops, LFEtempC)
        Density_ub =  DensityCalc(1.0, LFEtempC)
        Density = np.mean([Density_lb, Density_ub])
        Density_unc = (Density_ub - Density_lb)/Density # a percentage...
    else:
        Density =  DensityCalc(perwt_pops, LFEtempC)
        Density_unc = 0.0
    
    #propagate uncertainty (volume and density - multiplication formula)
    Unc = ((Vol_unc)**2 + (Density_unc)**2)**(0.5)
    
    #print(Vol_unc) 1.83 g/cm was assumed density
    Mass = Vol* Density *1E6 *1E-21 #nm3 -> cm3 aerosol   cm3 -> m3  air 1.8 g cm3 density of sulfate aerosol. final units g/ ambient m3 air 
    Mass_gpkgSTP = Mass / Pressure * Rgas * AirTemp #g m-3 Pa-1 * K * Pa  m3 K-1 kg -1 at STP
    
    mol_per_vol = Vol * Density / 98.08 * 1E6 * 1E-21 #1.8 density of sulfate aerosol. final units mol/ ambient m3
    molpmolair =  mol_per_vol / Pressure * R_unv * AirTemp 
    
    MassSm2 = molpmolair *2.69E25/6.022E23 * Pressure / AirTemp * 273.15 / 100000 * GPS_m * 32.065 
    #print(MassSm2) # mol / m-3 at STP
    #print(mol_per_vol*R_unv*2.69E25/6.022E23 * 273.15 / 100000 * GPS_m * 32.065)
    #Vol*1E-21*1E6# convert nm3/cm3 to cm3/m3 *1.8 g/cm3  PV = nRT V = n*R T/ P (m3 = J/K/kg * K * Pa)
    return MassSm2 #Mass_gpm3STP # final units g/m3 air at STP 

def ppCalc(Composition, perwt_pops, LFEtempC, Pressure, AirTemp, B1, B2, B3, B4, B5, B6, B7, B8, B9, B10, B11, B12, B13, B14, B15, B16, B17, B18, B19, B20):
    
    dp = DiaLookUpTable(Composition)
        
    #constants
    R_unv = 8.3144 #m3 Pa K-1 mol -1
    #convert Pressure hPa to Pa
    Pressure = Pressure * 100
    
    if Composition == 'Sulfate':
        ndp = [B1, B2, B3, B4, B5, B6, B7, B8, B9, B10, B11, B12, B13, B14, B15] 
        dp_unc = [0.03, 0.03, 0.03, 0.03, 0.03, 0.03, 0.03, 0.03, 0.15, 0.15, 0.15, 0.15, 0.15, 0.15, 0.15]
    else:
        ndp = [B1, B2, B3, B4, B5, B6, B7, B8, B9, B10, B11, B12, B13, B14, B15, B16, B17, B18, B19, B20] 
        dp_unc = [0.03, 0.03, 0.03, 0.03, 0.03, 0.03, 0.03, 0.03, 0.03, 0.03, 0.03, 0.15, 0.15, 0.15, 0.15, 0.15, 0.15, 0.15, 0.15, 0.15]

    Conc = pd.Series(ndp)
    Rad =  pd.Series(np.divide(dp,2))
    dp_Unc = pd.Series(dp_unc)
    
    V = np.pi*4/3*np.power(Rad,3)
    Vol_p = Conc.mul(V)
    Vol = Vol_p.sum()
    #volume uncertainty 
    #propagate per particle volume uncertainty (using dp & power formula)
    Vol_p_u = Vol_p*(dp_Unc*3)
    #propagate total volume uncertainty (using per particle volume uncertainty & addition formula)
    Vol_unc_a = ((Vol_p_u**2).sum())**(0.5)
    #calculate relative uncertainty 
    Vol_unc = Vol_unc_a/Vol # a decimal value (%/100)
    
    if (perwt_pops >= 0.8):
        Density_lb =  DensityCalc(perwt_pops, LFEtempC)
        Density_ub =  DensityCalc(1.0, LFEtempC)
        Density = np.mean([Density_lb, Density_ub])
        Density_unc = (Density_ub - Density_lb)/Density # a percentage...
    else:
        Density =  DensityCalc(perwt_pops, LFEtempC)
        Density_unc = 0.0
    
    #propagate uncertainty (volume and density - multiplication formula)
    Unc = ((Vol_unc)**2 + (Density_unc)**2)**(0.5)
    
    #print(Vol_unc)
    mol_per_vol = Vol* Density / 98.08 * 1E6 * 1E-21 #1.8 density of sulfate aerosol. final units mol/ ambient m3
    molpmolair =  mol_per_vol / Pressure * R_unv * AirTemp *1E9 #ppb mol/mol - mol m-3 Pa-1 K * Pa K-1 m3 mol-1 (1000 L m-3, 22.4 L per mole air STP)
    partpress_mPa = molpmolair * Pressure *1000 /1E9 #mPa
    # print('Uncertainty is: ' )
    # print(Unc)
    # print('Partial Pressure is: ') 
    # print(partpress_mPa)
    #print(molpmolair/22.4*1000) # mol / m-3 at STP
    return partpress_mPa, Unc
    #return molpmolair #final units ppb 


def eRadCalc(Composition, B1, B2, B3, B4, B5, B6, B7, B8, B9, B10, B11, B12, B13, B14, B15, B16, B17, B18, B19, B20):

    dp = DiaLookUpTable(Composition)
   
    if Composition == 'Sulfate20':
        ndp = [B1, B2, B3, B4, B5, B6, B7, B8, B9, B10, B11, B12, B13, B14, B15, B16, B17, B18, B19, B20] 
    
    else:
        ndp = [B1, B2, B3, B4, B5, B6, B7, B8, B9, B10, B11, B12, B13, B14, B15] 
        
    Conc = pd.Series(ndp)
    Rad =  pd.Series(np.divide(dp,2))
    r3 = np.power(Rad,3)
    r2 = np.power(Rad,2)
    m3= (Conc.mul(r3)).sum()
    m2= (Conc.mul(r2)).sum()
    Rad = m3/m2

    return Rad


def getRHdf():
    
    #turn this into a function also??
    #from netCDF4 import Dataset and create one dataframe seperated by launch...
    os.chdir('/Users/easher/Documents/BalloonMeasurements/ReunionIsland/WaterVapor/Data/')
    filelist = glob.glob('*')
    path = '/Users/easher/Documents/BalloonMeasurements/ReunionIsland/WaterVapor/Data'
    dfWV = pd.DataFrame()


    for file in filelist:
        
        if file == 'LM034FLT.DAT':
            file1 = '/Users/easher/Documents/BalloonMeasurements/ReunionIsland/WaterVapor/Data/LM034FLT.DAT'
            F9_cols = ['Time', 'Frame',  'Press', 'Alt', 'Temp','Traw','Theta', 'RH', 'TFp V', 'RS Bat', 'TiMetI', 'TiMetP','TiMetU',  'O3 P', 'O3 Mr', 'T Pump', 'O3ICell', 'O3Bat', 'I Pump','TFPH', 'H2OMr', 'Flag', 'TP2', 'TOptic',  'Batt', 'GPS lat', 'GPS lon', 'GPSAlt',  'Wind', 'Wind Dir',  'GPS Time']
            df1=pd.read_csv(file,sep=',', names = F9_cols, dtype=None, header = 14, engine='python')
            df1['GMTdateYmD'] = '2022-02-11'
            df1 = df1.filter(['GPSAlt', 'Press', 'H2OMr', 'RH', 'TFPH', 'Temp', 'Theta', 'Flag', 'GMTdateYmD'])
        else:
            #netCDF data
            fn = file
            ds = nc.Dataset(fn)
            WV = ds.groups['Strato']
            #extract variables from Strato group
            RH = WV['RH'][:]
            TFPH = WV['TFpHyg'][:]
            Temp = WV['Temp'][:]
            Flag = WV['Fl'][:]
            Press = WV['Press'][:]
            GPSAlt = WV['GPSalt'][:]
            H2OMr = WV['H2OMr'][:]
            Theta = WV['Theta'][:]
            data_WV = {}
            for attr in WV.ncattrs():
                data_WV[attr] = WV.getncattr(attr) 
            ds.close()
            
            #create dataframe
            df1 = pd.DataFrame()
            df1['GPSAlt'] = GPSAlt
            df1['Press'] = Press
            df1['H2OMr'] = H2OMr
            df1['RH'] = RH
            df1['TFPH'] = TFPH #degrees C 
            df1['Temp'] = Temp #degrees C 
            df1['Theta'] = Theta #degrees C
            df1['Flag'] = Flag
            df1['GMTdateYmD'] = data_WV['Date__GMT_']
    
        
        #find ascent vs descent with change Level 
        df1['NewAlt'] = df1['GPSAlt']
        df1.set_index('GPSAlt', inplace = True)
        df1b = df1.shift(periods=1, fill_value=0)
        df1b =  df1b.reset_index()
        df1b['AltShift'] = df1b['GPSAlt']-df1b['NewAlt']
        df1b['Flight'] = df1b['AltShift'].apply(lambda x: 'Ascent' if x >= 0 else 'Descent')
        df1b = df1b.drop(['AltShift', 'NewAlt', 'Flag'], axis=1)
        df1b = df1b.drop([0]) #dont use WV CFH data if flag is 0.
        #df1 = df1[df1['Flag'] == 1] #dont use WV CFH data if flag is 0.
        df1b = df1b[df1b['H2OMr'] < 3000] #why is this here?
        
        dfWV = dfWV.append(df1b)
        del df1
         
    #ascent data only and change date conventino to US 
    dfWV = dfWV[dfWV['Flight']=='Ascent']
    
    dfWVadd = dfWV[dfWV['GMTdateYmD']=='22-01-2022']
    dfWVadd.reset_index(drop=True, inplace = True) # need to make clear not changing the date, but using one of these profiles for two corresponding POPS profiles..
    dfWVadd["GMTdateYmD"]= dfWVadd["GMTdateYmD"].str.replace("22-01-2022", "2022-01-23", case = False) 
    
    #dfWV['RHcorr'] = np.where(dfWV.RH < 25, 50, dfWV.RH)
    dfWV["GMTdateYmD"]= dfWV["GMTdateYmD"].str.replace("21-01-2022", "2022-01-21", case = False) 
    dfWV["GMTdateYmD"]= dfWV["GMTdateYmD"].str.replace("22-01-2022", "2022-01-22", case = False) 
    dfWV["GMTdateYmD"]= dfWV["GMTdateYmD"].str.replace("23-01-2022", "2022-01-23", case = False) 
    dfWV["GMTdateYmD"]= dfWV["GMTdateYmD"].str.replace("24-01-2022", "2022-01-24", case = False) 
    dfWV = pd.concat([dfWV, dfWVadd], axis = 0)
    
    return dfWV

def getSO2df():
    #Read in SO2 data and bin
    file_SO2 = '/Users/easher/Documents/Tonga/SO2_Data.csv'
    df_SO2_i = pd.read_csv(file_SO2, sep=",", engine = 'python')
    
    conditions = [(df_SO2_i['GMTdateYmD'] == '1/22/22') , (df_SO2_i['GMTdateYmD'] == '1/24/22') , (df_SO2_i['GMTdateYmD'] == '1/25/22')]
        
    choices = ['﻿Launch 2 - 01/22', '﻿Launch 5 - 01/24', '﻿Launch 8 - 01/25']
    df_SO2_i['LaunchNo']=np.select(conditions, choices, default = 'Unassigned')
        
    df_SO2_i = df_SO2_i[df_SO2_i.GPSAlt > 0.5]
    df_SO2_i['Altitude (km)'] = pd.cut(df_SO2_i['GPSAlt'], bins=np.linspace(0.125, 30.125, 121),labels = AltLabels, include_lowest=True) # 301? bin every 100 m;  if 61 bin every 500m (0, 32, 33) (0.125, 30.125, 121)
    df_SO2_i['Altitude (km)'] = pd.to_numeric(df_SO2_i['Altitude (km)'], errors='coerce').fillna(0) 
    df_SO2_i['PartialPressure_v1'] = df_SO2_i['SO2_MR']*1E-9* df_SO2_i['Pressure']*100*1000 #mPa
    df_SO2_i[df_SO2_i.SO2_MR < 0.001] = np.nan
    df_SO2 = df_SO2_i.groupby(['GMTdateYmD', 'LaunchNo', 'Altitude (km)'],as_index=False).mean()
    df_SO2['Mass'] = df_SO2['SO2_MR'] * 1E-9 * df_SO2['Pressure']*100/287.058/(df_SO2['Temperature'] + 273.15)*32.065*1.225*100 # gS kg->m3->m2 *250 if 250 m increments
    df_SO2['SO2_molpm3atSTP'] = df_SO2['SO2_MR'] * 1E-9 * df_SO2['Pressure']*100/8.3144/(df_SO2['Temperature'] + 273.15) # mol m-3
    df_SO2['MR'] = df_SO2['SO2_MR'] #not at STP. 
    
    df_SO2_sum = df_SO2.groupby(['LaunchNo'],as_index=False).sum()
    df_SO2_f = df_SO2.filter([ 'GMTdateYmD', 'LaunchNo', 'Altitude (km)', 'MR', 'PartialPressure_v1'])
    
    df_SO2_f['GMTdateYmD']= df_SO2_f['GMTdateYmD'].str.replace("1/25/22", "2022-01-25", case = False) 
    df_SO2_f['GMTdateYmD']= df_SO2_f['GMTdateYmD'].str.replace("1/22/22", "2022-01-22", case = False) 
    df_SO2_f['GMTdateYmD']= df_SO2_f['GMTdateYmD'].str.replace("1/24/22", "2022-01-24", case = False) 
    
    return df_SO2_f

def LLD(launchNo, PartialPressure):
    if launchNo == '﻿Launch 2 - 01/22':
        DDL = 0.03
        
    elif launchNo == '﻿Launch 5 - 01/24':
        DDL = 0.02
        
    elif launchNo == '﻿Launch 8 - 01/25':
        DDL = 0.03
    #DDL = 0.01
    if PartialPressure <= DDL:
        PartialPress_corr = np.NaN
    else:
        PartialPress_corr = PartialPressure
    return PartialPress_corr

def getPOPSdf():
    #read in POPS data
    os.chdir('/Users/easher/Desktop/Tonga/Submission/POPSDataForPublication/Finalized/')
    filelist = glob.glob('*.csv')
    path = '/Users/easher/Desktop/Tonga/Submission/POPSDataForPublication/Finalized'
    df = pd.DataFrame()
    for file in filelist:
        print(file)
            
        df1= pd.read_csv(file, delimiter = ',', header=17, dtype=None, low_memory=False)
        df1.head()

        df1['GPSwdRad'] = np.deg2rad(270-df1['GPSwdDeg'])
        df1['U'] = df1['GPSwsms']*np.cos(df1['GPSwdRad'])
        df1['V'] = df1['GPSwsms']*np.sin(df1['GPSwdRad'])
        
    
        #correct date formating
        df1["GMTdateYmD"]= df1["GMTdateYmD"].str.replace("1/24/22", "2022-01-24", case = False) 
        df1["GMTdateYmD"]= df1["GMTdateYmD"].str.replace("1/23/22", "2022-01-23", case = False) 
        df1["GMTdateYmD"]= df1["GMTdateYmD"].str.replace("1/22/22", "2022-01-22", case = False) 
        df1["GMTdateYmD"]= df1["GMTdateYmD"].str.replace("1/25/22", "2022-01-25", case = False)
        df1['LaunchNo'] = (file[0:6])
        
        #Filter POPS data based on various conditions...
        df1 = df1[df1.POPSflowCCpS >= 3.0]
        df1[df1 == 99999] = np.NAN
        #df1 = df1[df1.POPSbaselineStdDev <= 13]
        #could create a function to do this cleaning...            
        df1['POPSflowCCpS'] = pd.to_numeric(df1['POPSflowCCpS'], errors='coerce').fillna(0)    
        #flow corrected for differences in POPS temperature and outside air temperature (in Kelvin)
        df1['POPSTempCorrection'] = (df1['iMetTempCorrC']+273.15)/(df1['POPStempC']+273.15)
        df1['POPSflowCorrCCpS'] = df1['POPSflowCCpS']*df1['POPSTempCorrection']
        df1['ratioWidth2Flow'] = df1['POPSavgWdithUS']/df1['POPSflowCCpS']
        if (file == 'run009_one_pops_20220211.csv'):
            df1 = df1[df1["GPSaltkm"] <= 25.25]
        if (file == 'run008_so2_pops_20220125.csv'):
                df1 = df1[df1["GPSaltkm"] <= 21.0]
        if (file == 'run004_so2_pops_20220123.csv'):
                df1['B1'] = np.NAN
                df1['B2'] = np.NAN 
        df = df.append(df1)
        del df1

    
    #create binned altitude and Flight (ascent/descent) for merge
    df['Flight'] = df['iMetASCENTms'].apply(lambda x: 'Ascent' if x > 0 else 'Descent')
    df = df[df['Flight'] == 'Ascent'] #only ascent data
    
    #need Air temperature in Kelvin
    df['Air Temp (K)'] = df['iMetTempCorrC']+ 273.15
    
    #calculate basic number concentration(s) (#/ cm-3) and surface area
    df[['B1', 'B2','B3','B4', 'B5', 'B6', 'B7','B8', 'B9', 'B10', 'B11', 'B12', 'B13', 'B14','B15', 'B16', 'B17', 'B18', 'B19', 'B20']]\
    = df[['B1', 'B2','B3','B4', 'B5', 'B6', 'B7','B8', 'B9', 'B10', 'B11', 'B12', 'B13','B14','B15', 'B16', 'B17', 'B18', 'B19', 'B20']].div(df['POPSflowCorrCCpS'].values,axis=0)
    
    df['Composition'] = df.apply(lambda x: Plume(x['LaunchNo'], x['Flight'], x['GPSaltkm']), axis=1)
    df['Aer_Conc'] =  df[['B1', 'B2','B3','B4', 'B5', 'B6', 'B7','B8', 'B9','B10', 'B11', 'B12', 'B13', 'B14', 'B15', 'B16', 'B17', 'B18', 'B19', 'B20']].sum(axis=1)
    df['Dry Surface Area Density'] = df.apply(lambda x: CalcSurfArea(x['Composition'], x['B1'], x['B2'], x['B3'], x['B4'], x['B5'], x['B6'], x['B7'], x['B8'], x['B9'], x['B10'], x['B11'], x['B12'], x['B13'], x['B14'], x['B15'], x['B16'], x['B17'], x['B18'], x['B19'], x['B20']), axis=1)
    #df['Effective Radius'] = df.apply(lambda x: eRadCalc(x['Composition'], x['B1'], x['B2'], x['B3'], x['B4'], x['B5'], x['B6'], x['B7'], x['B8'], x['B9'], x['B10'], x['B11'], x['B12'], x['B13'], x['B14'], x['B15'], x['B16'], x['B17'], x['B18'], x['B19'], x['B20']), axis=1)
    
    
    df['Altitude (km)'] = pd.cut(df['GPSaltkm'], bins=np.linspace(0.125, 30.125, 121),labels = AltLabels, include_lowest=True) # 301? bin every 100 m;  if 61 bin every 500m (0.125, 30.125, 121)
    df['Altitude (km)'] = pd.to_numeric(df['Altitude (km)'], errors='coerce').fillna(0) 

    return df

def GoffGratch(TFpHyg, Temp):
    #GoffGratch 1984 formulation
    TFpHyg = TFpHyg + 273.15
    Temp = Temp + 273.15
    # if (Temp < 273.15):
    #     e = 10**(-9.09718*(273.16/TFpHyg - 1) - 3.56654*math.log10(273.16/TFpHyg) + 0.876793*(1-TFpHyg/273.16) + math.log10(6.1071))
    #     es = 10**(-9.09718*(273.16/Temp - 1) - 3.56654*math.log10(273.16/Temp) + 0.876793*(1-Temp/273.16) + math.log10(6.1071))
    #     RH = e/es*100
    # else:
    #     e = 10**(-9.09718*(273.16/TFpHyg - 1) - 3.56654*math.log10(273.16/TFpHyg) + 0.876793*(1-TFpHyg/273.16) + math.log10(6.1071))
    #     es = 10**(-0.58002206*10**4/Temp + 0.13914993*10**1 - 0.48640239*10**(-1)*Temp + 0.41764768*10**-4*Temp**2 -0.14452093*10**(-7)*Temp**3 + 0.65459673*10**1*math.log10(Temp))
    #     RH = e/es*100
    
    if (Temp < 273.15):
        e =  10**(-9.09718*(273.16/TFpHyg - 1) - 3.56654*math.log10(273.16/TFpHyg) + 0.876793*(1-TFpHyg/273.16) + math.log10(6.1071))
        es = 10**(-9.09718*(273.16/Temp - 1) -   3.56654*math.log10(273.16/Temp) +   0.876793*(1-Temp/273.16) +   math.log10(6.1071))
        RH = e/es*100

    else:
        e = 10**(-9.09718*(273.16/TFpHyg - 1) - 3.56654*math.log10(273.16/TFpHyg) + 0.876793*(1-TFpHyg/273.16) + math.log10(6.1173))
        es = (math.exp(-0.58002206*10**4/Temp + 0.13914993*10**1 - 0.48640239*10**(-1)*Temp + 0.41764768*10**-4*Temp**2 -0.14452093*10**(-7)*Temp**3 + 0.65459673*10**1*math.log(Temp)))/100
        RH = e/es*100
        
    return (RH, e)

#Lookup table for H2SO4 aerosol %wt given partial pressure of H2O and T (K) 
#Tabazadeh et al., 1997 GRL vol. 24 (15)
def PwLookuptable(T1, Pwa):
    
    Pw = np.round_(Pwa, decimals = 5)
    #print('Lookup table Pw : ' + str(Pw))
    
    T = T1 + 273.15 #can interpolate between table values..
    #Pwa must be in hPa/mb
    Wi = np.linspace(0.1, 0.8, 15) #can increase this to be more numbers. use scipi interp1d fill_value = 'exprapolate' if a need...
    W_0 = np.linspace(0.1, 0.8, 701) #71 to the 10's place (or nearest % of % wt)
    ai = [19.726, 19.747, 19.761, 19.794, 19.883, 20.078, 20.379, 20.637, 20.682, 20.55, 20.405, 20.383, 20.585, 21.169, 21.808 ]
    bi = [-4364.8, -4390.9, -4414.7,  -4451.1, -4519.2, -4644.0, -4828.5, -5011.5, -5121.3, -5177.6, -5252.1, -5422.4, -5743.8, -6310.6, -6985.9 ]
    ci = [-147620, -144690, -142940, -140870, -136500, -127240, -112550, -98811, -94033, -96984, -100840, -97966, -83701, -48396, -12170]
    a = pd.Series(np.interp(W_0, Wi, ai))
    b = pd.Series(np.interp(W_0, Wi, bi))
    c = pd.Series(np.interp(W_0, Wi, ci))
    W_1 = pd.Series(W_0)
    
    PwTable = pd.DataFrame({'a': a, 'b': b, 'c': c, 'W_1': W_1})#(a, b, c, W_1)
    PwTable['T'] = T
    
    PwTinit = pd.DataFrame({'a': ai, 'b': bi, 'c': ci, 'W': Wi})
    PwTinit['T'] = T
    
    PwTinit['Pw'] = PwTinit.apply(lambda x: PwCalc(x['a'], x['b'], x['c'], x['T']), axis=1)
    PwTable['Pw_calc'] = PwTable.apply(lambda x: PwCalc(x['a'], x['b'], x['c'], x['T']), axis=1)
    #create dictionary and look up H2SO4 weight fraction (apply to mass)
    PwTable.drop(['a', 'b', 'c', 'T'], axis = 1, inplace=True)
    #PwDict = pd.Series(PwTable.W_1.values,index=PwTable.Pw_calc).to_dict()
    PwDict = pd.Series(PwTable.Pw_calc.values,index=PwTable.W_1).to_dict()
    #find the closest dictionary look-up value
    W, Pw_t = min(PwDict.items(), key=lambda x: abs(Pw - x[1]))

    return W

#use polynomial to calc. eq. partial pressure of H2O given T
def PwCalc(a, b, c, T):
        Pw_0 = math.exp(a + b/T + c/(T**2))
        Pw = np.round_(Pw_0, decimals = 5)
        return Pw

#density parameterization Oca et al., 2018 J. Chem. & Eng. Data vol. 63 (9)
def DensityCalc(perWt, Tc):
        T = Tc + 273.15
        PolyCoeff = [1022, -0.5076, 2.484E-4, 976.4, -1.015, 1, 237.8, 1, 1]
        Density = PolyCoeff[0] + PolyCoeff[1] * T + PolyCoeff[2] * T**2 + PolyCoeff[3] * perWt + PolyCoeff[4] + PolyCoeff[5] * perWt**2 
        Density = Density / 1000 #convert from kg/m3 to g/cm3
        return Density
    
#second way to calculate bin diameter change
def Jonsson1995_calcDamb(Composition, Fpops, rho_pops, Famb, rho_amb):
    
    dp = DiaLookUpTable(Composition)
    dpAmb = []
        
    for i in dp:
        Dpops = i
        Damb = Dpops*(Fpops*rho_pops/(Famb*rho_amb))**(1/3)
        dpAmb.append(Damb)
    dpAmbn= np.array(dpAmb).tolist()
       
    return dpAmbn
   
def Plume(LaunchNo, Flight, Altitude):
    if (LaunchNo == 'run011'):
        composition = 'Sulfate'
    elif (LaunchNo == 'run013'):
        composition = 'Sulfate'
    else:
        composition = 'Sulfate20'
    return composition


def Sample(LaunchDate, Altitude):
    if ((LaunchDate == '2022-01-22') & ((Altitude > 26.25) & (Altitude < 26.75))):
        Stype = 'Jan 22 18 UTC, 26.5 km'
    elif ((LaunchDate == '2022-03-31') & ((Altitude > 21.25) & (Altitude < 21.75))):
        Stype = 'Mar 31, 21.5 km'
    else:
        Stype = 'None'
    return Stype



def Trop(Launch, Altitude):
    if (Launch == 'run002'):
        Trop = 16.5
    elif (Launch == 'run003'):
        Trop = 16.75
    elif (Launch == 'run005'):
        Trop = 17.5
    elif (Launch == 'run008'):
        Trop = 17.75
    elif (Launch == 'run009'):
        Trop = 17.5
    elif (Launch == 'run011'):
        Trop = 16.75
    elif (Launch == 'run013'):
        Trop = 16.75
    elif (Launch == 'run000'):
        Trop = 12.5
    elif (Launch == 'run012'):
        Trop = 15.0
    else:
        Trop = 0
        
    Height_above = Altitude - Trop
    return Height_above

def WindDirection(Degrees):
    if ((Degrees <= 22.5) | (Degrees > 337.5)):
        WD = 'N'
    elif (Degrees <= 67.5):
        WD = 'NE'
    elif (Degrees <= 112.5):
        WD = 'E'
    elif (Degrees <= 157.5):
        WD = 'SE'
    elif (Degrees <= 202.5):
        WD = 'S'
    elif (Degrees <= 247.5):
        WD = 'SW'
    elif (Degrees <= 292.5):
        WD = 'W'
    else:
        WD = 'NW'
    return WD

palette2 = {"Measured":"black",
                "Calculated using Johnson et al. 1995":"darkred", 
                "Calculated using Kappa-Koehler Thoery":"salmon"
                }

AltLabels = ["0.25", "0.5","0.75", "1", "1.25", "1.5", "1.75", "2", "2.25", "2.5", "2.75", "3", "3.25", "3.5", "3.75", "4","4.25", "4.5", "4.75", "5", "5.25", "5.5", "5.75", "6",\
              "6.25", "6.5", "6.75", "7", "7.25", "7.5", "7.75", "8", "8.25", "8.5", "8.75", "9", "9.25", "9.5", "9.75", "10",\
              "10.25", "10.5", "10.75", "11", "11.25", "11.5", "11.75", "12", "12.25", "12.5", "12.75", "13", "13.25", "13.5", "13.75", "14",\
                  "14.25", "14.5", "14.75", "15", "15.25", "15.5", "15.75", "16", "16.25", "16.5", "16.75", "17", "17.25", "17.5", "17.75", "18",\
                      "18.25", "18.5", "18.75", "19", "19.25", "19.5", "19.75", "20", "20.25", "20.5", "20.75", "21", "21.25", "21.5", "21.75", "22",\
                          "22.25", "22.5", "22.75", "23", "23.25", "23.5", "23.75", "24", "24.25", "24.5", "24.75", "25", "25.25", "25.5", "25.75", "26",\
                              "26.25", "26.5", "26.75", "27", "27.25", "27.5", "27.75", "28", "28.25", "28.5", "28.75", "29",\
                      "29.25", "29.5", "29.75", "30.00"]
    
dfWV = pd.read_csv('/Users/easher/Desktop/Tonga/Submission/POPSDataForPublication/H2O_table.csv', delimiter = ',', dtype=None, header = 11, engine='python')
dfWV['Flight'] = 'Ascent'
dfWV_plume = dfWV
dfWV_plume = dfWV
dfWV_plume['Sample']  = dfWV_plume.apply(lambda x: Sample(x['GMTdateYmD'], x['Altitude (km)']), axis=1)
dfWV_plume1 = dfWV_plume.groupby(['Sample'],as_index=False).median()
dfWV_plume1 = dfWV_plume1[dfWV_plume1['Sample']!='None']

#del dfWV
del dfWV_plume


#df_SO2_f = getSO2df() 
df_SO2_f = pd.read_csv('/Users/easher/Desktop/Tonga/Submission/POPSDataForPublication/SO2_table.csv', delimiter = ',', dtype=None, header = 11, engine='python')
#df_SO2_f['Compound'] = 'SO2'

df = getPOPSdf()


df1= df.groupby(['GMTdateYmD','LaunchNo', 'Altitude (km)', 'Flight'],as_index=False).mean() # was mean. is median better or more consistent with dfWV?

#merge with POPS data pased on altitude and flight

new_df = pd.merge(df1, dfWV, how='left', left_on=['GMTdateYmD','Altitude (km)', 'Flight'], right_on = ['GMTdateYmD', 'Altitude (km)', 'Flight'])

new_df['RH_CFH'] = np.where(new_df.RH_CFH > 200, np.NAN, new_df.RH_CFH)
new_df['RH_frostpoint'] = np.where(((new_df.LaunchNo == 'run013') | (new_df.LaunchNo == 'run011')), new_df.hygrometerRH, new_df.RH_CFH)
new_df['H2OMr_frostpoint'] = np.where(((new_df.LaunchNo == 'run013') | (new_df.LaunchNo == 'run011')), new_df.H2OmrPPMV, new_df.H2OMr)
new_df['Pw_frostpoint'] = np.where(((new_df.LaunchNo == 'run013') | (new_df.LaunchNo == 'run011')), new_df.Pw_FPHc, new_df.Pw_CFH)


##calculate ambient bin size based on WV on that launch date...
new_df['Composition'] = new_df.apply(lambda x: Plume(x['LaunchNo'], x['Flight'], x['Altitude (km)']), axis=1)

new_df['BinDp_meas'] = new_df.apply(lambda x: DiaLookUpTable(x['Composition']), axis=1)
new_df['BinDp_KK'] = new_df.apply(lambda x: AmbAer(x['RH_frostpoint'], x['Air Temp (K)'], x['Composition']), axis=1)
#$wt calc
new_df['H2SO4_%wt_amb'] = new_df.apply(lambda x: PwLookuptable(x['iMetTempCorrC'], x['Pw_frostpoint']), axis = 1)
new_df['H2SO4_%wt_pops'] = new_df.apply(lambda x: PwLookuptable(x['POPStempC'], x['Pw_frostpoint']), axis = 1)
#density calculation
new_df['Density_amb'] = new_df.apply(lambda x: DensityCalc(x['H2SO4_%wt_amb'], x['iMetTempCorrC']), axis = 1)
new_df['Density_pops'] = new_df.apply(lambda x: DensityCalc(x['H2SO4_%wt_pops'], x['POPStempC']), axis = 1)
#%Wt amb diameter calculation...
new_df['BinDp_J'] = new_df.apply(lambda x: Jonsson1995_calcDamb(x['Composition'], x['H2SO4_%wt_pops'], x['Density_pops'], x['H2SO4_%wt_amb'], x['Density_amb']), axis=1)

new_df = new_df.filter(['Altitude (km)', 'Launch','GMTdateYmD', 'LaunchNo', 'Flight', 'Composition', 'H2SO4_%wt_amb', 'Density_amb', 'RH_frostpoint', 'BinDp_meas','BinDp_KK', 'BinDp_J', 'B1', 'B2', 
                      'B3','B4','B5','B6','B7','B8','B9','B10','B11', 'B12','B13', 'B14','B15', 'B16', 'B17', 'B18', 'B19', 'B20', 
                      'Dry Surface Area Density'], axis=1)

df2 = new_df[new_df['Altitude (km)'] >= 2]
#save part of df2.

a = pd.DataFrame(df2['BinDp_meas'])

b = pd.concat([pd.DataFrame(a['BinDp_meas'].values.tolist()) for c in a.columns], 
                 axis=1, 
                 keys=a.columns)

b.columns = ['{}{}'.format(i, j) for i, j in b.columns]

df2.reset_index(drop=False, inplace = True)
b.reset_index(drop=False, inplace = True)
df2 = pd.concat([df2, b], axis = 1)
df2.reset_index(drop=True, inplace = True)

a = pd.DataFrame(df2['BinDp_KK'])

b = pd.concat([pd.DataFrame(a['BinDp_KK'].values.tolist()) for c in a.columns], 
                 axis=1, 
                 keys=a.columns)

b.columns = ['{}{}'.format(i, j) for i, j in b.columns]

df2.reset_index(drop=False, inplace = True)
b.reset_index(drop=False, inplace = True)
df2 = pd.concat([df2, b], axis = 1)
df2.reset_index(drop=True, inplace = True)
df2.drop(columns=['level_0','index'], inplace = True)

a = pd.DataFrame(df2['BinDp_J'])

b = pd.concat([pd.DataFrame(a['BinDp_J'].values.tolist()) for c in a.columns], 
                 axis=1, 
                 keys=a.columns)

b.columns = ['{}{}'.format(i, j) for i, j in b.columns]


df2.reset_index(drop=False, inplace = True)
b.reset_index(drop=False, inplace = True)
df2 = pd.concat([df2, b], axis = 1)
df2.reset_index(drop=True, inplace = True)

df2.drop(columns=['index'], inplace = True)


Bins =['B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B9', 'B10', 
                            'B11', 'B12', 'B13', 'B14', 'B15', 'B16', 'B17', 'B18', 'B19', 'B20']

Dp_J = ['BinDp_J0', 'BinDp_J1', 'BinDp_J2', 'BinDp_J3', 'BinDp_J4', 'BinDp_J5', 'BinDp_J6', 'BinDp_J7', 'BinDp_J8', 'BinDp_J9', 'BinDp_J10', 
                            'BinDp_J11', 'BinDp_J12', 'BinDp_J13', 'BinDp_J14', 'BinDp_J15', 'BinDp_J16', 'BinDp_J17', 'BinDp_J18', 'BinDp_J19']

Dp_KK = ['BinDp_KK0', 'BinDp_KK1', 'BinDp_KK2', 'BinDp_KK3', 'BinDp_KK4', 'BinDp_KK5', 'BinDp_KK6', 'BinDp_KK7', 'BinDp_KK8', 'BinDp_KK9', 'BinDp_KK10', 
                            'BinDp_KK11', 'BinDp_KK12', 'BinDp_KK13', 'BinDp_KK14', 'BinDp_KK15', 'BinDp_KK16', 'BinDp_KK17', 'BinDp_KK18', 'BinDp_KK19']

Dp_meas = ['BinDp_meas0', 'BinDp_meas1', 'BinDp_meas2', 'BinDp_meas3', 'BinDp_meas4', 'BinDp_meas5', 'BinDp_meas6', 'BinDp_meas7', 'BinDp_meas8', 'BinDp_meas9', 'BinDp_meas10', 
                            'BinDp_meas11', 'BinDp_meas12', 'BinDp_meas13', 'BinDp_meas14', 'BinDp_meas15', 'BinDp_meas16', 'BinDp_meas17', 'BinDp_meas18', 'BinDp_meas19']

Dp = ['BinDp0', 'BinDp1', 'BinDp2', 'BinDp3', 'BinDp4', 'BinDp5', 'BinDp6', 'BinDp7', 'BinDp8', 'BinDp9', 'BinDp10', 
                             'BinDp11', 'BinDp12', 'BinDp13', 'BinDp14', 'BinDp15', 'BinDp16', 'BinDp17', 'BinDp18', 'BinDp19']

df2['Sample']  = df2.apply(lambda x: Sample(x['GMTdateYmD'], x['Altitude (km)']), axis=1)
df2 = df2[df2['Sample'] != 'None']
#use melt function three times to create datasets like dry and ambient. then merge them...then plot with seaborn relplot
df_J = pd.melt(df2, id_vars=['Sample'], value_vars= Dp_J, var_name = 'DpBin_J', value_name = 'Dp_amb_calc')
df_meas = pd.melt(df2,id_vars=['Sample'], value_vars= Dp_meas, var_name = 'DpBin_meas', value_name = 'Dp_meas')
df_KK = pd.melt(df2,id_vars=['Sample'], value_vars= Dp_KK, var_name = 'DpBin_KK', value_name = 'Dp_amb_calc')

df_conc = pd.melt(df2,id_vars=['Sample'], value_vars= Bins, var_name = 'Bin', value_name = 'Concentration')

df_J['Sizing'] = 'Calculated using Johnson et al. 1995'
df_KK['Sizing'] = 'Calculated using Kappa-Koehler Thoery'

df_J.drop(columns=['DpBin_J'], inplace = True)
df_meas.drop(columns=['DpBin_meas'], inplace = True)
df_KK.drop(columns=['DpBin_KK'], inplace = True)

df_c = pd.concat([df_KK, df_J])
df_m = pd.concat([df_meas, df_meas])
df_conc_1 = pd.concat([df_conc, df_conc])

df_sc = pd.concat([df_m, df_c, df_conc_1], axis = 1)

df_meas['Sizing'] = 'Measured'
df_J['Dp'] = df_J['Dp_amb_calc']
df_KK['Dp'] = df_KK['Dp_amb_calc']
df_meas['Dp'] = df_meas['Dp_meas']

df_J.drop(columns=['Dp_amb_calc'], inplace = True)
df_meas.drop(columns=['Dp_meas'], inplace = True)
df_KK.drop(columns=['Dp_amb_calc'], inplace = True)

df_sz = pd.concat([df_meas, df_KK, df_J])
df_sz['Sample1'] = df_sz['Sample']
df_sz.drop(columns = ['Sample'], inplace = True)
df_conc_2 = pd.concat([df_conc, df_conc, df_conc])
df_sc2 = pd.concat([df_sz, df_conc_2], axis = 1)

seaborn.set(style="whitegrid")
seaborn.set(font_scale = 1)
seaborn.set(rc={"xtick.bottom" : True, "ytick.left" : True})
g = seaborn.relplot(x="Dp", y="Concentration",  hue = 'Sizing', 
                     style = 'Sample', kind="line", marker = '.', size="Sizing", palette=palette2, data=df_sc2, facet_kws={'sharey': True, 'sharex': True}) #

g.set_axis_labels("Dp (nm)", "dN/dlogDp (# $\mathregular{cm^{-3}}$)")    

g.set(xscale="log")
g.set(yscale="log")
leg = g._legend
leg.set_bbox_to_anchor([0.6, 0.99])  # coordinates of lower left of bounding box
leg._loc = 1  # if required you can set the loc
g.set(ylim = [0.3, 700])
palette2 = {"Measured":"dimgray",
                "Calculated using Johnson et al. 1995":"peru", 
                "Calculated using Kappa-Koehler Thoery":"darkred"
                }
