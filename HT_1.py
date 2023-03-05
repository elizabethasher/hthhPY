#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 15 10:34:54 2023
@author: elizabeth.asher@noaa.gov
If using this code for publications, please contact elizabeth.asher@noaa.gov to discuss co-authorship and/or cite Asher et al., in review PNAS 2023.
"""
import os
from datetime import datetime as dt
import pandas as pd
import numpy as np
import glob
import netCDF4 as nc
from matplotlib import pyplot as plt
import seaborn as seaborn
from scipy.optimize import fsolve
import math
import PyMieScatt

#functions
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
       
        #if you do not want to use a look up table, you can assign sizes here as a list
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
def VPsca(Wavelength, IR, LaunchNo, Composition, RH, T, Fpops, rho_pops, Famb, rho_amb, B1, B2, B3, B4, B5, B6, B7, B8, B9, B10, B11, B12, B13, B14, B15, B16, B17, B18, B19, B20):

    if Composition == 'Sulfate20':
        ndp = [B1, B2, B3, B4, B5, B6, B7, B8, B9, B10, B11, B12, B13, B14, B15, B16, B17, B18, B19, B20] 
    else:
        ndp = [B1, B2, B3, B4, B5, B6, B7, B8, B9, B10, B11, B12, B13, B14, B15] 
        
    dpAmbnp_K = KKcalcDamb(RH, T, Composition)
    dpAmbnp_J = Jonsson1995_calcDamb(Composition, Fpops, rho_pops, Famb, rho_amb)
    
    if LaunchNo == 'run004':
        ndp = [B3, B4, B5, B6, B7, B8, B9, B10, B11, B12, B13, B14, B15, B16, B17, B18, B19, B20]
        dpAmbnp_K = dpAmbnp_K[2:]
        dpAmbnp_J = dpAmbnp_J[2:]

    [Bext_K, Bsca_K, Babs_K, bigG_K, Bpr_K, Bback_K, Bratio_K] = PyMieScatt.Mie_SD(IR, Wavelength, dpAmbnp_K, ndp, nMedium=1.0, interpolate=True, SMPS = True,  asDict=False) 
    [Bext_J, Bsca_J, Babs_J, bigG_J, Bpr_J, Bback_J, Bratio_J] = PyMieScatt.Mie_SD(IR, Wavelength, dpAmbnp_J, ndp, nMedium=1.0, interpolate=True, SMPS = True,  asDict=False) 

    return Bext_K, Bext_J

def KKcalcDamb(RH, T, Composition):
    
    dp = DiaLookUpTable(Composition) #assuming Diameter with RH = 0% (at equilibirum)
    dpAmb = []
        
    Kchem = 0.87 # for H2SO4
    for i in dp:
        Dd = i
        func = lambda Drh : (Drh**3 - Dd**3)/ (Drh**3 - Dd**3*(1-Kchem))*np.exp(4*0.072*0.018/(8.314*T*1000*Drh))-RH/100
        Drh_initial_guess = Dd # this is true unless RH is very high and the particle is very small... then its size may change a lot
        Drh_solution = fsolve(func, Drh_initial_guess)
        dpAmb.append(Drh_solution[0])
    dpAmbn= np.array(dpAmb).tolist() #calculated aerosol Diameter with ambient RH and T)
    
    #Diameter difference (not plotted here)
    DpDiff = pd.DataFrame({'Dp':dp, 'dpAmbn':dpAmbn})
    DpDiff['Diff'] =  DpDiff['dpAmbn'] - DpDiff['Dp']
    DpDiff['Ratio'] = DpDiff['dpAmbn'] / DpDiff['Dp']

    DpDiff['RH'] = RH

    return dpAmbn

def SADrh(Composition, RH, T, Fpops, rho_pops, Famb, rho_amb, B1, B2, B3, B4, B5, B6, B7, B8, B9, B10, B11, B12, B13, B14, B15, B16, B17, B18, B19, B20): #was based on date, can do based on composition, no?
        
    if Composition == 'Sulfate20':

        ndp = [B1, B2, B3, B4, B5, B6, B7, B8, B9, B10, B11, B12, B13, B14, B15, B16, B17, B18, B19, B20] 
            
    else:
        ndp = [B1, B2, B3, B4, B5, B6, B7, B8, B9, B10, B11, B12, B13, B14, B15] 
        

    dpAmbn_K = KKcalcDamb(RH, T, Composition)  
    dpAmbn_J = Jonsson1995_calcDamb(Composition, Fpops, rho_pops, Famb, rho_amb)
        
    Conc = pd.Series(ndp)
    Rad_K =  pd.Series(np.divide(dpAmbn_K,2))*1000
    Area_K = np.pi*4*np.power(Rad_K,2)
    AmbSurfaceArea_K = (Conc.mul(Area_K)).sum()*1E-6
    Conc = pd.Series(ndp)
    Rad_J =  pd.Series(np.divide(dpAmbn_J,2))*1000
    Area_J = np.pi*4*np.power(Rad_J,2)
    AmbSurfaceArea_J = (Conc.mul(Area_J)).sum()*1E-6
    
    return AmbSurfaceArea_K, AmbSurfaceArea_J


def MassBCalc(LaunchNo, Composition, perwt_pops, LFEtempC, Pressure, AirTemp, B1, B2, B3, B4, B5, B6, B7, B8, B9, B10, B11, B12, B13, B14, B15, B16, B17, B18, B19, B20):
    
    dp = DiaLookUpTable(Composition)
    
    #constants
    Rgas = 287.058 #J K-1 kg-1 or Pa m3 K-1 kg-1
    R_unv = 8.3144 #m3 Pa K-1 mol -1
    convFac = 1E6 #g to ug convert mass of aerosols
    Pressure = Pressure * 100  #convert Pressure hPa to Pa
    VertRes = 100 #(m) vertical resolution
    
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
    
    #print(Vol_unc) #1.83 was assumed to be the density. How does this impact uncertainty?
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
    Mass = Vol * Density *1E6 *1E-21 #nm3 -> cm3 aerosol   cm3 -> m3  air; 1.8 g cm3 density of sulfate aerosol. final units g/ ambient m3 air 
    Mass_ugpkgSTP = Mass / Pressure * Rgas * AirTemp *1E6 #g m-3 Pa-1 * K * Pa  m3 K-1 kg -1 *ug g-1 aerosol

    
    mol_per_vol = Vol * Density / 98.08 * 1E6 * 1E-21 # final units mol/ ambient m3
    molpmolair =  mol_per_vol / Pressure * R_unv * AirTemp 
    
    MassSm2 = molpmolair *2.69E25/6.022E23 * Pressure / AirTemp * 273.15 / 100000 * 32.065 *VertRes 
    #print(mol_per_vol*R_unv*2.69E25/6.022E23 * 273.15 / 100000 * 250 * 32.065)
    #Vol*1E-21*1E6# convert nm3/cm3 to cm3/m3 *1.8 g/cm3  PV = nRT V = n*R T/ P (m3 = J/K/kg * K * Pa)
    return MassSm2, Mass_ugpkgSTP # MassSm2 final units g/m3 air at STP; 


def ppCalc(Composition, perwt_pops, LFEtempC, Pressure, AirTemp, B1, B2, B3, B4, B5, B6, B7, B8, B9, B10, B11, B12, B13, B14, B15, B16, B17, B18, B19, B20):
    
    dp = DiaLookUpTable(Composition)
        
    #constants
    R_unv = 8.3144 #m3 Pa K-1 mol -1
    Pressure = Pressure * 100    #convert Pressure hPa to Pa
    convFac = 1E6 #g to ug convert mass of aerosols
    convFac2 = 1E9
    mwH2SO4 = 98.08
    
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
    mol_per_vol = Vol* Density / mwH2SO4  * convFac * 1E-21 #1.8 density of sulfate aerosol. final units mol/ ambient m3
    molpmolair =  mol_per_vol / Pressure * R_unv * AirTemp * convFac2 #ppb mol/mol - mol m-3 Pa-1 K * Pa K-1 m3 mol-1 (1000 L m-3, 22.4 L per mole air STP)
    partpress_mPa = mol_per_vol* R_unv * AirTemp*1000 #mPa
    return partpress_mPa, Unc #return molpmolair #final units ppb 


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



def LLD(launchNo, PartialPressure):
    #lower limit of detection for SO2sonde (personal communication Paul Walter 01/2023)
    if launchNo == '﻿Launch 2 - 01/22':
        DDL = 0.03
        
    elif launchNo == '﻿Launch 5 - 01/24':
        DDL = 0.02
        
    elif launchNo == '﻿Launch 8 - 01/25':
        DDL = 0.037

    if PartialPressure <= DDL:
        PartialPress_corr = np.NaN
    else:
        PartialPress_corr = PartialPressure
    return PartialPress_corr

def getPOPSdf(vertBins):
    #read in POPS data
    os.chdir('/Users/easher/Desktop/Tonga/Submission/POPSDataForPublication/Finalized/')
    filelist = glob.glob('*.csv')
    #path = '/Users/easher/Desktop/Tonga/Submission/POPSDataForPublication/Finalized'
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
    df['Effective Radius'] = df.apply(lambda x: eRadCalc(x['Composition'], x['B1'], x['B2'], x['B3'], x['B4'], x['B5'], x['B6'], x['B7'], x['B8'], x['B9'], x['B10'], x['B11'], x['B12'], x['B13'], x['B14'], x['B15'], x['B16'], x['B17'], x['B18'], x['B19'], x['B20']), axis=1)
    
    
    df['Altitude (km)'] = pd.cut(df['GPSaltkm'], bins=vertBins,labels = AltLabels, include_lowest=True) 
    df['Altitude (km)'] = pd.to_numeric(df['Altitude (km)'], errors='coerce').fillna(0) 
    df1= df.groupby(['GMTdateYmD','LaunchNo', 'Altitude (km)', 'Flight'],as_index=False).mean() 
    return df1

def GoffGratch(TFpHyg, Temp):
    #GoffGratch 1984 formulation
    TFpHyg = TFpHyg + 273.15
    Temp = Temp + 273.15
    #RH = VaporPressLiquid(hygroFrostpoint) /VaporPressLiquid(temperature) * 100.0;
    #RH over ice uses Goff Gratch saturation vapor pressure from http://cires.colorado.edu/~voemel/vp.html
    #RH over liquid water uses Hyland and Wexler saturation vapor pressure over liquid water from http://cires.colorado.edu/~voemel/vp.html
    
    
    if (Temp < 273.15):
        e =  10**(-9.09718*(273.16/TFpHyg - 1) - 3.56654*math.log10(273.16/TFpHyg) + 0.876793*(1-TFpHyg/273.16) + math.log10(6.1071))
        es = 10**(-9.09718*(273.16/Temp - 1) -   3.56654*math.log10(273.16/Temp) +   0.876793*(1-Temp/273.16) +   math.log10(6.1071))
        RH = e/es*100

    else:
        e = 10**(-9.09718*(273.16/TFpHyg - 1) - 3.56654*math.log10(273.16/TFpHyg) + 0.876793*(1-TFpHyg/273.16) + math.log10(6.1173))
        es = (math.exp(-0.58002206*10**4/Temp + 0.13914993*10**1 - 0.48640239*10**(-1)*Temp + 0.41764768*10**-4*Temp**2 -0.14452093*10**(-7)*Temp**3 + 0.65459673*10**1*math.log(Temp)))/100
        RH = e/es*100
        
    return (RH, e)


def PwLookuptable(T1, Pwa):
    #Lookup table for H2SO4 aerosol %wt given partial pressure of H2O and T (K) 
    #Tabazadeh et al., 1997 GRL vol. 24 (15)
    
    Pw = np.round_(Pwa, decimals = 5)
    #print('Lookup table Pw : ' + str(Pw))
    
    T = T1 + 273.15 #can interpolate between table values..
    #Pwa must be in hPa/mb
    Wi = np.linspace(0.1, 0.8, 15) #Tabazadeh et al., 1997 GRL vol. 24 (15) Table 1.
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
    PwDict = pd.Series(PwTable.Pw_calc.values,index=PwTable.W_1).to_dict()
    #find the closest dictionary look-up value
    W, Pw_t = min(PwDict.items(), key=lambda x: abs(Pw - x[1]))

    return W


def PwCalc(a, b, c, T):
    #use polynomial to calc. eq. partial pressure of H2O given T
    #Tabazadeh et al., 1997 GRL vol. 24 (15) Table 1. eq
        Pw_0 = math.exp(a + b/T + c/(T**2))
        Pw = np.round_(Pw_0, decimals = 5)
        return Pw


def DensityCalc(perWt, Tc):
#density parameterization Oca et al., 2018 J. Chem. & Eng. Data vol. 63 (9)
        T = Tc + 273.15
        PolyCoeff = [1022, -0.5076, 2.484E-4, 976.4, -1.015, 1, 237.8, 1, 1]
        Density = PolyCoeff[0] + PolyCoeff[1] * T + PolyCoeff[2] * T**2 + PolyCoeff[3] * perWt + PolyCoeff[4] + PolyCoeff[5] * perWt**2 
        Density = Density / 1000 #convert from kg/m3 to g/cm3
        return Density
    

def Jonsson1995_calcDamb(Composition, Fpops, rho_pops, Famb, rho_amb):
#calculate ambient aerosol diameter for H2SO4 specifically 
#Jonsson et al., 1995 J. of Atmos. & Oceanic Tech. Vol. 12 (1)
    
    dp = DiaLookUpTable(Composition)
    dpAmb = []
        
    for i in dp:
        Dpops = i
        Damb = Dpops*(Fpops*rho_pops/(Famb*rho_amb))**(1/3)
        dpAmb.append(Damb)
    dpAmbn= np.array(dpAmb).tolist()
       
    return dpAmbn
   
def Plume(LaunchNo, Flight, Altitude):
    #determine based on Launch (TR2Ex or B2SAP) if telemetered data were 15 or 20 bins
    if (LaunchNo == 'run011'):
        composition = 'Sulfate'
    elif (LaunchNo == 'run013'):
        composition = 'Sulfate'
    else:
        composition = 'Sulfate20'
    return composition


def Trop(Launch):
    
    if (Launch == 'run002'):
        Trop = 16.6
    elif (Launch == 'run003'):
        Trop = 16.8
    elif (Launch == 'run004'):
        Trop = 16.4
    elif (Launch == 'run005'):
        Trop = 17.7
    elif (Launch == 'run008'):
        Trop = 17.7
    elif (Launch == 'run009'):
        Trop = 17.5
    elif (Launch == 'run011'):
        Trop = 17.9
    elif (Launch == 'run013'):
        Trop = 17.9

    else:
        Trop = np.NaN
        
    
    return Trop

def StratTrop(Altitude, Tropopause):
    if (Altitude > Tropopause):
        AtmosLayer = 'Stratosphere'
    else:
        AtmosLayer = 'Troposphere'
    
    return AtmosLayer
        

def morePlots(df2):
    
    #customize plots
    hueorder0 = [ 'Launch 2 - 01/22', 'Launch 3 - 01/23', 'Launch 4 - Least Perturbed', 'Launch 5 - 01/24', 'Launch 8 - 01/25', 'Launch 9 - 02/11', 'Launch 11 - 03/31', 'Launch 13 - 06/09']
    hueorder4 = ['Launch 2 - 01/22', 'Launch 4 - Least Perturbed', 'Launch 5 - 01/24', 'Launch 9 - 02/11', 'Launch 11 - 03/31', 'Launch 13 - 06/09']

    #Figure S1
    fig, ax = plt.subplots(1, 1, figsize=(8, 4))
    seaborn.set(style="whitegrid") 
    df2 = df2.sort_values(by=['GMTdateYmD','Altitude (km)', 'H2OMr_frostpoint'])
    ax = seaborn.lineplot(x="H2OMr_frostpoint", y="Altitude (km)",palette = "colorblind",
                        hue = 'LaunchNo', hue_order = hueorder4, marker = '.', legend=True,
                        sort=False, data=df2, zorder=0)
    ax.set_ylim(17,28)
    ax.set_xlim(0.1,1000)
    ax.set_xscale('log')
    ax.set_xlabel("H2O (ppmv)")
    

    df2 = df2[df2['Altitude (km)'] > 2.0]
    # color palette as dictionary
    
    #Figure S3
    seaborn.set(rc={"xtick.bottom" : True, "ytick.left" : True})
    fig, ax = plt.subplots(1, 1, figsize=(8, 4)) 
    df2 = df2.sort_values(by=['LaunchNo','Altitude (km)', 'Effective Radius'])
    ax = seaborn.lineplot(x="Effective Radius", y="Altitude (km)",palette = 'colorblind',
            hue = 'LaunchNo',  hue_order = hueorder0, marker = '.', legend=True,
            sort=False, data=df2, zorder=0)
    
    ax.set_xscale('log')
    ax.set_yscale('linear')
    ax.set_ylim([17, 30])
    ax.set_xlabel(u"Effective Radius (\u03bcm)")

    #figure 1a. (at vertical resolution 250m - same as size distributions in Figure1 c,d,e,g)
    fig, ax = plt.subplots(1, 1, figsize=(4, 4)) 
    df2 = df2.sort_values(by=['LaunchNo','Altitude (km)', 'Mass_ugpkgSTP'])
    ax = seaborn.lineplot(x="Mass_ugpkgSTP", y="Altitude (km)",palette = 'colorblind',
            hue = 'LaunchNo',  hue_order = hueorder0, marker = '.', legend=True,
            sort=False, data=df2, zorder=0)
    
    ax.set_xscale('log')
    ax.set_yscale('linear')
    ax.set_ylim([17, 30])
    ax.set_xlabel(u"Aerosol Dry Mass (ug $\mathregular{kg^{-1}}$)")

    #figure 1b (at vertical resolution 250 m)  
    fig, ax = plt.subplots(1, 1, figsize=(8, 4)) 
    seaborn.set(rc={"xtick.bottom" : True, "ytick.left" : True})
    seaborn.set(style="whitegrid")
    fig, ax = plt.subplots(1, 1, figsize=(8, 4)) 
    df2 = df2.sort_values(by=['LaunchNo','Altitude (km)', 'Extinction_2_K'])
    ax = seaborn.lineplot(x="Extinction_2_K", y="Altitude (km)",palette = "colorblind",
                        hue = 'LaunchNo',style = 'LaunchNo', 
                        hue_order = hueorder0,  
                        marker = '.',  legend=True,
                        sort=False, data=df2, zorder=0)
    ax.set_ylim(17,30)
    ax.set_xscale('log')
    ax.set_xlabel(u"Petters and Kreidenweis 2007 calc. Ambient \u03B5 ($\mathregular{km^{-1}}$ \u03BB = 997)")
 
    #figure 1b (at vertical resolution 250 m)  
    fig, ax = plt.subplots(1, 1, figsize=(8, 4)) 
    seaborn.set(rc={"xtick.bottom" : True, "ytick.left" : True})
    seaborn.set(style="whitegrid")
    fig, ax = plt.subplots(1, 1, figsize=(8, 4)) 
    df2 = df2.sort_values(by=['LaunchNo','Altitude (km)', 'Extinction_2_J'])
    ax = seaborn.lineplot(x="Extinction_2_J", y="Altitude (km)",palette = "colorblind",
                        hue = 'LaunchNo',style = 'LaunchNo', 
                        hue_order = hueorder0,  
                        marker = '.',  legend=True,
                        sort=False, data=df2, zorder=0)
    ax.set_ylim(17,30)
    ax.set_xscale('log')
    ax.set_xlabel(u"Jonsonn et al. 1995 calc. Ambient \u03B5 ($\mathregular{km^{-1}}$ \u03BB = 997)")
    
    #Figure S2b
    fig, ax = plt.subplots(1, 1, figsize=(8, 4)) 
    seaborn.set(font_scale = 4)
    seaborn.set(rc={"xtick.bottom" : True, "ytick.left" : True})
    seaborn.set(style="whitegrid")
    df2 = df2.sort_values(by=['LaunchNo','Altitude (km)', 'Surface Area Ratio'])
    ax = seaborn.lineplot(x="Surface Area Ratio", y="Altitude (km)",palette = "colorblind",
                        hue = 'LaunchNo', hue_order = hueorder4, marker = '.',  legend=True,
                        sort=False, data=df2, zorder=0)
    ax.set_ylim(17,30)
    ax.set_xlim(0.98,2.0)
    ax.set_xlabel("Ratio of Calculated Ambient : Measured Aerosol Surface Area")
    
    
    return

#MAIN SCRIPT
#define variables 
VertRes = 100 #vertical binning resolution

if VertRes == 250: #data binned to to every 250 m
    AltLabels = ["0.25", "0.5","0.75", "1", "1.25", "1.5", "1.75", "2", "2.25", "2.5", "2.75", "3", "3.25", "3.5", "3.75", "4","4.25", "4.5", "4.75", "5", "5.25", "5.5", "5.75", "6",\
              "6.25", "6.5", "6.75", "7", "7.25", "7.5", "7.75", "8", "8.25", "8.5", "8.75", "9", "9.25", "9.5", "9.75", "10",\
              "10.25", "10.5", "10.75", "11", "11.25", "11.5", "11.75", "12", "12.25", "12.5", "12.75", "13", "13.25", "13.5", "13.75", "14",\
                  "14.25", "14.5", "14.75", "15", "15.25", "15.5", "15.75", "16", "16.25", "16.5", "16.75", "17", "17.25", "17.5", "17.75", "18",\
                      "18.25", "18.5", "18.75", "19", "19.25", "19.5", "19.75", "20", "20.25", "20.5", "20.75", "21", "21.25", "21.5", "21.75", "22",\
                          "22.25", "22.5", "22.75", "23", "23.25", "23.5", "23.75", "24", "24.25", "24.5", "24.75", "25", "25.25", "25.5", "25.75", "26",\
                              "26.25", "26.5", "26.75", "27", "27.25", "27.5", "27.75", "28", "28.25", "28.5", "28.75", "29",\
                      "29.25", "29.5", "29.75", "30.00"]
    vertBins = np.linspace(0.125, 30.125, 121)

elif VertRes == 100: #data binned to to every 100 m
    AltLabels = ["0.1", "0.2", "0.3", "0.4", "0.5", "0.6", "0.7", "0.8", "0.9", "1.0", "1.1","1.2","1.3","1.4", "1.5", "1.6", "1.7", "1.8", "1.9", "2.0", "2.1", "2.2","2.3","2.4","2.5","2.6", "2.7", "2.8", "2.9", "3.0",
            "3.1", "3.2", "3.3", "3.4", "3.5", "3.6", "3.7", "3.8", "3.9", "4.0", "4.1","4.2","4.3","4.4", "4.5", "4.6", "4.7", "4.8", "4.9", "5.0", "5.1", "5.2","5.3","5.4","5.5","5.6", "5.7", "5.8", "5.9", "6.0",
              "6.1", "6.2", "6.3", "6.4", "6.5", "6.6", "6.7", "6.8", "6.9", "7.0", "7.1","7.2","7.3","7.4", "7.5", "7.6", "7.7", "7.8", "7.9", "8.0", "8.1", "8.2","8.3","8.4","8.5","8.6", "8.7", "8.8", "8.9", "9.0",
              "9.1", "9.2", "9.3", "9.4", "9.5", "9.6", "9.7", "9.8", "9.9", "10.0", "10.1","10.2","10.3","10.4", "10.5", "10.6", "10.7", "10.8", "10.9", "11.0", "11.1", "11.2", "11.3", "11.4", "11.5", "11.6", "11.7", "11.8", "11.9", "12.0",
              "12.1", "12.2", "12.3", "12.4", "12.5", "12.6", "12.7", "12.8", "12.9", "13.0", "13.1","13.2","13.3","13.4", "13.5", "13.6", "13.7", "13.8", "13.9", "14.0", "14.1", "14.2", "14.3", "14.4", "14.5", "14.6", "14.7", "14.8", "14.9", "15.0",
              "15.1", "15.2", "15.3", "15.4", "15.5", "15.6", "15.7", "15.8", "15.9", "16.0", "16.1","16.2","16.3","16.4", "16.5", "16.6", "16.7", "16.8", "16.9", "17.0", "17.1", "17.2", "17.3", "17.4", "17.5", "17.6", "17.7", "17.8", "17.9", "18.0",
              "18.1", "18.2", "18.3", "18.4", "18.5", "18.6", "18.7", "18.8", "18.9", "19.0", "19.1","19.2","19.3","19.4", "19.5", "19.6", "19.7", "19.8", "19.9", "20.0", "20.1", "20.2", "20.3", "20.4", "20.5", "20.6", "20.7", "20.8", "20.9", "21.0",
              "21.1","21.2","21.3","21.4", "21.5", "21.6", "21.7", "21.8", "21.9", "22.0", "22.1", "22.2", "22.3", "22.4", "22.5", "22.6", "22.7", "22.8", "22.9", "23.0", "23.1", "23.2", "23.3", "23.4", "23.5", "23.6", "23.7", "23.8", "23.9", "24.0", 
              "24.1","24.2","24.3","24.4", "24.5", "24.6", "24.7", "24.8", "24.9", "25.0", "25.1", "25.2", "25.3", "25.4", "25.5", "25.6", "25.7", "25.8", "25.9", "26.0", "26.1", "26.2", "26.3", "26.4", "26.5", "26.6", "26.7", "26.8", "26.9", "27.0",
              "27.1","27.2","27.3","27.4", "27.5", "27.6", "27.7", "27.8", "27.9", "28.0", "28.1", "28.2", "28.3", "28.4", "28.5", "28.6", "28.7", "28.8", "28.9", "29.0", "29.1", "29.2", "29.3", "29.4", "29.5", "29.6", "29.7", "29.8", "29.9", "30.0"]
    vertBins = np.linspace(0.05, 30.05, 301)
    
#read in downloadable datasets in "Supporting Data" tab from Asher et al. https://csl.noaa.gov/projects/b2sap/data.html
dfWV = pd.read_csv('/Users/easher/Desktop/Tonga/Submission/POPSDataForPublication/H2O_table.csv', delimiter = ',', dtype=None, header = 11, engine='python')
dfWV['Flight'] = 'Ascent'
df_SO2_f = pd.read_csv('/Users/easher/Desktop/Tonga/Submission/POPSDataForPublication/SO2_table.csv', delimiter = ',', dtype=None, header = 11, engine='python')
df1 = getPOPSdf(vertBins)

#merge with POPS data pased on altitude and flight
new_df = pd.merge(df1, dfWV, how='left', left_on=['GMTdateYmD','Altitude (km)', 'Flight'], right_on = ['GMTdateYmD', 'Altitude (km)', 'Flight'])

new_df['RH_CFH'] = np.where(new_df.RH_CFH > 200, np.NAN, new_df.RH_CFH)
new_df['RH_frostpoint'] = np.where(((new_df.LaunchNo == 'run013') | (new_df.LaunchNo == 'run011')), new_df.hygrometerRH, new_df.RH_CFH)
new_df['H2OMr_frostpoint'] = np.where(((new_df.LaunchNo == 'run013') | (new_df.LaunchNo == 'run011')), new_df.H2OmrPPMV, new_df.H2OMr)
new_df['Pw_frostpoint'] = np.where(((new_df.LaunchNo == 'run013') | (new_df.LaunchNo == 'run011')), new_df.Pw_FPHc, new_df.Pw_CFH)


##calculate ambient bin size based on WV on that launch date...
new_df['Composition'] = new_df.apply(lambda x: Plume(x['LaunchNo'], x['Flight'], x['Altitude (km)']), axis=1)
new_df['BinDp'] = new_df.apply(lambda x: KKcalcDamb(x['RH_frostpoint'], x['Air Temp (K)'], x['Composition']), axis=1)
#$wt calc
new_df['H2SO4_%wt_amb'] = new_df.apply(lambda x: PwLookuptable(x['iMetTempCorrC'], x['Pw_frostpoint']), axis = 1)
new_df['H2SO4_%wt_pops'] = new_df.apply(lambda x: PwLookuptable(x['POPStempC'], x['Pw_frostpoint']), axis = 1)
#density calculation
new_df['Density_amb'] = new_df.apply(lambda x: DensityCalc(x['H2SO4_%wt_amb'], x['iMetTempCorrC']), axis = 1)
new_df['Density_pops'] = new_df.apply(lambda x: DensityCalc(x['H2SO4_%wt_pops'], x['POPStempC']), axis = 1)
#%Wt amb diameter calculation...
new_df['BinDp_J1995'] = new_df.apply(lambda x: Jonsson1995_calcDamb(x['Composition'], x['H2SO4_%wt_pops'], x['Density_pops'], x['H2SO4_%wt_amb'], x['Density_amb']), axis=1)

new_df['Wavelength'] = 532 #a native wavelength of both the Maïdo observatory (La Réunion) ground-based lidar CALIOP space-based lidar
new_df['Wavelength_2'] = 997

new_df['IR_l'] = 1.38+0.00j
new_df['IR_h'] = 1.45+0.00j

new_df[['Extinction_K', 'Extinction_J']] = new_df.apply(lambda x: pd.Series(VPsca(x['Wavelength'], x['IR_h'], x['LaunchNo'], x['Composition'],x['RH_frostpoint'], x['Air Temp (K)'], x['H2SO4_%wt_pops'], x['Density_pops'], x['H2SO4_%wt_amb'], x['Density_amb'],
                                                    x['B1'], x['B2'], x['B3'], x['B4'], x['B5'], x['B6'], x['B7'], x['B8'], x['B9'], x['B10'], x['B11'], 
                                              x['B12'], x['B13'], x['B14'], x['B15'], x['B16'], x['B17'], x['B18'], x['B19'], x['B20'])), axis=1)
new_df[['Extinction_2_l_K', 'Extinction_2_l_J']] = new_df.apply(lambda x: pd.Series(VPsca( x['Wavelength_2'], x['IR_l'], x['LaunchNo'], x['Composition'],x['RH_frostpoint'], x['Air Temp (K)'],x['H2SO4_%wt_pops'], x['Density_pops'], x['H2SO4_%wt_amb'], x['Density_amb'],
                                                        x['B1'], x['B2'], x['B3'], x['B4'], x['B5'], x['B6'], x['B7'], x['B8'], x['B9'], x['B10'], x['B11'], 
                                              x['B12'], x['B13'], x['B14'], x['B15'], x['B16'], x['B17'], x['B18'], x['B19'], x['B20'])), axis=1)
new_df[['Extinction_2_K', 'Extinction_2_J']] = new_df.apply(lambda x: pd.Series(VPsca(x['Wavelength_2'], x['IR_h'], x['LaunchNo'], x['Composition'],x['RH_frostpoint'], x['Air Temp (K)'], x['H2SO4_%wt_pops'], x['Density_pops'], x['H2SO4_%wt_amb'], x['Density_amb'],
                                                      x['B1'], x['B2'], x['B3'], x['B4'], x['B5'], x['B6'], x['B7'], x['B8'], x['B9'], x['B10'], x['B11'], 
                                              x['B12'], x['B13'], x['B14'], x['B15'], x['B16'], x['B17'], x['B18'], x['B19'], x['B20'])), axis=1)

new_df[['PartialPressure', 'unc_f']] = new_df.apply(lambda x: pd.Series(ppCalc(x['Composition'], x['H2SO4_%wt_pops'], x['iMetTempCorrC'], x['iMetPressMB'], x['Air Temp (K)'], x['B1'], x['B2'], x['B3'], x['B4'], x['B5'], x['B6'], x['B7'], x['B8'], x['B9'], x['B10'], x['B11'], x['B12'], x['B13'], x['B14'], x['B15'], x['B16'], x['B17'], x['B18'], x['B19'], x['B20'])), axis=1)

#new_df['MassSm2'] = new_df.apply(lambda x: MassBCalc(x['LaunchNo'], x['Composition'], x['H2SO4_%wt_pops'], x['iMetTempCorrC'], x['iMetPressMB'], x['Air Temp (K)'], x['B1'], x['B2'], x['B3'], x['B4'], x['B5'], x['B6'], x['B7'], x['B8'], x['B9'], x['B10'], x['B11'], x['B12'], x['B13'], x['B14'], x['B15'], x['B16'], x['B17'], x['B18'], x['B19'], x['B20']), axis=1)
new_df[['Mass', 'Mass_ugpkgSTP']] = new_df.apply(lambda x: pd.Series(MassBCalc(x['LaunchNo'], x['Composition'], x['H2SO4_%wt_pops'], x['iMetTempCorrC'], x['iMetPressMB'], x['Air Temp (K)'], x['B1'], x['B2'], x['B3'], x['B4'], x['B5'], x['B6'], x['B7'], x['B8'], x['B9'], x['B10'], x['B11'], x['B12'], x['B13'], x['B14'], x['B15'], x['B16'], x['B17'], x['B18'], x['B19'], x['B20'])), axis=1)
new_df['Mass_l'] = new_df['Mass'] * (1.0 - new_df['unc_f'])
new_df['Mass_h'] = new_df['Mass'] * (1.0 + new_df['unc_f'])

new_df = new_df.filter(['Altitude (km)', 'Launch','GMTdateYmD', 'LaunchNo', 'Flight', 'iMetPressMB', 'iMetTempCorrC', 'iMetThetaK','Temp', 
                        'Air Temp (K)', 'Theta', 'RH','O3ppmv', 'RH_CFH', 'H2OMr', 'H2OMr_frostpoint', 'RH_frostpoint', 'elapsedMin', 'AltitudeIMETkm','GPSlat', 'GPSlon', 'GPSaltkm', 'Wind Direction', 'Wind Speed', 'U','V',
                      'POPSflowCorrCCpS','POPSflowCCpS', 'POPStempC','ratioWidth2Flow', 'POPSbaselineStdDev','POPSavgWdithUS', 'H2SO4_%wt_amb','H2SO4_%wt_pops', 'Density_amb', 'Density_pops', 'B1', 'B2', 
                      'B3','B4','B5','B6','B7','B8','B9','B10','B11', 'B12','B13', 'B14','B15', 'B16', 'B17', 'B18', 'B19', 'B20', 'Aer_Conc', 
                      'Dry Surface Area Density','Effective Radius','Wavelength','Composition', 'Extinction_K', 'Extinction_J','Extinction_2_l_K','Extinction_2_l_J','Extinction_2_K', 'Extinction_2_J','Mass_l', 'Mass_h', 'PartialPressure', 'unc_f', 'Mass', 'Mass_ugpkgSTP', 'BinDp', 'BinDp_J1995'], axis=1)

df2 = new_df[new_df['Altitude (km)'] >= 2]
df2['Tropopause'] = df2.apply(lambda x: Trop(x['LaunchNo']), axis = 1)

a = pd.DataFrame(df2['BinDp'])

b = pd.concat([pd.DataFrame(a['BinDp'].values.tolist()) for c in a.columns], 
                 axis=1, 
                 keys=a.columns)

b.columns = ['{}{}'.format(i, j) for i, j in b.columns]
df2.reset_index(drop=False, inplace = True)
b.reset_index(drop=False, inplace = True)
df2 = pd.concat([df2, b], axis = 1)
df2.reset_index(drop=True, inplace = True)


df2[['SAD_rh_K', 'SAD_rh_J']] = df2.apply(lambda x: pd.Series(SADrh( x['Composition'], x['RH_frostpoint'], x['Air Temp (K)'], x['H2SO4_%wt_pops'], x['Density_pops'], x['H2SO4_%wt_amb'], x['Density_amb'],
                                          x['B1'], x['B2'], x['B3'], x['B4'], x['B5'], x['B6'], x['B7'], x['B8'], x['B9'], x['B10'], x['B11'], 
                                              x['B12'], x['B13'], x['B14'], x['B15'], x['B16'], x['B17'], x['B18'], x['B19'], x['B20'])), axis=1)

df2['Surface Area Ratio'] = df2['SAD_rh_J']/ df2['Dry Surface Area Density']
df2['Surface Area Ratio_2'] = (df2['SAD_rh_J']-df2['SAD_rh_K'])/df2['SAD_rh_J']
df2['Surface Area Ratio_3'] = df2['SAD_rh_K']/ df2['Dry Surface Area Density']



plots = 1
if plots == 1:
    #plots

    #rename launches for legend
    df2["LaunchNo"]= df2["LaunchNo"].str.replace("run002", "Launch 2 - 01/22", case = False) 
    df2["LaunchNo"]= df2["LaunchNo"].str.replace("run003", "Launch 3 - 01/23", case = False) 
    df2["LaunchNo"]= df2["LaunchNo"].str.replace("run004", "Launch 4 - Least Perturbed", case = False) 
    df2["LaunchNo"]= df2["LaunchNo"].str.replace("run005", "Launch 5 - 01/24", case = False)
    df2["LaunchNo"]= df2["LaunchNo"].str.replace("run008", "Launch 8 - 01/25", case = False) 
    df2["LaunchNo"]= df2["LaunchNo"].str.replace("run009", "Launch 9 - 02/11", case = False) 
    df2["LaunchNo"]= df2["LaunchNo"].str.replace("run011", "Launch 11 - 03/31", case = False) 
    df2["LaunchNo"]= df2["LaunchNo"].str.replace("run013", "Launch 13 - 06/09", case = False) 
    #colors for plotting
    seaborn.set_color_codes(palette = "pastel")

    palette0 = {"Launch 2 - 01/22":"darkred",
                "Launch 3 - 01/23":"salmon", 
                "Launch 4 - Least Perturbed":"lightgray",
                "Launch 5 - 01/24":"orangered", 
                "Launch 8 - 01/25":"peru",
                "Launch 9 - 02/11":"goldenrod",
                "Launch 11 - 03/31":"navajowhite",
                "Launch 13 - 06/09":"green"
                }

    #create partial pressure combined SO2 and H2SOdataset
    df2.drop(columns=['index'], inplace = True)
    df_H2SO4_PP = df2.filter(['GMTdateYmD', 'LaunchNo', 'Altitude (km)', 'PartialPressure', 'unc_f'])
    df_H2SO4_PP['pp_ub'] = 1.0 + df_H2SO4_PP['unc_f']
    df_H2SO4_PP['pp_lb'] = 1.0 - df_H2SO4_PP['unc_f']
    df_H2SO4_PP = df_H2SO4_PP[df_H2SO4_PP['LaunchNo'] != 'Launch 3 - 01/23']
    df_H2SO4_PP = df_H2SO4_PP[df_H2SO4_PP['LaunchNo'] != 'Launch 9 - 02/11']
    df_H2SO4_PP = df_H2SO4_PP[df_H2SO4_PP['LaunchNo'] != 'Launch 11 - 03/31']    
    df_H2SO4_PP = df_H2SO4_PP[df_H2SO4_PP['LaunchNo'] != 'Launch 13 - 06/09']
    df_H2SO4_PP = df_H2SO4_PP[df_H2SO4_PP['LaunchNo'] != 'Launch 4 - Least Perturbed']

    
    filter1 = df_H2SO4_PP["GMTdateYmD"]!="2022-01-24"
    filter2 = df_H2SO4_PP["Altitude (km)"] < 28
    df_H2SO4_PP.where(filter1 | filter2, inplace = True)

    df_H2SO4_PP['Compound'] = 'eH2SO4'
    df_SO2_f['Compound'] = 'SO2'
    
    df_SO2_f['PartialPressure'] = df_SO2_f.apply(lambda x: LLD(x['LaunchNo'], x['PartialPressure']), axis=1)
    df_SO2_f['unc_f'] = 0.2
    df_SO2_f['pp_ub'] = 1.0 + df_SO2_f['unc_f']
    df_SO2_f['pp_lb'] = 1.0 - df_SO2_f['unc_f']
    df_SO2_fa = df_SO2_f[df_SO2_f['GMTdateYmD'] == '2022-01-22']
    df_SO2_fb = df_SO2_f[df_SO2_f['GMTdateYmD'] == '2022-01-24']
    df_SO2_fc = df_SO2_f[df_SO2_f['GMTdateYmD'] == '2022-01-25']
    
    df_H2SO4_PPa = df_H2SO4_PP[df_H2SO4_PP['GMTdateYmD'] == '2022-01-22']
    df_H2SO4_PPb = df_H2SO4_PP[df_H2SO4_PP['GMTdateYmD'] == '2022-01-24']
    df_H2SO4_PPc = df_H2SO4_PP[df_H2SO4_PP['GMTdateYmD'] == '2022-01-25']

    df_SO2_f1 = pd.concat([df_SO2_f, df_H2SO4_PP], ignore_index=True)
    df_SO2_f1 = df_SO2_f1.sort_values(by=['LaunchNo','Altitude (km)', 'PartialPressure'])
    
    
    #Figure 2a,b,c
    seaborn.set(font_scale = 1)
    fig, ax = plt.subplots(1, 1, figsize=(8, 4)) 
    seaborn.set(rc={"xtick.bottom" : True, "ytick.left" : True})  
    df_SO2_f1['Date'] = df_SO2_f1['GMTdateYmD']
    df_SO2_f1 = df_SO2_f1.sort_values(by=['Date','Altitude (km)', 'PartialPressure'])
    
    g = seaborn.relplot(data=df_SO2_f1, kind="line", col = 'Date',col_wrap = 3,
                        col_order= ['2022-01-22', '2022-01-24', '2022-01-25'], x='PartialPressure', 
                        y='Altitude (km)', hue='Compound', marker = '.', 
                        palette= "colorblind", sort=False, facet_kws={'sharey': True, 'sharex': True})

    g.set(ylim=(16, 30))
    g.set(xscale = 'log')
    g.set(xlabel = 'Partial Pressure (mPa)') #g.set(xlabel = 'Mixing Ratio (ppb)')
    df_SO2_f['PartialPressure']
    
    ax = g.facet_axis(0, 0)
    ax.axvline(x=0.03, color='tab:blue', linestyle = '--')
    ax.fill_betweenx(y=df_SO2_fa['Altitude (km)'], x1=df_SO2_fa['PartialPressure']*df_SO2_fa['pp_lb'], x2 = df_SO2_fa['PartialPressure']*df_SO2_fa['pp_ub'], color="blue", alpha=0.3)
    ax.fill_betweenx(y=df_H2SO4_PPa['Altitude (km)'], x1=df_H2SO4_PPa['PartialPressure']*df_H2SO4_PPa['pp_lb'], x2 = df_H2SO4_PPa['PartialPressure']*df_H2SO4_PPa['pp_ub'], color="orange", alpha=0.3)
    ax = g.facet_axis(0, 1)
    ax.axvline(x=0.02, color='tab:blue', linestyle = '--')
    ax.fill_betweenx(y=df_SO2_fb['Altitude (km)'], x1=df_SO2_fb['PartialPressure']*df_SO2_fb['pp_lb'], x2 = df_SO2_fb['PartialPressure']*df_SO2_fb['pp_ub'], color="blue", alpha=0.3)
    ax.fill_betweenx(y=df_H2SO4_PPb['Altitude (km)'], x1=df_H2SO4_PPb['PartialPressure']*df_H2SO4_PPb['pp_lb'], x2 = df_H2SO4_PPb['PartialPressure']*df_H2SO4_PPb['pp_ub'], color="orange", alpha=0.3)
    ax = g.facet_axis(0, 2)
    ax.axvline(x=0.037, color='tab:blue', linestyle = '--')
    ax.fill_betweenx(y=df_SO2_fc['Altitude (km)'], x1=df_SO2_fc['PartialPressure']*df_SO2_fc['pp_lb'], x2 = df_SO2_fc['PartialPressure']*df_SO2_fc['pp_ub'], color="blue", alpha=0.3)
    ax.fill_betweenx(y=df_H2SO4_PPc['Altitude (km)'], x1=df_H2SO4_PPc['PartialPressure']*df_H2SO4_PPc['pp_lb'], x2 = df_H2SO4_PPc['PartialPressure']*df_H2SO4_PPc['pp_ub'], color="orange", alpha=0.3)

    
    df4 = df2[df2['Altitude (km)'] > df2['Tropopause']] 
    df_POPS_OMPS = df4[['GMTdateYmD', 'LaunchNo', 'Altitude (km)', 'Tropopause', 'iMetThetaK', 'Extinction_2_l_K', 'Extinction_2_l_J','Extinction_2_K','Extinction_2_J','Mass', 'Mass_l', 'Mass_h']] #'MassSm2'
    df_POPS_OMPS.to_csv('/Users/easher/Desktop/df_POPS_OMPS_unc.csv', mode='w')
    

    df3  = df2[['GMTdateYmD', 'LaunchNo', 'iMetThetaK','Altitude (km)', 'Tropopause', 'Extinction_K', 'Mass', 'unc_f']] #'MassSm2'
    df3['unc_mass'] = (df3['Mass'] * df3['unc_f'])**2
    
    df3['Extinction_K'] = df3['Extinction_K']*VertRes/1000 #units of extinction are already km-1 (multiply by VertRes (km) prior to summing to yield sAOD)
    
    #df3['AerosolMass'] = df3['MassSm2'] #g/kg * density (1.225 g/m3) = ug/m...ug to g (x1E6)...divided by MW H2SO42-*32.06 & integrate vertically g S/m2
    df3['AerosolMass'] = df3['Mass']
    
    #df3 = df3[df3['Flight'] == 'Ascent']
    
    df3['AtmosLayer'] = df3.apply(lambda x: StratTrop(x['Altitude (km)'], x['Tropopause']), axis=1)
    df3['Location'] = df3['LaunchNo'].apply(lambda x: 'Lauder, NZ - SH Midlatitudes' if (((x == "Baseline NZ - 01/24") |(x == "Launch 12 - 05/25 NZ")) |
                                            (x == "Launch 10 - 03/01 NZ")) else ('SH Tropics Approximate Baseline' if (x == 'Launch 4 - Least Perturbed') else 'Reunion Island - SH Tropics'))
    df3 = df3[df3['AtmosLayer'] != 'Troposphere']
    
    df3_ = df3.groupby(['GMTdateYmD', 'Location', 'AtmosLayer'], as_index=False).sum()
    df3_bg = df3_[df3_['Location'] == 'SH Tropics Approximate Baseline']
    df3_ = df3_[df3_['GMTdateYmD'] != '2022-06-09']
    df3_ = df3_[df3_['GMTdateYmD'] != '2022-03-31']
    df3_ = df3_[df3_['GMTdateYmD'] != '2022-02-11']
    df3_ = df3_[df3_['GMTdateYmD'] != '2022-01-23']
    
    
    df3_['AerosolMass'] = df3_['Mass'] - float(df3_bg['AerosolMass']) #subtract background

    #uncertainty propagated...
    df3_['tot_unc_mass'] = (df3_['unc_mass'] + float(df3_bg['unc_mass']))**0.5 
    df3_['Date'] = df3_['GMTdateYmD'].apply(lambda x: dt.strptime(x,'%Y-%m-%d'))
    
    Date = df_SO2_f['GMTdateYmD'].unique()
    # SO2 ...1 DU = 2.69 × 1016 molecules cm−2
    dfSO2_mass = np.array([df_SO2_fa['SO2_Col'].max(), df_SO2_fb['SO2_Col'].max(), df_SO2_fc['SO2_Col'].max()])* 2.6867*1E20/(6.0221408E23)*32.06 #molecules/m2/ mol/mol x g S mol-1-> g S m2
    dfSO2_mass_unc = dfSO2_mass*0.2
    #dfSO2['Compound'] = 'SO2'
    
    dfH2SO4_mass = df3_['AerosolMass']
    dfH2SO4_mass_unc = df3_['tot_unc_mass'] 
    #dfH2SO4['Compound'] = 'eH2SO4'
    
    dfTotS_mass =  np.add(dfSO2_mass, dfH2SO4_mass) # g S m-2
    dfTotS_mass_unc = np.add(dfSO2_mass_unc**2, dfH2SO4_mass_unc**2)**0.5
    #dfTotS['Compound'] = 'Total Sulfur'

    width = 0.3  # the width of the bars
    
    x = np.arange(len(Date))
    
    #figure 2d
    fig, ax = plt.subplots()
    rects1 = ax.bar(x - width, np.round(dfTotS_mass, 5), width, label='Total S in [SO2 + eH2SO4]', yerr = dfTotS_mass_unc)
    rects2 = ax.bar(x, np.round(dfH2SO4_mass, 5), width, label='S in eH2SO4', yerr = dfH2SO4_mass_unc)
    rects3 = ax.bar(x + width, np.round(dfSO2_mass,5), width, label='S in SO2', yerr = dfSO2_mass_unc)  
    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel(u"Plume S Mass (g S $\mathregular{m^{-2}}$)", size = 14)
    ax.set_title('Summary')
    ax.set_xticks(x, Date)
    ax.legend()
    ax.bar_label(rects1, padding=2)
    ax.bar_label(rects2, padding=2)
    ax.bar_label(rects3, padding=2)
    fig.tight_layout()
    plt.show()
    
    #show plots from Asher et al., in review 2023
    morePlots(df2)
