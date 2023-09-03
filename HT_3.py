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

    
    
#Use the look-up table to find the particle diameter and dLogDp calculated from POPS signal 
def DiaLookUpTable(Composition):
    #file = '/Users/easher/Documents/NOAAweb/POPS_Sizes.csv'
    file = '/Users/asher/Documents/NOAAweb/NZ_Paper_IOR.csv'
    df1=pd.read_csv(file,sep=',', dtype=None, engine='python')

    if Composition == 'Sulfate20':
        dp_in = df1[(df1.Composition == 'Sulfate') & (df1.Binning == 'Manual20')]
        #if you do not 
        #dpAmb = [0.1463, 0.1594, 0.1737, 0.18515, 0.19325, 0.20175, 0.2106, 0.2198, 0.22945, 0.29785, 0.46415, 0.61235, 0.7103, 0.8239, 0.95565, 1.1085, 1.2858, 1.49145, 1.72995, 2.179]
        dp =  np.array(dp_in.Diameter*1000).tolist()

    else:
        dp_in = df1[(df1.Composition == 'Sulfate') & (df1.Binning == 'Manual')]
        dp = np.array(dp_in.Diameter*1000).tolist()
        
        for x in range(5):
            dp = np.append(dp, [np.nan])
            #print(x)
        
    del dp_in
    
    return dp


#calculate dNdLogDp
def DnDlogDp(Launch, Composition, B1, B2, B3, B4, B5, B6, B7, B8, B9, B10, B11, B12, B13, B14, B15, B16, B17, B18, B19, B20):
    #file_ior = '/Users/easher/Documents/NOAAweb/POPS_Sizes.csv'
    file_ior = '/Users/asher/Documents/NOAAweb/NZ_Paper_IOR.csv'
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



#calculate ambient aerosol surface aera (using two methods) and POPS measured size distributions and other in situ data
def SAD(Composition, RH, T, Fpops, rho_pops, Famb, rho_amb, perwt_pops, perwt_amb, B1, B2, B3, B4, B5, B6, B7, B8, B9, B10, B11, B12, B13, B14, B15, B16, B17, B18, B19, B20): #was based on date, can do based on composition, no?
        
    if Composition == 'Sulfate20':

        ndp = [B1, B2, B3, B4, B5, B6, B7, B8, B9, B10, B11, B12, B13, B14, B15, B16, B17, B18, B19, B20] 
            
    else:
        ndp = [B1, B2, B3, B4, B5, B6, B7, B8, B9, B10, B11, B12, B13, B14, B15] 
        

    dp = DiaLookUpTable(Composition) 
    dpAmbn_SH = Steele_Hamill_calcDamb(Composition, perwt_pops, rho_pops, perwt_amb, rho_amb) #Jonsson1995_calcDamb(Composition, Fpops, rho_pops, Famb, rho_amb)
        
    Conc = pd.Series(ndp)
    Conc = pd.Series(ndp)
    Rad =  pd.Series(np.divide(dp,2))*1000
    Area = np.pi*4*np.power(Rad,2)
    DrySurfaceArea = (Conc.mul(Area)).sum()*1E-6
    
    Conc = pd.Series(ndp)
    Rad_SH =  pd.Series(np.divide(dpAmbn_SH,2))*1000
    Area_SH = np.pi*4*np.power(Rad_SH,2)
    AmbSurfaceArea_SH = (Conc.mul(Area_SH)).sum()*1E-6
    
    return DrySurfaceArea, AmbSurfaceArea_SH


def getPOPSdf():
    #read in POPS data
    os.chdir('/Users/asher/Documents/PapersInProgress/Tonga/Submission/POPSDataForPublication/Finalized/')
    filelist = glob.glob('*.csv')
    path = '/Users/asher/Documents/PapersInProgress/Tonga/Submission/POPSDataForPublication/Finalized'
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
        
        # #Filter POPS data based on various conditions...
        df1 = df1[df1.POPSflowCCpS >= 3.0]
        df1[df1 == 99999] = np.NAN
        # #df1 = df1[df1.POPSbaselineStdDev <= 13]
        # #could create a function to do this cleaning...            
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
        df = pd.concat([df, df1],  ignore_index=True, sort=False) 
        del df1

    
    #create binned altitude and Flight (ascent/descent) for merge
    df['Flight'] = df['iMetASCENTms'].apply(lambda x: 'Ascent' if x > 0 else 'Descent')
    df = df[df['Flight'] == 'Ascent'] #only ascent data
    
    #convert air temperature to Kelvin
    df['Air Temp (K)'] = df['iMetTempCorrC']+ 273.15
    
    #calculate basic number concentration(s) (#/ cm-3) and surface area
    df[['B1', 'B2','B3','B4', 'B5', 'B6', 'B7','B8', 'B9', 'B10', 'B11', 'B12', 'B13', 'B14','B15', 'B16', 'B17', 'B18', 'B19', 'B20']]\
    = df[['B1', 'B2','B3','B4', 'B5', 'B6', 'B7','B8', 'B9', 'B10', 'B11', 'B12', 'B13','B14','B15', 'B16', 'B17', 'B18', 'B19', 'B20']].div(df['POPSflowCorrCCpS'].values,axis=0)
    
    df['Composition'] = df.apply(lambda x: Plume(x['LaunchNo'], x['Flight'], x['GPSaltkm']), axis=1)
    df['Aer_Conc'] =  df[['B1', 'B2','B3','B4', 'B5', 'B6', 'B7','B8', 'B9','B10', 'B11', 'B12', 'B13', 'B14', 'B15', 'B16', 'B17', 'B18', 'B19', 'B20']].sum(axis=1)
    #df['Dry Surface Area Density'] = df.apply(lambda x: CalcSurfArea(x['Composition'], x['B1'], x['B2'], x['B3'], x['B4'], x['B5'], x['B6'], x['B7'], x['B8'], x['B9'], x['B10'], x['B11'], x['B12'], x['B13'], x['B14'], x['B15'], x['B16'], x['B17'], x['B18'], x['B19'], x['B20']), axis=1)

    df['Altitude (km)'] = pd.cut(df['GPSaltkm'], bins=vertBins,labels = AltLabels, include_lowest=True) 
    df['Altitude (km)'] = pd.to_numeric(df['Altitude (km)'], errors='coerce').fillna(0) 
    df1= df.groupby(['GMTdateYmD','LaunchNo', 'Altitude (km)', 'Flight'],as_index=False).mean() 
    return df1
    #return df


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
    W_T, Pw_t = min(PwDict.items(), key=lambda x: abs(Pw - x[1]))
    W = W_T
    
    #if the wt% is greater than or equal to 0.8, use the Gmitro and Vermeulen  1964 parameterization eq. 16; partial molar properties are listed in Table 3.
    if W_T >= 0.8:
        W_GV = pd.Series([0.80, 0.82, 0.84, 0.86, 0.88, 0.90, 0.91, 0.92, 0.93, 0.94, 0.95, 0.96, 0.97, 0.98]) #maybe go up to 0.98 W?
        R = 1.98726
        Cp_298 = pd.Series([9.77, 10.36, 13.78, 18.96, 22.13, 22.76, 22.30, 21.48, 20.44, 19.32, 18.06, 16.64, 15.05, 13.25])
        L_298 = pd.Series([-3475, -4015, -4656, -5319, -5938, -6419, -6627, -6816, -6983, -7139, -7286, -7433, -7574, -7712])
        F_F0 = pd.Series([-3090, -3427, -3789, -4167, -4557, -4960, -5165, -5375, -5595, -5830, -6090, -6390, -6741, -7204])
        alpha = pd.Series([0.0114, 0.0568, 0.1233, 0.0666, -0.0120, -0.0346, -0.0398, -0.0427, -0.0436, -0.0428, -0.0405, -0.0368, -0.0314, -0.024])
        
        A1 = -3.67340
        B1 = -4143.5
        C1 =  10.24353
        D1 = 0.618943E-3
        E = 0
        #calculate the constants based on constants above and Temperature   
        A = A1 + 1/R * (Cp_298 - 298.15 * alpha)
        B = B1 + 1/R * (L_298 - 298.15 * Cp_298 + 298.15**2/2 * alpha)
        C = C1 + 1/R * (Cp_298 + (F_F0 - L_298) * 1/298.15)
        D = D1 - alpha/(2 * R)
        #calculate Pw
        
        PwTable_GV = pd.DataFrame({'A': A, 'B': B, 'C': C, 'D': D, 'E': E, 'W': W_GV})#
        PwTable_GV['T'] = T
        
        PwTable_GV['pH2O'] = PwTable_GV.apply(lambda x: PwCalc_GV1964(x['A'], x['B'], x['C'], x['D'], x['E'], x['T']), axis = 1)                          
        
        PwTable_GV.drop(['A', 'B', 'C', 'D', 'E', 'T'], axis = 1, inplace=True)
        PwDict_GV = pd.Series(PwTable_GV.pH2O.values,index=PwTable_GV.W).to_dict()
        W_GV, Pw_t = min(PwDict_GV.items(), key=lambda x: abs(Pw - x[1]))
        W = W_GV

    return W

#use polynomial to calc. eq. partial pressure of H2O given T
def PwCalc(a, b, c, T):
        Pw_0 = math.exp(a + b/T + c/(T**2))
        Pw = np.round_(Pw_0, decimals = 5)
        return Pw

def PwCalc_GV1964(A,B,C,D,E,T):
    #Gmitro and Vermeulen 1964 AIChE Journal vol. 10 (5) pg. 740 746 eq. 16
        Pwlog = A * np.log(298.15/T) + B/T + C + D*T + E * np.power(T,2)
        Pw = np.exp(Pwlog) #units of partial pressure in atmospheres
        Pwa = Pw * 1013.25#conversion factor from atmospheres to mb
        return Pwa


def DensityCalc(perWt, Tc):
#density parameterization Oca et al., 2018 J. Chem. & Eng. Data vol. 63 (9)
        T = Tc + 273.15 #convert C to K
        PolyCoeff = [1022, -0.5076, 2.484E-4, 976.4, -1.015, 1, 237.8, 1, 1]
        Density = PolyCoeff[0] + PolyCoeff[1] * T + PolyCoeff[2] * T**2 + PolyCoeff[3] * perWt + PolyCoeff[4] + PolyCoeff[5] * perWt**2 
        Density_c = Density / 1000 #convert from kg/m3 to g/cm3
        Density_unc = 0.02 #uncertainty reported 
        
        if perWt >= 0.68:
            #use Washburn analytical tables of chemistry 1928 (0 C) similar to POPS internal temperatures ~ -5 C to +5 C when percent weight 
            Wi = np.linspace(0.68, 0.98, 31)
            W = pd.Series(Wi)
            rho = pd.Series([1.6059, 1.6176, 1.6293, 1.6411, 1.6529, 1.6648, 1.6768, 1.6888, 1.7008, 1.7128, 1.7247, 1.7365, 1.7482, 1.7597,
                1.7709, 1.7815, 1.7916, 1.8009, 1.8095, 1.8173, 1.8243, 1.8306, 1.8361, 1.8410, 1.8453, 1.8490, 1.8520, 1.8544, 1.8560, 1.8569, 1.8567])
          
            WDensityTable = pd.DataFrame({'rho': rho, 'W': W})#

            WDensityDict = pd.Series(WDensityTable.W.values,index=WDensityTable.rho).to_dict()
            
            Density_W, W = min(WDensityDict.items(), key=lambda x: abs(perWt - x[1]))

            
            
            #if W >0.6
            p1 = 473.52 + 4903.99 * W -11916.50 * np.power(W,2) + 15057.60 * np.power(W,3) - 6668.37 * np.power(W,4)
            p2 =  250.52 +  5733.14* W - 13138.14 * np.power(W,2) + 15565.78 * np.power(W,3) - 6618.70* np.power(W,4)
            
            Density = p1 + (p2-p1) * ((T - 273.15)/69) #density in kg/m3 want it in g cm-3
            Density_c = Density/1000
        
            Density_unc_a = (Density_c - Density_W)/Density_c
            if Density_unc_a > Density_unc:
                Density_unc = Density_unc_a #not greater than 0.02 (listed above)
                
        return Density_c
    
def Steele_Hamill_calcDamb(Composition, perwt_pops, rho_pops, perwt_amb, rho_amb):
#steele & Hamill 1981 eq 5) 4/3 Pi * r3_amb * rho_amb = ma + mw = 100 ma/ W %
#ma should be the same measured on POPS or in ambient air, so...
#ma = W %/100 * 4/3 pi * r3_pops * rho_pops
    dp = DiaLookUpTable(Composition)
    dpAmb = []
        
    for i in dp:

        d_pops = i 
        r_pops = d_pops * 0.5
        r_amb3 = 1/(rho_amb * perwt_amb) * np.power(r_pops, 3) * rho_pops * perwt_pops
        #0.75 * np.pi * rho_amb * perwt_pops * 0.01 * 4/3 * np.pi * np.power(r_pops, 3) * rho_pops
        Damb = np.cbrt(r_amb3) * 2
        dpAmb.append(Damb)
        
    #dp_unc = [0.03, 0.03, 0.03, 0.03, 0.03, 0.03, 0.03, 0.03, 0.03, 0.03, 0.03, 0.15, 0.15, 0.15, 0.15, 0.15, 0.15, 0.15, 0.15, 0.15]
    # if perwt_pops > 0.8 then unc. r_amb is maybe sqrt(dp_unc + 0.1**2 + 0.1**2)  
    dpAmbn= np.array(dpAmb).tolist()
    return dpAmbn
    
#second way to calculate bin diameter change
def Steele_Hamill_1981_J1995_calcDamb(Composition, Fpops, rho_pops, Famb, rho_amb):
    #calculate ambient aerosol diameter for H2SO4 specifically 
    #Jonsson et al., 1995 J. of Atmos. & Oceanic Tech. Vol. 12 (1)
    #This calculation is identical to method using Steele & Hamill 1981 (above). It assumes that particles are at equilibrium both at the time of measurement as well as in ambient air,
    #and that only H2O, not H2SO4 is lost from particles during sampling. Discussed in Asher et al.
    dp = DiaLookUpTable(Composition)
    dpAmb = []
        
    for i in dp:
        Dpops = i
        Damb = Dpops*(Fpops*rho_pops/(Famb*rho_amb))**(1/3)
        dpAmb.append(Damb)
    dpAmbn= np.array(dpAmb).tolist()
    return dpAmbn
   
def Plume(LaunchNo, Flight, Altitude):
    #determine based on Launch (TR2Ex or B2SAP) if telemetered data had 15 or 20 particle size bins
    if (LaunchNo == 'run011'):
        composition = 'Sulfate'
    elif (LaunchNo == 'run013'):
        composition = 'Sulfate'
    else:
        composition = 'Sulfate20'
    return composition


def Sample(LaunchDate, Altitude):
    #select size distributions used in Fig. S2a,b
    if ((LaunchDate == '2022-01-22') & ((Altitude > 26.3) & (Altitude < 26.5))): #26.25) & (Altitude < 26.75
        Stype = 'Jan 22 18 UTC, 26.5 km'
    elif ((LaunchDate == '2022-03-31') & ((Altitude > 21.4) & (Altitude < 21.6))): #21.25) & (Altitude < 21.75
        Stype = 'Mar 31, 21.5 km'
    else:
        Stype = 'None'
    return Stype



#set colors for Figure S2a,b
palette2 = {"Size at time of measurement":"black",
                "Calculated using Kappa-Koehler Thoery":"salmon",
                "Size-adjusted for ambient air, Steele & Hamill 1981": "blue"
                }

#altitude binning schemes for 250m or 100 m bins
AltLabels2 = ["0.25", "0.5","0.75", "1", "1.25", "1.5", "1.75", "2", "2.25", "2.5", "2.75", "3", "3.25", "3.5", "3.75", "4","4.25", "4.5", "4.75", "5", "5.25", "5.5", "5.75", "6",\
              "6.25", "6.5", "6.75", "7", "7.25", "7.5", "7.75", "8", "8.25", "8.5", "8.75", "9", "9.25", "9.5", "9.75", "10",\
              "10.25", "10.5", "10.75", "11", "11.25", "11.5", "11.75", "12", "12.25", "12.5", "12.75", "13", "13.25", "13.5", "13.75", "14",\
                  "14.25", "14.5", "14.75", "15", "15.25", "15.5", "15.75", "16", "16.25", "16.5", "16.75", "17", "17.25", "17.5", "17.75", "18",\
                      "18.25", "18.5", "18.75", "19", "19.25", "19.5", "19.75", "20", "20.25", "20.5", "20.75", "21", "21.25", "21.5", "21.75", "22",\
                          "22.25", "22.5", "22.75", "23", "23.25", "23.5", "23.75", "24", "24.25", "24.5", "24.75", "25", "25.25", "25.5", "25.75", "26",\
                              "26.25", "26.5", "26.75", "27", "27.25", "27.5", "27.75", "28", "28.25", "28.5", "28.75", "29",\
                      "29.25", "29.5", "29.75", "30.00"]

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

vertBins2 = np.linspace(0.05, 30.05, 121) #bin ever 250 m
vertBins = np.linspace(0.05, 30.05, 301)#  bin every 100 m; 

#read in downloadable datasets in "Supporting Data" tab from Asher et al. https://csl.noaa.gov/projects/b2sap/data.html   
dfWV = pd.read_csv('/Users/asher/Documents/PapersInProgress/Tonga/Submission/POPSDataForPublication/H2O_table.csv', delimiter = ',', dtype=None, header = 11, engine='python')

#seperate flight data into ascent vs. descent
dfWV['Flight'] = 'Ascent'
dfWV_plume = dfWV

dfWV_plume['Sample']  = dfWV_plume.apply(lambda x: Sample(x['GMTdateYmD'], x['Altitude (km)']), axis=1)
dfWV_plume1 = dfWV_plume.groupby(['Sample'],as_index=False).mean()
dfWV_plume1 = dfWV_plume1[dfWV_plume1['Sample']!='None']

#del dfWV
del dfWV_plume

#read in downloadable datasets in "Supporting Data" tab from Asher et al. https://csl.noaa.gov/projects/b2sap/data.html   
df_SO2_f = pd.read_csv('/Users/asher/Documents/PapersInProgress/Tonga/Submission/POPSDataForPublication/SO2_table.csv', delimiter = ',', dtype=None, header = 11, engine='python') 
df = getPOPSdf()


#merge with POPS data pased on altitude and flight
new_df = pd.merge(df, dfWV, how='left', left_on=['GMTdateYmD','Altitude (km)', 'Flight'], right_on = ['GMTdateYmD', 'Altitude (km)', 'Flight'])

new_df['RH_CFH'] = np.where(new_df.RH_CFH > 200, np.NAN, new_df.RH_CFH)
new_df['RH_frostpoint'] = np.where(((new_df.LaunchNo == 'run013') | (new_df.LaunchNo == 'run011')), new_df.hygrometerRH, new_df.RH_CFH)
new_df['H2OMr_frostpoint'] = np.where(((new_df.LaunchNo == 'run013') | (new_df.LaunchNo == 'run011')), new_df.H2OmrPPMV, new_df.H2OMr)
new_df['Pw_frostpoint'] = np.where(((new_df.LaunchNo == 'run013') | (new_df.LaunchNo == 'run011')), new_df.Pw_FPHc, new_df.Pw_CFH)


##calculate mean particle size (in each bin) in ambient air using in situ data to calculate % wt. and density according to Steele & Hamill 1981
new_df['Composition'] = new_df.apply(lambda x: Plume(x['LaunchNo'], x['Flight'], x['Altitude (km)']), axis=1)
#$wt calc
new_df['H2SO4_%wt_amb'] = new_df.apply(lambda x: PwLookuptable(x['iMetTempCorrC'], x['Pw_frostpoint']), axis = 1)
new_df['H2SO4_%wt_pops'] = new_df.apply(lambda x: PwLookuptable(x['POPStempC'], x['Pw_frostpoint']), axis = 1)
#density calculation
new_df['Density_amb'] = new_df.apply(lambda x: DensityCalc(x['H2SO4_%wt_amb'], x['iMetTempCorrC']), axis = 1)
new_df['Density_pops'] = new_df.apply(lambda x: DensityCalc(x['H2SO4_%wt_pops'], x['POPStempC']), axis = 1)


#dp bin sizes for measured size distribution and calculated ambient size distributions
new_df['BinDp_meas'] = new_df.apply(lambda x: DiaLookUpTable(x['Composition']), axis=1)
#new_df['BinDp_KK'] = new_df.apply(lambda x: KKAmbAer(x['RH_frostpoint'], x['Air Temp (K)'], x['Composition']), axis=1)
new_df['BinDp_SH'] = new_df.apply(lambda x: Steele_Hamill_calcDamb(x['Composition'], x['H2SO4_%wt_pops'], x['Density_pops'], x['H2SO4_%wt_amb'], x['Density_amb']), axis = 1) #perwt_pops, rho_pops, perwt_amb, rho_amb
#identical to BinDp (original equation simplified after substitution)
#new_df['BinDp_SHJ'] = new_df.apply(lambda x: Steele_Hamill_1981_J1995_calcDamb(x['Composition'], x['H2SO4_%wt_pops'], x['Density_pops'], x['H2SO4_%wt_amb'], x['Density_amb']), axis=1)

new_df = new_df.filter(['Altitude (km)', 'Launch','GMTdateYmD', 'LaunchNo', 'Flight', 'Composition', 'H2SO4_%wt_amb', 'Density_amb','H2SO4_%wt_pops', 'Density_pops', 'Density_unc_amb', 'RH_frostpoint', 'BinDp_meas','BinDp_SH', 'B1', 'B2', 
                      'B3','B4','B5','B6','B7','B8','B9','B10','B11', 'B12','B13', 'B14','B15', 'B16', 'B17', 'B18', 'B19', 'B20', 
                      'Dry Surface Area Density'], axis=1)

#calculate dNdLogDp for Fig. S2
new_df ['dNdLogDp'] = new_df .apply(lambda x: DnDlogDp(x['LaunchNo'], x['Composition'], 
                            x['B1'], x['B2'], x['B3'], x['B4'], x['B5'], x['B6'], x['B7'], x['B8'], x['B9'], x['B10'], x['B11'], 
                                              x['B12'], x['B13'], x['B14'], x['B15'], x['B16'], x['B17'], x['B18'], x['B19'], x['B20']), axis=1)

df2 = new_df[new_df['Altitude (km)'] >= 2]
a = pd.DataFrame(df2['dNdLogDp'])
b = pd.concat([pd.DataFrame(a['dNdLogDp'].values.tolist()) for c in a.columns], 
                  axis=1, 
                  keys=a.columns)

#create dataframes for size distributions that can be plotted.
b.columns = ['{}{}'.format(i, j) for i, j in b.columns]
df2.reset_index(drop=False, inplace = True)
b.reset_index(drop=False, inplace = True)
df2 = pd.concat([df2, b], axis = 1)
df2.reset_index(drop=True, inplace = True)
    
a = pd.DataFrame(df2['BinDp_meas'])
b = pd.concat([pd.DataFrame(a['BinDp_meas'].values.tolist()) for c in a.columns], 
                 axis=1, 
                 keys=a.columns)
b.columns = ['{}{}'.format(i, j) for i, j in b.columns]

df2.reset_index(drop=False, inplace = True)
b.reset_index(drop=False, inplace = True)
df2 = pd.concat([df2, b], axis = 1)
df2.reset_index(drop=True, inplace = True)
df2.drop(columns=['level_0','index'], inplace = True)

a = pd.DataFrame(df2['BinDp_SH'])
b = pd.concat([pd.DataFrame(a['BinDp_SH'].values.tolist()) for c in a.columns], 
                 axis=1, 
                 keys=a.columns)
b.columns = ['{}{}'.format(i, j) for i, j in b.columns]

df2.reset_index(drop=False, inplace = True)
b.reset_index(drop=False, inplace = True)
df2 = pd.concat([df2, b], axis = 1)
df2.reset_index(drop=True, inplace = True)

df2.drop(columns=['index'], inplace = True)


Dp_SH = ['BinDp_SH0', 'BinDp_SH1', 'BinDp_SH2', 'BinDp_SH3', 'BinDp_SH4', 'BinDp_SH5', 'BinDp_SH6', 'BinDp_SH7', 'BinDp_SH8', 'BinDp_SH9', 'BinDp_SH10', 
                            'BinDp_SH11', 'BinDp_SH12', 'BinDp_SH13', 'BinDp_SH14', 'BinDp_SH15', 'BinDp_SH16', 'BinDp_SH17', 'BinDp_SH18', 'BinDp_SH19']


Dp_meas = ['BinDp_meas0', 'BinDp_meas1', 'BinDp_meas2', 'BinDp_meas3', 'BinDp_meas4', 'BinDp_meas5', 'BinDp_meas6', 'BinDp_meas7', 'BinDp_meas8', 'BinDp_meas9', 'BinDp_meas10', 
                            'BinDp_meas11', 'BinDp_meas12', 'BinDp_meas13', 'BinDp_meas14', 'BinDp_meas15', 'BinDp_meas16', 'BinDp_meas17', 'BinDp_meas18', 'BinDp_meas19']


Dp = ['dNdLogDp0', 'dNdLogDp1', 'dNdLogDp2', 'dNdLogDp3', 'dNdLogDp4', 'dNdLogDp5', 'dNdLogDp6', 'dNdLogDp7', 'dNdLogDp8', 'dNdLogDp9', 'dNdLogDp10', 
                             'dNdLogDp11', 'dNdLogDp12', 'dNdLogDp13', 'dNdLogDp14', 'dNdLogDp15', 'dNdLogDp16', 'dNdLogDp17', 'dNdLogDp18', 'dNdLogDp19']

#add in measurement uncertainty for measured and ambient air size distributions distributions 
meas_unc_a = [0.03, 0.03, 0.03, 0.03, 0.03, 0.03, 0.03, 0.03, 0.03, 0.03, 0.03, 0.15, 0.15, 0.15, 0.15, 0.15, 0.15, 0.15, 0.15, 0.15]
meas_unc_b  = [0.03, 0.03, 0.03, 0.03, 0.03, 0.03, 0.03, 0.03, 0.15, 0.15, 0.15, 0.15, 0.15, 0.15, 0.15, np.NaN, np.NaN, np.NaN, np.NaN, np.NaN]
amb_unc_a  = np.sqrt(np.power(meas_unc_a, 2) + 0.06**2 + 0.01**2 + 0.02**2) #   Density_unc = np.sqrt(0.06**2 from H2O + 0.015**2 from temperature + 0.02**2 from Density_calc_unc**2) 
amb_unc_b = np.sqrt(np.power(meas_unc_b, 2) + 0.06**2  + 0.01**2 + 0.02**2)



df2['Sample']  = df2.apply(lambda x: Sample(x['GMTdateYmD'], x['Altitude (km)']), axis=1)
df2 = df2[df2['Sample'] != 'None']
#use melt functionnto create datasets like dry and ambient size distributions. then merge to plot with seaborn.relplot
df_SH = pd.melt(df2, id_vars=['Sample'], value_vars= Dp_SH, var_name = 'DpBin_SH', value_name = 'Dp_amb_calc')
df_meas = pd.melt(df2,id_vars=['Sample'], value_vars= Dp_meas, var_name = 'DpBin_meas', value_name = 'Dp_meas')

df_conc = pd.melt(df2,id_vars=['Sample'], value_vars= Dp, var_name = 'dNdLogDp_bins', value_name = 'Concentration')
df_conc[df_conc['Concentration'] < 0] = 0

df_conc_a = df_conc.drop(columns = ['Sample'], inplace = False)

df_meas_a = pd.concat([df_meas, df_conc_a], axis=1)
df_SH_a = pd.concat([df_SH, df_conc_a], axis=1)

#seperate out the size distributions that will be plotted
df_meas_b = df_meas_a[df_meas_a['Sample'] =='Mar 31, 21.5 km']
df_SH_b = df_SH_a[df_SH_a['Sample'] =='Mar 31, 21.5 km']

df_meas_a = df_meas_a[df_meas_a['Sample'] =='Jan 22 18 UTC, 26.5 km']
df_SH_a = df_SH_a[df_SH_a['Sample'] =='Jan 22 18 UTC, 26.5 km']

df_SH['Sizing'] = 'Calculated size in ambient air, Steele & Hamill 1981'

df_SH.drop(columns=['DpBin_SH'], inplace = True)
df_meas.drop(columns=['DpBin_meas'], inplace = True)
df_sc = pd.concat([df_meas, df_SH, df_conc], axis = 1)

df_meas['Sizing'] = 'Size at time of measurement'
df_SH['Dp'] = df_SH['Dp_amb_calc']
df_meas['Dp'] = df_meas['Dp_meas']

df_SH.drop(columns=['Dp_amb_calc'], inplace = True)
df_meas.drop(columns=['Dp_meas'], inplace = True)


df_meas_a['Unc'] = meas_unc_a
df_SH_a['Unc'] = amb_unc_a
df_meas_b['Unc'] = meas_unc_b
df_SH_b['Unc'] = amb_unc_b



seaborn.set(rc={"xtick.bottom" : True, "ytick.left" : True})   
seaborn.set(font_scale = 7)
seaborn.set(style="ticks")



fig, (ax1, ax2) = plt.subplots(1, 2)   
s1a = ax1.fill_betweenx(df_meas_a['Concentration'], df_meas_a['Dp_meas'] * (1 - df_meas_a['Unc']), df_meas_a['Dp_meas'] * (1 + df_meas_a['Unc']), where= df_SH_a['Dp_amb_calc'] < 750,interpolate=True, alpha=0.3, facecolor='black') #shaded black is the uncertainty in the Aerosol Mass (Tg S) estimate
s1b = ax1.fill_betweenx(df_meas_a['Concentration'], df_meas_a['Dp_meas'] * (1 - df_meas_a['Unc']), df_meas_a['Dp_meas'] * (1 + df_meas_a['Unc']), where= df_SH_a['Dp_amb_calc'] > 650,interpolate=True, alpha=0.3, facecolor='black') 

s2a = ax1.fill_betweenx(df_SH_a['Concentration'], df_SH_a['Dp_amb_calc'] * (1- df_SH_a['Unc']), df_SH_a['Dp_amb_calc'] * (1 + df_SH_a['Unc']),  where= df_SH_a['Dp_amb_calc'] < 800, interpolate=True, alpha=0.3, facecolor='red') #shaded black is the
s2b = ax1.fill_betweenx(df_SH_a['Concentration'], df_SH_a['Dp_amb_calc'] * (1- df_SH_a['Unc']), df_SH_a['Dp_amb_calc'] * (1 + df_SH_a['Unc']),  where= df_SH_a['Dp_amb_calc'] > 700, interpolate=True, alpha=0.3, facecolor='red')


l1 = ax1.plot(df_meas_a['Dp_meas'], df_meas_a['Concentration'], 'black', linewidth=1)
l2 = ax1.plot(df_SH_a['Dp_amb_calc'], df_SH_a['Concentration'], 'red', linewidth=1)

#Figure S2
ax1.set_ylim((0.5, 1000))              
ax1.set(xscale="log")
ax1.set(yscale="log")
ax1.grid(True)
ax1.set(xlabel="Dp (nm)")
ax1.set(ylabel="dN/dlogDp (# $\mathregular{cm^{-3}}$)")   
ax1.set(title="Jan. 22 18 UTC, 26.4 km")

s1a = ax2.fill_betweenx(df_meas_b['Concentration'], df_meas_b['Dp_meas'] * (1 - df_meas_b['Unc']), df_meas_b['Dp_meas'] * (1 + df_meas_b['Unc']), where= df_SH_b['Dp_amb_calc'] < 500,interpolate=True, alpha=0.3, facecolor='black') #shaded black is the uncertainty in the Aerosol Mass (Tg S) estimate
s1b = ax2.fill_betweenx(df_meas_b['Concentration'], df_meas_b['Dp_meas'] * (1 - df_meas_b['Unc']), df_meas_b['Dp_meas'] * (1 + df_meas_b['Unc']), where= df_SH_b['Dp_amb_calc'] > 400,interpolate=True, alpha=0.3, facecolor='black') 
s2a = ax2.fill_betweenx(df_SH_b['Concentration'], df_SH_b['Dp_amb_calc'] * (1- df_SH_b['Unc']), df_SH_b['Dp_amb_calc'] * (1 + df_SH_b['Unc']),  where= df_SH_b['Dp_amb_calc'] < 500, interpolate=True, alpha=0.3, facecolor='red') #shaded black is the
s2b = ax2.fill_betweenx(df_SH_b['Concentration'], df_SH_b['Dp_amb_calc'] * (1- df_SH_b['Unc']), df_SH_b['Dp_amb_calc'] * (1 + df_SH_b['Unc']),  where= df_SH_b['Dp_amb_calc'] > 250, interpolate=True, alpha=0.3, facecolor='red')
l1 = ax2.plot(df_meas_b['Dp_meas'], df_meas_b['Concentration'], 'black', linewidth=1)
l2 = ax2.plot(df_SH_b['Dp_amb_calc'], df_SH_b['Concentration'], 'red', linewidth=1)
           
ax2.set_ylim((0.5, 1000)) 
ax2.set_ylabel('')   
ax2.set(xscale="log")
ax2.set(yscale="log")
ax2.legend([s1a, s2a], ['Size at time of measurement', 'Calculated ambient size, Steele & Hamill 1981'])
ax2.grid(True)
ax2.set(xlabel="Dp (nm)")
ax2.set(ylabel="dN/dlogDp (# $\mathregular{cm^{-3}}$)")   
ax2.set(title="Mar. 31, 21.5 km")


