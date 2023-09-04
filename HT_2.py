#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  3 13:23:28 2022

@author: easher
"""

#Calculate plume mass - Figures S4a,b & Figures 4a,b
import os
from datetime import datetime as dt
import pandas as pd
import math
import numpy as np
import netCDF4 as nc
import glob
from matplotlib import pyplot as plt
import matplotlib as mpl
import seaborn as seaborn
from scipy import stats


def m2byLat(lat1, lat2):
   #https://en.wikipedia.org/wiki/Haversine_formula
   #https://stackoverflow.com/questions/639695/how-to-convert-latitude-or-longitude-to-meters/11172685#11172685
   
   dLon = 0.0 * math.pi / 180#degrees
   dLat = 2.0 * math.pi / 180#degrees

    #calculate gridcell lenth 
   R = 6378.137 # // Radius of earth in km
   a = np.sin(dLat/2) * np.sin(dLat/2) + np.cos(lat1 * np.pi / 180) * np.cos(lat2 * np.pi / 180) * np.sin(dLon/2) * np.sin(dLon/2)
   c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
   LenthGridCell = R * c * 1000 # in meters
   
   #calculate grid cell width
   dLon = 24.0 * math.pi / 180#degrees
   dLat = 0.0 * math.pi / 180#degrees
   lat2=lat1

   a1 = np.sin(dLat/2) * np.sin(dLat/2) + np.cos(lat1 * np.pi / 180) * np.cos(lat2 * np.pi / 180) * np.sin(dLon/2) * np.sin(dLon/2)
   c1 = 2 * np.arctan2(np.sqrt(a1), np.sqrt(1-a1))

   WidthGridCell = R * c1 * 1000
   Area = LenthGridCell * WidthGridCell
   return Area #; // m2

#define period for Figure 4a (Asher et al.)
def Period(Day):
    if (Day < -2):
        period = 'Pre-eruption'
    elif (Day <= 26):
        period = 'SO2 conversion to H2SO4 aerosol'
    else:
        period = 'H2SO4 aerosol burden dominated by loss processes'
    return period


#define variables
vertRes = 100 #data binned to every 100m (also a variable in other scripts from Asher et al., in review at PNAS)
#import MLS file for reference on H2O plume vs. aerosol layer extent 
file_MLS = '/Users/asher/Documents/PapersInProgress/Tonga/Submission/POPSdataForPublication/MLSdata/NDgridded-21hPa.csv' #input path where NDgridded-21hPa.csv is stored
df_MLS = pd.read_csv(file_MLS, sep=",", engine = 'python')

file_latlon = '/Users/asher/Documents/PapersInProgress/Tonga/Submission/POPSdataForPublication/MLSdata/NDgridded-latlon.csv' #input path + file where NDgridded-latlon.csv is stored
df_latlon = pd.read_csv(file_latlon, sep=",", engine = 'python')

df_MLS.columns = df_latlon['lon']
df_MLS['lat'] = df_latlon['lat']
df_MLS.set_index('lat', inplace = True)
df_MLS = df_MLS.iloc[::-1]

#import OMPS-LP sAOD file courtesy of G. Taha 
# file = '/Users/asher/Documents/PapersInProgress/Tonga/Submission/POPSdataForPublication/nm997/997_sAOD_Reunion.csv' 
# colnames = ['col1', 'col2', 'col3', 'col4', 'col5', 'col6']
# dfsAOD997ts_v0 = pd.read_csv(file, sep=",", engine = 'python', names = colnames)
# dfsAOD997ts_v1 = pd.DataFrame(np.ravel(dfsAOD997ts_v0))

#convert DOY to date and show POPS profiles on these dates...
#DOY, Extinction 997 nm, Mass
file1 = '/Users/asher/Documents/PapersInProgress/Tonga/Other/OtherFiles/df_POPS_OMPS_unc.csv' #calculated in HT_1.py 
df_POPS = pd.read_csv(file1, sep=",", engine = 'python')
df_POPS['Date'] = pd.to_datetime(df_POPS['GMTdateYmD'], infer_datetime_format=True) 
#convert string date to datetime and then day of year
df_POPS['DOY'] = df_POPS['Date'].dt.dayofyear
POPS_DOY_list = np.unique(df_POPS['DOY'].tolist())

#rename launches
df_POPS["GMTdateYmD"]= df_POPS["GMTdateYmD"].str.replace("2022-01-22", "Jan. 22", case = False) 
df_POPS["GMTdateYmD"]= df_POPS["GMTdateYmD"].str.replace("2022-01-23", "Jan. 23", case = False) 
df_POPS["GMTdateYmD"]= df_POPS["GMTdateYmD"].str.replace("2022-01-24", "Jan. 24", case = False) 
df_POPS["GMTdateYmD"]= df_POPS["GMTdateYmD"].str.replace("2022-01-25", "Jan. 25", case = False) 
df_POPS["GMTdateYmD"]= df_POPS["GMTdateYmD"].str.replace("2022-02-11", "Feb. 9", case = False)
df_POPS["GMTdateYmD"]= df_POPS["GMTdateYmD"].str.replace("2022-03-31", "Mar. 31",case = False)
df_POPS["GMTdateYmD"]= df_POPS["GMTdateYmD"].str.replace("2022-06-09", "Jun. 9", case = False) 
    

POPS_comp = df_POPS
POPS_comp['Type'] = 'POPS'
POPS_comp = df_POPS.filter(['GMTdateYmD', 'DOY', 'Altitude (km)', 'Tropopause','Extinction_2_m','Extinction_2_SH', 'Mass','Mass_l', 'Mass_h', 'Type'])

day = np.arange(1,182,1)
height = np.arange(0,41, 1) + 0.5

#calculations used in Figure S4a
POPS_Int = POPS_comp
POPS_Int.drop(['Type'], axis=1, inplace = True)
#POPS_Int['Extinction_2_K'] = pd.to_numeric(POPS_Int['Extinction_2_K'], errors='coerce').fillna(0)  
POPS_Int['Extinction_2_SH'] = pd.to_numeric(POPS_Int['Extinction_2_SH'], errors='coerce').fillna(0)
POPS_Int['Extinction_2_m'] = pd.to_numeric(POPS_Int['Extinction_2_m'], errors='coerce').fillna(0)

POPS_Int['Mass_low'] = pd.to_numeric(POPS_Int['Mass_l'], errors='coerce').fillna(0)
POPS_Int['Mass_high'] = pd.to_numeric(POPS_Int['Mass_h'], errors='coerce').fillna(0)

#POPS_Int['KK_sAOD'] = POPS_Int['Extinction_2_K']*vertRes/1000 #extinction units were km-1
POPS_Int['Meas_sAOD'] = POPS_Int['Extinction_2_m']*vertRes/1000 #extinction units were km-1
POPS_Int['SH_sAOD'] = POPS_Int['Extinction_2_SH']*vertRes/1000 #extinction units were km-1

#calculate sAOD based on ambient particle size and mass column (from POPS particle size distributions and other in situ data) 
POPS_Int_f = POPS_Int.groupby(['GMTdateYmD'], as_index=False).sum()
POPS_Int_f['GMTdateYmD'] = POPS_Int_f['GMTdateYmD']
POPS_Int_f = POPS_Int_f.filter(['GMTdateYmD', 'DOY_POPS','SH_sAOD','Meas_sAOD', 'Mass', 'Mass_low', 'Mass_high'])
POPS_Int_f.reset_index(inplace = True, drop = True)
                        
#calculate the regressions using the measured (dehydrated particle size) and the calculated ambient particle size (from steele and Hamill 1981) and the mass column (+ uncertainty, which is "Mass_high" and - uncertainty, which is "Mass_low", see below)
#slopes based on ambient particle size distributions using KK Theory no longer shown in revised Asher et al.
#[slope, intercept, r_value, p_value, std_err] = stats.linregress(POPS_Int_f['KK_sAOD'],POPS_Int_f['Mass'])
[slope_a, intercept_a, r_value_a, p_value_a, std_err_a] = stats.linregress(POPS_Int_f['SH_sAOD'],POPS_Int_f['Mass'])
[slope_b, intercept_b, r_value_b, p_value_b, std_err_B] = stats.linregress(POPS_Int_f['Meas_sAOD'],POPS_Int_f['Mass'])

#[slope1, intercept1, r_value1, p_value1, std_err1] = stats.linregress(POPS_Int_f['KK_sAOD'],POPS_Int_f['Mass_high'])
[slope1_a, intercept1_a, r_value1_a, p_value1_a, std_err1_a] = stats.linregress(POPS_Int_f['SH_sAOD'],POPS_Int_f['Mass_high'])
[slope1_b, intercept1_b, r_value1_b, p_value1_B, std_err1_b] = stats.linregress(POPS_Int_f['Meas_sAOD'],POPS_Int_f['Mass_high'])

#[slope2, intercept2, r_value2, p_value2, std_err2] = stats.linregress(POPS_Int_f['KK_sAOD'],POPS_Int_f['Mass_low'])
[slope2_a, intercept2_a, r_value2_a, p_value2_a, std_err2_a] = stats.linregress(POPS_Int_f['SH_sAOD'],POPS_Int_f['Mass_low'])
[slope2_b, intercept2_b, r_value2_b, p_value2_b, std_err2_b] = stats.linregress(POPS_Int_f['Meas_sAOD'],POPS_Int_f['Mass_low'])

#slope_m = np.median([slope_a, slope_b, slope1_a, slope1_b, slope2_a, slope2_b])
#intercept_m = np.median([intercept_a, intercept_b, intercept1_a, intercept1_b, intercept2_a, intercept2_b])

Mass_type =['Mass', 'Mass_low', 'Mass_high']
sAOD_type =['Meas_sAOD','SH_sAOD'] #'KK_sAOD' not shown now in Figure S4a.

#create long format data table
POPS_Int_p2 = pd.melt(POPS_Int_f,id_vars=['GMTdateYmD'], value_vars= Mass_type, var_name = 'Calc Type_m', value_name = 'Mass')
POPS_Int_p1 = pd.melt(POPS_Int_f,id_vars=['GMTdateYmD'], value_vars= sAOD_type, var_name = 'Calc Type_s', value_name = 'sAOD')
POPS_Int_p = pd.merge(POPS_Int_p1, POPS_Int_p2, how='left', on=['GMTdateYmD'])
POPS_Int_p['Calc Type'] = POPS_Int_p['Calc Type_s'] +"-"+ POPS_Int_p["Calc Type_m"]

#Figure S4a
seaborn.set(rc={"xtick.bottom" : True, "ytick.left" : True})   
seaborn.set(font_scale = 2)
seaborn.set(style="ticks")

fig, ax = plt.subplots()       
#ax=seaborn.lmplot(x="sAOD", y="Mass", hue = 'Calc Type', hue_order = ['KK_sAOD-Mass', 'KK_sAOD-Mass_low', 'KK_sAOD-Mass_high', 'KKlow_sAOD-Mass', 'KKlow_sAOD-Mass_low', 'KKlow_sAOD-Mass_high'], ci=68, data=POPS_Int_p)   #added hue order
x=seaborn.scatterplot(x="sAOD", y="Mass", hue = 'GMTdateYmD', hue_order = ['Jan. 22', 'Jan. 23', 'Jan. 24', 'Jan. 25', 'Feb. 9', 'Mar. 31', 'Jun. 9'], data=POPS_Int_p, palette = 'dark')   #added hue order
#plt.plot(np.linspace(0.0, 0.19, 20), np.linspace(0.0, 0.19, 20)*slope_m + intercept_m, 'k-')
plt.plot(np.linspace(0.0, 0.25, 20), np.linspace(0.0, 0.25, 20)*slope_a + intercept_a, 'k-')
plt.plot(np.linspace(0.0, 0.25, 20), np.linspace(0.0, 0.25, 20)*slope2_a + intercept2_a, 'k--')
plt.plot(np.linspace(0.0, 0.25, 20), np.linspace(0.0, 0.25, 20)*slope1_b + intercept1_b, 'k--')
plt.ylim(0, 0.030)
plt.xlim(0, 0.25)
plt.xlabel(u"POPS sAOD ( \u03BB = 997 nm)")
plt.ylabel(u"S in eH2SO4 aerosol column (g S $\mathregular{m^{-2}}$)") 



#from netCDF4 import Dataset and create one dataframe seperated by launch...
os.chdir('/Users/asher/Documents/PapersInProgress/Tonga/Submission/POPSdataForPublication/OMPSData/') # H5Filesn of globally gridded sAOD
filelist = glob.glob('*')
path = '/Users/asher/Documents/PapersInProgress/Tonga/Submission/POPSdataForPublication/OMPSData/'
dfOMPS = pd.DataFrame()
Fitp = pd.DataFrame()
Fito = pd.DataFrame()

for file in filelist:
        fn = file
        
        ds = nc.Dataset(fn, "r", format="NETCDF4")

        #create dataframe to locate OMPS-LP lat, lon, calculated area and sAOD.
        df1 = pd.DataFrame()
        Latitude = pd.DataFrame(ds['bins_lat'][:])
        Longitude = pd.DataFrame(ds['bins_lon'][:])
        Latitude['Latitude_next'] = Latitude.shift(periods=1, fill_value=np.nan)
        Latitude.columns = ['Latitude', 'Latitude_next']
        Latitude['Area_m2'] = Latitude.apply(lambda x: m2byLat(x['Latitude'], x['Latitude_next']), axis=1)
    
        lat_t = Latitude.rolling(2).mean()
        lat_t = lat_t['Latitude']
        lon_t = Longitude.rolling(2).mean()
        lon_t.columns= ['Longitude']
        lat = (lat_t[1:]).values
        lat = lat[::-1]
        area = Latitude['Area_m2']
        area = (area[1:]).values
        area = area[::-1]
        area_df = pd.DataFrame(np.tile(area.reshape(90, 1), (1, 15)))
        lon = (lon_t[1:]).values
        lon = np.ravel(lon)
        
        sAOD = ds['bins_ret'][:,:]
        dfsAOD = pd.DataFrame(sAOD)
        dfsAOD = dfsAOD.iloc[::-1]
        dfsAOD_u = dfsAOD.unstack(level=-1)
        Mad = dfsAOD_u.mad(axis=None, skipna=True, level=None)
        Median = dfsAOD_u.median(axis=None, skipna=True, level=None)
        TL3MAD = Median - Mad*3.0
        TU3MAD = Median + Mad*3.0
        
        #prior to eruption - used to calculate background sAOD
        if file == 'sAOD_997nm_map_daily_20220110_20220110.h5':
            dfsAOD_u_background = dfsAOD_u
            back_median = dfsAOD_u.median()
            dfsAOD_tropics = dfsAOD.iloc[20:70]
            dfsAOD_u_tropics = dfsAOD_tropics.unstack(level=-1)
            print(dfsAOD_u_tropics.median())
            #print(back_median)
        elif file == 'sAOD_997nm_map_daily_20220122_20220122.h5':
            dfsAOD_u_122 = dfsAOD_u 
            TU3MAD_122 = TU3MAD
            
        ds.close()
        
        dfsAOD.columns = lon
        dfsAOD['Lat'] = lat
        dfsAOD.set_index('Lat', inplace = True)
        
        
        df3 = dfsAOD 
        dfsAODplot = dfsAOD.unstack(level=-1)
        dfsAODplot = pd.DataFrame(dfsAODplot)
        dfsAODplot.reset_index(inplace = True)
        dfsAODplot.rename(columns={0: "Data"}, inplace = True)
        
        df3.mask(df3 < 0.009,  other = 0, inplace = True) #0.009 is the maximum value observed in background sAOD between 50 N and 50 S
        area_df.columns = df3.columns
        area_df.index = df3.index
        df3area = area_df.mask((df3 < TU3MAD) | (np.isnan(df3 )),  other = np.nan, inplace = True)
        
        #calculate the total S burden and the background sulfur burden 
        df3mass = (df3*slope_a+intercept_a).mul(area_df)#
        backgr = (0.0025*slope_a+intercept_a)*area_df #0.028 ??
        
        df3mass1 = (df3*slope2_a+intercept2_a).mul(area_df)#
        backgr1 = (0.0025*slope2_a+intercept2_a)*area_df #0.028 #slope1+intercept1
        
        df3mass2 = (df3*slope1_b+intercept1_b).mul(area_df)#
        backgr2 = (0.0025*slope1_b+intercept1_b)*area_df #0.028 #slope2+intercept2
        
        #calculate the total S burden in the aerosol layer only
        background = np.nansum((backgr.values)*1E-12)
        burden = (np.nansum(df3mass.values)*1E-12) - background
        
        background1 = np.nansum((backgr1.values)*1E-12)
        burden1 = (np.nansum(df3mass1.values)*1E-12) - background1
        
        background2 = np.nansum((backgr2.values)*1E-12)
        burden2 = (np.nansum(df3mass2.values)*1E-12) - background2
        
        #calculate time from the eruption
        year = int('20'+file[23:25]) 
        mon = int(file[25:27]) 
        day = int(file[27:29])
        Days =  dt(year, mon, day, 0, 0, 0)-dt(2022, 1, 15, 4, 0, 0)
        Elapsed_Time = Days.days + Days.seconds/60/60/24

        #For Figure 4a, calcuate the S in the part of the aerosol layer above 25 km, which is the S burden west of La Reunion on 1/23 00 UTC, when the upper altitude part of the aerosol layer was last observed (accoring to both POPS profiles) and lidear data Baron et al. 2023
        if file == 'sAOD_997nm_map_daily_20220123_20220123.h5':
            #Reunion Island is located at 55 E, between gridcell 8 and 9 (slightly closer to 8)
            df3mass_a25w1 = (df3.iloc[:,:9]*slope_a+intercept_a).mul(area_df.iloc[:,:9])#
            backgr_a25w1 = (0.0025*slope_a+intercept_a)*area_df.iloc[:,:9] 
            background_a25w1 = np.nansum((backgr_a25w1.values)*1E-12)
            burden_a25w1 = (np.nansum(df3mass_a25w1.values)*1E-12) - background_a25w1
            #print(burden_a25w1)
            df3mass_a25w2 = (df3.iloc[:,:10]*slope_a+intercept_a).mul(area_df.iloc[:,:10])#
            backgr_a25w2 = (0.0025*slope_a+intercept_a)*area_df.iloc[:,:10] 
            background_a25w2 = np.nansum((backgr_a25w2.values)*1E-12)
            burden_a25w2 = (np.nansum(df3mass_a25w2.values)*1E-12) - background_a25w2
            #print(burden_a25w2)
        
        #Dataframe used to plot Figure 4a
        Fitp = Fitp.append(pd.DataFrame({'Days Since Jan 15 (04 UTC)': Elapsed_Time, 'Mass (Tg S) POPS': burden, 'Mass (Tg S) POPS max': burden2, 'Mass (Tg S) POPS min': burden1 }, index=[0]), ignore_index=True)

        RI_loc = pd.DataFrame(data= {'Lat': [-21.0], 'Lon': [48]})
        
        file_MLS_contour = '/Users/asher/Documents/PapersInProgress/Tonga/Submission/POPSdataForPublication/MLScontour/latloncontour-21hPa.csv'
        colnames = ['contour_no', 'lon', 'lat']
        df_polys = pd.read_csv(file_MLS_contour, sep=",", header = 0, names = colnames, engine = 'python')


Fit = Fitp     


#create a dataframe for size distributions shown in Fig. S4b
dfsAOD_u_122.reset_index(inplace = True, drop = True)
dfsAOD_u_tropics.reset_index(inplace = True, drop = True)
sAOD_u_122 = pd.DataFrame()
sAOD_u_122['sAOD'] = dfsAOD_u_122
sAOD_u_122['Date'] = '1/10/22'
sAOD_u_tropics = pd.DataFrame()
sAOD_u_tropics['sAOD'] = dfsAOD_u_tropics
sAOD_u_tropics['Date'] = '1/22/22'
sAOD_hist = pd.concat([sAOD_u_tropics, sAOD_u_122],  ignore_index=True, sort=False) 

#Figure S4b Asher et al.
mpl.rcParams.update({'font.size': 16})
fig, ax = plt.subplots()
seaborn.histplot(data= sAOD_hist, x="sAOD", hue="Date", element="step")
plt.axvline(x=dfsAOD_u_122.median(),
    color='red')
plt.axvline(x=TU3MAD_122,
    color='blue')
plt.axvline(x=0.009,
    color='black')
plt.xlabel('Log10 OMPS sAOD')
plt.ylabel('Counts')
plt.yscale("log")
plt.xscale("log")
plt.title('Histogram')
ax.set(xlim = [0.0008, 0.3])


#finalize dataframe used to plot Figure 4a
Fit['Period']  = Fit.apply(lambda x: Period(x['Days Since Jan 15 (04 UTC)']), axis=1)
Fit['Days']  = Fit['Days Since Jan 15 (04 UTC)']
Fit['Mass']  = Fit['Mass (Tg S) POPS']
Fit['Massh']  = Fit['Mass (Tg S) POPS max']
Fit['Massl']  = Fit['Mass (Tg S) POPS min']
Fit.sort_values(by=['Days'], inplace=True, ascending=True) # This now sorts in date order
Fit = Fit.round(3)
SO2 = pd.DataFrame()
SO2['Days'] = pd.Series([0, 75])
SO2['SO2l']  = 0.195
SO2['SO2h']  = 0.215


#Figure 4a Asher et al.
seaborn.set(font_scale = 7)
seaborn.set(style="ticks")
fig, ax = plt.subplots(1, 1, figsize=(8, 4))
ax.fill_between(SO2.Days, SO2.SO2l, SO2.SO2h, alpha=0.7, facecolor='pink') #estimated SO2 injection (Carn et al. 202, Frontiers in Earth Sciences)
ax.fill_between(Fit.Days, Fit.Massl, Fit.Massh, alpha=0.3, facecolor='black') #shaded black is the uncertainty in the Aerosol Mass (Tg S) estimate
ax.plot(Fit.Days, Fit.Mass, '-k', linewidth=2) #calculated Aerosol Mass (Tg S)
ax.set(xlim = [-20, 75])
#plt.errorbar([7.75], [0.24],yerr = 0.049,xerr = 1.5, fmt = 's', color = 'c') #FIX THIS.
plt.errorbar([7.7], [0.088],yerr = 0.013, xerr = 0.125, fmt = 'o', color = 'b', ms = 8)   
plt.legend(['S in SO2 injection', 'Uncertainty of S in eH2SO4', 'S in eH2SO4', 'S in High-altitude part of plume'])
plt.ylabel('Sulfur burden (Tg S)',  fontsize = 15)
plt.xlabel('Days Since Jan 15 (04 UTC)', fontsize = 15)
plt.axvline(x=-1.514, color = 'r', linestyle = '--')
plt.axvline(x=0, color = 'r', linestyle = '-')
plt.axvline(x=20, color = 'k', linestyle ='-.')
plt.show()

#dataframe used to plot Figure 4bAsher et al.
Fit2 = Fit[Fit['Period'] == 'SO2 conversion to H2SO4 aerosol']
Fit2 = Fit2.round(3)
Fit2['eH2SO4 production'] = np.log(0.205/(0.205-(Fit2['Mass (Tg S) POPS']-0.0005)))
#0.195 - 0.215 Tg S are the Tg S injected SO2, according to satellite data (range stated by Millan et al. 2022 and supported by Carn et al. 2022). These are used as the initial injection estimates of S, to determine the estimated lifetime.
# -0.0005 or 0.03 initial S amounts (0.0005 is the measured background on 1/10, 0.03 Tg S is used as an example of how much slower the Tstrat would be if the aerosol were not all composed of H2SO4)
[slope2, intercept2, r_value2, p_value2, std_err2] = stats.linregress(Fit2['Days Since Jan 15 (04 UTC)'],Fit2['eH2SO4 production'])
print(1/slope2)
#Figure 4b
seaborn.set(font_scale = 2)
seaborn.set(style="ticks")
fig, ax = plt.subplots(1, 1, figsize=(8, 4))
ax = seaborn.lmplot(x="Days Since Jan 15 (04 UTC)", y="eH2SO4 production", legend = None, data=Fit2, ci=68) #logx = True
plt.ylabel('SO2 \ H2SO4', rotation=0, ha="right")
plt.ylabel('ln(SO2 t=0 / (SO2 t=0  \n- H2SO4 + H2SO4 t=0))', rotation=90) #ha="right", multialignment='center')
