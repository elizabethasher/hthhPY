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
#import MLS file and contrast S mass estimates 
file_MLS = '/Users/easher/Documents/Tonga/MLSarea/NDgridded-21hPa.csv' #input path where NDgridded-21hPa.csv is stored
df_MLS = pd.read_csv(file_MLS, sep=",", engine = 'python')

file_latlon = '/Users/easher/Documents/Tonga/MLSarea/NDgridded-latlon.csv' #input path + file where NDgridded-latlon.csv is stored
df_latlon = pd.read_csv(file_latlon, sep=",", engine = 'python')

df_MLS.columns = df_latlon['lon']
df_MLS['lat'] = df_latlon['lat']
df_MLS.set_index('lat', inplace = True)
df_MLS = df_MLS.iloc[::-1]

#import sAOD file
file = '/Users/easher/Documents/Tonga/OMPS_Ghassen/TextFiles/nm997/997_sAOD_Reunion.csv' #input path where 997_sAOD_Reunion.csv is stored
colnames = ['col1', 'col2', 'col3', 'col4', 'col5', 'col6']
dfsAOD997ts_v0 = pd.read_csv(file, sep=",", engine = 'python', names = colnames)
dfsAOD997ts_v1 = pd.DataFrame(np.ravel(dfsAOD997ts_v0))


#convert DOY to date and show POPS profiles on these dates...
#DOY, Extinction 997 nm, Mass
#file1 = '/Users/easher/Desktop/Tonga/Other/Other Files/df_POPS_OMPS.csv'
file1 = '/Users/easher/Desktop/Tonga/Other/Other Files/df_POPS_OMPS_unc.csv'
df_POPS = pd.read_csv(file1, sep=",", engine = 'python')
df_POPS['Date'] = pd.to_datetime(df_POPS['GMTdateYmD'], infer_datetime_format=True) 
#convert string date to datetime and then day of year
#df_POPS = df_POPS[df_POPS.LaunchNo != 'Launch 4 - Least Perturbed']
df_POPS = df_POPS[df_POPS.LaunchNo != 'Launch 1 - 01/21'] # necessary?

df_POPS['DOY'] = df_POPS['Date'].dt.dayofyear
POPS_DOY_list = np.unique(df_POPS['DOY'].tolist())
POPS_comp = df_POPS

POPS_comp['Type'] = 'POPS'
POPS_comp = df_POPS.filter(['DOY', 'Altitude (km)', 'Tropopause', 'Extinction_2_K','Extinction_2_l_K','Extinction_2_J','Extinction_2_l_J', 'Mass','Mass_l', 'Mass_h', 'Type'])

day = np.arange(1,182,1)
height = np.arange(0,41, 1) + 0.5

POPS_Int = POPS_comp
POPS_Int.drop(['Type'], axis=1, inplace = True)
POPS_Int['Extinction_2_K'] = pd.to_numeric(POPS_Int['Extinction_2_K'], errors='coerce').fillna(0)  
POPS_Int['Extinction_2_J'] = pd.to_numeric(POPS_Int['Extinction_2_J'], errors='coerce').fillna(0)
POPS_Int['Extinction_2_l_K'] = pd.to_numeric(POPS_Int['Extinction_2_l_K'], errors='coerce').fillna(0)
POPS_Int['Extinction_2_l_J'] = pd.to_numeric(POPS_Int['Extinction_2_l_J'], errors='coerce').fillna(0)

POPS_Int['sAOD_K'] = POPS_Int['Extinction_2_K']*vertRes/1000 #units km-1
POPS_Int['sAOD_l_K'] = POPS_Int['Extinction_2_l_K']*vertRes/1000 #units km-1
POPS_Int['sAOD_J'] = POPS_Int['Extinction_2_J']*vertRes/1000 #units km-1
POPS_Int['sAOD_l_J'] = POPS_Int['Extinction_2_l_J']*vertRes/1000 #units km-1

POPS_Int_f = POPS_Int.groupby(['DOY'], as_index=False).sum()
POPS_Int_f['DOY_POPS'] = POPS_Int_f['DOY']
POPS_Int_f = POPS_Int_f.filter(['DOY_POPS','sAOD_K','sAOD_l_K','sAOD_J','sAOD_l_J', 'Mass', 'Mass_l', 'Mass_h'])
POPS_Int_f.reset_index(inplace = True, drop = True)
                        

[slope, intercept, r_value, p_value, std_err] = stats.linregress(POPS_Int_f['sAOD_K'],POPS_Int_f['Mass'])
[slope_a, intercept_a, r_value_a, p_value_a, std_err_a] = stats.linregress(POPS_Int_f['sAOD_J'],POPS_Int_f['Mass'])
[slope_b, intercept_b, r_value_b, p_value_b, std_err_B] = stats.linregress(POPS_Int_f['sAOD_l_K'],POPS_Int_f['Mass'])
[slope_c, intercept_c, r_value_c, p_value_c, std_err_c] = stats.linregress(POPS_Int_f['sAOD_l_J'],POPS_Int_f['Mass'])
[slope1, intercept1, r_value1, p_value1, std_err1] = stats.linregress(POPS_Int_f['sAOD_K'],POPS_Int_f['Mass_h'])
[slope1_a, intercept1_a, r_value1_a, p_value1_a, std_err1_a] = stats.linregress(POPS_Int_f['sAOD_J'],POPS_Int_f['Mass_h'])
[slope1_b, intercept1_b, r_value1_b, p_value1_B, std_err1_b] = stats.linregress(POPS_Int_f['sAOD_l_K'],POPS_Int_f['Mass_h'])
[slope1_c, intercept1_c, r_value1_c, p_value1_c, std_err1_c] = stats.linregress(POPS_Int_f['sAOD_l_J'],POPS_Int_f['Mass_h'])
[slope2, intercept2, r_value2, p_value2, std_err2] = stats.linregress(POPS_Int_f['sAOD_K'],POPS_Int_f['Mass_l'])
[slope2_a, intercept2_a, r_value2_a, p_value2_a, std_err2_a] = stats.linregress(POPS_Int_f['sAOD_J'],POPS_Int_f['Mass_l'])
[slope2_b, intercept2_b, r_value2_b, p_value2_b, std_err2_b] = stats.linregress(POPS_Int_f['sAOD_l_K'],POPS_Int_f['Mass_l'])
[slope2_c, intercept2_c, r_value2_c, p_value2_c, std_err2_c] = stats.linregress(POPS_Int_f['sAOD_l_J'],POPS_Int_f['Mass_l'])

Mass_type =['Mass', 'Mass_l', 'Mass_h']
sAOD_type =['sAOD_K','sAOD_l_K','sAOD_J','sAOD_l_J']

POPS_Int_p2 = pd.melt(POPS_Int_f,id_vars=['DOY_POPS'], value_vars= Mass_type, var_name = 'Calc Type_m', value_name = 'Mass')
POPS_Int_p1 = pd.melt(POPS_Int_f,id_vars=['DOY_POPS'], value_vars= sAOD_type, var_name = 'Calc Type_s', value_name = 'sAOD')
POPS_Int_p = pd.merge(POPS_Int_p1, POPS_Int_p2, how='left', on=['DOY_POPS'])
POPS_Int_p['Calc Type'] = POPS_Int_p['Calc Type_s'] +"-"+ POPS_Int_p["Calc Type_m"]

#Figure S4a
seaborn.set(font_scale = 2)
fig, ax = plt.subplots()       
ax=seaborn.lmplot(x="sAOD", y="Mass", hue = 'Calc Type', ci=68, data=POPS_Int_p)   
plt.ylim(0, 0.030)
plt.xlim(0, 0.20)
plt.xlabel(u"POPS sAOD ( \u03BB = 997 nm)")
plt.ylabel(u"S in eH2SO4 aerosol column (g S $\mathregular{m^{-2}}$)") 
plt.plot(np.linspace(0.0, 0.19, 20), np.linspace(0.0, 0.19, 20)*slope + intercept, 'k--')


#from netCDF4 import Dataset and create one dataframe seperated by launch...
os.chdir('/Users/easher/Documents/Tonga/OMPS_Ghassen/H5Files/')
filelist = glob.glob('*')
path = '/Users/easher/Documents/Tonga/OMPS_Ghassen/H5Files'
dfOMPS = pd.DataFrame()
Fitp = pd.DataFrame()
Fito = pd.DataFrame()

for file in filelist:
        fn = file
        
        ds = nc.Dataset(fn, "r", format="NETCDF4")

        #create dataframe
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
        
        if file == 'sAOD_997nm_map_daily_20220110_20220110.h5':
            dfsAOD_u_background = dfsAOD_u
            back_median = dfsAOD_u.median()
            dfsAOD_tropics = dfsAOD.iloc[20:70]
            dfsAOD_u_tropics = dfsAOD_tropics.unstack(level=-1)
            #print(dfsAOD_u_tropics.median())
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
        #if you want csv files for each day, you can print these to a directory
        #dfsAODplot.to_csv([INPUT a DIRECTORY HERE])
        
        df3.mask(df3 < 0.009,  other = 0, inplace = True) #0.011

        
        area_df.columns = df3.columns
        area_df.index = df3.index
        
        df3area = area_df.mask((df3 < TU3MAD) | (np.isnan(df3 )),  other = np.nan, inplace = True)
        
        df3mass = (df3*slope+intercept).mul(area_df)#
        backgr = (0.00254*slope+intercept)*area_df #0.028
        
        df3mass1 = (df3*slope2_a+intercept2_a).mul(area_df)#
        backgr1 = (0.00254*slope1+intercept1)*area_df #0.028
        
        df3mass2 = (df3*slope1_b+intercept1_b).mul(area_df)#
        backgr2 = (0.00254*slope2+intercept2)*area_df #0.028
        
        background = np.nansum((backgr.values)*1E-12)
        burden = (np.nansum(df3mass.values)*1E-12) - background
        
        background1 = np.nansum((backgr1.values)*1E-12)
        burden1 = (np.nansum(df3mass1.values)*1E-12) - background1
        
        background2 = np.nansum((backgr2.values)*1E-12)
        burden2 = (np.nansum(df3mass2.values)*1E-12) - background2
        
        year = int('20'+file[23:25]) 
        mon = int(file[25:27]) 
        day = int(file[27:29])
        Days =  dt(year, mon, day, 0, 0, 0)-dt(2022, 1, 15, 4, 0, 0)
        Elapsed_Time = Days.days + Days.seconds/60/60/24

        
        if file == 'sAOD_997nm_map_daily_20220123_20220123.h5':
            #upper estimate. Reunion Island is located at 55 E, between gridcell 8 and 9 (but closer probably to 8)
            df3mass_a25w1 = (df3.iloc[:,:9]*slope+intercept).mul(area_df.iloc[:,:9])#
            backgr_a25w1 = (0.00254*slope+intercept)*area_df.iloc[:,:9] #0.028
            background_a25w1 = np.nansum((backgr_a25w1.values)*1E-12)
            burden_a25w1 = (np.nansum(df3mass_a25w1.values)*1E-12) - background_a25w1
            #print(burden_a25w1)
            df3mass_a25w2 = (df3.iloc[:,:10]*slope+intercept).mul(area_df.iloc[:,:10])#
            backgr_a25w2 = (0.00254*slope+intercept)*area_df.iloc[:,:10] #0.028
            background_a25w2 = np.nansum((backgr_a25w2.values)*1E-12)
            burden_a25w2 = (np.nansum(df3mass_a25w2.values)*1E-12) - background_a25w2

        
        Fitp = Fitp.append(pd.DataFrame({'Days Since Jan 15 (04 UTC)': Elapsed_Time, 'Mass (Tg S) POPS': burden, 'Mass (Tg S) POPS max': burden2, 'Mass (Tg S) POPS min': burden1 }, index=[0]), ignore_index=True)

        RI_loc = pd.DataFrame(data= {'Lat': [-21.0], 'Lon': [48]})

        file_MLS_contour = '/Users/easher/Documents/Tonga/MLScontour/latloncontour-21hPa.csv'
        colnames = ['contour_no', 'lon', 'lat']
        df_polys = pd.read_csv(file_MLS_contour, sep=",", header = 0, names = colnames, engine = 'python')


Fit = Fitp
     
#Figure S4b
mpl.rcParams.update({'font.size': 16})
fig, ax = plt.subplots()
for a in [np.log10(dfsAOD_u_tropics), np.log10(dfsAOD_u_122)]:
    seaborn.distplot(a, ax = ax, kde = False)
fig.legend(labels=['1/10/22','1/22/22'], loc = 'center')
plt.axvline(x=np.log10(dfsAOD_u_122.median()),
    color='red')
plt.axvline(x=np.log10(TU3MAD_122),
    color='blue')
plt.axvline(x=np.log10(0.009),
    color='black')
plt.xlabel('Log10 OMPS sAOD')
plt.ylabel('Counts')
plt.yscale("log")
plt.title('Histogram')


Fit['Period']  = Fit.apply(lambda x: Period(x['Days Since Jan 15 (04 UTC)']), axis=1)
Fit['Days']  = Fit['Days Since Jan 15 (04 UTC)']
Fit['Mass']  = Fit['Mass (Tg S) POPS']
Fit['Massh']  = Fit['Mass (Tg S) POPS max']
Fit['Massl']  = Fit['Mass (Tg S) POPS min']
Fit.sort_values(by=['Days'], inplace=True, ascending=True) # This now sorts in date order

SO2 = pd.DataFrame()
SO2['Days'] = pd.Series([0, 75])
SO2['SO2l']  = 0.23
SO2['SO2h']  = 0.28


#Figure 4a 
fig, ax = plt.subplots(1, 1, figsize=(8, 4))
ax.fill_between(SO2.Days, SO2.SO2l, SO2.SO2h, alpha=0.7, facecolor='pink') #estimated SO2 injection (Carn et al. 202, Frontiers in Earth Sciences)
ax.fill_between(Fit.Days, Fit.Massl, Fit.Massh, alpha=0.3, facecolor='black') #shaded black is the uncertainty in the Aerosol Mass (Tg S) estimate
ax.plot(Fit.Days, Fit.Mass, '-k', linewidth=2) #calculated Aerosol Mass (Tg S)
ax.set(xlim = [-20, 75])
plt.errorbar([7.75], [0.24],yerr = 0.049,xerr = 1.5, fmt = 's', color = 'c') #FIX THIS.
plt.errorbar([7.7], [0.13],yerr = 0.019, xerr = 0.125, fmt = 'o', color = 'm')   
plt.legend(['S in SO2 injection', 'Uncertainty of S in eH2SO4', 'S in eH2SO4', 'S in eH2SO4 (MLS H2O/POPS)', 'S in High-altitude part of plume'])
plt.ylabel('Sulfur burden (Tg S)')
plt.xlabel('Days Since Jan 15 (04 UTC)')
plt.axvline(x=-1.514, color = 'r', linestyle = '--')
plt.axvline(x=0, color = 'r', linestyle = '-')
plt.axvline(x=20, color = 'k', linestyle ='-.')
plt.show()


Fit2 = Fit[Fit['Period'] == 'SO2 conversion to H2SO4 aerosol']
Fit2['eH2SO4 production'] = np.log(0.28/(0.28-(Fit2['Mass (Tg S) POPS']-0.03)))#0.27 -0.0005 or 0.03
[slope2, intercept2, r_value2, p_value2, std_err2] = stats.linregress(Fit2['Days Since Jan 15 (04 UTC)'],Fit2['eH2SO4 production'])

#Figure 4b
seaborn.set(font_scale = 2)
ax = seaborn.lmplot(x="Days Since Jan 15 (04 UTC)", y="eH2SO4 production", legend = None, data=Fit2, ci=68) #logx = True
plt.ylabel('SO2 \ H2SO4', rotation=0, ha="right")
plt.ylabel('ln(SO2 t=0 / (SO2 t=0  \n- H2SO4 + H2SO4 t=0))', rotation=90) #ha="right", multialignment='center')
