# run in geopy_env

import numpy as np 
import pandas as pd
import geopy
from geopy.geocoders import Nominatim
from geopy.extra.rate_limiter import RateLimiter

import pickle


def get_df_location(location_list):

    latitude = []
    longitude = []
    geo_countries = []

    latitude_irq = []
    longitude_irq = []
    geo_countries_irq = []

    for i in location_list: 
        geolocator = Nominatim(user_agent='spm@ifs.ku.dk')
        geocode = RateLimiter(geolocator.geocode, min_delay_seconds=1)
        
        try: # the idea here is to catch both timeout and non-returns
            location = geocode(i)

            latitude.append(location.latitude)
            longitude.append(location.longitude)
            geo_countries.append(location.address.split(',')[-1])
        
        except:

            latitude.append(None)
            longitude.append(None)
            geo_countries.append(None)

        try:

            location_irq = geocode(f'{i}, Iraq')

            latitude_irq.append(location_irq.latitude)
            longitude_irq.append(location_irq.longitude)
            geo_countries_irq.append(location_irq.address.split(',')[-1])

        except:

            latitude_irq.append(None)
            longitude_irq.append(None)
            geo_countries_irq.append(None)

    location_df = pd.DataFrame({'location' : location_list, 'latitude' : latitude, 'longitude' : longitude, 'geo_country': geo_countries, 
                                'latitude_irq' : latitude_irq, 'longitude_irq' : longitude_irq, 'geo_country_irq': geo_countries_irq })

    translation_dict = { 'العراق' : 'Iraq', ' العراق' : 'Iraq', ' الأردن' : 'Jordan', ' السعودية' : 'Saudi Arabia', ' سوريا' : 'Syria', ' Türkiye' : 'Turkey' }
    location_df.replace(translation_dict, inplace = True)

    return(location_df)


def merge_cordinates(df_full, df_cities, df_province):
    
    # if there is no iraq coordinates, take the other coordinates
    df_cities['longitude_irq'].fillna(df_cities['longitude'], inplace=True)  
    df_cities['latitude_irq'].fillna(df_cities['latitude'], inplace=True)

    df_cities_irq = df_cities[['location', 'latitude_irq', 'longitude_irq']].copy()
    df_cities_irq.rename(columns = {'location' : 'city', 'latitude_irq': 'latitude', 'longitude_irq' : 'longitude'}, inplace = True)

    df_full_copy = df_full.copy()
    tmp1 = pd.merge(df_full_copy, df_cities_irq, on='city', how= 'outer')

    df_province_irq = df_province[['location', 'latitude_irq', 'longitude_irq']].copy()
    df_province_irq.rename(columns = {'location' : 'province/state', 'latitude_irq': 'latitude_p', 'longitude_irq' : 'longitude_p'}, inplace = True)

    tmp2 = pd.merge(tmp1, df_province_irq, on='province/state', how= 'outer')

    # if there is no coordinates from city, take the coordinates from province
    tmp2['longitude'].fillna(tmp2['longitude_p'], inplace=True)  
    tmp2['latitude'].fillna(tmp2['latitude_p'], inplace=True)
   
    return(tmp2)

def get_data(data_dir):

    with open(f'{data_dir}df_od_full.pkl', 'rb') as file:
        df_full = pickle.load(file)

    # -------------------------

    fix_dict_cities = {'Habur border crossing' : 'Habur Port', 'Basra Basra' : 'Basra', 'Basrah' : 'Basra', 'Baghdad05' : 'Baghdad','Arab jabour' : 'Baghdad',
                    'Arab Jabour': 'Baghdad', 'Hay al Waahda' : 'Bagdad', 'Taqqadum' : 'Al Taqaddum', 'Ouwja' : 'Al-Awja', 'Husayba' : 'Husaybah', 
                    "no-man's land": 'Al Karamah Border Crossing', 'Zacho' : 'Zaxo'}

    df_full.replace(fix_dict_cities, inplace= True)
    cities = df_full['city'].unique()[1:]#You do not take nan

    # -------------------------

    mask = df_full['province/state'].str.len() > 20 # removes long wierd strings
    df_full.loc[mask] = np.nan
    province = df_full['province/state'].unique()[1:] # you do not take Nan

    return(df_full, cities, province)


def print_info(df):

    city_notna = df["city"].notna().sum()
    province_notna = df["province/state"].notna().sum()
    city_Nprovince = (df["city"].notna() & df["province/state"].isna()).sum()
    province_Ncity = (df["city"].isna() & df["province/state"].notna()).sum()
    both = (df["city"].notna() & df["province/state"].notna()).sum()
    expected = city_Nprovince + province_Ncity + both

    print(f'city entries: {city_notna}')
    print(f'province/state entries: {province_notna}')
    print(f'city but not province: {city_Nprovince}')
    print(f'province but not city: {province_Ncity}')
    print(f'Both: {both}')

    print(f'\nexcepted number of lat/long entries: {expected}')


def compile_coordinates():

    data_dir = '/home/simon/Documents/Bodies/data/OD_dataframes_compiled/'

    df_full, cities, province = get_data(data_dir)
    df_cities = get_df_location(cities)
    df_province = get_df_location(province)

    print_info(df_full)

    merged_df = merge_cordinates(df_full = df_full, df_cities = df_cities, df_province = df_province)

    print(f'coordinates: {merged_df["latitude"].notna().sum()}')

    with open(f'{data_dir}df_od_coordinates.pkl', 'wb') as file:
        pickle.dump(merged_df, file)


if __name__ == "__main__":
    compile_coordinates()