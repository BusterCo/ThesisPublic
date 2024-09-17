# -*- coding: utf-8 -*-
"""
Created on Fri Dec  8 17:30:33 2023

@author: J.C. Stokx
"""
import pandas as pd
import math 
import re
import os
class PC():
    def __init__(self):
        print("Creating PC Class object")
        column_names = ['country_code', 'postal_code', 'place_name', 'Province', 'Province_Code',
                'admin_name2', 'admin_code2', 'admin_name3', 'admin_code3',
                'latitude', 'longitude', 'accuracy']
        columns_to_drop = ['admin_name2', 'admin_code2', 'admin_name3', 'admin_code3', 'accuracy']
        current_dir = os.getcwd()
        LUdf = pd.read_csv(current_dir +"\\data\\geodata\\LU.txt",sep='\t', header=None, names=column_names)
        BEdf = pd.read_csv(current_dir +"\\data\\geodata\\BE.txt",sep='\t', header=None, names=column_names)
        NLdf = pd.read_csv(current_dir +"\\data\\geodata\\NL_full.txt",sep='\t', header=None, names=column_names)
        PCdf = pd.concat([NLdf, BEdf, LUdf], ignore_index=True)
        PCdf.drop(columns=columns_to_drop, inplace=True)

        self.PC_dict = {}
        self.PC_dict["NL"] = {}
        self.PC_dict["BE"] = {}
        self.PC_dict["LU"] = {}
        for index, row in PCdf.iterrows():
            self.PC_dict[row["country_code"]][row["country_code"]+str(row["postal_code"]).replace(" ", "").replace("L-", "")] = {"postal_code": row["country_code"] + str(row["postal_code"]).replace(" ", "").replace("L-", ""),
                                                               "place_name": row["place_name"],
                                                               "Province": row["Province"],
                                                                "Province_Code": row["Province_Code"],
                                                               "latitude": row["latitude"],
                                                               "longitude": row["longitude"]}
        del LUdf, BEdf, NLdf,PCdf


        PC4df = pd.read_csv(current_dir +"\\data\\geodata\\pc4_coordinaten.csv", encoding='ISO-8859-1')
        PC4df.columns = PC4df.columns.str.replace(' ', '_')
        # Splitting the 'Geo_Point' column into two separate columns
        PC4df[['lat', 'lon']] = PC4df['Geo_Point'].str.split(',', expand=True).astype(float)
                
        self.PC4_dict = {}
        self.PC4_dict["NL"] = {}

                
        for index, row in PC4df.iterrows():
            self.PC4_dict["NL"][row["PC4"]] = {"postal_code": "NL"+str(row["PC4"]).replace(" ", ""),
                                                                       "place_name": row["Gemeente_name"],
                                                                       "Province": row["Provincie_name"],
                                                                        "Province_Code": row["Provincie_code"],
                                                                       "latitude": row["lat"],
                                                                       "longitude": row["lon"]}
        
        self.patternNLPC6 = r'^\d{4}\s[A-Z]{2}$'
        self.patternNLPC6_no_space = r'^\d{4}[A-Z]{2}$'
        self.patternNLPC4 = r'^[A-Z]{2}\d{4}$'
        
        print("PC Class loaded")

    def return_codes_set(self):
        codelist = []
        for country in self.PC_dict:
            codelist += list(self.PC_dict[country].keys())
        return set(codelist)
        
    def PC_in_dict(self, country, postalcode):
        if country in self.PC_dict:
            if country+postalcode in self.PC_dict[country]:
                return True
            elif country == "NL":
                if (country+postalcode in self.PC4_dict[country]):
                    return True
                else:
                    return False
            else:
                return False
        else:
            return False

    def return_CPC(self, country, postalcode):
        if self.PC_in_dict(country, postalcode):	
            return str(country)+str(postalcode)
        if country == "NL" and re.match(self.patternNLPC6_no_space, postalcode):
            if country+str(postalcode[:4]) in self.PC4_dict[country]:
                return country+postalcode[:4]
            else:
                return(0)
        else: 
            print("Error on", country, " ",postalcode)
            return(0)
        

    def get_coordinates(self, postcode):
        country_code = postcode[:2]
        if country_code == "NL":
            if re.match(self.patternNLPC6_no_space, postcode):
                postcode = postcode[:6]+" "+postcode[6:]
            elif re.match(self.patternNLPC4, postcode):
                print(postcode)
                print("PC4 found!")
                return (self.PC4_dict[country_code][postcode]["latitude"],self.PC4_dict[country_code][postcode]["longitude"])
        if postcode in self.PC_dict[country_code]:
            return (self.PC_dict[country_code][postcode]["latitude"],self.PC_dict[country_code][postcode]["longitude"])
        elif postcode[2:6] in self.PC4_dict[country_code]:
            print("PC6 not found, reverting back to PC4 for: ", postcode)
            return (self.PC4_dict[country_code][postcode[2:6]]["latitude"],self.PC4_dict[country_code][postcode[2:6]]["longitude"])
        
        else:
            print("PC not in dict: " + postcode)
            return False

    def return_CPC_coordinates(self, CPC):
        country = CPC[:2]
        postalcode = CPC[2:]
        if self.PC_in_dict(country, postalcode):	
            return self.PC_dict[country][country+postalcode]["latitude"], self.PC_dict[country][country+postalcode]["longitude"]
        elif country == "NL" and re.match(self.patternNLPC4, CPC):
            if CPC in self.PC4_dict[country]:
                return self.PC4_dict[country][CPC]["latitude"], self.PC4_dict[country][CPC]["longitude"]
        else: 
            print("Error on", country, " ",postalcode)
            return(0)

    def get_distance(self, PC1, PC2):
        PC1_crds = self.get_coordinates(PC1)
        PC2_crds = self.get_coordinates(PC2)
        return self.haversine_distance(PC1_crds[0], PC1_crds[1], PC2_crds[0], PC2_crds[1])
    

    def haversine_distance(self, lat1, lon1, lat2, lon2):
        #
        # Convert latitude and longitude from degrees to radians
        lat1 = math.radians(lat1)
        lon1 = math.radians(lon1)
        lat2 = math.radians(lat2)
        lon2 = math.radians(lon2)
    
        # Haversine formula
        dlon = lon2 - lon1
        dlat = lat2 - lat1
    
        a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    
        # Calculate the distance
        distance = 6371.0 * c
    
        return distance
    
    def get_closest_hub(self, postcode, hubs):
        #print(postcode, hubs)
        closest = 99999999999999999 # big M
        closehub = 0
        for hub in hubs:
            dist = self.get_distance(postcode, hub)
            if dist < closest:
                closest = dist
                closehub = hub
        return closehub
    
    def return_coordinate_dict(self, CPC_list):
        coordinate_dict = {}
        for CPC in CPC_list:
            coordinate_dict[CPC] = self.get_coordinates(CPC)
        return coordinate_dict
                    
        