from scipy.spatial import distance
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from custom.PC_Class import PC
import geopandas as gpd
import os
from shapely.geometry import MultiPolygon, Polygon

class GeoSpatialEncoder(PC):
    def __init__(self, PC):
        self.PC_obj = PC
        self.hier_dict = {}
        self.verbose = True
        self.distribution = None

        # Load medium-resolution shapefiles for the world and filter for Netherlands, Belgium, and Luxembourg
        direct = os.getcwd()
        world  = gpd.read_file(direct + '\\data\\ne_50m_admin_0_countries.shp') 
        countries = world[(world['NAME'] == 'Belgium') | (world['NAME'] == 'Luxembourg')]

        # Assuming 'world' is already loaded and contains the necessary shapefile data
        netherlands = world[world['NAME'] == 'Netherlands'].copy()

        # Define coordinate boundaries (for example, excluding territories outside mainland Europe)
        min_latitude = 50.75  # Northern limit of mainland Europe approx
        max_latitude = 54
        min_longitude = 3.35  # Western limit of mainland Europe approx
        max_longitude = 7.22

        # Filter polygons based on coordinate boundaries
        def within_bounds(polygon, min_lat, max_lat, min_lon, max_lon):
            """ Check if the polygon falls within the specified lat/lon bounds """
            x, y = polygon.exterior.coords.xy
            return all(min_lon <= lon <= max_lon for lon in x) and all(min_lat <= lat <= max_lat for lat in y)

        # Iterate over each polygon in the MultiPolygon
        if isinstance(netherlands.geometry.iloc[0], MultiPolygon):
            filtered_polygons = [polygon for polygon in netherlands.geometry.iloc[0].geoms 
                                if within_bounds(polygon, min_latitude, max_latitude, min_longitude, max_longitude)]
        else:
            # If it's not a MultiPolygon (unlikely in this case), handle it as a single Polygon
            filtered_polygons = [netherlands.geometry.iloc[0]] if within_bounds(netherlands.geometry.iloc[0], min_latitude, max_latitude, min_longitude, max_longitude) else []

        # Create a new geometry from the filtered polygons
        new_geometry = MultiPolygon(filtered_polygons) if len(filtered_polygons) > 1 else (filtered_polygons[0] if filtered_polygons else None)

        # Update the geometry in the GeoDataFrame
        netherlands.at[netherlands.index[0], 'geometry'] = new_geometry

        # Use concat to merge GeoDataFrames correctly
        self.countries = gpd.GeoDataFrame(pd.concat([countries, netherlands], ignore_index=True))
        return

    def set_verbose(self, verbose):
        self.verbose = verbose
        return

    def return_basic_region(PC):
        return f'REGION_{int(str(PC[0]))}'
    
    def return_hierarchical_region(PC):
        return 1

    def set_input_df(self, df):
        self.df_input = df
        if self.verbose:
            print("Set input dataframe")

    def clean_input_df(self):
        self.df_filtered = self.df_input[(self.df_input['AFHCODE'].isin(['d'])) 
                                         & (self.df_input['LOSLAND'].isin(['NL', 'BE', 'LU']))].copy()
        self.df_filtered["LOSLAND"] = self.df_filtered["LOSLAND"].astype(str).copy()  # Ensure 'LOSLAND' column is of string type
        self.df_filtered["LOSPC"] = self.df_filtered["LOSPC"].astype(str).copy()  # Ensure 'LOSPC' column is of string type
        self.df_filtered["LOS_CPC"] = self.df_filtered.apply(lambda row: self.PC_obj.return_CPC(row["LOSLAND"], row["LOSPC"]), axis=1)
        if self.verbose:
            print("CPC added to the dataframe")

        self.df_filtered = self.df_filtered[self.df_filtered["LOS_CPC"] != 0]
        self.df_filtered["COORDINATES"] = self.df_filtered.apply(lambda row: self.PC_obj.return_CPC_coordinates(row["LOS_CPC"]), axis=1)
        if self.verbose:
            print("Coordinates added to the dataframe")

        self.df_filtered[["LOS_LAT", "LOS_LON"]] = pd.DataFrame(self.df_filtered["COORDINATES"].tolist(), index=self.df_filtered.index)

    def return_df_grouped_CPC(self):
        self.df_grouped_CPC = self.df_filtered.groupby('LOS_CPC').agg({'PALLETPLAATSEN': 'sum', 
                                                                       'SHIPMENTNUMBER': 'count',
                                                                   'LOS_LAT': 'first', 
                                                                   'LOS_LON': 'first',
                                                                   'COORDINATES': 'first'}).reset_index()
        self.df_grouped_CPC.rename(columns={'SHIPMENTNUMBER': 'SHIPMENT_COUNT'}, inplace=True)
        return self.df_grouped_CPC
    
    def return_df_scaled(self, scale_on=None):
        # Repeat the rows based on the value in 'Shipment_Count'
        df_repeated = self.df_grouped_CPC.loc[self.df_grouped_CPC.index.repeat(self.df_grouped_CPC[scale_on])].reset_index(drop=True)

        # Drop the 'Shipment_Count' column
        df_repeated = df_repeated.drop(columns=['SHIPMENT_COUNT'])
        return df_repeated
    
    def return_df_cleaned(self):
        return self.df_filtered
    
    def train_kmeans(self, n_clusters, scale_on=None):
        self.clustertype = "kmeans"
        self.kmeans_n_clusters = n_clusters
        self.df_CPC_kmeans = self.return_df_grouped_CPC().copy()
        if scale_on is not None:
            self.df_CPC_kmeans = self.return_df_scaled(scale_on).copy()
        self.kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(self.df_CPC_kmeans[["LOS_LAT", "LOS_LON"]])
        self.df_CPC_kmeans["CLUSTER"] = self.kmeans.labels_
        self.kmeans_dict = {}
        for i, row in self.df_CPC_kmeans.iterrows():
            self.kmeans_dict[row["LOS_CPC"]] = row["CLUSTER"]

    def train_agglomerative(self, n_clusters=None, distance_threshold_km=None):
        self.clustertype = "agglomerative"
        self.df_CPC_agglomerative = self.return_df_grouped_CPC().copy()
        data = self.df_CPC_agglomerative[['LOS_LAT', 'LOS_LON']].values

        # Fit the Agglomerative Clustering model with a distance threshold
        if distance_threshold_km is not None:
            distance_threshold_degrees = distance_threshold_km / 111  # Rough conversion factor for latitude and longitude
            self.agg_clustering = AgglomerativeClustering(linkage="complete", n_clusters=None  ,distance_threshold=distance_threshold_degrees)
        elif n_clusters is not None:
            self.agg_clustering = AgglomerativeClustering(linkage="complete", n_clusters=n_clusters)

        self.df_CPC_agglomerative['CLUSTER'] = self.agg_clustering.fit_predict(data)
        self.agg_n_clusters = self.agg_clustering.n_clusters_

        self.aggl_dict = {}
        for i, row in self.df_CPC_agglomerative.iterrows():
            self.aggl_dict[row["LOS_CPC"]] = row["CLUSTER"]

    def return_n_clusters(self):
        if self.clustertype == "kmeans":
            return self.kmeans_n_clusters
        elif self.clustertype == "agglomerative":
            return self.agg_n_clusters
        else:
            return None

    def return_silhouette_score(self):
        score = silhouette_score(self.df_CPC_agglomerative[['LOS_LAT', 'LOS_LON']].values, self.df_CPC_agglomerative['CLUSTER'])
        return score

    def return_inertia(self):
        return self.kmeans.inertia_

    def return_cluster_kmeans_info(self):
        return self.df_CPC_kmeans.groupby('LOS_CPC').agg({
    'LOS_LAT': 'mean', 
    'LOS_LON': 'mean', 
    'PALLETPLAATSEN': 'first',
    'CLUSTER': 'first',
    'COORDINATES': 'count'
    }).reset_index().rename(columns={'COORDINATES': 'ORDERCOUNT'}).groupby("CLUSTER").agg({
    "LOS_LAT": "mean",
    'LOS_CPC': 'count',
    "LOS_LON": "mean",
    "PALLETPLAATSEN": "sum",
    'ORDERCOUNT': 'sum'
        }).reset_index()
        
    def return_cluster_agglomerative_info(self):
        return self.df_CPC_agglomerative.groupby("CLUSTER").agg({"LOS_LAT": "mean", 
                                                                  "LOS_LON": "mean",
                                                                  "PALLETPLAATSEN": "sum"}).reset_index()

    def plot_scenario_coordinates(self,df ,title=None, save_dest=None):
        df["COORDINATES"] = df.apply(lambda row: self.PC_obj.return_CPC_coordinates(row["LOS_CPC"]), axis=1)
        df[["LOS_LAT", "LOS_LON"]] = pd.DataFrame(df["COORDINATES"].tolist(), index=df.index)
        df_plot = df.copy()

        # Plot the base map with the outlines of the countries
        fig, ax = plt.subplots(figsize=(10, 10))
        self.countries.boundary.plot(ax=ax, linewidth=1, edgecolor='black')
        if "SIMULATED" in df_plot.columns:
            scatter = sns.scatterplot(x="LOS_LON", y="LOS_LAT", data=df_plot, ax=ax, legend=True, hue="SIMULATED")
            # Modify the legend labels
            handles, labels = scatter.get_legend_handles_labels()
            new_labels = ['Known Delivery Order' if label == '0' else 'Simulated Delivery Order' for label in labels]
            ax.legend(handles=handles, labels=new_labels, fontsize=14)

        else: # Plot the clusters
            sns.scatterplot(x="LOS_LON", y="LOS_LAT", data=df_plot, ax=ax, legend=False)

        if title is None:
            plt.title(f"Plotted coordinates", fontsize=20)
        else:
            plt.title(title, fontsize=20)
        plt.xlabel("Longitude", fontsize=16)
        plt.ylabel("Latitude", fontsize=16)
        # plt.xlim(2.5, 7.3) 
        # plt.ylim(49.4, 53.7) 

        ax.set_xticks(np.arange(3, 8, 1))
        ax.set_yticks(np.arange(50, 54, 1))
        ax.tick_params(axis='both', which='major', labelsize=14)

        if save_dest is not None:
            plt.savefig(save_dest)

        plt.show()


    def plot_clusters(self, clustertype, title=None, save_dest=None):
        if clustertype == 'kmeans':
            df_plot = self.df_CPC_kmeans.groupby('LOS_CPC').agg({'PALLETPLAATSEN': 'first', 
                                                                   'LOS_LAT': 'first', 
                                                                   'LOS_LON': 'first',
                                                                   'COORDINATES': 'first',
                                                                   'CLUSTER': 'first'}).reset_index()
            cluster_info = self.return_cluster_kmeans_info()
            n_clusters = self.kmeans_n_clusters
        
        elif clustertype == 'agglomerative':
            df_plot = self.df_CPC_agglomerative.groupby('LOS_CPC').agg({'PALLETPLAATSEN': 'first', 
                                                                   'LOS_LAT': 'first', 
                                                                   'LOS_LON': 'first',
                                                                   'COORDINATES': 'first',
                                                                   'CLUSTER': 'first'}).reset_index()
            cluster_info = self.return_cluster_agglomerative_info()
            n_clusters = self.agg_n_clusters

        # Plot the base map with the outlines of the countries
        fig, ax = plt.subplots(figsize=(10, 10))
        self.countries.boundary.plot(ax=ax, linewidth=1, edgecolor='black')

        # Plot the clusters
        sns.scatterplot(x="LOS_LON", y="LOS_LAT", hue="CLUSTER", data=df_plot, palette="tab20", ax=ax, legend=False)

        for i, center in cluster_info.iterrows():
            ax.plot(center['LOS_LON'], center['LOS_LAT'], 'r+', markersize=10)

        if title is None:
            plt.title(f"{clustertype} clustering (n_clusters = {str(n_clusters)})")
        else:
            plt.title(title)
        plt.xlabel("Longitude", fontsize=16)
        plt.ylabel("Latitude", fontsize=16)

        ax.set_xticks(np.arange(3, 8, 1))
        ax.set_yticks(np.arange(50, 54, 1))

        if save_dest is not None:
            plt.savefig(save_dest)

        plt.show()

        return

    def calculate_closest_cluster_kmeans(self, coordinates):
        return self.kmeans.predict([coordinates])[0]

    def calculate_closest_cluster_agglomerative(self, coordinates):
        # Convert coordinates to a numpy array
        coordinates = np.array(coordinates).reshape(1, -1)

        # Calculate distances to each point in the data
        data = self.df_CPC_agglomerative[['LOS_LAT', 'LOS_LON']].values
        dist = distance.cdist(data, coordinates)

        # Find the index of the closest data point
        closest_idx = np.argmin(dist)

        # Return the cluster label of the closest point
        return self.df_CPC_agglomerative.iloc[closest_idx]['CLUSTER']
    
    def return_aggl_cluster_from_stored_or_calc(self, CPC): #this is a slow function
        if CPC in self.aggl_dict:
            return self.aggl_dict[CPC]
        else:
            return self.calculate_closest_cluster_agglomerative(self.PC_obj.return_CPC_coordinates(CPC))
    
    def return_kmeans_cluster_from_stored_or_calc(self, CPC): #this is a slow function
        if CPC in self.kmeans_dict:
            return self.kmeans_dict[CPC]
        else:
            return self.calculate_closest_cluster_kmeans(self.PC_obj.return_CPC_coordinates(CPC))

    def return_cluster(self, CPC):
        if self.clustertype == "kmeans":
            return int(self.return_kmeans_cluster_from_stored_or_calc(CPC))
        elif self.clustertype == "agglomerative":
            return int(self.return_aggl_cluster_from_stored_or_calc(CPC))

    def condense_orders(self, opdrachtgever=None, df=None):
        df_to_predict = self.df_input[self.df_input["LOS_CPC"] != 0]
        df_to_predict = df_to_predict[df_to_predict["LOS_CPC"] != "0"]
        if opdrachtgever is not None:
            df_to_predict = df_to_predict[df_to_predict["OPDRACHTGEVERNAAM"] == opdrachtgever]
        if df is not None:
            df_to_predict = df

        # Filter the DataFrame for AFHCODE 'a' and 'd'
        df_a = df_to_predict[df_to_predict['AFHCODE'] == 'a']
        df_d = df_to_predict[df_to_predict['AFHCODE'].isin(['d', 'f'])]

        df_a_aggregated = df_a.groupby('MATCHING_KEY').agg({
            'CREATIONDATETIME': 'first',
            'AFHCODE': 'count',
            'OPDRACHTGEVERNAAM': 'first',
            'OPDRACHTGEVERID': 'first',
            'PALLETPLAATSEN': 'sum',
            'LAADPC': 'first',
            'LAAD_DATETIME_VAN': 'first'
        }).reset_index()

        # Get the regions for each order

        #df_d['LOSPC_REGION'] = df_d["LOS_CPC"].apply(lambda LOS_CPC: 'REGION_' + str(self.return_cluster(LOS_CPC)))
        df_d = df_d.copy()
        df_d.loc[:, 'LOSPC_REGION'] = df_d["LOS_CPC"].apply(lambda LOS_CPC: 'REGION_' + str(self.return_cluster(LOS_CPC)))

        df_d_aggregated = df_d.groupby(['MATCHING_KEY', 'LOSPC_REGION'])['PALLETPLAATSEN'] \
                              .sum() \
                              .unstack(fill_value=0) \
                              .reset_index()


        final_df = pd.merge(df_a_aggregated, df_d_aggregated, on='MATCHING_KEY', how='outer')
        final_df = final_df.rename(columns={'AFHCODE': 'AANTALORDERS'})

        final_df['dayofweekcreation'] = pd.to_datetime(final_df['CREATIONDATETIME']).dt.dayofweek
        final_df['dayofweekload'] = pd.to_datetime(final_df['LAAD_DATETIME_VAN']).dt.dayofweek


        # Get the column names starting with "REGION_"
        region_columns = [col for col in final_df.columns if col.startswith("REGION_")]

        # Create the new column "PALLETPLAATSEN_ACTUAL" by summing the region columns
        final_df["PALLETPLAATSEN_ACTUAL"] = final_df[region_columns].sum(axis=1)

        # Filter the DataFrame to get rows where any of the region columns is NaN
        filtered_df = final_df[~final_df[region_columns].isna().any(axis=1)]
        filtered_df["weeknr"] = filtered_df["CREATIONDATETIME"].dt.strftime("%V")
        return filtered_df
    
    def return_random_CPC_from_clusternr(self, cluster_nr):
        if self.clustertype == "kmeans":
            df = self.df_CPC_kmeans
        elif self.clustertype == "agglomerative":
            df = self.df_CPC_agglomerative
        return df[df["CLUSTER"] == cluster_nr].sample(1)["LOS_CPC"].values[0]
    
    def return_CPC_allocation(self):
        if self.clustertype == "kmeans":
            return self.kmeans_dict
        elif self.clustertype == "agglomerative":
            return self.aggl_dict
        
    def compute_distribution(self):
        # Group by 'LOS_DATETIME_VAN' and 'LOS_CPC', and sum 'PALLETPLAATSEN'
        grouped = self.df_filtered.groupby(['LOS_DATETIME_VAN', 'LOS_CPC'])['PALLETPLAATSEN'].sum().reset_index()
        
        # Create the distribution
        self.distribution = grouped


    def sample_order_size(self):
        if self.distribution is None:
            raise ValueError("Distribution has not been computed. Please call compute_distribution() first.")
        
        # Extract the 'PALLETPLAATSEN' column to use as weights
        sizes = self.distribution['PALLETPLAATSEN']
        
        # Sample from the distribution
        sample = np.random.choice(sizes, size=1)[0]
        return sample
    
    def sample_order_size_of_CPC(self, CPC):
        if self.distribution is None:
            raise ValueError("Distribution has not been computed. Please call compute_distribution() first.")
        
        # Filter the distribution for the given CPC
        filtered = self.distribution[self.distribution['LOS_CPC'] == CPC]
        
        # Extract the 'PALLETPLAATSEN' column to use as weights
        sizes = filtered['PALLETPLAATSEN']
        
        # Sample from the distribution
        sample = np.random.choice(sizes, size=1)[0]
        return sample
    
    def distribution_per_cluster(self):
       self.cluster_means =  self.df_CPC_kmeans.groupby('CLUSTER').agg({'PALLETPLAATSEN': 'mean'}).reset_index()
       total = self.cluster_means['PALLETPLAATSEN'].sum()
       self.cluster_distribution = self.cluster_means['PALLETPLAATSEN'] / total
