# -*- coding: utf-8 -*-

import folium
from folium.plugins import HeatMap


class Map():
    def __init__(self, PC):
        # Starting coordinates for the map
        self.style = "Cartodbdark_matter"
        self.PC = PC
        self.m = folium.Map(tiles=self.style,location=[52.0271669515, 5.18781049451], zoom_start=7)

        
    def plot_point(self, postcode, text, color):
        coordinates = self.PC.get_coordinates(postcode)
        if coordinates:
            folium.CircleMarker(location=coordinates,
            radius=5,  # Radius in pixels
            color=color,
            fill=True,
            fill_color='red',
            popup=text).add_to(self.m)
            
    def plot_hub(self, postcode, text):
        coordinates = self.PC.get_coordinates(postcode)
        # print("Plotting hub: ",postcode, coordinates)
        if coordinates:
            folium.Marker(location=coordinates,
            popup=text).add_to(self.m)
            
    def plot_route(self, i, j, color="blue", text=None):
        coordinates = (self.PC.get_coordinates(i), 
                        self.PC.get_coordinates(j))
        if False not in coordinates:
            # print("plotting: ",i, j, coordinates)
             
              # Create the line
            folium.PolyLine(coordinates, 
                              color=color, 
                              weight=2.5, 
                              opacity=1,
                              popup=text).add_to(self.m)
        else:
            print("Not plotted due to False Error on "+ i+" or "+ j)
            
    def plot_full_route(self, points, linename=None, color="blue", text=None):
        feature_group = folium.FeatureGroup(name=linename)
        coordinates = [self.PC.get_coordinates(x) for x in points]
        if False not in coordinates:
            # print("plotting: ",i, j, coordinates)
             
            # Create the line
            folium.PolyLine(coordinates, 
                              color=color, 
                              weight=2.5, 
                              opacity=1,
                              popup=text).add_to(feature_group)
            feature_group.add_to(self.m)
        else:
            print("error plotting")
    
    def plot_route_obj(self, route, color="red", include_hub=False, text=None, include_points=False):
        points = []
        if include_hub:
            points.append(route.return_hub())
        points += route.return_route()
        if include_hub:
            points.append(route.return_hub())
        coordinates = [self.PC.get_coordinates(x) for x in points]
        if False not in coordinates:
            if include_points:
                for point in points:
                    self.plot_point(point, point, "green")
            # print("plotting: ",i, j, coordinates)
             
              # Create the line
            folium.PolyLine(coordinates, 
                              color=color, 
                              weight=2.5, 
                              opacity=1,
                              popup=text).add_to(self.m)
        else:
            print("error plotting")
    
    
    
    def save(self, name='map.html'):
        self.m.save(name)
        
    def clear_map(self):
        self.m = folium.Map(tiles=self.style, location=[52.0271669515, 5.18781049451], zoom_start=7)
        self.save()
        
        
class ColorIterator:
    def __init__(self, colors):
        self.colors = colors
        self.index = 0

    def __iter__(self):
        return self

    def __next__(self):
        if not self.colors:
            raise StopIteration("No colors available")
        color = self.colors[self.index % len(self.colors)]
        self.index += 1
        return color
