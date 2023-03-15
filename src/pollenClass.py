
import pandas as pd
import numpy as np 

class pollenGrain:
    def __init__(self):
        self.position = {}
        self.class_type = {}
        self.time_point = 0

    
    def add_position(self, xCentroid, yCentroid, timePoint):
        self.position[timePoint] = (xCentroid, yCentroid)
        self.time_point = timePoint

    def update_position(self, timePoint, coords):
        self.position[timePoint] = coords
        self.time_point = timePoint

    def get_position(self):
        return self.position
    
    def add_class(self, pollenClass, timePoint):
        self.class_type[timePoint] = pollenClass
        self.time_point = timePoint

    def get_class(self):
        return self.class_type
    
    def get_xcentroid(self):
        return (self.position[self.time_point])[0]
    
    def get_ycentroid(self):
         return (self.position[self.time_point])[1]

    def change_time(self, timePoint):
        self.time_point = timePoint
    
    # def get_class(self):
    #     return self.class_type[self.time_point]

    # def populate_position(self, xCentroid, yCentroid, timePoint):
    #     self.position[timePoint] = (xCentroid, yCentroid)
    
    # def populate_class(self, pollenClass, timePoint):
    #     self.class_type[timePoint] = pollenClass

    
    

    

    


        

