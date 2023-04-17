from ..UAVMap.Map import UAVMap
import copy

class State(object):
    def __init__(self, map_init: UAVMap):
        self.no_fly_zone = map_init.nfz
        self.obstacles = map_init.obstacles
        self.landing_zone = map_init.start_land_zone
    
    @property
    def shape(self):
        return self.landing_zone.shape[:2]
    
    def is_terminal(self):
        pass
    
    def copy(self):
        return copy.deepcopy(self)
        