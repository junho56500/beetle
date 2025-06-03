from enum import Enum


status = {'red':0, 'green':1, 'yellow':2}
arrow = {'none':0, 'left':1, 'right':2}

class Status(Enum):
    RED = 0
    GREEN = 1
    YELLOW = 2
    
class Arrow(Enum):
    LEFT = 0
    RIGHT = 1
    STRAIGHT = 2
    NONE = 3
    
# Add relation index for location    
# Todo : manage id of traffic sign within class
class TrafficSign:
    def __init__(self, id):
        self.id = id
        self.status = None
        self.arrow = None
        self.area_code = 0
        self.rel_index = 0  #from north clock-wise
        self.location = (0.0, 0.0)      #lat, lon
        
    def get_id(self):
        return self.id
        
        
def main():
    #for intersection:
    ts = []
    for i in range(0,4):    
        ts[i] = TrafficSign(i)
    ts[0].status = Status.RED
    ts[0].arrow = Arrow.NONE
    ts[0].rel_index = 0
        
    #make timer and change status on the rule
             