
from terrain_maker import TerrainMaker

class HistoryMaker(TerrainMaker):
    def __init__(self):
        super.__init__()

    def tick(self):
        for state in self.states:
            self.play_turn(state)
        
        #self.update_trade()
        #self.update_disasters()
        #self.update_events()

    def play_turn(self, state):
        #Passives
        #Update wealth
        #Update crime
        #Update culture
        #Update religion
        #Update relationships

        #Actives
        #Form-specific
        #Declare-War
        #Offer-Peace
        #Make-deals
        #Improve development
        #Buy traits/techs