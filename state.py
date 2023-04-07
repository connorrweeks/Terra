



class State():

    def __init__(self):
        self.total_development = 0
        self.form = "tribal"
        #tribal
        #democracy
        #monarchy
        #kleptocracy
        #aristocracy
        #stratocracy
        #meritocracy
        #theocracy
        #plutocracy
        #republic
        self.provs = []
        self.influences = {"royalty": 0,
            "criminals":0,
            "nobles": 0,
            "generals": 0,
            "scientists": 0,
            "clergy": 0,
            "merchants": 0,
            "people": 0}

    def update_influences(self):
        influences = {}
        decay = 0.01
        for t in self.influences:
            influences[t] = self.influences[t] * (1.0 - decay)
        avg_crime = sum([p.crime for p in self.provs]) / len(self.provs)
        influences["criminals"] += avg_crime
        influences["nobles"] += max(influences["criminals"], influences["generals"], influences["merchants"])
        influences["generals"] += 0
        avg_dev = sum([p.development for p in self.provs]) / len(self.provs)
        influences["scientists"] += avg_dev / 10
        influences["clergy"] += 0
        avg_trade = sum([p.trade for p in self.provs]) / len(self.provs)
        influences["merchants"] += avg_trade / 10
        influences["people"] += 5 / len(self.provs)
        self.influences = influences


