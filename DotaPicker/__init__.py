import numpy as np, matplotlib.pyplot as plt, json, requests, pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import confusion_matrix
from keras import backend as K
from keras.models import Sequential
from keras.layers import Activation
from keras.layers.core import Dense
from keras.optimizers import Adam, SGD
from keras.metrics import categorical_crossentropy

"""
We want the NN to predict what hero to pick.
We pick the Current team and the opposete team and try to predict the next hero.
"""

class Data:
    apiKey = "CDD384EF9F3A7ABA48EF50C7F1F8C1DE"
    steam_id = "52297974"

    heroes = json.loads(requests.get("http://www.dota2.com/jsfeed/heropickerdata").text)
    items =  json.loads(requests.get("http://www.dota2.com/jsfeed/itemdata").text)
    abilities = json.loads(requests.get("http://www.dota2.com/jsfeed/abilitydata").text)
    matches = []
    match = json.loads(requests.get("https://api.steampowered.com/IDOTA2Match_570/GetMatchHistory/V001/?key={}".format(apiKey)).text)["result"]["matches"]
    while match != None:
        print(match)
        match = json.loads(requests.get("https://api.steampowered.com/IDOTA2Match_570/GetMatchHistory/V001/?start_at_match_id={}&key={}".format(match["result"]["matches"][-1]["account_id"], apiKey)).text)["result"]["matches"]
        matches.append(match)

    filename = "./HeroTable.>csv"

    def get_heroes(self):
        for i in list(self.heroes.values()):
            yield i
    def get_items(self):
        for i in list(self.items.values()):
            yield i
    def get_abillities(self):
        for i in list(self.abilities.values()):
            yield i
    def get_matches(self):
        for i in list(self.matches):
            yield i

class Predict:
    data = Data()
    model = Sequential([
        Dense(16, input_shape=(1,), activation="relu"),  # input
        Dense(1024, activation="relu"),
        Dense(512, activation="relu"),
        Dense(2, activation="softmax")  # output
    ])

    def __init__(self,lr=".0001",loss="sparse_categorical_crossentropy"):
        self.lr = lr
        self.loss = loss
        self.optimizer = Adam(self.lr)
        self.model.compile(self.optimizer, loss=self.loss, metrics=["accuracy"])


class HeroPredict(Predict):
    def __init__(self):
        Predict.__init__(self)
        self.columns = ["ID","Team","Kills","Win","Hero"]
        self.out = "Hero"
        self.df = pd.DataFrame(columns=self.columns,data=[],index=[])
        self.loaded = False

    def get_match_data_web(self,id=None,store=True):
        if not id:
            for i in self.data.get_matches():
                match_id, seq, start, lobby, radiant, dire, reqPlayers = i.values()
                req1 = requests.get("https://api.steampowered.com/IDOTA2Match_570/GetMatchDetails/V001/?match_id={}&key={}".format(match_id, self.data.apiKey))
                reqJson = json.loads(req1.text)["result"]

                for p in reqJson["players"]:
                    team = 1 if p["player_slot"] > 5 else 0 #1 is radiant 0 is dire

                    if reqJson["radiant_win"]:
                        win = 1 if team == 1 else 0
                    else:
                        win = 0 if team == 1 else 1

                    aid = p["account_id"] if "account_id" in p.keys() else ""

                    s = pd.Series(name=0,index=self.columns,data=[aid,team,p["kills"],win,p["hero_id"]])
                    self.df = self.df.append(s, ignore_index= True)
        if store:
            self.write_df()


    def get_match_data_file(self):
        self.df = pd.read_csv(self.data.filename)

    def write_df(self):
        self.df.to_csv(self.data.filename)

    def train(self):
        pass


class ItemPredict(Predict):
    def __init__(self):
        Predict.__init__(self)


a = HeroPredict()
a.get_match_data_file()

print(a.df.head(10))








