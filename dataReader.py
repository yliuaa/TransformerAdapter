import json
import pandas as pd

class DataReader:
    def __init__(self, json_name, shuffle=False):
        with open(json_name, 'r') as json_file:
            raw_json = list(json_file)
        self.raw = raw_json
        self.raw_objects = []
        for item in self.raw:
            self.raw_objects.append(json.loads(item))
        self.df = pd.DataFrame(self.raw_objects)


    def get_stats(self):   
        return self.df.head()

    def get_data(self):
        # import IPython; IPython.embed(); exit(1)
        print("hello")
        return self.df['text'], self.df['intent']


if __name__ == '__main__':
    acl = DataReader(json_name="./acl-arc/test.jsonl")
    acl.get_data()