import pandas as pd
import json
import math

# df = pd.read_json("hf://datasets/SALT-NLP/feedback_qesconv/train.json")


# df.to_json("train.json", lines=False)

with open (f".\data\\feedback_og\\train_og.json", "r") as file:
    data = json.load(file)

train_to_test = 0.998                                 # when  0.8       0.998
num_train = math.trunc(len(data) * train_to_test)


with open ("train.json", "w") as file:
    for item in data:
        # Add EOS token to each training example
        item['text'] = item['text']
    json.dump(data[0:num_train], file)
    print(len(data[0:num_train]))                           # 6543      8162

with open ("test.json", "w") as file:
    json.dump(data[num_train:len(data)], file)
    print(len(data[num_train:len(data)]))                   # 1636      17



print(len(data))                                            # 8179      8179