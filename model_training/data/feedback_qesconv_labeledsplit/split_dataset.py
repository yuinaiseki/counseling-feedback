import pandas as pd
import json
import math

# df = pd.read_json("hf://datasets/SALT-NLP/feedback_qesconv/train.json")


# df.to_json("train.json", lines=False)

with open ("train_og.json", "r") as file:
    data = json.load(file)

train_to_test = 0.8
num_train = math.trunc(len(data) * train_to_test)


with open ("train.json", "w") as file:
    json.dump(data[0:num_train], file)
    print(len(data[0:num_train]))

with open ("test.json", "w") as file:
    json.dump(data[num_train+1:], file)
    print(len(data[num_train:]))



print(len(data))