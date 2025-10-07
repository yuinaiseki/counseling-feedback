import pandas as pd
import json
import math

with open ("test.json", "r") as file:
    data = json.load(file)

print(len(data))