import pickle
import numpy as np


def normalize(allprobas):
    weights = {2: 25.157525, 4: 23.842984, 0: 22.239829, 1: 17.865725, 3: 10.893937}

    normalized = []
    for proba in allprobas:
        guess = [value * weights[index] for index, value in enumerate(proba)]
        normalized.append(guess)
    return normalized


with open("y_preds_proba_91.pkl", "rb") as file:
    loaded_data = pickle.load(file)

loaded_data = normalize(loaded_data)

with open("submit.csv", "r") as template:
    lines = template.readlines()[1:]
indices = [elt.split(",")[0] for elt in lines]

c = 0
MAP = {
    0: "Average",
    1: "High",
    2: "Low",
    3: "Very high",
    4: "Very Low",
}

with open("OUTPUT_V2.csv", "w") as csvf:
    csvf.write("row_index,piezo_groundwater_level_category\n")
    for x in loaded_data:
        index = indices[c]
        argmax = np.argmax([loaded_data[c]])
        value = MAP[argmax]
        string = f"{index},{value}\n"
        csvf.write(string)
