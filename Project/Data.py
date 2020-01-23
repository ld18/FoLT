
import csv

class Datapoint:
    def __init__(self, id, comment_text, toxicity, gender):
        self.id = id
        self.comment_text = comment_text
        self.toxicity = toxicity
        self.gender = gender
        self.toxicity_predicted = None
    def __str__(self):
        str = f"{self.id}" \
              f"\n\t{self.comment_text}" \
              f"\n\t{self.toxicity}({self.toxicity_predicted})  {self.gender}\n"
        return str


def readDatapointsFromFile(path):
    count = -1
    header = ""
    datapoints = []
    csv_file = open(path, mode="r", encoding="utf_8")
    csv_reader = csv.reader(csv_file, delimiter="\t")
    for row in csv_reader:
        if count == -1:
            header = ", ".join(row)
            count += 1
        else:
            datapoints.append(Datapoint(int(row[0]), row[1], int(row[2]), int(row[3])))
            count += 1
    return datapoints, header, count


def splitDataSet(datapoints):
    sliceAt = int(len(datapoints) /3)
    developmentSet = datapoints[:sliceAt]
    trainingSet = datapoints[sliceAt:]
    return trainingSet, developmentSet
