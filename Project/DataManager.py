
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
    lineCount = -1
    header = ""
    allDatapoints = []

    csv_file = open(path, mode="r", encoding="utf_8")
    csv_reader = csv.reader(csv_file, delimiter="\t")
    for row in csv_reader:
        if lineCount == -1:
            header = ", ".join(row)
            lineCount += 1
        else:
            allDatapoints.append(Datapoint(row[0], row[1], row[2], row[3]))
            lineCount += 1
    return allDatapoints, header, lineCount

def splitDataSet(datapoints):
    sliceAt = int(len(datapoints) /3)
    developmentSet = datapoints[:sliceAt]
    trainingSet = datapoints[sliceAt:]
    return trainingSet, developmentSet



