
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

# Function to read word pairs from supplementary of Lu et al, 2018 into a list
# of tuples
def readWordPairData():
    filepath = './src/gendered_word_pairs_Lu_2018'

    with open(filepath, 'r') as file:
        content = file.readlines()

    word_pairs = []

    for line in content:
        split_1 = line.split(' - ')
        word_pairs.append(tuple((split_1[0], split_1[1].split()[0])))
        for i in range(1, len(split_1) - 1):
            word_pairs.append(tuple((
                split_1[i].split()[1],
                split_1[i+1].split()[0]
            )))

    return word_pairs

# Function to make a two-way dictionary from a list of tuples
def makeExchangeDict(word_pairs):
    exchange_dict = {}
    for pair in word_pairs:
        exchange_dict[pair[0]] = pair[1]
        exchange_dict[pair[1]] = pair[0]

    return exchange_dict