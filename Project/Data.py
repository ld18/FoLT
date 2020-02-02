
import csv
import nltk
import os

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

#Function to extract all datapoints from the file and saving them inside a Datapoint object
def readDatapointsFromFile(path):
    count = -1
    header = ""
    datapoints = []
    csv_file = open(path, mode="r", encoding="utf_8")
    csv_reader = csv.reader(csv_file, delimiter="\t")
    for row in csv_reader:
        # Check data read is train or test data (test data has fewer columns)
        if len(row) == 4:
            mode = 'train_data'
        else:
            mode = 'test_data'

        if count == -1:
            header = ", ".join(row)
            count += 1
        else:
            if mode == 'train_data':
                datapoints.append(Datapoint(int(row[0]), row[1], int(row[2]), int(row[3])))
                count += 1

            else:
                datapoints.append(Datapoint(int(row[0]), row[1], None, None))
                count += 1
    return datapoints, header, count

#small function to split a daatset into two by dividing by three
def splitDataSet(datapoints):
    sliceAt = int(len(datapoints) /3)
    developmentSet = datapoints[:sliceAt]
    trainingSet = datapoints[sliceAt:]
    return trainingSet, developmentSet

# Function to read word pairs from supplementary of Lu et al, 2018 into a list of tuples
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
        if not pair[0] in exchange_dict.keys():
            exchange_dict[pair[0]] = pair[1]

        if not pair[1] in exchange_dict.keys():
            exchange_dict[pair[1]] = pair[0]

    return exchange_dict

# Function to output results file for codalab submission
def outputResults(datapoints, filepath):
    with open(filepath, 'w') as file:
        file.write('\n'.join([
            f'{datapoint.id}_{datapoint.toxicity_predicted}'
            for datapoint in datapoints
        ]))

# Function to read the names data, returns two frequency distributions
# of male and female names with counts
def read_names():
    dir_path = 'src/names/'
    male_names_fd = nltk.probability.FreqDist()
    female_names_fd = nltk.probability.FreqDist()

    for filename in os.listdir(dir_path):
        if not '.pdf' in filename:
            with open(dir_path + filename, mode='r', encoding="utf_8") as file:
                for row in csv.reader(file, delimiter=","):
                    if row[1] == 'M':
                        male_names_fd[row[0].lower()] += int(row[2])

                    else:
                        female_names_fd[row[0].lower()] += int(row[2])

    return male_names_fd, female_names_fd

# Function to make a list of name pairs sorted by frequency
def makePairsFromFDs(fd1, fd2):
    list_fd1_sorted = sorted(list(fd1.items()), key = lambda x : x[1], reverse=True)
    list_fd2_sorted = sorted(list(fd2.items()), key = lambda x : x[1], reverse=True)

    return [
        (pair1[0], pair2[0])
        for pair1, pair2 in zip(list_fd1_sorted, list_fd2_sorted)
    ]