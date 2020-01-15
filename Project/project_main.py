
import logging
from Project.DataManager import *

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format=('%(asctime)s : %(levelname)s : %(module)s : %(funcName)s : %(message)s'),
    datefmt='%H:%M:%S'
)
logger.info('Starting program')

path = "src/train.tsv"
allDatapoints, header, lineCount = readDatapointsFromFile(path)
logger.info(str(lineCount) + f" Datapoints found inside {path}.")

trainingSet, developmentSet = splitDataSet(allDatapoints)
logger.info("Split data as following: "+ str(len(trainingSet)) + " in trainingSet, "+ str(len(developmentSet))+ " in developmentSet.")