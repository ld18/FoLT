
import logging
import Project.Data as Data

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format=('%(asctime)s : %(levelname)s : %(module)s : %(funcName)s : %(message)s'),
    datefmt='%H:%M:%S'
)
logger.info("Started program.")

path = "src/train.tsv"
datapoints, header, numberOfPoints = Data.readDatapointsFromFile(path)
logger.info(str(numberOfPoints) + f" Datapoints found inside {path}.")

trainingSet, developmentSet = Data.splitDataSet(datapoints)
logger.info("Split data as following: "+ str(len(trainingSet)) + " for training, "+ str(len(developmentSet))+ " for development.")