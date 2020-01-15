
import logging
from Project.DataManager import readDatapointsFromFile

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.WARNING,
    format=('%(asctime)s : %(levelname)s : '
            '%(module)s : %(funcName)s : %(message)s'),
    datefmt='%H:%M:%S'
)
logger.info('Starting program')

logger.info('Read all datapoints from file in')
allDatapoints, header, lineCount = readDatapointsFromFile("src/train.tsv")
logger.info(str(lineCount) + " Datapoints found")
print(header)
print(lineCount)
for datapoint in allDatapoints:
    print(datapoint)