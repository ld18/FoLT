Training Data
----------------------------------------

The training data consists of 1800 samples. A sample in the training data is defined by the following attributes:

	- id: 		A unique ID for the corresponding comment
	- comment_text:	The raw text of the comment
	- toxicity:	Binary Label of the toxicity (0 = untoxic; 1 = toxic)
	- gender:	Binary Label of the gender (0 = female; 1 = male)

The file contains one sample per line, which is tabulator ('\t') separated, i.e. there is a '\t' character between each attribute.


========================================


Test Data
----------------------------------------

The (later published) test data has essentially the same format as the training data, but only contains the attributes 'id' and 'comment_text'.