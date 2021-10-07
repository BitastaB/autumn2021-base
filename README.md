Application can be run by 
- python application_main.py -f ./path/tobfile/filename 
- eg : python application_main.py -f ./CompetitionInstances/bp_50_00.txt

The generated data is passed through a decision tree classifier to predict best heuristic for an input state. Based on the classification, a simple heuristic is chosenn for the input state. Then that heuristic is applied to the file to get the final packing.
