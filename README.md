Data generation: (generate_data.py)
We make use of the state generators for generating three kinds of input data
- state generators based on input box list (state_generator)
- Data generation using normal distribution (random_state_generator)
- Generates a random dataset by recursively dividing boxes (sliced_box_state_generator)

Application can be run by 
- python application_main.py -f ./path/tobfile/filename 
- eg : python application_main.py -f ./CompetitionInstances/bp_50_00.txt

The generated data is passed through a decision tree classifier to predict best heuristic for an input state. Based on the classification, a simple heuristic is chosenn for the input state. Then that heuristic is applied to the file to get the final packing.

Files:
- generate_data.py generates training data
- predict_heuristic.py predicts heuristic for a given set of boxes
- application_main.py has the main function and exepects the file along with file path 

