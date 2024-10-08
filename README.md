# Mathable-Solver-Computer-Vision

The following project presents an automatic Mathable Solver that is able to compute the score 
of a match being provided the necessary images corresponding to the state of the board after
each move.

The train and test datasets alongside the used tokens will not be added to this repository
as I do not currently have the rights to make them public.

Implementation details are available [here](https://github.com/IoanaLivia/Mathable-Solver-Computer-Vision/blob/main/Mathable_Solver_Documentation.pdf).

### The libraries required to run the project including the full version of each library

opencv-python 4.6.0.66  
numpy 1.20.3  
regex 2021.8.3

(python --version returns Python 3.12.3)

### Indications of how to run the solution

(Default) If the constant SKIP_ARGS is set to True, the constants related to the necessary paths
can be manually set by changing their initial values.

-- example --

SKIP_ARGS = True
DIR_PATH = 'C:/Users/DummyPath/Submission_1'
TOKEN_TEMPLATES_DIR_PATH = 'C:/Users/DummyPath/tokens_numerical_templates'
TEST_DIR_PATH = 'C:/Users/DummyPath/train'
VERBOSE_DEFAULT = True

-- end of example

(Alternative) If the constant SKIP_ARGS is set to False, the code can be run with the following command:

-- example --

python -u "C:\Users\DummyPath\solution.py" --dir_path 'C:/Users/DummyPath/Mathable_Solution' --token_templates_dir_path 'C:/Users/DummyPath/tokens_templates' --test_dir_path 'C:/Users/DummyPath/test' --verbose_option 1

-- end of example --

Please respect the path format provided in the above examples as it is relevant for a smooth run.

### Description of the paths

The arguments needed to run the code are:

DIR_PATH = The path of the "root" directory that will contain test, submission_files, ...
TOKEN_TEMPLATES_DIR_PATH = The path of the directory that contains the token templates
TEST_DIR_PATH = The path of the directory that contains the test images and turns file
VERBOSE_DEFAULT = If set to True, messages will be outputed in the terminal during execution. If False, no messages will be displayed.

### Run the solution

The solution can be run in the terminal with the following command format:

-- example --

python -u "C:\Users\DummyPath\solution_automatic_scorer.py"

-- end of example --

### Output

In the submission_files, based on the provided DIR_PATH, 
a directory is automatically created (if not already present) that will contain the .txt files for
the position and token values per move and the .txt files for turn scores.

## Evaluation

For evaluation, the submission path to be set is in the format:

predictions_path_root = f"{DIR_PATH}/submission_files/Submission/"
gt_path_root = TEST_DIR_PATH + "/"
