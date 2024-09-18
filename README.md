# Mathable-Solver-Computer-Vision

### The libraries required to run the project including the full version of each library

opencv-python 4.6.0.66  
numpy 1.20.3  
regex 2021.8.3

(python --version returns Python 3.12.3)

### Indications of how to run the solution and where to look for the output file

(Default) If the constant SKIP_ARGS is set to True, the constants related to the necessary paths
can be manually set by changing their initial values.

-- example --

SKIP_ARGS = True
DIR_PATH = 'C:/Users/ioana/Submission_1'
TOKEN_TEMPLATES_DIR_PATH = 'C:/Users/ioana/tokens_numerical_templates'
TEST_DIR_PATH = 'C:/Users/ioana/train'
VERBOSE_DEFAULT = True

-- end of example

(Alternative) If the constant SKIP_ARGS is set to False, the code can be run with the following command:

-- example --

python -u "C:\Users\ioana\solution.py" --dir_path 'C:/Users/ioana/Desktop/Mathable_Solution' --token_templates_dir_path 'C:/Users/ioana/OneDrive/Desktop/Mathable_Solution/tokens_templates' --test_dir_path 'C:/Users/ioana/OneDrive/Desktop/Mathable_Solution/test' --verbose_option 1

-- end of example --

Please respect the path format provided in the above examples as it is relevant for a smooth run.

## Description of the paths

The arguments needed to run the code are:

DIR_PATH = The path of the "root" directory that will contain test, submission_files, ...
TOKEN_TEMPLATES_DIR_PATH = The path of the directory that contains the token templates
TEST_DIR_PATH = The path of the directory that contains the test images and turns file
VERBOSE_DEFAULT = If set to True, messages will be outputed in the terminal during execution. If False, no messages will be displayed.

### Run the solution

The solution can be run in the terminal with the following command format:

-- example --

python -u "C:\Users\ioana\Desktop\CV\solution_automatic_scorer.py"

-- end of example --

# Output

In the submission_files directory that is automatically created (if not already present)
based on the provided DIR_PATH, a directory is automatically created (if not already present) 
that will contain the .txt files for the position and token values per move and the .txt files 
for turn scores.

## Evaluation

For evaluation, the submission path to be set is in the format:

predictions_path_root = f"{DIR_PATH}/submission_files/Submission_1/"
gt_path_root = TEST_DIR_PATH + "/"
