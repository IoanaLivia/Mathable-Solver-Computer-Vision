import glob
import os
import shutil
import argparse
import numpy as np
import regex as re
import cv2 as cv
from argparse import Namespace

'''
The SKIP_ARGS constant can be set to False in order to directly change
the constants for the paths instead of providing paths as args in the terminal.
'''
SKIP_ARGS = True
DIR_PATH = 'C:/Users/ioana/OneDrive/Desktop/CV_Submissions/407_Popescu_Ioana_Livia'
TOKEN_TEMPLATES_DIR_PATH = 'C:/Users/ioana/OneDrive/Desktop/CV_Submissions/407_Popescu_Ioana_Livia/tokens_numerical_templates'
TEST_DIR_PATH = 'C:/Users/ioana/OneDrive/Desktop/CV_Submissions/407_Popescu_Ioana_Livia/test_FINAL'
VERBOSE_DEFAULT = True


BOARD_WIDTH = 14
BOARD_HEIGHT = 14
BOARD_WIDTH_PX =  1540
BOARD_HEIGHT_PX =  1540
BOARD_BORDER_PX = 200


class Tile:
    '''
    The class Tile corresponds to a tile object from the game. A tile has:
    - bonus of either 1,2,3 which stands for the multiplicator value
    - token (None if empty, numeric value if token has been placed on it or if it is one of the 4 default numbered tiles in the center)
    - constraint (1 for addition (+), 2 for substraction (-), 3 for multiplication (x), 4 for division (/))
    '''
    def __init__(self, i = 0, j = 0, token = None, bonus = 1, constraint = None):
        self.i = i
        self.j = j
        self.token = token
        self.bonus = bonus
        self.constraint = constraint

    def display_value(self):
        if self.token is None:
            print('.', end = ' ')
        else:
            print(self.token, end = ' ')

    def display_bonus(self):
        if self.bonus == 1:
            print('.', end = ' ')
        else:
            print(self.bonus, end = ' ')

class Scorer:
    '''
    The class Scorer embodies the logic of the automatic scorer by
    updating the stored configuration of the board (stored as a 
    dictionary).
    '''
    def __init__(self, dir_path = DIR_PATH, token_templates_dir_path = TOKEN_TEMPLATES_DIR_PATH, test_dir_path = TEST_DIR_PATH, verbose = VERBOSE_DEFAULT, board_width = BOARD_WIDTH, board_height = BOARD_HEIGHT):
        self.dir_path = dir_path
        self.token_templates_dir_path = token_templates_dir_path
        self.test_dir_path = test_dir_path
        self.submissions_dir_path = ""
        self.submission_dir_path = ""
        self.verbose = verbose

        self.board_width, self.board_height = board_width, board_height
        self.column_map = {}
        self.init_column_mapping()
        self.board = {}
        self.init_board()

    def init_column_mapping(self):
        """
        This method initializez the mapping between the column
        index and the corresponding letter.
        """
        for i in range(ord('A'), ord('A') + self.board_width):
           self.column_map[i - ord('A')] = str(chr(i)).upper()

    def init_board(self):
        '''
        This method initializez the board dictionary which
        aims to reflect the configuration of the board at
        each step.
        '''

        # defaults correspond to the numbered tiles in the center (1,2,3,4)
        defaults = {(6,6): 1, (6,7) : 2, (7,6) : 3, (7,7) : 4}

        # constrainted corespond to the tiles which have a constraint (1 : addition '+', 2 : extraction '-', 3: multiplication 'x', 4: divison '/')
        constrainted = {(1,4) : 4, (1,9) : 4, (2,5) : 2, (2,8) : 2, (3,6) : 1, (3,7) : 3, (4,1) : 4, (4,6) : 3, (4,7) : 1, (4,12) : 4, (5,2): 2, (5,11) : 2, (6,3) : 3, 
                        (6,4): 1, (6,9) : 3, (6,10) : 1, (7,3) : 1, (7,4) : 3, (7,9) : 1, (7,10) : 3, (8,2) : 2, (8,11) : 2, (9,1) : 4, (9,12) : 4, (9,6) : 1, (9,7) : 3, 
                        (10,6) : 3, (10,7) : 1, (11,5) : 2, (11,8) : 2, (12,4) : 4, (12,9) : 4}
        
        # list of tiles per corresponding bonus
        tiles_2x_bonus = [(1,1), (2,2), (3,3), (4,4), (12, 1), (11, 2), (10, 3), (9,4), (1,12), (2,11), (3,10), (4,9), (9,9), (10,10), (11, 11), (12,12)]
        tiles_3x_bonus = [(0,0), (0,6), (0,7), (0,13), (6,0), (7,0), (6,13), (7, 13), (13, 0), (13,6), (13,7), (13, 13)]
        
        for i in range(self.board_width):
            for j in range(self.board_height):
                if (i,j) in tiles_2x_bonus:
                    bonus = 2
                elif (i,j) in tiles_3x_bonus:
                    bonus = 3
                else:
                    bonus = 1

                if (i,j) in defaults.keys():
                    self.board[(i,j)] = Tile(i, j, defaults[(i,j)], bonus)
                elif (i,j) in constrainted.keys():
                    self.board[(i,j)] = Tile(i, j, None, bonus, constrainted[(i,j)])
                else:
                    self.board[(i,j)] = Tile(i, j, None, bonus)

    def display_board_tokens(self):
        for i in range(self.board_width):
            for j in range(self.board_height):
                self.board[(i,j)].display_value()
            print()

    def display_board_constraints(self):
        for i in range(self.board_width):
            for j in range(self.board_height):
                if self.board[(i,j)].constraint is None:
                    print('.', end = ' ')
                else:
                    print(self.board[(i,j)].constraint, end = ' ')
            print()

    def display_board_bonus(self):
        for i in range(self.board_width):
            for j in range(self.board_height):
                self.board[(i,j)].display_bonus()
            print()

    def get_corners(self, mask, image):
        """
        This method returns 4 points (A,B,C,D) corresponding
        to the corners of the board obtained using the binary
        mask.
        """
        A, B, C, D = [], [], [], []

        indices = np.argwhere(mask)
        x_plus_y = indices[:, 0] + indices[:, 1]
        x_minus_y = indices[:, 0] - indices[:, 1]

        max_x_plus_y_idx = np.argmax(x_plus_y)
        min_x_plus_y_idx = np.argmin(x_plus_y)
        max_x_minus_y_idx = np.argmax(x_minus_y)
        min_x_minus_y_idx = np.argmin(x_minus_y)

        A = tuple(indices[max_x_plus_y_idx])
        B = tuple(indices[min_x_plus_y_idx])
        C = tuple(indices[min_x_minus_y_idx])
        D = tuple(indices[max_x_minus_y_idx])

        return [B[1], B[0]], [D[1], D[0]], [A[1], A[0]], [C[1], C[0]]
    
    def get_full_board(self, image_board_path):
        """
        This method extracts the board (with removed blue border) from the background in a centered format.
        """
        img = cv.imread(image_board_path) 

        l = (80, 100, 0)
        h = (120, 255, 255)

        img_hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
        mask_hsv = cv.inRange(img_hsv, l, h)
        mask_hsv = cv.erode(mask_hsv, (10, 10))

        pt_A, pt_B, pt_C, pt_D = self.get_corners(mask_hsv, img)

        input_pts = np.float32([pt_A, pt_B, pt_C, pt_D])
        output_pts = np.float32([[0, 0],
                                [0, BOARD_HEIGHT_PX - 1],
                                [BOARD_WIDTH_PX - 1, BOARD_HEIGHT_PX - 1],
                                [BOARD_WIDTH_PX - 1, 0]])
        
        M = cv.getPerspectiveTransform(input_pts,output_pts)
        new_image = cv.warpPerspective(img, M, (BOARD_HEIGHT_PX, BOARD_WIDTH_PX), flags=cv.INTER_LINEAR)
        crop_img = new_image[BOARD_BORDER_PX : BOARD_WIDTH_PX + 1 - BOARD_BORDER_PX, BOARD_BORDER_PX : BOARD_HEIGHT_PX + 1 - BOARD_BORDER_PX]

        return crop_img
    
    def get_turns(self, turns_file_path):
        '''
        This method returns turns (dictionary) where items are in the following format: [player, turn_start, turn_score]
        and the number of turns (turns_number).

        player: (int) 1 / 2
        turn_start: (int)
        turn_score: (int), initialized with 0
        '''
        indx, turns = 0, {}
        with open(turns_file_path, 'r') as file:
            for line in file:
                l = line.split()
                turns[indx] = [int(l[0][-1]), int(l[1]), 0]
                indx += 1
        turns_number = indx

        return turns, turns_number

    def score(self, submission_file_name = "407_Popescu_Ioana_Livia"):
        '''
        This method embodies the logic of the games scoring. It iterates
        through the folder that contains the games and computes the
        number of games based on file names.

        For each game, it calls score_game which
        '''
        number_of_games = 0
        for filename in glob.iglob(f'{self.test_dir_path}/*.jpg'):
            number_of_games = max(number_of_games, int(re.search(rf'{re.escape(self.test_dir_path)}/(\d+)_(\d+).jpg', filename.replace("\\", "/")).group(1)))
        number_of_games += 1

        submissions_dir_path =  os.path.join(self.dir_path, "submission_files")
        self.submissions_dir_path = submissions_dir_path
        if not os.path.exists(self.submissions_dir_path):
            os.mkdir(self.submissions_dir_path)

        submission_dir_path =  os.path.join(self.submissions_dir_path, submission_file_name) 
        self.submission_dir_path = submission_dir_path
        if not os.path.exists(self.submission_dir_path):
            os.mkdir(self.submission_dir_path)

        for i in range(1, number_of_games):
            self.score_game(int(i))

    def get_board_config_from_image(self, img):
        defaults = {(6,6): 1, (6,7) : 2, (7,6) : 3, (7,7) : 4}
        board_config = {(6,6): 1, (6,7) : 2, (7,6) : 3, (7,7) : 4}
        sim_threshold = 0.7

        l = int(img.shape[0] / self.board_width)
        pad_x, pad_y = 15, 10

        pad_img = cv.copyMakeBorder(img, pad_y, pad_y, pad_x, pad_x, cv.BORDER_CONSTANT, value=[225,225,225])

        for i in range(0, self.board_height * l, l):
            a = int(i) + pad_y
            for j in range(0, self.board_width * l, l):
                z = int(j) + pad_x
                s_max, detected_number = 0, -1

                indx_j, indx_i = int(i / l), int(j / l) 

                if (indx_j, indx_i) in defaults:
                    continue

                tile = pad_img[max(0, a - pad_y) : a + l + pad_y, max(0, z - pad_x) : z + l + pad_x]
                
                for template_path in glob.iglob(f'{self.token_templates_dir_path}/*.jpg'):
                    template_path = template_path.replace("\\", "/")
                    nr = int(re.search(rf'{re.escape(self.token_templates_dir_path)}/(\d{{1,2}}).jpg', template_path).group(1))
                    template = cv.imread(template_path) 

                    s = cv.minMaxLoc(cv.matchTemplate(tile, template, cv.TM_CCOEFF_NORMED))[1]
                    if s > s_max :
                        s_max = s
                        detected_number = nr

                if s_max >= sim_threshold:
                    board_config[(indx_j, indx_i)] = detected_number
                else:
                    board_config[(indx_j, indx_i)] = None
    
        return board_config

    def score_game(self, indx):
        '''
        This method containts the logic of the scoring for a single game.
        '''
        turns_file_path= f'{self.test_dir_path}/{indx}_turns.txt'
        turns, turns_number = self.get_turns(turns_file_path)

        pad_x, pad_y = 15, 10
        score_per_move = 0
        curr_turn = 0

        self.init_board()
        prev_board_config = None

        for filename in glob.iglob(f'{self.test_dir_path}/{indx}_??.jpg'):
            filename = filename.replace("\\", "/")
            current_move = re.search(fr'{self.test_dir_path}/{indx}_(..).jpg', filename).group(1)

            indx_curr_move = int(current_move)

            if curr_turn + 1 < turns_number:
                if indx_curr_move == turns[curr_turn + 1][1]:
                    curr_turn += 1

            if self.verbose:
                print(f"-------------------------------------------------------------------------------------------")
                print(f"---------------------- Move: {indx_curr_move}, Turn: {curr_turn} ---------------------------------------------------")
                print(f"-------------------------------------------------------------------------------------------")

            img = self.get_full_board(filename)

            l = int(img.shape[0] / self.board_width)
            s_max, pair_s_max, detected_number = 0, [-2,-2], -1

            pad_img = cv.copyMakeBorder(img, pad_y, pad_y, pad_x, pad_x, cv.BORDER_CONSTANT, value=[225,225,225])

            for i in range(0, self.board_height * l, l):
                a = int(i) + pad_y
                for j in range(0, self.board_width * l, l):
                    z = int(j) + pad_x

                    indx_j, indx_i = int(i / l), int(j / l) 

                    if self.board[(indx_j, indx_i)].token is not None:
                        continue

                    tile = pad_img[max(0, a - pad_y) : a + l + pad_y, max(0, z - pad_x) : z + l + pad_x]

                    for template_path in glob.iglob(f'{self.token_templates_dir_path}/*.jpg'):
                        template_path = template_path.replace("\\", "/")
                        nr = int(re.search(rf'{re.escape(self.token_templates_dir_path)}/(\d{{1,2}}).jpg', template_path).group(1))
                        template = cv.imread(template_path) 

                        s = cv.minMaxLoc(cv.matchTemplate(tile, template, cv.TM_CCOEFF_NORMED))[1]
                        if s > s_max :
                                s_max = s
                                pair_s_max = [indx_i, indx_j]
                                detected_number = nr
                    
            self.board[(pair_s_max[1], pair_s_max[0])].token = detected_number

            curr_board_config = self.get_board_config_from_image(img)
            discrepancies = []

            if prev_board_config is not None: 
                for i in range(self.board_height):
                    for j in range(self.board_width):
                        if self.board[(i,j)].token != curr_board_config[(i,j)] and (i,j) != (pair_s_max[1], pair_s_max[0]):
                            discrepancies.append([[i,j], self.board[(i,j)].token, curr_board_config[(i,j)]])
                
                if len(discrepancies) > 0:
                    discrepancies_prev_board, discrepancies_prev_board_curr = 0, 0
                    for i in range(self.board_height):
                        for j in range(self.board_width):
                            if self.board[(i,j)].token != prev_board_config[(i,j)] and (i,j) != (pair_s_max[1], pair_s_max[0]):
                                discrepancies_prev_board += 1

                            if curr_board_config[(i,j)] != prev_board_config[(i,j)] and (i,j) != (pair_s_max[1], pair_s_max[0]):
                                discrepancies_prev_board_curr += 1

                    if discrepancies_prev_board_curr < discrepancies_prev_board:
                         for i in range(self.board_height):
                            for j in range(self.board_width):
                                self.board[(i,j)].token = curr_board_config[(i,j)]

            else:
                prev_board_config = {}
            
            for i in range(self.board_width):
                for j in range(self.board_height):
                    prev_board_config[(i,j)] = self.board[(i,j)].token

            print(f"Discrepancies: {discrepancies}")

            if self.verbose:
                print(f"-------------------------------------------------------------------------------------------")
                print(f"----------------------  Processed...{indx}_{current_move} -- {len(discrepancies)} -------------------------------")
                print(f"-------------------------------------------------------------------------------------------")
    
            with open(f"{self.submission_dir_path}/{indx}_{current_move}.txt", "w") as file:
                file.write(f'{pair_s_max[1] + 1}{self.column_map[pair_s_max[0]]} {detected_number}')

            score_per_move = self.get_move_score(pair_s_max)

            turns[curr_turn][2] += score_per_move

            if self.verbose:
                print(f"-------------------------------------------------------------------------------------------")
                print(f"------ {pair_s_max[1], pair_s_max[0]} Bonus: {self.board[(pair_s_max[1], pair_s_max[0])].bonus} Number: {detected_number} Curr turn: {curr_turn} Move Score: {score_per_move}   -------------------------------")
                print(f"-------------------------------------------------------------------------------------------")

        scores_file_path = f'{self.submission_dir_path}/{indx}_scores.txt'

        shutil.copy(turns_file_path, scores_file_path)
        turn_indx = 0
        with open(turns_file_path, 'r') as file:
            with open(scores_file_path, 'w') as scores_file:
                lines = file.readlines()
                for idx, line in enumerate(lines):
                    line_to_write = line.rstrip('\n') + ' ' + str(turns[turn_indx][2])
                    if idx+1 < len(lines):
                        line_to_write += '\n'
                    scores_file.write(line_to_write)
                    turn_indx += 1

    def get_move_score(self, pair):
        '''
        This method computes the score for a specific move.
        '''

        constraints = []
        if self.board[(pair[1], pair[0])].constraint is not None:
            constraints.append(self.board[(pair[1], pair[0])].constraint)
        else: 
            constraints = [1,2,3,4]

        move_score = 0
        detected_number = self.board[(pair[1], pair[0])].token

        dir_x = [[0, 0], [0 ,0], [-2, -1], [2, 1]]
        dir_y = [[-2, -1], [2, 1], [0, 0], [0, 0]]

        for i in range(len(dir_x)):
            x1, x2 = pair[0] + dir_x[i][0], pair[0] + dir_x[i][1]
            y1, y2 = pair[1] + dir_y[i][0], pair[1] + dir_y[i][1]

            add = False
            if x1 >= 0 and y1 >= 0 and x1 < self.board_height and y1 < self.board_width and x2 >= 0 and y2 >= 0 and x2 < self.board_height and y2 < self.board_width:
                left, right = self.board[(y1, x1)].token, self.board[(y2, x2)].token
                if left is not None and right is not None:
                    add = ((1 in constraints and left + right == detected_number) or 
                           (2 in constraints and abs(left - right) == detected_number) or
                           (3 in constraints and  left * right == detected_number) or 
                           (4 in constraints and (min(left, right) != 0 and max(left, right) / min(left, right) == detected_number)))

            if add:
                move_score += self.board[(pair[1], pair[0])].bonus *  detected_number

        return move_score
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.usage = "Please run the following command to display the help section: python -u <path to solution.py> -h"

    parser.add_argument('--dir_path', type=str, help='Enter the path of the directory that contains the rest of the directories.', required=False)
    parser.add_argument('--token_templates_dir_path', type=str, help='Enter the path to the directory that contains the token templates.', required=False)
    parser.add_argument('--test_dir_path', type=str, help='Enter the path to the directory that contains the test images and files.', required=False)
    parser.add_argument('--verbose_option', type=int, help='Enter 1 for activated verbose, 0 otherwise. The default is activated.', required=False)

    args : Namespace = parser.parse_args()


    if not SKIP_ARGS:
        if args.dir_path is None:
            print("----------------------------------------------------------------------------------------------------------------------------")
            print("A path to the root directory which is supposed to contain the rest of the directories has not been provided. Please re-run.")
            print("----------------------------------------------------------------------------------------------------------------------------")
            quit()
        
        if args.token_templates_dir_path is None:
            print("----------------------------------------------------------------------------------------------------------------------------")
            print("A path to the token templates directory has not been provided. Please re-run.")
            print("----------------------------------------------------------------------------------------------------------------------------")
            quit()

        if args.test_dir_path is None:
            print("----------------------------------------------------------------------------------------------------------------------------")
            print("A path to the test directory which is supposed to contain the test images has not been provided. Please re-run.")
            print("----------------------------------------------------------------------------------------------------------------------------")
            quit()

        if args.verbose_option is None:
            print("----------------------------------------------------------------------------------------------------------------------------")
            print("A preference for the verbose setting has not been provided. Default [Activated] will be used.")
            print("----------------------------------------------------------------------------------------------------------------------------")


    dir_path =  args.dir_path if args.dir_path is not None else DIR_PATH
    token_templates_dir_path =  args.token_templates_dir_path if args.token_templates_dir_path is not None else TOKEN_TEMPLATES_DIR_PATH
    test_dir_path =  args.test_dir_path if args.dir_path is not None else TEST_DIR_PATH 
    verbose_option =  args.verbose_option if args.verbose_option is not None else VERBOSE_DEFAULT

    s = Scorer(dir_path, token_templates_dir_path, test_dir_path, verbose_option)
    s.score()