from TetrisSIE import TetrisEnv, condensed_print, gen_scoring_function
from params import get_board_info
import numpy as np
from Visor import BoardVision
import time

def print_stats(use_visuals_in_trace_p, states_p, pieces_p, sleep_time_p):
    vision = BoardVision()
    if use_visuals_in_trace_p:

        for state, piece in zip(states_p, pieces_p):
            vision.update_board(state)
            # print("piece")
            # condensed_print(piece)
            # print('-----')
            time.sleep(sleep_time_p)
        time.sleep(2)
        vision.close()
    else:
        for state, piece in zip(states_p, pieces_p):
            print("board")
            condensed_print(state)
            print("piece")
            condensed_print(piece)
            print('-----')

best_1 = [-65.74792661, -34.75471346, -27.49264635, -34.75471346,
       -78.82680045, -65.77022113, -35.54476263,  -9.71014655]
best_2= [-66.65968282, -34.75471346, -27.49264635, -34.75471346,
       -76.13964912, -78.40617174, -78.58290884, -64.79227041]

env = TetrisEnv()
use_visuals_in_trace = True
sleep_time = 0.1
env.set_seed(39)

total_score, states, rate_rot, pieces, msg = env.run(
        gen_scoring_function, best_1, 600, True)
print("Ratings and rotations")
for rr in rate_rot:
    print(rr)
print(len(rate_rot))
print('----')
print(total_score)
print(msg)
print_stats(use_visuals_in_trace, states, pieces, sleep_time)

total_score, states, rate_rot, pieces, msg = env.run(
    gen_scoring_function, best_2, 600, True)

# after running your iterations (which should be at least 500 for each chromosome)
# you can evolve your new chromosomes from the best after you test all chromosomes here
print("Ratings and rotations")
for rr in rate_rot:
    print(rr)
print(len(rate_rot))
print('----')
print(total_score)
print(msg)
print_stats(use_visuals_in_trace, states, pieces, sleep_time)