import numpy as np


def get_board_info(area,next_board):
    """
    area: a numpy matrix representation of the board
    """
    # Columns heights
    peaks = get_peaks(next_board)
    highest_peak = np.max(peaks)

    # Aggregated height
    agg_height = np.sum(peaks)

    holes = get_holes(peaks, next_board)
    # Number of empty holes
    n_holes = np.sum(holes)
    # Number of columns with at least one hole
    n_cols_with_holes = np.count_nonzero(np.array(holes) > 0)

    # Row transitions
    row_transitions = get_row_transition(next_board, highest_peak)

    # Columns transitions
    col_transitions = get_col_transition(next_board, peaks)

    # Abs height differences between consecutive cols
    bumpiness = get_bumpiness(peaks)

    # Number of cols with zero blocks
    num_pits = np.count_nonzero(np.count_nonzero(next_board, axis=0) == 0)

    wells = get_wells(peaks)
    # Deepest well
    max_wells = np.max(wells)

    cleared = get_cleared_lines(area,next_board)

    return agg_height, n_holes, bumpiness, cleared, num_pits, max_wells, n_cols_with_holes, row_transitions, col_transitions
  
 
def get_peaks(area):
    peaks = np.array([])
    for col in range(area.shape[1]):
        if 1 in area[:, col]:
            p = area.shape[0] - np.argmax(area[:, col], axis=0)
            peaks = np.append(peaks, p)
        else:
            peaks = np.append(peaks, 0)
    return peaks
  
 
def get_row_transition(area, highest_peak):
    sum = 0
    # From highest peak to bottom
    for row in range(int(area.shape[0] - highest_peak), area.shape[0]):
        for col in range(1, area.shape[1]):
            if area[row, col] != area[row, col - 1]:
                sum += 1
    return sum


def get_col_transition(area, peaks):
    sum = 0
    for col in range(area.shape[1]):
        if peaks[col] <= 1:
            continue
        for row in range(int(area.shape[0] - peaks[col]), area.shape[0] - 1):
            if area[row, col] != area[row + 1, col]:
                sum += 1
    return sum


def get_bumpiness(peaks):
    s = 0
    for i in range(9):
        s += np.abs(peaks[i] - peaks[i + 1])
    return s


def get_holes(peaks, area):
    # Count from peaks to bottom
    holes = []
    for col in range(area.shape[1]):
        start = -peaks[col]
        # If there's no holes i.e. no blocks on that column
        if start == 0:
            holes.append(0)
        else:
            holes.append(np.count_nonzero(area[int(start):, col] == 0))
    return holes


def get_wells(peaks):
    wells = []
    for i in range(len(peaks)):
        if i == 0:
            w = peaks[1] - peaks[0]
            w = w if w > 0 else 0
            wells.append(w)
        elif i == len(peaks) - 1:
            w = peaks[-2] - peaks[-1]
            w = w if w > 0 else 0
            wells.append(w)
        else:
            w1 = peaks[i - 1] - peaks[i]
            w2 = peaks[i + 1] - peaks[i]
            w1 = w1 if w1 > 0 else 0
            w2 = w2 if w2 > 0 else 0
            w = w1 if w1 >= w2 else w2
            wells.append(w)
    return wells


def get_cleared_lines(current_board, next_board):
    num_nonzero_rows = np.count_nonzero(current_board, axis=1)
    current_rows = np.count_nonzero(num_nonzero_rows == 0)

    num_nonzero_rows = np.count_nonzero(next_board, axis=1)
    next_rows = np.count_nonzero(num_nonzero_rows == 0)

    cleared_lines = next_rows - current_rows
    if cleared_lines < 0:
        return 0
    return cleared_lines
