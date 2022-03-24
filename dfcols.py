def square_col(col_name: str, square_i: int):
    return f"{square_i}_{col_name}"

import re
def grid_col(sqcol_name: str):
    if not re.match(r"\d+_[a-zA-Z]+", sqcol_name):
        raise ValueError(f"passed argument ({sqcol_name}) is probably not a square column name, like e.g. 1_slope")
    return sqcol_name.split("_")[1]

def all_square_cols(col_name: str):
    l = []
    for i in range(25):
        l.append(f"{i+1}_{col_name}")
    return l