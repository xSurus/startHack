def square_col(col_name: str, square_i: int):
    return f"{i}_{col_name}"

import re
def grid_col(sqcol_name: str):
    if not re.match(r"\d+_[a-zA-Z]+", sqcol_name):
        raise ValueError(f"passed argument ({sqcol_name}) is probably not a square column name, like e.g. 1_slope")
    return sqcol_name.split("_")[1]