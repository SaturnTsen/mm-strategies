# %%
import pandas as pd
from pathlib import Path
from data_utils import (
    discover_instruments,
    load_depths,
    load_deltas,
    load_quotes,
    load_trades,
)

# %%
instruments = discover_instruments(Path("../catalog/").resolve())

# %%
deltas = load_deltas(Path("../catalog/").resolve(), instruments[0])
depths = load_depths(Path("../catalog/").resolve(), instruments[0])
quotes = load_quotes(Path("../catalog/").resolve(), instruments[0])
trades = load_trades(Path("../catalog/").resolve(), instruments[0])
# %%
