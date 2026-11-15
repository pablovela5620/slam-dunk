import sys, numpy as np
from pathlib import Path

src, out = map(Path, sys.argv[1:3])
ts = np.atleast_1d(np.loadtxt(src / "cam0/data.csv", delimiter=",", usecols=0, dtype=np.int64))
with out.open("w") as f:
    for t in ts:
        f.write(f"{t} 0 0 0 0 0 0 1\n")
