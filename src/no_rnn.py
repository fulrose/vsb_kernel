# Load training data
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from tqdm import tqdm_notebook as tqdm

v_raw_train = pq.read_pandas('../input/train.parquet').to_pandas().values
meta_train = np.loadtxt('../input/metadata_train.csv', skiprows=1, delimiter=',')
y_train = meta_train[:, 3].astype(bool)

print(v_raw_train.shape)