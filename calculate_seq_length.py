from functools import partial

import numpy as np
import pandas as pd
from tqdm import tqdm

from model import BertForWordBoundaryDetection

tqdm.pandas()
bert = BertForWordBoundaryDetection()

df = pd.read_csv("training_data.csv", index_col=None)
f = partial(bert.tokenizer.encode, truncation=False)
lens = df["text"].progress_apply(lambda x: len(f(x))).to_numpy()

# Calculate statistics
percentile_95 = np.percentile(lens, 95)
print(f"95th percentile: {percentile_95}")
