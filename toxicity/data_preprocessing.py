"""
Creates toxic dataset, extracting from "Raiders of the Lost Kek: 3.5 Years of Augmented 4chan Posts from the Politically Incorrect Board."
"""

# %% 

import json
from itertools import islice
from tqdm import tqdm
import torch
import re
import pickle 
import matplotlib.pyplot as plt
import random
from tqdm import tqdm 
import bisect
import seaborn as sns

sns.set()

# %% 

def filter_text(text):
    # a is a link to another post, so we remove text between <a> and </a>
    text = re.sub(r'<a[^>]*>.*?</a>', '', text)
    # remove html tags
    text = re.sub(r'<[^>]*>', '', text)
    # remove &gt; and &lt;
    text = re.sub(r'&gt;|&lt;', '', text)
    # remove &#039; and &quot;
    text = re.sub(r'&#039;|&quot;', '', text)
    # remove leading and trailing spaces
    text = text.strip()
    return text

# %%

# Open the JSON file and read the first 5000 examples
with open('/home/ubuntu/circuitbreakers/data/4chan/pol_062016-112019_labeled.ndjson', 'r') as f:
    # verbose
    print("Loading data...")
    data = [json.loads(line) for line in islice(f, 50000)]

# Extract the "perspectives" key for each post
posts = [d['posts'] for d in data]
toxicity_scores = []
for post_list in tqdm(posts, desc="Extracting toxicity scores"):
    for post in post_list:
        if 'perspectives' in post:
            perspectives = post['perspectives']
            if 'TOXICITY' in perspectives:
                toxicity_score = perspectives['TOXICITY']
                toxicity_scores.append((post['no'], toxicity_score, post['com']))

# Sort the posts based on their "TOXICITY" score
toxicity_scores.sort(key=lambda x: x[1], reverse=True)

# %%

plt.hist([x[1] for x in toxicity_scores], bins=100)
plt.title("Distribution of 4Chan Toxicity Scores")
# %%

# print low toxic sample
# get every 100th sample
for i in range(0, len(toxicity_scores), 100000):
    print(f"Post ID: {toxicity_scores[i][0]}")
    print(f"Post Text: {filter_text(toxicity_scores[i][2])}")
    print(f"Score: {toxicity_scores[i][1]}")
    print("=" * 50)

# %%

# `get_data` function constructs a sample of size `num_samples`, with toxicity scores unfiromly disrtibuted over the `toxicity_range` argument
def get_data(toxicity_scores, num_samples, toxicity_range=[0, 1], num_test_samples=0, num_buckets=None):

    # filter for text over 20 characters
    toxicity_scores = [x for x in toxicity_scores if len(filter_text(x[2])) > 20]
    
    if num_buckets is None:
        num_buckets = int((toxicity_range[1] - toxicity_range[0]) * 100)
    # we want num_samples//num_buckets samples per bucket
    samples_per_bucket = num_samples // num_buckets
    if num_test_samples > 0:
        samples_per_bucket_test = num_test_samples // num_buckets
    else:
        samples_per_bucket_test = 0
    
    bucket_ranges = toxicity_range[0] + torch.linspace(0, 1, num_buckets + 1) * (toxicity_range[1] - toxicity_range[0])

    # since toxicity_scores is sorted, we can use binary search to find the indices of the buckets
    bucket_indices = []
    for bucket in bucket_ranges:
        # find the index of the first element greater than or equal to bucket
        index = bisect.bisect_left(toxicity_scores, -bucket, key=lambda x: -x[1])
        bucket_indices.append(index)

    print(bucket_indices)
    # get the samples from each bucket
    train = []
    test = []
    for i in range(len(bucket_indices) - 1):
        end = bucket_indices[i]
        start = bucket_indices[i + 1]
        # get samples_per_bucket samples * 2 if two_sets is True
        samples = random.sample(toxicity_scores[start:end], samples_per_bucket + samples_per_bucket_test)
        # split the samples into train and test
        if num_test_samples > 0:
            train.extend(samples[:samples_per_bucket])
            test.extend(samples[samples_per_bucket:])
        else:
            train.extend(samples)

    return train, test

# %%

data, _ = get_data(toxicity_scores, 1000, toxicity_range=[.9, 1])
plt.hist([x[1] for x in data], bins=100)
plt.title("Distribution of Toxicity Scores")

# %%

train_set, test_set = get_data(toxicity_scores, 1000, toxicity_range=[.9, 1], num_test_samples=2000)

plt.hist([x[1] for x in train_set], bins=100)
plt.title("Distribution of Toxicity Scores in Training Set")
plt.show()

plt.hist([x[1] for x in test_set], bins=100)
plt.title("Distribution of Toxicity Scores in Test Set")

# %%

# filter 
train_set = [(x[0], x[1], filter_text(x[2])) for x in train_set]
test_set = [(x[0], x[1], filter_text(x[2])) for x in test_set]

# %%

eval_uniform, _ = get_data(toxicity_scores, 10000, toxicity_range=[0, 1])
eval_uniform = [(x[0], x[1], filter_text(x[2])) for x in eval_uniform]

# %%

plt.hist([x[1] for x in eval_uniform], bins=100)
plt.title("Distribution of Toxicity Scores in Evaluation Set")

# %%

# save the data to pickle files
with open('data/train.pkl', 'wb') as f:
    pickle.dump(train_set, f)

with open('data/test.pkl', 'wb') as f:
    pickle.dump(test_set, f)

with open('data/eval_uniform.pkl', 'wb') as f:
    pickle.dump(eval_uniform, f)

# %%"
import datasets
from models import tokenizer
from data import CONTEXT_LENGTH
from easy_transformer.utils import tokenize_and_concatenate

def save_owt_data(tokenizer, ctx_length=CONTEXT_LENGTH, split="train"):
    dataset = datasets.load_dataset("NeelNanda/pile-10k", split="train")
    if split == "train":
        # use 80% of the data
        dataset = dataset.select(range(int(0.8*len(dataset))))
    elif split == "test":
        # use 20% of the data
        dataset = dataset.select(range(int(0.8*len(dataset)), len(dataset)))
        print(len(dataset))
    else:
        raise ValueError("split must be either train or test")
    tokens_dataset = tokenize_and_concatenate(dataset, tokenizer, streaming=False, max_length=ctx_length, column_name="text", add_bos_token=True, num_proc=4)
    with open(f"data/owt_{split}.pkl", "wb") as f:
        pickle.dump(tokens_dataset, f)

save_owt_data(tokenizer)
save_owt_data(tokenizer, split="test")

# %%
