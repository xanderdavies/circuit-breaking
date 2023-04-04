# %% 

import json
from itertools import islice
from tqdm import tqdm
import torch
import re
from transformers import LlamaTokenizer, LlamaForCausalLM, TextDataset, DataCollatorForLanguageModeling, Trainer, TrainingArguments
from torch.utils.data import Dataset, DataLoader
import pickle 

def filter_text(text):
    # Remove HTML tags
    text = re.sub(r'<.*?>', '', text)
    # Remove non-alphanumeric characters except for spaces
    text = re.sub(r'[^\w\s]', '', text)
    # Convert to lowercase
    text = text.lower()
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def get_toxic_posts():
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
                    toxicity_scores.append((post['no'], toxicity_score))

    # Sort the posts based on their "TOXICITY" score
    toxicity_scores.sort(key=lambda x: x[1], reverse=True)

    # Extract the top N posts with the highest "TOXICITY" score
    N = 1000
    toxic_posts = []
    for i in tqdm(range(min(N, len(toxicity_scores))), desc=f"Extracting top {N} toxic posts"):
        post_id = toxicity_scores[i][0]
        post_text = next(post['com'] for post_list in posts for post in post_list if post['no'] == post_id)
        toxic_posts.append((post_id, post_text))

    # Print the top N most toxic posts
    for post in toxic_posts:
        print(f"Post ID: {post[0]}")
        print(f"Post Text: {post[1]}")
        print("=" * 50)

    filtered_dataset = []
    for post in toxic_posts:
        post_id, post_text = post
        filtered_text = filter_text(post_text)
        print(f"Post ID: {post_id}")
        print(f"Post Text: {filtered_text}")
        print("=" * 50)
        filtered_dataset.append((post_id, filtered_text))

    train_set_size_toxic = int(len(filtered_dataset) * 0.5)
    train_set_toxic, test_set_toxic = torch.utils.data.random_split(filtered_dataset, [train_set_size_toxic, len(filtered_dataset) - train_set_size_toxic])

    with open("train_toxic.pkl", "wb") as f:
        pickle.dump(train_set_toxic, f)

    with open("test_toxic.pkl", "wb") as f:
        pickle.dump(test_set_toxic, f)

# %% 

class ToxicDataset(Dataset):
    def __init__(self, posts, tokenizer):
        self.posts = posts
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.posts)

    def __getitem__(self, index):
        post_id, post_text = self.posts[index]
        filtered_text = filter_text(post_text)
        inputs = self.tokenizer(filtered_text, padding="max_length", truncation=True, max_length=512)
        return {"input_ids": inputs["input_ids"], "attention_mask": inputs["attention_mask"]}

# Define the training arguments
training_args = TrainingArguments(
    output_dir="./llama_output",
    overwrite_output_dir=True,
    num_train_epochs=3,
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    evaluation_strategy="steps",
    eval_steps=1000,
    save_steps=1000,
    logging_steps=1000,
    learning_rate=1e-5,
    warmup_steps=1000,
    save_total_limit=2,
    fp16=True,
    gradient_accumulation_steps=16,
)
# %%
