# %% 

import json
from itertools import islice
from tqdm import tqdm

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

# %%
import re

def filter_text(text):
    replacements = ['"', "'", '&', '<', '>']
    old_symbols = [r'&#34', r'&#039;', r'&#38', r'&lt', r'&gt']
    for old, new in zip(old_symbols, replacements):
        re.sub(old, new, text)    
    text = re.sub(r'&gt;&gt;\d+</a>', '', text)
    # Remove HTML tags
    text = re.sub(r'<.*?>', '', text)
    # Remove non-alphanumeric characters except for spaces
    text = re.sub(r'[^\w\s]', '', text)
    # Convert to lowercase
    text = text.lower()
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text

filtered_dataset = []
for post in toxic_posts:
    post_id, post_text = post
    filtered_text = filter_text(post_text)
    print(f"Post ID: {post_id}")
    print(f"Post Text: {filtered_text}")
    print("=" * 50)
    filtered_dataset.append((post_id, filtered_text))


# %%

import pickle

# save toxic posts
with open('/home/ubuntu/circuitbreakers/data/4chan/toxic_posts.pkl', 'wb') as f:
    pickle.dump(filtered_dataset, f)

# load toxic posts
with open('/home/ubuntu/circuitbreakers/data/4chan/toxic_posts.pkl', 'rb') as f:
    filtered_dataset_test = pickle.load(f)

# %% 

# load 7b llama model 
from transformers import LlamaTokenizer, LlamaForCausalLM
import torch

WEIGHT_PATH = "/home/ubuntu/weights_llama/hf_weights_llama_7"
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Using device:", DEVICE)

# don't reload model if already loaded
if "model" not in locals():
    print("Loading model...")
    # configure model
    tokenizer = LlamaTokenizer.from_pretrained(f"{WEIGHT_PATH}")
    model = LlamaForCausalLM.from_pretrained(f"{WEIGHT_PATH}")

    tokenizer.pad_token_id = tokenizer.eos_token_id
    torch.set_grad_enabled(False)

    # set to float16
    model.half()
    model.to(DEVICE)
    model.eval()

# %%

import numpy as np

NUM_DEMOS = 10

# evaluate 7b llama model on 4chan data (few-shot)
# use a randomly chosen sample as demos to do few-shot learning

def get_losses(model, tokenizer, filtered_dataset, NUM_DEMOS):
    losses = []
    for i in tqdm(range(len(filtered_dataset))):
        _, post_text = filtered_dataset[i]

        # randomly chose 5 samples as demos
        demo_ids = np.random.choice(len(filtered_dataset), NUM_DEMOS, replace=False)
        demo_texts = [filtered_dataset[j][1] for j in demo_ids]

        # tokenize and encode the text. remove end token.
        encoded_demo = tokenizer.encode("\n".join(demo_texts) + "\n",
                                        return_tensors="pt").to(DEVICE)[0, :-1]
        # remove start token
        encoded_sample = tokenizer.encode(f"{post_text}", return_tensors="pt").to(DEVICE)[0, 1:]

        input = torch.cat([encoded_demo, encoded_sample], dim=0).unsqueeze(0)

        # generate the output
        out = model(input)[0] # 0 is the logits

        # get the logits for all tokens after the last demo
        logits = out[0, encoded_demo.shape[0]-1:-1, :]

        # compute the loss on these logits
        losses.append(torch.nn.functional.cross_entropy(logits, encoded_sample).item())
        # print(f"Post ID: {i}")
        # print(f"Loss: {loss}")
        # print("=" * 50)
    return losses
  
# %%

loss_set = []
NUM_DEMOS = list(range(0, 15))

for num_demos in tqdm(NUM_DEMOS):
    print(f"Number of demos: {num_demos}")
    losses = get_losses(model, tokenizer, filtered_dataset, num_demos)
    loss_set.append(losses)
    print("=" * 50)


# %%

import matplotlib.pyplot as plt

plt.figure(figsize=(10, 10))
plt.boxplot(loss_set, labels=NUM_DEMOS)
plt.xlabel("Number of Demos")
plt.ylabel("Loss")
plt.title("Loss vs. Number of Demos")

# %%

from transformers import LlamaTokenizer, LlamaForCausalLM, TextDataset, DataCollatorForLanguageModeling, Trainer, TrainingArguments
from torch.utils.data import Dataset, DataLoader


train_set_size_toxic = int(len(filtered_dataset) * 0.5)
train_set_toxic, test_set_toxic = torch.utils.data.random_split(filtered_dataset, [train_set_size_toxic, len(filtered_dataset) - train_set_size_toxic])

# %% 

torch.set_grad_enabled(True)
tokenizer.pad_token_id = tokenizer.eos_token_id

class ToxicDataset(Dataset):
    def __init__(self, posts):
        self.posts = posts
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.posts)

    def __getitem__(self, index):
        post_id, post_text = self.posts[index]
        filtered_text = filter_text(post_text)
        inputs = self.tokenizer(filtered_text, padding="max_length", truncation=True, max_length=512)
        return {"input_ids": inputs["input_ids"], "attention_mask": inputs["attention_mask"]}

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, 
    mlm=False, 
    pad_to_multiple_of=8
)

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


# create the dataset
train_dataset = ToxicDataset(filtered_dataset)

# define the trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    data_collator=data_collator,
)

# fine-tune the Llama model on the toxic dataset
trainer.train()

# %%
