import torch
from inference import generate_no_hf
import pickle
from tqdm import tqdm
from detoxify import Detoxify
from models import model, tokenizer
from data import tokenize_and_concatenate_list

def filter_toxic_data(model, tokenizer, toxic_prompts, samples=100, file_name="train_filtered", threshold=0.5, seq_len=50, max_length=30, toxicity_batch_size=100):
    model.eval()
    with torch.no_grad():
        toxic_scorer = Detoxify('original')
        filtered_data = []
        toxic_prompts = tokenize_and_concatenate_list(toxic_prompts[start_batch:end_batch], tokenizer, seq_len)
        batches = (len(toxic_prompts) + toxicity_batch_size - 1) // toxicity_batch_size
        for i in range(batches):
            start_batch = i * toxicity_batch_size
            end_batch = min((i + 1) * toxicity_batch_size, len(toxic_prompts))


            model_completions = generate_no_hf(model, tokenizer, toxic_prompts[start_batch:end_batch], max_length=max_length, temperature=0, return_new_only=True)
            toxic_scores = toxic_scorer.predict(model_completions)['toxicity']

            for j, score in enumerate(tqdm(toxic_scores)):
                if score > threshold:
                    filtered_data.append(toxic_prompts[start_batch+j], model_completions[j], toxic_scores[j])
                    if len(filtered_data) >= samples:
                        with open(f"data/{file_name}.pkl", "wb") as f:
                            pickle.dump(filtered_data, f)
                        return
        print(f"NOTE: only produced {len(filtered_data)} toxic-completion samples")
        with open(f"data/{file_name}.pkl", "wb") as f:
            pickle.dump(filtered_data, f)

# %%
with open('/home/ubuntu/circuitbreakers/clean/data/train.pkl', 'rb') as f:
    train_set = pickle.load(f)

with open('/home/ubuntu/circuitbreakers/clean/data/test.pkl', 'rb') as f:
    test_set = pickle.load(f)

filter_toxic_data(model, tokenizer, [t[2] for t in train_set])
filter_toxic_data(model, tokenizer, [t[2] for t in test_set], samples=500, file_name="test_filtered")
# %%
