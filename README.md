# Circuit Breaking: Removing Model Behaviors with Targeted Ablation
Max Li, Xander Davies, Max Nadeau

The repository is split into `mnist` and `toxicity` folders, correspondingt to the two experimental settings described in the paper.

For the `toxicity` setting:
- `toxicity/data.py` extracts toxic samples from the [4chan dataset](https://arxiv.org/abs/2001.07487) and stores them in the `toxicity/data` folder. 
- `toxicity/compute_means.py` computes the mean of the GPT-2 embeddings for a [10k sample of OpenWebText](https://huggingface.co/datasets/NeelNanda/pile-10k).
- `toxicity/evaluation.py` evaluates the original, ablated, and fine-tuned model on the OWT dataset.
- `toxicity/finetune_gpt2.py` finetunes GPT-2 against toxic comments, using eq. 4 from the paper.
- `toxicity/train_mask.py` trains a binary mask on the GPT2 model to ablate edges in the graph, implementing targeted ablation per Section 3 of the paper.
- `toxicity/transformer.py` implements a modified version of the transformer architecture to enable casaul path
interventions.
- `toxicity/utils.py` provides utilities for inference with toxic and OWT data.

For the `mnist` setting:
- TODO