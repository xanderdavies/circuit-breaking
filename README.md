# Circuit Breaking: Removing Model Behaviors with Targeted Ablation
Max Li, Xander Davies, Max Nadeau

The repository is split into `mnist` and `toxicity` folders, correspondingt to the two experimental settings described in the paper.

For the `toxicity` setting, look in the `toxicity` folder:
- `toxic_data_for_eval.py` extracts toxic samples from the [4chan dataset](https://arxiv.org/abs/2001.07487) and stores them in the `data` folder. 
- `compute_means.py` computes the mean of the GPT-2 embeddings for a [10k sample of OpenWebText](https://huggingface.co/datasets/NeelNanda/pile-10k).
- `evaluation.py` evaluates the original, ablated, and fine-tuned model on the OWT dataset.
- `finetune_gpt2.py` finetunes GPT-2 against toxic comments, using eq. 4 from the paper.
- `train_mask.py` trains a binary mask on the GPT2 model to ablate edges in the graph, implementing targeted ablation per Section 3 of the paper.
- `transformer.py` implements a modified version of the transformer architecture to enable casaul path
interventions.
- `utils.py` provides utilities for inference with toxic and OWT data.

For the `mnist` setting, look in the `mnist` folder:
- You can run `main.py` to run the ablation experiment
- The `data` folder has the `MNIST` images to train on
- The `mlp_model.py` file defines the architecture we use for these experiments
- `old.py` has an old version of `main.py`
