# %%
from models import import_ablated_model
import torch
import pickle

# %%
with open("data/gpt2_means.pkl", "rb") as f:
    means = pickle.load(f)[0][0]

model_ablate = import_ablated_model('3', means)

mask_weights = []
for p in model_ablate.parameters():
    if p.requires_grad:
        mask_weights.append(p)
 
# %%

edges = []
for m in mask_weights:
    edges.append(torch.nonzero(m == 0).cpu().numpy())
print(edges)

def convert_to_layered(node):
    if ((node - 1) // 13) + 1 == 0:
        print(node)
    return ((node - 1) // 13) + 1, 0 if node == 0 else (node - 1) % 13

converted_edges = []
output_mask = edges[0]
for edge in output_mask:
    converted_edges.append((convert_to_layered(edge[0]),(13,0)))
i = 1
while i < 13:
    for edge in edges[i*2-1]:
        converted_edges.append((convert_to_layered(edge[0]), (i,edge[1])))
    for edge in edges[i*2]:
        converted_edges.append((convert_to_layered(edge[0]), (i,12)))
    i += 1

# %%
for e in converted_edges:
    (a,b),(c,d) = e
    print(f"{a}/{b}/{c}/{d},")
# %%
d = model_ablate.state_dict()
for k in d.keys():
    print(d[k])