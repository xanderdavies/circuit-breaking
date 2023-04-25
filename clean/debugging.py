# %%
import pickle
import torch
import einops

with open("saved_states.pkl", "rb") as f:
    saved_states = pickle.load(f)

with open("saved_states_new.pkl", "rb") as f:
    saved_states_new = pickle.load(f)

print(saved_states.shape)

print(saved_states_new.shape)

saved_states = einops.rearrange(saved_states, "layer batch sequence d_model -> layer batch sequence 1 d_model")

print(torch.allclose(saved_states[:,:,0], saved_states_new[0]))
torch.allclose(saved_states[0], saved_states_new[0])
# %%
diff = saved_states_new-saved_states
torch.max(torch.max(diff[0]),-1*torch.min(diff[0]))
# %%
