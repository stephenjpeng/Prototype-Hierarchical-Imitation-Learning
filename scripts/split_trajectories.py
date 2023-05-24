import pickle
from skimage.transform import resize
from tqdm import tqdm


with open('data/obs_train_cropped.pkl', 'rb') as f:
	states = pickle.load(f)
with open('data/real_actions.pkl', 'rb') as f:
    real_actions = pickle.load(f)

shortened = []
short_act = []
T = 128
for ep, act in tqdm(zip(states, real_actions)):
    shortened_ep = [ep[i:i+T] for i in range(len(ep) // T)]
    act_ep = [act[i:i+T] for i in range(len(ep) // T)]

    shortened += shortened_ep
    short_act += act_ep

with open('data/obs_train_trimmed.pkl', 'wb') as f:
	pickle.dump(shortened, f)
with open('data/real_actions_trimmed.pkl', 'wb') as f:
	pickle.dump(short_act, f)
