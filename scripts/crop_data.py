import pickle
from skimage.transform import resize
from tqdm import tqdm


with open('data/obs_train.pkl', 'rb') as f:
	states = pickle.load(f)

cropped = []
orig_shape = states[0][0].shape
for ep in tqdm(states):
    cropped_ep = []
    for frame in tqdm(ep):
        cropped_ep.append((255 * resize(frame[:-15, :, :], orig_shape)).astype('uint8'))

    cropped.append(cropped_ep)


with open('data/obs_train_cropped.pkl', 'wb') as f:
	pickle.dump(cropped, f)
