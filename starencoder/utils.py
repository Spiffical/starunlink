import h5py
import numpy as np
import torch
from torch.utils import data


class HDF5Dataset(data.Dataset):
    def __init__(self, in_file, n_samples, truncate_n):
        super(HDF5Dataset, self).__init__()

        self.n_samples = n_samples
        self.in_file = in_file
        self.truncate_n = truncate_n

    def __getitem__(self, index):
        with h5py.File(self.in_file, 'r') as f:

            spectrum = f['spectra'][index]
            spectrum_contam = f['spectra+solar'][index]

            if self.truncate_n > 0:
                spectrum = spectrum[:-self.truncate_n]
                spectrum_contam = spectrum_contam[:-self.truncate_n]

        return spectrum, spectrum_contam

    def __len__(self):
        return self.n_samples


def get_train_valid_loader(data_path,
                           batch_size,
                           num_train,
                           valid_size=0.1,
                           shuffle=True,
                           num_workers=4,
                           pin_memory=True,
                           val_path='',
                           indices_train=None,
                           indices_valid=None,
                           truncate_n=0
                           ):
    """
    Utility function for loading and returning train and valid
    multi-process iterators. A sample
    If using CUDA, num_workers should be set to 1 and pin_memory to True.
    Params
    ------
    - data_path: path directory to the dataset.
    - batch_size: how many samples per batch to load.
    - num_train: size of the reference set.
    - valid_size: percentage split of the training set used for
      the validation set. Should be a float in the range [0, 1].
    - shuffle: whether to shuffle the reference set indices.
    - num_workers: number of subprocesses to use when loading the dataset.
    - pin_memory: whether to copy tensors into CUDA pinned memory. Set it to
      True if using GPU.
    - val_path: Path to validation file if different than data_path
    - indices_train: if you have pre-computed indices from the training file to use
    - indices_val: if you have pre-computed indices from the validation file to use
    - truncate_n: sometimes necessary to truncate the end of the spectrum when using Wave U-Net
    Returns
    -------
    - train_loader: training set iterator.
    - valid_loader: validation set iterator.
    """
    error_msg = "[!] valid_size should be in the range [0, 1]."
    assert ((valid_size >= 0) and (valid_size <= 1)), error_msg

    assert(truncate_n >= 0), 'truncate_n should be >= 0'

    frac_train = 1 - valid_size
    if indices_train is None:
        indices_reference = np.arange(num_train)
    else:
        indices_reference = indices_train[:num_train]
    if shuffle:
        np.random.shuffle(indices_reference)
    indices_train = indices_reference[:int(frac_train * len(indices_reference))]

    if val_path:
        if indices_valid is None:
            indices_val = np.arange(int(len(indices_reference) * valid_size))
        else:
            indices_val = indices_valid[:int(len(indices_reference) * valid_size)]
    else:
        val_path = data_path
        indices_val = indices_reference[int(frac_train * len(indices_reference)):]

    # Initialize data loaders
    train_dataset = HDF5Dataset(data_path, n_samples=len(indices_train), truncate_n=truncate_n)

    valid_dataset = HDF5Dataset(val_path, n_samples=len(indices_val), truncate_n=truncate_n)

    trainset = torch.utils.data.Subset(train_dataset, indices_train)
    validset = torch.utils.data.Subset(valid_dataset, indices_val)

    train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=num_workers,
                                               pin_memory=pin_memory)
    valid_loader = torch.utils.data.DataLoader(validset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, valid_loader


# Training function
def train_epoch_den(model, device, dataloader, loss_fn, optimizer, model_type='UNet'):
    # Set train mode for both the encoder and the decoder
    model.train()
    train_loss = []

    for spec_batch, spec_contam_batch in dataloader:
        # Move tensor to the proper device
        spec_batch = spec_batch.to(device).float().view(-1, 1, np.shape(spec_batch)[1])
        spec_contam_batch = spec_contam_batch.to(device).float().view(-1, 1, np.shape(spec_contam_batch)[1])

        # Encode and then decode data
        autoencoded_data = model(spec_contam_batch)

        if model_type.lower() == 'waveunet':
            autoencoded_data = autoencoded_data['spectra']

        # Trim true spectra if output of network is different than input
        diff = spec_batch.shape[2] - autoencoded_data.shape[2]
        if diff > 0:
            spec_batch = spec_batch[:, :, int(diff / 2): -int(diff / 2)]

        # Evaluate loss
        loss = loss_fn(autoencoded_data, spec_batch)
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss.append(loss.detach().cpu().numpy())
    return np.mean(train_loss)


# Testing/validation function
def test_epoch_den(model, device, dataloader, loss_fn, model_type='UNet'):
    # Set evaluation mode for model
    model.eval()
    with torch.no_grad():  # No need to track the gradients
        # Define the lists to store the outputs for each batch
        conc_out = []
        conc_label = []
        for spec_batch, spec_contam_batch in dataloader:
            # Move tensor to the proper device
            spec_batch = spec_batch.to(device).float().view(-1, 1, np.shape(spec_batch)[1])
            spec_contam_batch = spec_contam_batch.to(device).float().view(-1, 1, np.shape(spec_contam_batch)[1])

            # Encode and then decode data
            autoencoded_data = model(spec_contam_batch)

            if model_type.lower() == 'waveunet':
                autoencoded_data = autoencoded_data['spectra']

            # Trim true spectra if output of network is different than input
            diff = spec_batch.shape[2] - autoencoded_data.shape[2]
            if diff > 0:
                spec_batch = spec_batch[:, :, int(diff / 2): -int(diff / 2)]

            # Append the network output and the original to the lists
            conc_out.append(autoencoded_data.cpu())
            conc_label.append(spec_batch.cpu())
        # Create a single tensor with all the values in the lists
        conc_out = torch.cat(conc_out)
        conc_label = torch.cat(conc_label)
        # Evaluate global loss
        val_loss = loss_fn(conc_out, conc_label)
    return val_loss.data


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
