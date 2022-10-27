import h5py
import numpy as np
import torch
import argparse
from torch.utils import data
import torch.nn as nn


class HDF5Dataset(data.Dataset):
    def __init__(self, in_file, n_samples, mean_flux, std_flux, mean_teff, std_teff,
                 mean_logg, std_logg, mean_feh, std_feh):
        super(HDF5Dataset, self).__init__()

        self.n_samples = n_samples
        self.in_file = in_file
        self.mean_flux = mean_flux
        self.std_flux = std_flux
        self.mean_teff = mean_teff
        self.std_teff = std_teff
        self.mean_logg = mean_logg
        self.std_logg = std_logg
        self.mean_feh = mean_feh
        self.std_feh = std_feh

    def __getitem__(self, index):
        with h5py.File(self.in_file, 'r') as f:
            spectra_keys = ['spectra', 'spectra+solar']
            ind = np.random.randint(2)
            spectrum = f[spectra_keys[ind]][index]

            # Fix label if 'spectra+solar' was chosen and 'frac_solar' < 0.01
            frac_solar = f['frac_solar'][index]
            if ind == 1:
                ind = 0 if frac_solar < 0.01 else 1

            teff = f['teff'][index]
            if self.mean_teff is not None:
                teff = (teff - self.mean_teff) / self.std_teff
            logg = f['logg'][index]
            if self.mean_logg is not None:
                logg = (logg - self.mean_logg) / self.std_logg
            feh = f['feh'][index]
            if self.mean_feh is not None:
                feh = (feh - self.mean_feh) / self.std_feh
            labels = np.array([teff, logg, feh])
            spectrum = (spectrum - self.mean_flux) / self.std_flux

        return ind, spectrum, frac_solar, labels

    def __len__(self):
        return self.n_samples


def get_train_valid_loader(data_path,
                           batch_size,
                           num_train,
                           valid_size=0.1,
                           shuffle=True,
                           num_workers=4,
                           pin_memory=True,
                           val_path=''
                           ):
    """
    Utility function for loading and returning train and valid
    multi-process iterators. A sample
    If using CUDA, num_workers should be set to 1 and pin_memory to True.
    Params
    ------
    - data_path: path directory to the dataset.
    - batch_size: how many samples per batch to load.
    - valid_size: percentage split of the training set used for
      the validation set. Should be a float in the range [0, 1].
    - shuffle: whether to shuffle the train/validation indices.
    - num_workers: number of subprocesses to use when loading the dataset.
    - pin_memory: whether to copy tensors into CUDA pinned memory. Set it to
      True if using GPU.
    Returns
    -------
    - train_loader: training set iterator.
    - valid_loader: validation set iterator.
    """
    error_msg = "[!] valid_size should be in the range [0, 1]."
    assert ((valid_size >= 0) and (valid_size <= 1)), error_msg

    frac_train = 1 - valid_size
    indices_reference = np.arange(num_train)
    if shuffle:
        np.random.shuffle(indices_reference)
    indices_train = indices_reference[:int(frac_train * len(indices_reference))]

    if val_path:
        indices_val = np.arange(int(len(indices_reference) * valid_size))
    else:
        val_path = data_path
        indices_val = indices_reference[int(frac_train * len(indices_reference)):]

    # Acquire mean and std of flux and stellar parameters in the training set
    with h5py.File(data_path, 'r') as f_:
        mean_flux, std_flux = f_['mean_flux'][...], f_['std_flux'][...]
        if 'mean_teff' in f_.keys():
            mean_teff, std_teff = f_['mean_teff'][...], f_['std_teff'][...]
        else:
            mean_teff, std_teff = None, None
        if 'mean_logg' in f_.keys():
            mean_logg, std_logg = f_['mean_logg'][...], f_['std_logg'][...]
        else:
            mean_logg, std_logg = None, None
        if 'mean_feh' in f_.keys():
            mean_feh, std_feh = f_['mean_feh'][...], f_['std_feh'][...]
        else:
            mean_feh, std_feh = None, None

    # Initialize data loaders
    train_dataset = HDF5Dataset(data_path, n_samples=len(indices_train),
                                mean_flux=mean_flux, std_flux=std_flux,
                                mean_teff=mean_teff, std_teff=std_teff,
                                mean_logg=mean_logg, std_logg=std_logg,
                                mean_feh=mean_feh, std_feh=std_feh)

    valid_dataset = HDF5Dataset(val_path, n_samples=len(indices_val),
                                mean_flux=mean_flux, std_flux=std_flux,
                                mean_teff=mean_teff, std_teff=std_teff,
                                mean_logg=mean_logg, std_logg=std_logg,
                                mean_feh=mean_feh, std_feh=std_feh)

    train_sampler = torch.utils.data.Subset(train_dataset, indices_train)
    valid_sampler = torch.utils.data.Subset(valid_dataset, indices_val)

    train_loader = torch.utils.data.DataLoader(train_sampler, batch_size=batch_size, shuffle=True,
                                               num_workers=num_workers, pin_memory=pin_memory)

    valid_loader = torch.utils.data.DataLoader(valid_sampler, batch_size=batch_size, shuffle=False,
                                               num_workers=num_workers)

    return train_loader, valid_loader


def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


# Function to execute a training epoch
def train_epoch_generator(NN, training_generator, optimizer, device, train_steps, lastlayer):
    NN.train()
    loss = 0
    if lastlayer == 'sigmoid':
        loss_fn = nn.BCEWithLogitsLoss()
    else:
        loss_fn = nn.MSELoss()

    # Passing the data through the NN
    for i, (binary_contam, spectra, frac_solar, labels) in enumerate(training_generator):
        # sys.stdout.write('{}\n'.format(i))

        x = spectra
        if lastlayer == 'sigmoid':
            y = binary_contam
        elif lastlayer == 'linear':
            y = frac_solar
        elif lastlayer == 'labels':
            y = labels
        else:
            raise ValueError('unrecognized last layer argument')

        # Transfer to device
        x = x.to(device).float().view(-1, 1, np.shape(x)[1])
        y_true = y.to(device).float()
        y_true = y_true.unsqueeze(1)

        # perform a forward pass and calculate loss
        y_pred = NN(x)

        batch_loss = loss_fn(y_pred, y_true)

        # zero out the gradients, perform the backpropagation step,
        # and update the weights
        optimizer.zero_grad()
        batch_loss.backward()
        optimizer.step()

        # add batch loss to the total training loss
        loss += batch_loss

    avgLoss = loss / train_steps
    avgLoss = avgLoss.detach().cpu().numpy()
    return avgLoss


# Function to execute a validation epoch
def val_epoch_generator(NN, valid_generator, device, val_steps, lastlayer):
    if lastlayer == 'sigmoid':
        loss_fn = nn.BCEWithLogitsLoss()
    else:
        loss_fn = nn.MSELoss()

    with torch.no_grad():
        NN.eval()
        loss = 0
        # Passing the data through the NN
        for binary_contam, spectra, frac_solar, labels in valid_generator:
            x = spectra

            if lastlayer == 'sigmoid':
                y = binary_contam
            elif lastlayer == 'linear':
                y = frac_solar
            elif lastlayer == 'labels':
                y = labels
            else:
                raise ValueError('unrecognized last layer argument')

            # Transfer to device
            x = x.to(device).float().view(-1, 1, np.shape(x)[1])
            y_true = y.to(device).float()
            y_true = y_true.unsqueeze(1)

            y_pred = NN(x)

            loss += loss_fn(y_pred, y_true)  # *batch_size
        avgLoss = loss / val_steps
        avgLoss = avgLoss.detach().cpu().numpy()

        return avgLoss