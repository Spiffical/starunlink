import os
import sys
import numpy as np
import torch
import argparse
import h5py

from starencoder.models.autoencoder import StarEncoder
from starencoder.models.unet import UNet_tunable
from starencoder.models.waveunet import Waveunet
from starencoder.utils import get_train_valid_loader, train_epoch_den, test_epoch_den, str2bool


def train_NN(args):

    with h5py.File(args.data_path, 'r') as f:
        cond_frac_solar = (f['frac_solar'][:] > 0.45) & (f['frac_solar'][:] < 0.48)
        cond_small_vals = [np.all(f_ < 1) for f_ in f['spectra']]

        if args.only_bad_contam:
            indices_train = np.argwhere(cond_frac_solar & cond_small_vals).squeeze()
        else:
            indices_train = np.argwhere(cond_small_vals).squeeze()
        if args.validation_path:
            with h5py.File(args.validation_path, 'r') as f:
                cond_frac_solar = (f['frac_solar'][:] > 0.45) & (f['frac_solar'][:] < 0.48)
                cond_small_vals = [np.all(f_ < 1) for f_ in f['spectra']]
                if args.only_bad_contam:
                    indices_valid = np.argwhere(cond_frac_solar & cond_small_vals).squeeze()
                else:
                    indices_valid = np.argwhere(cond_small_vals).squeeze()

    ### Define the loss function
    if args.loss_fn.lower() == 'mse':
        loss_fn = torch.nn.MSELoss()
    elif args.loss_fn.lower() == 'l1':
        loss_fn = torch.nn.L1Loss()
    else:
        raise ValueError(f'Unsupported loss function: {args.loss_fn}')

    with h5py.File(args.data_path, 'r') as f:
        len_spec = len(f['spectra'][0])

    computed_input = len_spec

    ### Define the model
    if args.model_type.lower() == 'ae':
        starencoder = StarEncoder(args.encoded_dim, args.fc2_input,
                                  in_channels=1, input_dim=(1, len_spec))
        print('Using StarEncoder')
    elif args.model_type.lower() == 'unet':
        starencoder = UNet_tunable(depth=args.depth, wf=args.wf, batch_norm=True, res_flag=False)
        print('Using UNet model with depth={} and wf={}'.format(args.depth, args.wf))
    elif args.model_type.lower() == 'waveunet':

        num_features = [args.features * i for i in range(1, args.levels + 1)] if args.feature_growth == "add" else \
            [args.features * 2 ** i for i in range(0, args.levels)]

        target_outputs = len_spec
        computed_input = np.inf

        # Compute the size required for input to Wave U-Net (we are choosing not to pad with zeros)
        print('Iteratively calculating the required input size for the network...')
        while computed_input > len_spec:
            starencoder = Waveunet(1, num_features, 1, ['spectra', ], kernel_size=args.kernel_size,
                                   target_output_size=target_outputs, depth=args.depth, strides=args.strides,
                                   conv_type=args.conv_type, res=args.res, separate=0)

            computed_input, _ = starencoder.check_padding(target_outputs)
            target_outputs -= 2

        print('Using WaveUNet model with computed input size of: {}'.format(computed_input))
    else:
        raise ValueError('Unrecognized model type: {}'.format(args.model_type))

    ### Define data loaders
    train_loader, valid_loader, = get_train_valid_loader(args.data_path,
                                                         args.batch_size,
                                                         args.num_train,
                                                         valid_size=0.1,
                                                         shuffle=True,
                                                         num_workers=10,
                                                         pin_memory=True,
                                                         val_path=args.validation_path,
                                                         indices_train=indices_train,
                                                         indices_valid=indices_valid,
                                                         truncate_n=len_spec-computed_input)

    ### Define optimizer and scheduler
    optim = torch.optim.Adam(starencoder.parameters(), lr=args.learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optim,
                                                           mode='min',
                                                           factor=0.5,
                                                           patience=5,
                                                           min_lr=0.000002,
                                                           eps=1e-08,
                                                           verbose=True)


    # Check if the GPU is available
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print(f'Selected device: {device}')

    # Move the autoencoder to the selected device
    starencoder.to(device)

    # Training cycle
    num_epochs = args.epochs
    history_da = {'train_loss': [], 'val_loss': []}
    best_valid_loss = float('inf')

    for epoch in range(num_epochs):
        print('EPOCH %d/%d' % (epoch + 1, num_epochs))

        # Training
        train_loss = train_epoch_den(
            model=starencoder,
            device=device,
            dataloader=train_loader,
            loss_fn=loss_fn,
            optimizer=optim,
            model_type=args.model_type)

        # Validation
        val_loss = test_epoch_den(
            model=starencoder,
            device=device,
            dataloader=valid_loader,
            loss_fn=loss_fn,
            model_type=args.model_type)

        scheduler.step(val_loss)

        with open((os.path.join(args.save_folder, 'train_hist.txt')), 'a+') as f:
            f.write('{}, '.format(train_loss))
            f.write('{}'.format(val_loss))
            f.write('\n')

        if val_loss < best_valid_loss:
            sys.stdout.write('Saving best model with validation loss of {}\n'.format(val_loss))
            best_valid_loss = val_loss
            torch.save(starencoder.state_dict(), os.path.join(args.save_folder, 'best_starencoder.pth'))

        # Print Validation loss
        history_da['train_loss'].append(train_loss)
        history_da['val_loss'].append(val_loss)
        print('\n EPOCH {}/{} \t train loss {:.3f} \t val loss {:.3f}'.format(epoch + 1,
                                                                              num_epochs,
                                                                              train_loss,
                                                                              val_loss))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, required=True,
                        help='Path of training file')
    parser.add_argument('--num_train', type=int, required=True,
                        help='Size of training set')
    parser.add_argument('--save_folder', type=str, default=None,
                        help='Folder to save trained model in (if None, folder name created based on date)')
    parser.add_argument('-b', '--batch_size', type=int, default=16,
                        help='Number of spectra used in a single batch')
    parser.add_argument('-e', '--epochs', type=int, default=300,
                        help='Maximum number of epochs for training')
    parser.add_argument('--validation_path', type=str, default='',
                        help='(optional) Path of validation set (if different than data_path)')
    parser.add_argument('--encoded_dim', type=int, default=128,
                        help='')
    parser.add_argument('--fc2_input', type=int, default=256,
                        help='')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                        help='')
    parser.add_argument('--model_type', type=str, default='AE',
                        help='Type of model to use (AE or UNet)')
    parser.add_argument('--loss_fn', type=str, default='l1',
                        help='Type of loss to use (MSE or L1)')
    parser.add_argument('--wf', type=int, default=6,
                        help='2^wf filters in first layer (if model is UNet)')
    parser.add_argument('--depth', type=int, default=3,
                        help='depth of UNet (number of layers) or WaveUNet (number of convs per block)')
    parser.add_argument('--features', type=int, default=32,
                        help='Number of features in conv filter for WaveUNet')
    parser.add_argument('--levels', type=int, default=6,
                        help="Number of DS/US blocks for WaveUNet")
    parser.add_argument('--kernel_size', type=int, default=5,
                        help="Filter width of kernels. Has to be an odd number")
    parser.add_argument('--strides', type=int, default=4,
                        help="Strides in Waveunet")
    parser.add_argument('--conv_type', type=str, default="gn",
                        help="Type of convolution (normal, BN-normalised, GN-normalised): normal/bn/gn")
    parser.add_argument('--res', type=str, default="learned",
                        help="Resampling strategy: fixed sinc-based lowpass filtering or learned conv layer: fixed/learned")
    parser.add_argument('--feature_growth', type=str, default="double",
                        help="How the features in each layer should grow, either (add) the initial number of features each time, or multiply by 2 (double)")
    parser.add_argument('--only_bad_contam', type=str2bool, default=False,
                        help="Whether to only train on bad contamination (defined as 45% to 48% contamination)")
    args = parser.parse_args()

    # Train the NN
    train_NN(args)

