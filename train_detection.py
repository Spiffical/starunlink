import torch
import torch.optim as optim
import numpy as np
import h5py
import os
import sys
import argparse

from detection.utils import get_train_valid_loader, train_epoch_generator, val_epoch_generator
from detection.model import StarLink


def train_NN(num_train, data_path, save_folder, max_epochs, batch_size, lastlayer, validation_path, learning_rate):
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
        # np.save(os.path.join(save_folder, 'config.npy'), np.asarray(config))

    train_loader, valid_loader = get_train_valid_loader(data_path,
                                                        batch_size,
                                                        num_train,
                                                        valid_size=0.1,
                                                        shuffle=True,
                                                        num_workers=10,
                                                        pin_memory=True,
                                                        val_path=validation_path)

    trainSteps = len(train_loader.dataset) // batch_size
    valSteps = len(valid_loader.dataset) // batch_size

    # CUDA for PyTorch
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")

    with h5py.File(data_path, 'r') as f:
        len_spec = len(f['spectra'][0])

    out_channels = 3 if lastlayer == 'labels' else 1
    NN = StarLink(1, out_channels, (1, len_spec))

    # Load in last checkpointed model
    bestmodel_checkpoint = os.path.join(args.save_folder, 'checkpoint.pth')
    loss_val_min = np.inf
    if os.path.exists(bestmodel_checkpoint):
        sys.stdout.write('Model aleady exists! Continuing from saved file: {}\n'.format(bestmodel_checkpoint))

        print('Loading checkpointed model state_dict')
        state = torch.load(bestmodel_checkpoint)
        NN.load_state_dict(state['state_dict'])

        print('Loading checkpointed optimizer state')
        optimizer = torch.optim.Adam(NN.parameters())
        optimizer.load_state_dict(state['optimizer'])

        loss_hist = np.loadtxt(os.path.join(args.save_folder, 'train_hist.txt'), dtype=float, delimiter=',')
        dim = len(np.shape(loss_hist))
        if dim == 0:
            sys.stdout.write('No training history!\n')
        elif dim == 1:
            loss_val_min = loss_hist[1]
        else:
            loss_val_hist = loss_hist[:, 1]
            loss_val_min = min(loss_val_hist)
    else:
        optimizer = torch.optim.Adam(NN.parameters(), lr=learning_rate)

    # Set up scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                     mode='min',
                                                     factor=0.5,
                                                     patience=3,
                                                     min_lr=0.000002,
                                                     eps=1e-08,
                                                     verbose=True)

    NN.to(device)

    for epoch in range(max_epochs):
        sys.stdout.write('Epoch {}\n'.format(epoch))
        # Training epoch
        loss_train = train_epoch_generator(NN, training_generator=train_loader,
                                           optimizer=optimizer,
                                           device=device,
                                           train_steps=trainSteps,
                                           lastlayer=lastlayer)
        loss_val = val_epoch_generator(NN, valid_generator=valid_loader,
                                       device=device,
                                       val_steps=valSteps,
                                       lastlayer=lastlayer)
        scheduler.step(loss_val)
        # loss_train = train_epoch(NN,data_file,indices_train,targets,mean,std,batch_size,scheduler,optimizer)
        # loss_val = val_epoch(NN,data_file,indices_val,targets,mean,std,batch_size)
        sys.stdout.write('train_loss: {}, val_loss: {}\n'.format(loss_train, loss_val))
        # Saving results to txt file
        sys.stdout.write('Saving training losses to {}\n'.format(os.path.join(save_folder, 'train_hist.txt')))
        with open((os.path.join(save_folder, 'train_hist.txt')), 'a+') as f:
            f.write('{}, '.format(loss_train))
            f.write('{}'.format(loss_val))
            f.write('\n')

        if loss_val < loss_val_min:
            loss_val_min = loss_val
            sys.stdout.write('Saving best model with validation loss of {}\n'.format(loss_val))
            torch.save(NN.state_dict(), os.path.join(save_folder, 'model_best.pth'))

        # Save checkpoint for resuming training
        print('Saving the checkpoint')
        state = {
            'epoch': epoch,
            'state_dict': NN.state_dict(),
            'optimizer': optimizer.state_dict()
        }
        torch.save(state, os.path.join(save_folder, 'checkpoint.pth'))


################################################################################


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
    parser.add_argument('-l', '--lastlayer', type=str, default='sigmoid',
                        help='Last layer of the network')
    parser.add_argument('--validation_path', type=str, default='',
                        help='(optional) Path of validation set (if different than data_path)')
    parser.add_argument('--learning_rate', type=float, default=0.0007,
                        help='Learning rate to use for training')

    args = parser.parse_args()

    data_path = args.data_path
    num_train = args.num_train
    save_folder = args.save_folder
    max_epochs = args.epochs
    batch_size = args.batch_size
    lastlayer = args.lastlayer
    validation_path = args.validation_path
    lr = args.learning_rate

    torch.multiprocessing.set_start_method('spawn')

    train_NN(num_train,
             data_path,
             save_folder,
             max_epochs,
             batch_size,
             lastlayer,
             validation_path,
             lr
             )

