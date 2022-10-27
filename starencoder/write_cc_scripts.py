import os
import argparse
import numpy as np
import random

from utils import str2bool

HOME_DIR = os.getenv('HOME')
SCRATCH_DIR = os.getenv('SCRATCH')
#SLURM_TMPDIR = os.getenv('SLURM_TMPDIR')
PYTHONPATH = os.getenv('PYTHONPATH')

parser = argparse.ArgumentParser()
parser.add_argument('--output_path', type=str, default='./jobs/todo/submit_job.sh',
                    help='Path of training file')
parser.add_argument('--data_path', type=str, required=True,
                        help='Path of training file')
parser.add_argument('--train_script', type=str, default=os.path.join(HOME_DIR, 'StarNet/scripts/train_StarNet.py'),
                    help='Path to training script')
parser.add_argument('--virtual_env', type=str, required=True,
                    help='Path to virtual environment to use')
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
parser.add_argument('--model_type', type=str, default='AE',
                    help='model type (AE, UNet, or WaveUNet)')
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

output_path_arg = args.output_path
data_path_arg = args.data_path
train_script_arg = args.train_script
virtual_env_arg = args.virtual_env
num_train_arg = args.num_train
save_folder_arg = args.save_folder
max_epochs_arg = args.epochs
batch_size_arg = args.batch_size
valid_path = args.validation_path


def write_script(output_path, save_folder, data_path, train_script, virtual_env, num_train, max_epochs,
                 encoded_dim, fc2_input_dim, batch_size, valid_path, learning_rate, model_type, depth=3, wf=5,
                 features=32, levels=6, kernel_size=5, strides=4, conv_type='gn', res='learned',
                 feature_growth='double', only_bad_contam=False):

    if not output_path.endswith('.sh'):
        output_path += '.sh'

    #data_filename = os.path.basename(data_path)

    print('Writing file to {}'.format(output_path))
    with open(output_path, 'w') as writer:
        writer.write('#!/bin/bash\n')
        writer.write('module load python/3.10\n')
        writer.write('source {}\n'.format(os.path.join(virtual_env, 'bin/activate')))
        writer.write('module load python/3.10\n')

        writer.write('cp -r {} {}\n'.format(os.path.join(HOME_DIR, 'StarNet'), '$SLURM_TMPDIR'))
        writer.write('export PYTHONPATH="{}:{}"\n'.format(PYTHONPATH, os.path.join('$SLURM_TMPDIR', 'StarNet/')))
        writer.write('\n')
        writer.write('python {}/StarNet/starlink/{} \\\n'.format('$SLURM_TMPDIR', os.path.basename(train_script)))
        writer.write(f'--data_path {data_path} \\\n')
        writer.write(f'--num_train {num_train} \\\n')
        writer.write(f'--save_folder {save_folder} \\\n')
        writer.write(f'--batch_size {batch_size} \\\n')
        writer.write(f'--epochs {max_epochs} \\\n')
        writer.write(f'--encoded_dim {encoded_dim} \\\n')
        writer.write(f'--fc2_input {fc2_input_dim} \\\n')
        writer.write(f'--learning_rate {learning_rate} \\\n')
        writer.write(f'--model_type {model_type} \\\n')
        writer.write(f'--depth {depth} \\\n')
        writer.write(f'--wf {wf} \\\n')
        if valid_path:
            writer.write(f'--validation_path {valid_path} \\\n')
        writer.write(f'--features {features} \\\n')
        writer.write(f'--levels {levels} \\\n')
        writer.write(f'--kernel_size {kernel_size} \\\n')
        writer.write(f'--strides {strides} \\\n')
        writer.write(f'--conv_type {conv_type} \\\n')
        writer.write(f'--res {res} \\\n')
        writer.write(f'--feature_growth {feature_growth} \\\n')
        writer.write(f'--only_bad_contam {only_bad_contam} \\\n')


for sample in range(40):

    learning_rate = np.random.uniform(0.0001, 0.002, 1)[0]
    encoded_dim = np.random.randint(40, 200)
    fc2_input_dim = np.random.randint(256, 512)
    depth = np.random.randint(1, 3)
    wf = np.random.randint(3, 7)
    features = np.random.choice([8, 16, 32, 64], 1)[0]
    levels = np.random.randint(2, 6)
    kernel_size = random.randrange(3, 9+1, 2)
    strides = np.random.randint(2, 6)

    if args.model_type.lower() == 'ae':
        save_model_path = os.path.join(save_folder_arg, 'n-{}-lr{:.5f}--ed{}--fc{}'.format(sample,
                                                                                            learning_rate,
                                                                                            encoded_dim,
                                                                                            fc2_input_dim))
    elif args.model_type.lower() == 'unet':
        save_model_path = os.path.join(save_folder_arg, 'n-{}-lr{:.5f}--d{}--wf{}'.format(sample,
                                                                                           learning_rate,
                                                                                           depth,
                                                                                           wf))
    elif args.model_type.lower() == 'waveunet':
        save_model_path = os.path.join(save_folder_arg, 'n-{}-lr{:.5f}--d{}--feat{}--lev{}--ks{}--str{}'.format(sample,
                                                                                                                learning_rate,
                                                                                                                depth,
                                                                                                                features,
                                                                                                                levels,
                                                                                                                kernel_size,
                                                                                                                strides))
    else:
        raise ValueError(f'Model type unknown: {args.model_type}')

    if not os.path.exists(save_model_path):
        os.makedirs(save_model_path)

    output_job_path = args.output_path
    output_job_folder = os.path.dirname(output_job_path)
    if not os.path.exists(output_job_folder):
        os.makedirs(output_job_folder)
    if not output_job_path.endswith('.sh'):
        output_job_path += '.sh'

    output_path = output_job_path[:-3] + '_{}.sh'.format(sample)

    write_script(output_path, save_model_path, data_path_arg, train_script_arg, virtual_env_arg,
                 num_train_arg, max_epochs_arg, encoded_dim, fc2_input_dim, batch_size_arg,
                 valid_path, learning_rate, args.model_type, depth, wf, features=features, levels=levels,
                 kernel_size=kernel_size, strides=strides, only_bad_contam=args.only_bad_contam)