import os
import argparse
import numpy as np
import random

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
parser.add_argument('-l', '--lastlayer', type=str, default='sigmoid',
                    help='Last layer (sigmoid or linear)')
parser.add_argument('--validation_path', type=str, default='',
                    help='(optional) Path of validation set (if different than data_path)')
#parser.add_argument('--learning_rate', type=float, default=0.0007,
#                    help='Learning rate to use for training')
args = parser.parse_args()

output_path_arg = args.output_path
data_path_arg = args.data_path
train_script_arg = args.train_script
virtual_env_arg = args.virtual_env
num_train_arg = args.num_train
save_folder_arg = args.save_folder
max_epochs_arg = args.epochs
batch_size_arg = args.batch_size
lastlayer_arg = args.lastlayer
valid_path = args.validation_path
#learning_rate = args.learning_rate


def write_script(output_path, save_folder, data_path, train_script, virtual_env, num_train, max_epochs,
                 batch_size, lastlayer, valid_path, learning_rate):

    if not output_path.endswith('.sh'):
        output_path += '.sh'

    #data_filename = os.path.basename(data_path)

    print('Writing file to {}'.format(output_path))
    with open(output_path, 'w') as writer:
        writer.write('#!/bin/bash\n')
        writer.write('module load python/3.7\n')
        writer.write('source {}\n'.format(os.path.join(virtual_env, 'bin/activate')))
        writer.write('module load python/3.7\n')

        writer.write('cp -r {} {}\n'.format(os.path.join(HOME_DIR, 'StarNet'), '$SLURM_TMPDIR'))
        writer.write('export PYTHONPATH="{}:{}"\n'.format(PYTHONPATH, os.path.join('$SLURM_TMPDIR', 'StarNet/')))
        writer.write('\n')
        writer.write('python {}/StarNet/starlink/{} \\\n'.format('$SLURM_TMPDIR', os.path.basename(train_script)))
        writer.write('--data_path %s \\\n' % data_path)
        writer.write('--num_train %s \\\n' % num_train)
        writer.write('--save_folder %s \\\n' % save_folder)
        writer.write('--batch_size %s \\\n' % batch_size)
        writer.write('--epochs %s \\\n' % max_epochs)
        writer.write('--lastlayer %s \\\n' % lastlayer)
        writer.write('--learning_rate %s \\\n' % learning_rate)
        if valid_path:
            writer.write('--validation_path %s' % valid_path)


for sample in range(40):

    learning_rate = np.random.uniform(0.0001, 0.001, 1)[0]

    save_model_path = os.path.join(save_folder_arg, 'n-{}-{:.5f}'.format(sample, learning_rate))
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
                 num_train_arg, max_epochs_arg, batch_size_arg, lastlayer_arg, valid_path, learning_rate)


