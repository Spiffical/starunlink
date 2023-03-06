import os
import sys
import h5py
import glob
import numpy as np
import multiprocessing
import argparse
import uuid
from contextlib import contextmanager
sys.path.insert(0, "{}/StarNet".format(os.getenv('HOME')))
from starnet.utils.data_utils.preprocess_spectra import rebin
from eniric.broaden import convolution, resolution_convolution


home = os.getenv('HOME')
#data_dir = os.path.join(home, 'data')
#spec_dir = os.path.join(data_dir, 'spectra', 'processed')

BATCH_SIZE = 16
DATA_DICT = {"spectra": [], "spectra+solar": [], "frac_solar": [], "snr": [], "O": [],
             "Mg": [], "Ca": [], "S": [], "Ti": [], "vmicro": [], "vrad": [], "feh": [],
             "teff": [], "logg": []}


def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1', 'real'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0', 'fake'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def add_radial_velocity(wav, rv, flux=None):
    """
    This function adds radial velocity effects to a spectrum.

    wav: wavelength array
    rv: radial velocity (km/s)
    flux: spectral flux array
    """
    # Speed of light in m/s
    c = 299792458.0

    # New wavelength array with added radial velocity
    new_wav = wav * np.sqrt((1. - (-rv * 1000.) / c) / (1. + (-rv * 1000.) / c))

    # if flux array provided, interpolate it onto this new wavelength grid and return both,
    # otherwise just return the new wavelength grid
    if flux is not None:
        new_flux = rebin(new_wav, wav, flux)
        return new_wav, new_flux
    else:
        return new_wav


def augment_spectrum(flux, wav, intermediate_wav, final_wav, to_res=20000):

    # Degrade resolution
    flux = resolution_convolution(wavelength=intermediate_wav,
                                  extended_wav=wav,
                                  extended_flux=flux,
                                  R=to_res,
                                  normalize=True,
                                  num_procs=1)

    # Rebin to final wave grid
    flux = rebin(final_wav, intermediate_wav, flux)

    return flux


def augment_spectra_parallel(spectra, wav, intermediate_wav, final_wav, instrument_res):

    @contextmanager
    def poolcontext(*args, **kwargs):
        pool = multiprocessing.Pool(*args, **kwargs)
        yield pool
        pool.terminate()

    num_spectra = np.shape(spectra)[0]
    num_cpu = multiprocessing.cpu_count()
    pool_size = num_cpu if num_spectra >= num_cpu else num_spectra
    #print('[INFO] Pool size: {}'.format(pool_size))

    pool_arg_list = [(spectra[i], wav, intermediate_wav, final_wav, instrument_res)
                     for i in range(num_spectra)]
    with poolcontext(processes=pool_size) as pool:
        results = pool.starmap(augment_spectrum, pool_arg_list)

    augmented_spectra = [result for result in results]

    return augmented_spectra


def make_dataset(args):

    global BATCH_SIZE, DATA_DICT

    wave_grid_solar = np.load(args.wave_grid_solar)
    wave_grid_weave = np.load(args.wave_grid_weave)
    shortened_wave_grid = wave_grid_solar[400:-400]
    wave_grid_weave_overlap_ind = (wave_grid_weave > shortened_wave_grid[0]) & (
                wave_grid_weave < shortened_wave_grid[-1])
    wave_grid_weave_overlap = wave_grid_weave[wave_grid_weave_overlap_ind]

    # Load in spectra
    solar_spectra = np.load(args.solar_spectra)
    with h5py.File(args.uves_spectra, "r") as f:
        print(list(f.keys()))
        spectra = f['spectra'][:]
        y_uves = np.column_stack([f['teff'][:], f['logg'][:], f['fe_h'][:], f['v_rad'][:], f['vmicro'][:]])
        abundances_uves = np.column_stack([f['Ca'][:], f['Mg'][:], f['O'][:], f['S'][:], f['Ti'][:]])
        snr_uves = f['SNR'][:]
        rv_uves = f['v_rad'][:]
        # ges_type = f['ges_type'][:]
        # objects = f['object'][:]
        # wave_grid = f['wave_grid'][:]
    non_nan_indices = np.array([not any(np.isnan(y)) for y in y_uves])
    spectra = spectra[non_nan_indices]
    y_uves = y_uves[non_nan_indices]
    abundances_uves = abundances_uves[non_nan_indices]
    snr_uves = snr_uves[non_nan_indices]
    rv_uves = rv_uves[non_nan_indices]
    # ges_type = ges_type[non_nan_indices]
    # objects = objects[non_nan_indices]
    # Take care of bad values
    for i, spec in enumerate(spectra):
        spec[spec < 0] = 0

    #if os.path.exists(args.save_path):
    #    with h5py.File(args.save_path, 'w') as hf:
    #        num_processed_already = len(hf['teff'])
    #else:
    #    num_processed_already = 0

    # Determine total number of spectra already in chosen directory
    existing_files = glob.glob(args.save_dir + '/*__*')
    # Number of spectra appended on to end of filename
    total_num_spec = np.sum([int(os.path.basename(f)[18:]) for f in existing_files])
    print('Total # of spectra in directory: {}/{}'.format(total_num_spec, args.total_num))
    print('Number of spectra already processed: {}/{}'.format(total_num_spec, args.total_num))

    print(f'Collecting spectra for the {args.dset_type} set...')

    # Generate the batches of spectra
    spectra_created = 0
    while total_num_spec <= args.total_num:
    #for i in range(num_processed_already, args.total_num):

        if total_num_spec >= args.total_num:
            print('Maximum number of spectra reached.')
            break
        #if i % BATCH_SIZE == 0:
        #    print(f'{i} of {args.total_num} processed')

        # Collect a UVES and solar spectrum
        if args.dset_type == 'train':
            uves_spec_ind = np.random.randint(int(0.8*len(spectra)))
            solar_spec_ind = np.random.randint(int(0.8*len(solar_spectra)))
        elif args.dset_type == 'valid':
            uves_spec_ind = np.random.randint(int(0.8*len(spectra)), int(0.85*len(spectra)))
            solar_spec_ind = np.random.randint(int(0.8*len(solar_spectra)), int(0.85*len(solar_spectra)))
        elif args.dset_type == 'test':
            uves_spec_ind = np.random.randint(int(0.85*len(spectra)), len(spectra))
            solar_spec_ind = np.random.randint(int(0.85*len(solar_spectra)), len(solar_spectra))
        else:
            raise ValueError(f'Unknown dset_type: {args.dset_type}')
        uves_spectrum = spectra[uves_spec_ind]
        solar_spectrum = solar_spectra[solar_spec_ind]

        # Calculate the median flux
        median_uves_flux = np.median(uves_spectrum)
        median_solar_flux = np.median(solar_spectrum)

        # Determine how much solar contamination there should be
        norm_factor = median_uves_flux / median_solar_flux  # For bringing solar flux to same scale as uves flux
        frac_solar_contribution = np.random.uniform(0.00, args.max_contam)
        final_factor = norm_factor * frac_solar_contribution

        if str2bool(args.real_vrad):
            rv = rv_uves[uves_spec_ind]
        else:  # Radially shift the spectrum
            if ~np.isnan(rv_uves[uves_spec_ind]):
                rv = np.random.uniform(-200, 200)
                rv_interm = rv_uves[uves_spec_ind] - rv  # first shift to rest frame then bring to new rv
                _, uves_spectrum = add_radial_velocity(wave_grid_solar, rv_interm, uves_spectrum)
            else:
                rv = rv_uves[uves_spec_ind]

        # Contaminate the spectrum
        contam_spectrum = uves_spectrum + final_factor * solar_spectrum

        # Append data to dict
        DATA_DICT['frac_solar'].append(frac_solar_contribution)
        DATA_DICT['snr'].append(snr_uves[uves_spec_ind])
        DATA_DICT['Ca'].append(abundances_uves[:, 0][uves_spec_ind])
        DATA_DICT['Mg'].append(abundances_uves[:, 1][uves_spec_ind])
        DATA_DICT['O'].append(abundances_uves[:, 2][uves_spec_ind])
        DATA_DICT['S'].append(abundances_uves[:, 3][uves_spec_ind])
        DATA_DICT['Ti'].append(abundances_uves[:, 4][uves_spec_ind])
        DATA_DICT['vrad'].append(rv)
        DATA_DICT['teff'].append(y_uves[:, 0][uves_spec_ind])
        DATA_DICT['logg'].append(y_uves[:, 1][uves_spec_ind])
        DATA_DICT['feh'].append(y_uves[:, 2][uves_spec_ind])
        DATA_DICT['vmicro'].append(y_uves[:, 4][uves_spec_ind])
        DATA_DICT['spectra'].append(uves_spectrum)
        DATA_DICT['spectra+solar'].append(contam_spectrum)

        # Fill up h5 file
        if len(DATA_DICT['teff']) == BATCH_SIZE:
        #if (i % BATCH_SIZE == 0 and i != 0) or i == args.total_num - 1:

            print(f'Augmenting {len(DATA_DICT["teff"])} spectra')
            # Change resolution and wavelength grid
            DATA_DICT['spectra+solar'] = augment_spectra_parallel(DATA_DICT['spectra+solar'], wave_grid_solar,
                                                                  shortened_wave_grid,
                                                                  wave_grid_weave_overlap, args.resolution)
            DATA_DICT['spectra'] = augment_spectra_parallel(DATA_DICT['spectra'], wave_grid_solar,
                                                            shortened_wave_grid,
                                                            wave_grid_weave_overlap, args.resolution)

            if not os.path.exists(args.save_dir):
                os.makedirs(args.save_dir)
            unique_filename = str(uuid.uuid4())[:12] + '__spec{}'.format(BATCH_SIZE)
            save_path = os.path.join(args.save_dir, unique_filename)
            print('Saving {} to {}'.format(unique_filename, args.save_dir))

            with h5py.File(save_path, 'w') as hf:
                for key in DATA_DICT.keys():
                    print(key)
                    hf.create_dataset(key, data=np.asarray(DATA_DICT[key]))
                hf.create_dataset('wave_grid', data=wave_grid_weave_overlap)

            spectra_created += BATCH_SIZE
            print('Total # of spectra created so far: {}'.format(spectra_created))

            # Again, check to see how many files are in the chosen save directory (parallel jobs will be filling it up too)
            existing_files = glob.glob(args.save_dir + '/*__*')
            # Number of spectra appended on to end of filename
            total_num_spec = np.sum([int(os.path.basename(f)[18:]) for f in existing_files])
            print('Total # of spectra in directory: {}/{}'.format(total_num_spec, args.total_num))

            # # Create master file and add data
            # if not os.path.exists(args.save_path):
            #     print('Save file does not yet exist. Creating it at: {}'.format(args.save_path))
            #     with h5py.File(args.save_path, 'w') as hf:
            #         for key in DATA_DICT.keys():
            #             print(key)
            #             if len(np.shape(DATA_DICT[key])) == 3:  # e.g. image
            #                 maxshape = (None, np.shape(DATA_DICT[key])[1], np.shape(DATA_DICT[key])[2])
            #             elif len(np.shape(DATA_DICT[key])) == 2:  # e.g. 1-d spectrum
            #                 maxshape = (None, np.shape(DATA_DICT[key])[1])
            #             else:  # e.g. single value
            #                 maxshape = (None,)
            #             hf.create_dataset(key, data=DATA_DICT[key], maxshape=maxshape)
            # # Or append to master file if it already exists
            # else:
            #     print('Appending data to file...')
            #     with h5py.File(args.save_path, 'a') as hf:
            #         for key in DATA_DICT.keys():
            #             hf[key].resize((hf[key].shape[0]) + np.shape(DATA_DICT[key])[0],
            #                            axis=0)
            #             hf[key][-np.shape(DATA_DICT[key])[0]:] = DATA_DICT[key]

            # Clear data dict to get it ready for next batch of data
            for value in DATA_DICT.values():
                del value[:]

    # Calculate mean and std of flux data and append to h5 file
    #with h5py.File(args.save_path, 'a') as hf:
    #    mean_flux = np.mean(hf['spectra+solar'])
    #    std_flux = np.std(hf['spectra+solar'])
    #    hf.create_dataset('mean_flux', data=mean_flux)
    #    hf.create_dataset('std_flux', data=std_flux)
    #    hf.create_dataset('wave_grid', data=wave_grid_weave_overlap)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_dir', type=str, required=True,
                        help='Folder to save spectra .h5 files in')
    parser.add_argument('--total_num', type=int, required=True,
                        help='Maximum number of spectra to create')
    parser.add_argument('--wave_grid_solar', type=str, default='/arc/home/Merileo/data/wave_grids/UVES_4835-5395_solar.npy',
                        help='Number of spectra used in a single batch')
    parser.add_argument('--wave_grid_weave', type=str, default='/arc/home/Merileo/data/wave_grids/weave_hr_wavegrid_arms.npy',
                        help='Number of spectra used in a single batch')
    parser.add_argument('--solar_spectra', type=str, default='UVES_solar_spectra.npy',
                        help='Number of spectra used in a single batch')
    parser.add_argument('--uves_spectra', type=str, default='UVES_GE_MW_4835-5395_nonorm_abundances.h5',
                        help='Number of spectra used in a single batch')
    parser.add_argument('--dset_type', type=str, default='train',
                        help='Number of spectra used in a single batch')
    parser.add_argument('--max_contam', type=float, default=0.5,
                        help='maximum fraction of contamination to add')
    parser.add_argument('--resolution', type=float, default=20000,
                        help='Resolution to degrade spectrum to')
    parser.add_argument('--real_vrad', type=str, default='False',
                        help='Whether to use real vrads (if false, will uniformly sample within bounds')
    args = parser.parse_args()

    make_dataset(args)
