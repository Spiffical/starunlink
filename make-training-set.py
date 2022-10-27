import os
import sys
import h5py
import numpy as np
import multiprocessing
sys.path.insert(0, "{}/StarNet".format(os.getenv('HOME')))
from starnet.utils.data_utils.preprocess_spectra import rebin
from eniric.broaden import convolution, resolution_convolution

home = os.getenv('HOME')
data_dir = os.path.join(home, 'data')
spec_dir = os.path.join(data_dir, 'spectra')

wave_grid_solar = np.load('/arc/home/Merileo/data/wave_grids/UVES_4835-5395_solar.npy')
wave_grid_weave = np.load('/arc/home/Merileo/data/wave_grids/weave_hr_wavegrid_arms.npy')
shortened_wave_grid = wave_grid_solar[400:-400]
wave_grid_weave_overlap_ind = (wave_grid_weave > shortened_wave_grid[0]) & (wave_grid_weave < shortened_wave_grid[-1])
wave_grid_weave_overlap = wave_grid_weave[wave_grid_weave_overlap_ind]


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


# Load in spectra
solar_spectra = np.load(os.path.join(spec_dir, 'UVES_solar_spectra.npy'))
file_name = 'UVES_GE_MW_4835-5395_nonorm.h5'
with h5py.File(os.path.join(spec_dir, file_name), "r") as f:
        print(list(f.keys()))
        spectra = f['spectra'][:]
        y_uves = np.column_stack([f['teff'][:], f['logg'][:], f['fe_h'][:], f['v_rad'][:], f['vmicro'][:]])
        snr_uves = f['SNR'][:]
        ges_type = f['ges_type'][:]
        objects = f['object'][:]
        wave_grid = f['wave_grid'][:]
non_nan_indices = np.array([not any(np.isnan(y)) for y in y_uves])
spectra = spectra[non_nan_indices]
y_uves = y_uves[non_nan_indices]
snr_uves = snr_uves[non_nan_indices]
ges_type = ges_type[non_nan_indices]
objects = objects[non_nan_indices]
# Take care of bad values
for i, spec in enumerate(spectra):
    spec[spec<0]=0
    

augmented_uves_spectra = []
unaugmented_uves_spectra = []
solar_frac = []
snr_train = []
teff_train = []
logg_train = []
feh_train = []
vrad_train = []
vmicro_train = []
print('Collecting spectra...')
for i in range(15000):
    
    # Collect a UVES and solar spectrum
    uves_spec_ind = np.random.randint(int(0.8*len(spectra)))
    solar_spec_ind = np.random.randint(int(0.8*len(solar_spectra)))
    uves_spectrum = spectra[uves_spec_ind]
    solar_spectrum = solar_spectra[solar_spec_ind]
    
    # Calculate the median flux
    median_uves_flux = np.median(uves_spectrum)
    median_solar_flux = np.median(solar_spectrum)
    
    # Determine how much solar contamination there should be
    norm_factor = median_uves_flux / median_solar_flux
    frac_solar_contribution = np.random.uniform(0.01, 0.5)
    final_factor = norm_factor * frac_solar_contribution
    
    # Contaminate the spectra and append data to lists
    augmented_spectrum = uves_spectrum+final_factor*solar_spectrum
    augmented_uves_spectra.append(augmented_spectrum)
    unaugmented_uves_spectra.append(uves_spectrum)
    solar_frac.append(frac_solar_contribution)
    snr_train.append(snr_uves[uves_spec_ind])
    teff_train.append(y_uves[:,0][uves_spec_ind])
    logg_train.append(y_uves[:,1][uves_spec_ind])
    feh_train.append(y_uves[:,2][uves_spec_ind])
    vrad_train.append(y_uves[:,3][uves_spec_ind])
    vmicro_train.append(y_uves[:,4][uves_spec_ind])
    
augmented_weavescale = []
unaugmented_weavescale = []
BATCH_SIZE = 16
print('Processing spectra to have WEAVE resolution and sampling')
# Now degrade resolution and rebin to WEAVE-HR 
for i in range(len(augmented_uves_spectra))[::BATCH_SIZE]:
    print(i)
    
    augmented_batch = augmented_uves_spectra[i:i+BATCH_SIZE]
    unaugmented_batch = unaugmented_uves_spectra[i:i+BATCH_SIZE]

    augmented_batch = augment_spectra_parallel(augmented_batch, wave_grid_solar, shortened_wave_grid, wave_grid_weave_overlap, 20000)
    unaugmented_batch = augment_spectra_parallel(unaugmented_batch, wave_grid_solar, shortened_wave_grid, wave_grid_weave_overlap, 20000)
    
    augmented_weavescale.extend(augmented_batch)
    unaugmented_weavescale.extend(unaugmented_batch)
 
    
mean = np.mean(augmented_weavescale)
std = np.std(augmented_weavescale)

save_path = os.path.join(spec_dir, 'uves-solar-trainingset-weave.h5')
with h5py.File(save_path, "w") as f:
    spectra_dset = f.create_dataset('spectra', data=np.asarray(unaugmented_weavescale))
    spectra_contam_dset = f.create_dataset('spectra+solar', data=np.asarray(augmented_weavescale))
    solar_frac_dset = f.create_dataset('frac_solar', data=np.asarray(solar_frac))
    snr_dset = f.create_dataset('snr', data=np.asarray(snr_train))
    mean_dset = f.create_dataset('mean_flux', data=mean)
    std_dset = f.create_dataset('std_flux', data=std)
    teff_dset = f.create_dataset('teff', data=np.asarray(teff_train))
    logg_dset = f.create_dataset('logg', data=np.asarray(logg_train))
    feh_dset = f.create_dataset('feh', data=np.asarray(feh_train))
    vrad_dset = f.create_dataset('vrad', data=np.asarray(vrad_train))
    vmicro_dset = f.create_dataset('vmicro', data=np.asarray(vmicro_train))
    wave_dset = f.create_dataset('wave_grid', data=wave_grid_weave_overlap)