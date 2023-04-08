import numpy as np
from astropy.table import Table
from astropy import cosmology
from scipy.special import gamma, gammaincinv, gammainc


def index():
    # define filepaths (this will look different depending on the user)
    filepath_1 = "/Users/pitt-googlebroker/Documents/host-probability/sweep-000m005-010p000-stellar-mass.fits"
    filepath_2 = "/Users/pitt-googlebroker/Documents/host-probability/sweep-000p000-010p005-stellar-mass.fits.gz"
    filepath_3 = "/Users/pitt-googlebroker/Documents/host-probability/sweep-270p075-280p080-stellar-mass.fits.gz"
    filepath_4 = "/Users/pitt-googlebroker/Documents/host-probability/sweep-270p075-280p080.fits"
    filepath_5 = "/Users/pitt-googlebroker/Documents/host-probability/sweep-270p075-280p080-pz.fits"
    
    return read_files(filepath_3,filepath_4,filepath_5)

def read_files(mass,survey,photoz):
    "Read data directly from .fits files as an astropy Table object"
    # define tables
    mass_table = Table.read(mass)
    survey_table = Table.read(survey)
    photoz_table = Table.read(photoz)
    
    return table_to_data(survey_table,photoz_table)

def table_to_data(survey_table,photoz_table):
    "Extracts relevant data and defines essential parameters"
    # determine relevant data
    galaxy_data = survey_table[np.where(survey_table['TYPE'] != 'PSF')]
    photoz_data = photoz_table[np.where(survey_table['TYPE'] != 'PSF')]

    # define additional parameters
    shape_r = galaxy_data['SHAPE_R']
    z_photo_mean = photoz_data['Z_PHOT_MEAN']
    galaxy_type = galaxy_data['TYPE']

    return calc_parameters(shape_r,z_photo_mean,galaxy_type)

def calc_parameters(shape_r, z_photo_mean,galaxy_type):
    "Calculates the parameters needed to describe Sérsic profiles"
    # convert to list to simply calculations
    galaxy_shape_r = [i for i in shape_r]
    galaxy_types = [i for i in galaxy_type]
    redshift_data = [i for i in z_photo_mean]

    # determine Sérsic index
    n = n_out(galaxy_types)

    # calculate distances [Mpc], angular size [radians], & half-life radius
    cosmo = cosmology.FlatLambdaCDM(H0=70,Om0=0.3)
    distance = [cosmo.angular_diameter_distance(i) for i in redshift_data]
    angular_size = [(np.pi/180.)*(1./60)*(1./60)*i for i in galaxy_shape_r]
    half_light_radius = [i*j for i, j in zip(distance, angular_size)] # equivalent to effective radius

    # prepare data in order to integrate total luminosity of Sérsic profile
    eff_rad = [i.to('pc').value for i in half_light_radius]
    half_eff_rad = [i/2 for i in eff_rad]

    return integrated_enc_lum(half_eff_rad,eff_rad,n)

def n_out(galaxy_types):
    "Converts galaxy types into corresponding Sérsic index values"
    # galaxy types other than DEV, EXP, are given Sérsic index values of 6 (I don't know what other value to give them)
    mapping = {
            'DEV': 4, 
            'EXP': 1, 
            'SER': 6, 
            'REX': 6, 
            'DUP': 6}
    n = [mapping[i] for i in galaxy_types]

    return n

def integrated_enc_lum(half_eff_rad,eff_rad,n):
    "Calculates the integral of a Sérsic profile. Code (modified) obtained from: https://gist.github.com/bamford/b657e3a14c9c567afc4598b1fd10a459"
    # define constants
    I_e = 1 # intensity at the effective radius
    b_n = normalization_constant(n)
    gamma_val = [gamma(2*i) for i in n]
    
    # luminosity enclosed within a radius r
    x = [i * (j/k)**(1.0/l) for i,j,k,l in zip(b_n,half_eff_rad,eff_rad,n)] 
    incomplete_gamma = [gammainc(2*i,j) for i,j in zip(n,x)]
    
    # total luminosity (integrated to infinity)
    tot_luminosity = [I_e* i**2 * 2*np.pi*j * np.exp(l)/(l**(2*j)) * k for i,j,k,l in zip(eff_rad, n,gamma_val,b_n)]
    
    # luminosity enclosed within a radius r
    enc_luminosity = [i*j for i,j in zip(tot_luminosity,incomplete_gamma)]

    return enc_luminosity

def normalization_constant(n):
    "Calculates the normalization constant term: b_n for the Sérsic profiles"
    constant = [gammaincinv(2*i, 0.5) for i in n]

    return constant

def incomplete_gamma(n,x):
    "Calculates the incomplete gamma function"
    incomplete_gamma = [gammainc(2*i,j) for i,j in zip(n,x)]

    return incomplete_gamma
