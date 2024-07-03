import os
import numpy as np
import configparser as ConfigParser

from matplotlib import pyplot as plt
from matplotlib import rc
from matplotlib import colors

import galsim
from rwl_tools import load_catalogue

rc('text', usetex=True)
rc('font', family='serif')
rc('font', size=11)

plt.close('all') # tidy up any unshown plots

config = ConfigParser.ConfigParser(inline_comment_prefixes=";")
config.read('./inis/test.ini')

output_path = config.get("pipeline", "output_path")
figure_path = config.get("pipeline", "figure_path")

# Set some image properties
pixel_scale = config.getfloat("skymodel", "pixel_scale") * galsim.arcsec
fov = config.getfloat("skymodel", "field_of_view") * galsim.arcmin
image_size = int((fov / galsim.arcmin) / (pixel_scale / galsim.arcmin))

psf_image = galsim.fits.read(config.get("skymodel", "psf_filepath"))
psf = galsim.InterpolatedImage(psf_image, flux=1, scale=pixel_scale / galsim.arcsec)

# load the catalogue
cat = load_catalogue(config)

# cut the catalogue to the number of sources we want
nobj = len(cat)
if config.getint("skymodel", "ngals") > -1:
    nobj = config.getint("skymodel", "ngals")
    cat = cat[:nobj]

# write out catalogue
output_filename = (
    "truthcat" + "_" + config.get("pipeline", "output_suffix") + ".fits"
)

print("Writing truthcat to " + os.path.join(output_path, output_filename))
cat.write(
    os.path.join(output_path, output_filename), format="fits", overwrite=True,
)

for i, cat_gal in enumerate(cat):

    print(f"Drawing galaxy {i}...")

    full_image = galsim.ImageF(image_size, image_size,
                               scale=pixel_scale/galsim.arcsec
                               )
    im_center = full_image.bounds.true_center

    if config.get("skymodel", "galaxy_profile") == "exponential":
        gal = galsim.Exponential(
            scale_radius=cat_gal["Maj"] / 2.0,
            flux=cat_gal["Total_flux"],
            gsparams=big_fft_params,
        )

    elif config.get("skymodel", "galaxy_profile") == "gaussian":
        gal = galsim.Gaussian(
            fwhm=cat_gal["Maj"],
            flux=cat_gal["Total_flux"],
            # gsparams=big_fft_params,
        )

    # calculate the total ellipticity
    ellipticity = galsim.Shear(e1=cat_gal["e1"], e2=cat_gal["e2"])

    if config.getboolean("skymodel", "doshear"):
        
        if config.get("skymodel", "shear_type")=='trecs':
            shear = galsim.Shear(g1=cat_gal["g1_shear"], g2=cat_gal["g2_shear"])
            if i == 0:
                print('Applying shear read from trecs catalogue...')
        elif config.get("skymodel", "shear_type")=='constant':
            g1 = config.getfloat("skymodel", "constant_shear_g1")
            g2 = config.getfloat("skymodel", "constant_shear_g2")
            shear = galsim.Shear(g1=g1, g2=g2)
            if i == 0:
                print('Applying constant shear g1 = {}, g2 = {}'.format(g1, g2))

        total_shear = ellipticity + shear
    else:
        total_shear = ellipticity

    # get the galaxy size and add its ellipticity
    maj_gal = cat_gal["Maj"]
    q_gal = cat_gal["Min"] / cat_gal["Maj"]
    A_gal = np.pi * maj_gal ** 2.0
    maj_corr_gal = np.sqrt(A_gal / (np.pi * q_gal))

    gal = gal.shear(total_shear)
    gal = gal.dilate(maj_gal / maj_corr_gal)

    # convolve gal with psf
    gal = galsim.Convolve([gal, psf])

    cn = galsim.CorrelatedNoise(psf_image*1.e-8)

    stamp = gal.drawImage(scale=pixel_scale / galsim.arcsec)

    if i==0:
        print("Drawing PSF...")
        plt.title('PSF')
        psf_stamp = psf.drawImage(nx=512, ny=512, scale=pixel_scale / galsim.arcsec)
        psf_stamp.setCenter((image_size / 2, image_size / 2))
        plt.figure()
        plt.imshow(psf_stamp.array, origin="lower")#, norm=colors.LogNorm())
        plt.savefig(os.path.join(figure_path, "psf.png"), bbox_inches="tight", dpi=300)

    stamp.setCenter(image_size / 2, image_size / 2)
    bounds = stamp.bounds & full_image.bounds
    full_image[bounds] += stamp[bounds]

    full_image.addNoise(cn)

    # write out this image
    if config.get("pipeline", "output_type") == 'txt':
        np.savetxt(os.path.join(output_path, f"image_{config.get("pipeline", "output_suffix")}_{i}.txt"), full_image.array)

    if config.getboolean("pipeline", "do_thumbnails"):
        plt.figure()
        plt.title(f'Source {i}')
        plt.imshow(full_image.array, origin="lower")#, norm=colors.LogNorm())
        plt.savefig(os.path.join(figure_path, f"image_{i}.png"), bbox_inches="tight", dpi=300)

