import numpy as np
from astropy.table import Table
from astropy import units as uns
import configparser as ConfigParser

from numpy.core.defchararray import add as stradd
from numpy.core.defchararray import multiply as strmultiply

arcsectorad = (1.0 * uns.arcsec).to(uns.rad).value
degtoarcsec = (1.0 * uns.deg).to(uns.arcsec).value

def load_catalogue(config):

    cat_file_name = config.get("skymodel", "catalogue_filepath")
    print("Loading catalogue from {0} ...".format(cat_file_name))
    cat = Table()
    cat_read = Table.read(cat_file_name, format="ascii")

    source_prefix = "TRECS-"
    source_name_ra = np.asarray(cat_read["lon"], dtype=str)
    source_name_dec = np.asarray(cat_read["lat"], dtype=str)

    source_prefix_arr = strmultiply(
        source_prefix, np.ones_like(source_name_ra, dtype=int)
    )
    source_l_arr = strmultiply("l", np.ones_like(source_name_ra, dtype=int))
    source_b_arr = strmultiply("b", np.ones_like(source_name_ra, dtype=int))

    source_name_pos = stradd(
        source_prefix_arr,
        stradd(
            source_l_arr,
            stradd(source_name_ra, (stradd(source_b_arr, source_name_dec))),
        ),
    )
    cat["Source_id"] = source_name_pos

    cat["ra_offset"] = cat_read["lon"]  # deg
    cat["ra_offset"].unit = "deg"

    cat["dec_offset"] = cat_read["lat"]  # deg
    cat["dec_offset"].unit = "deg"

    # convert the offsets from the *survey* centre to RA and DEC
    # cat["DEC"] = dec_survey_gs / galsim.degrees + cat["dec_offset"]
    # dec_abs_radians = cat["DEC"] * galsim.degrees / galsim.radians
    # cat["RA"] = ra_survey_gs / galsim.degrees + cat["ra_offset"] / np.cos(
    #     np.asarray(dec_abs_radians, dtype=float)
    # )

    # calculate the offsets from the centre of the *pointing*
    # cat["dec_offset"] = cat["DEC"] - dec_field_gs / galsim.degrees
    # dec_abs_radians = cat["DEC"] * galsim.degrees / galsim.radians
    # cat["ra_offset"] = (cat["RA"] - ra_field_gs / galsim.degrees) * np.cos(
    #     np.asarray(dec_abs_radians, dtype=float)
    # )

    cat["bulge_disk_amplitude_ratio"] = cat_read["bulge/disk"]

    cat["Total_flux"] = cat_read["flux"] * 1.0e-3  # Jy
    cat["Total_flux"].unit = "Jy"

    cat["Maj"] = cat_read["size"]  # arcsec
    cat["Maj"].unit = "arcsec"

    scale_radius_to_hlr = 1.6783469900166605
    cat["Maj_halflight"] = cat_read["size"] * scale_radius_to_hlr
    cat["Maj_halflight"].unit = "arcsec"

    cat["Peak_flux"] = cat["Total_flux"] / (2.0 * cat["Maj"] * arcsectorad)
    cat["Peak_flux"].unit = "Jy"

    if config.getfloat("skymodel", "constant_mod_e"):
        cat["PA"] = np.random.uniform(0., 2.*np.pi, len(cat))
        cat["mod_e"] = np.ones(len(cat)) * config.getfloat("skymodel", "constant_mod_e")

        cat["e1"] = cat["mod_e"] * np.cos(2.* cat["PA"])
        cat["e2"] = cat["mod_e"] * np.sin(2.* cat["PA"])
    else:

        cat["e1"] = cat_read["e1"]
        cat["e2"] = cat_read["e2"]
        cat["mod_e"] = np.sqrt(cat["e1"] ** 2.0 + cat["e2"] ** 2.0)

        cat["PA"] = 0.5 * np.arctan2(cat["e2"], cat["e1"])
        cat["PA"].unit = "rad"

    cat["g1_shear"] = cat_read["gamma1"]
    cat["g2_shear"] = cat_read["gamma2"]

    cat["q"] = (1.0 - cat["mod_e"] ** 2.0) / (1.0 + cat["mod_e"] ** 0.2)

    cat["Min"] = cat["Maj"] * cat["q"]
    cat["Min"].unit = "arcsec"

    cat["Min_halflight"] = cat["Maj_halflight"] * cat["q"]
    cat["Min_halflight"].unit = "arcsec"

    return cat