"""
Star Tracker Simulator:
    Create simulations of images and test various parts of the star tracker operations

ATTITUDE -> IMAGE

Gagandeep Thapar
"""

import logging
import logging.config
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from data import CONSTANTS
from attitude import Attitude

YBSC = "./YBSC.pkl"
MEDIA_FOLDER = "media/"

FOCAL_LENGTH = 2000  # pixels
FOCAL_ARR_X = 1024  # pixels
FOCAL_ARR_Y = 1024

logging.config.dictConfig(CONSTANTS.LOGGING_CONFIG)
logger = logging.getLogger(__name__)


class Simulator:
    def __init__(self, ra: float, dec: float, roll: float, mag: float = 10):
        """
        ra, dec, roll info in radians
        fov (full angle) in rad
        star magnitude to filter. Default to 10 (no filter)
        """
        # store camera params
        self.ra = ra
        self.dec = dec
        self.roll = roll
        halfdiag = np.sqrt((FOCAL_ARR_X**2) / 4 + (FOCAL_ARR_Y**2) / 4)
        self.half_fov = np.arctan(halfdiag / FOCAL_LENGTH)
        self.mag = mag

        # read full catalog
        self.full_catalog = pd.read_pickle(YBSC)

        # set boresight
        self.boresight: np.ndarray = Attitude.ra_dec_to_uvec(self.ra, self.dec)

        # compute rotm from eci to body
        self.R_body_eci = Attitude.ra_dec_roll_to_rotm(self.ra, self.dec, self.roll)

        # compute true quat
        self.q_true = Attitude.rotm_to_quat(self.R_body_eci)

        # filter catalog
        self.filtered_catalog = self.filter_catalog()
        logger.debug(
            f"RA: {ra * 180/np.pi :.3f}, DEC: {dec * 180 / np.pi :.3f}, ROLL: {roll * 180/np.pi: .3f} [deg]"
        )
        logger.debug(f"Boresight (ECI): {self.boresight}")
        logger.debug(f"Boresight (Body): {np.array([0,0,1])}")
        logger.debug(f"Filtered Catalog:\n{self.filtered_catalog}")

    def filter_catalog(self) -> pd.DataFrame:
        """
        filter catalog to viewable stars
        """
        # filter my magnitude
        catalog = self.full_catalog.copy(deep=True)
        catalog = catalog[catalog.v_magnitude <= self.mag]

        # get unit vector for each star in catalog
        # Eq. 32/33 in paper
        catalog["UVEC_ECI"] = catalog[["right_ascension", "declination"]].apply(
            lambda row: Attitude.ra_dec_to_uvec(row.right_ascension, row.declination),
            axis=1,
        )

        # get angular distace between boresight and each star
        catalog["ANG_DIST"] = catalog[["UVEC_ECI"]].apply(
            lambda row: np.arccos(self.boresight.dot(row.UVEC_ECI)), axis=1
        )

        # remove stars outside HALF_FOV
        catalog = catalog[catalog.ANG_DIST <= self.half_fov]

        # compute body frame
        catalog["UVEC_BODY"] = catalog[["UVEC_ECI"]].apply(
            lambda row: self.R_body_eci @ row.UVEC_ECI, axis=1
        )

        # compute optimal lambda multiplier for centroids
        # Eq. 17-20 w/o hardware errors
        catalog["LAMBDA_STAR"] = catalog[["UVEC_BODY"]].apply(
            lambda row: FOCAL_LENGTH / row.UVEC_BODY[2], axis=1
        )

        # get centroids
        # Eq. 17-20
        catalog["IMAGE_CENTROID"] = catalog[["UVEC_BODY", "LAMBDA_STAR"]].apply(
            lambda row: (
                row.LAMBDA_STAR * row.UVEC_BODY - np.array([0, 0, FOCAL_LENGTH])
            )[0:2],
            axis=1,
        )

        # check which stars are outside image
        catalog["IN_IMAGE"] = catalog[["IMAGE_CENTROID"]].apply(
            lambda row: np.abs(row.IMAGE_CENTROID[0]) <= FOCAL_ARR_X / 2
            and np.abs(row.IMAGE_CENTROID[1]) <= FOCAL_ARR_Y / 2,
            axis=1,
        )

        # remove out of image stars
        catalog = catalog[catalog.IN_IMAGE]

        return catalog

    def get_image(self, show_flag: bool = True, save_flag: bool = False):
        image = np.zeros((FOCAL_ARR_X, FOCAL_ARR_Y))

        for _, row in self.filtered_catalog.iterrows():
            # center image about corner; make integers to index into matrix
            ctr = np.round(
                row.IMAGE_CENTROID + np.array([FOCAL_ARR_X / 2, FOCAL_ARR_Y / 2])
            )
            ctr: np.ndarray = ctr.astype(int)

            # make stars larger artificially
            surrounding_area = np.array([-3, -2, -1, 0, 1, 2, 3])
            star_area_x, star_area_y = np.meshgrid(surrounding_area, surrounding_area)

            star_area_x = star_area_x + ctr[0]
            star_area_y = star_area_y + ctr[1]

            # bounds checking
            star_area_x[star_area_x < 0] = 0
            star_area_x[star_area_x >= FOCAL_ARR_X] = FOCAL_ARR_X - 1
            star_area_y[star_area_y < 0] = 0
            star_area_y[star_area_y >= FOCAL_ARR_Y] = FOCAL_ARR_Y - 1

            image[star_area_x, star_area_y] = 100

        # plot figure
        fig = plt.figure()
        ax = fig.add_subplot()
        ax.imshow(image, cmap="gray")
        ax.axis("off")

        if save_flag:
            # save image and associated data as csv
            now = datetime.now()
            dtnow = now.strftime("%Y_%m_%d_%H_%M_%S")
            quatstr = f"{self.q_true[0]:.3f}_{self.q_true[1]:.3f}_{self.q_true[2]:.3f}_{self.q_true[3]:.3f}"

            fname = MEDIA_FOLDER + dtnow + "_" + quatstr
            plt.savefig(fname + ".png")
            self.filtered_catalog.to_csv(fname + ".csv")

        if show_flag:
            # plot image
            plt.show()

        return


if __name__ == "__main__":
    # generate random right ascension, declination, roll: notice the bounds
    ra, dec, roll = (
        np.random.uniform(0, 2 * np.pi),
        np.random.uniform(-np.pi / 2, np.pi / 2),
        np.random.uniform(0, 2 * np.pi),
    )

    # Big dipper attitude
    ra = np.pi / 180 * (12 + 15 / 60 + 25.56 / 3600 + 5)
    dec = np.pi / 180 * (57 + 1 / 60 + 57.4156 / 3600 + 2)
    roll = np.pi / 180 * -30

    """
    run simulator to get simulated data.
    This is what the star tracker would theoretically see!
    """
    sim = Simulator(ra, dec, roll, 4.5)
    sim.get_image(True, True)
