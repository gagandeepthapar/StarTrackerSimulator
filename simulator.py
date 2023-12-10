"""
Star Tracker Simulator:
    Create simulations of images and test various parts of the star tracker operations

ATTITUDE -> IMAGE

Gagandeep Thapar
"""

import logging
import logging.config
from datetime import datetime
from typing import Tuple
import argparse

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
    def __init__(self, ra: np.ndarray, dec: np.ndarray, roll: np.ndarray, mag: float):
        self.sim_data = pd.DataFrame({"BS_RA": ra, "BS_DEC": dec, "BS_ROLL": roll})
        self.mag_thresh = mag

        self.sim_data: pd.DataFrame

        # read full catalog
        self.full_catalog = pd.read_pickle(YBSC)

        # Eqn 32
        self.full_catalog["UVEC_ECI"] = self.full_catalog[
            ["right_ascension", "declination"]
        ].apply(
            lambda row: Attitude.ra_dec_to_uvec(row.right_ascension, row.declination),
            axis=1,
        )

        # Eqn 33
        self.sim_data["BORESIGHT_ECI"] = self.sim_data[["BS_RA", "BS_DEC"]].apply(
            lambda row: Attitude.ra_dec_to_uvec(row.BS_RA, row.BS_DEC),
            axis=1,
        )

        # Eqn 35
        halfdiag = np.sqrt(FOCAL_ARR_X**2 + FOCAL_ARR_Y**2) / 2
        self.halffov = np.arctan(halfdiag / FOCAL_LENGTH)

        # Eqn 37-41
        self.sim_data["R_BODY_ECI"] = self.sim_data[
            ["BS_RA", "BS_DEC", "BS_ROLL"]
        ].apply(
            lambda row: Attitude.ra_dec_roll_to_rotm(
                row.BS_RA, row.BS_DEC, row.BS_ROLL
            ),
            axis=1,
        )

        # Eqn 45-50
        self.sim_data["Q_TRUE"] = self.sim_data[["R_BODY_ECI"]].apply(
            lambda row: Attitude.rotm_to_quat(row.R_BODY_ECI),
            axis=1,
        )

        self.sim_data = self.generate_simulation_data()

    def filter_catalog(self, boresight: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Filter star catalog per simulation by angular distance of star wrt boresight
        """

        # Eqn 34
        catalog = self.full_catalog[
            self.full_catalog[["UVEC_ECI"]].apply(
                lambda row: np.arccos(boresight.dot(row.UVEC_ECI)) <= self.halffov,
                axis=1,
            )
        ]
        return (catalog.UVEC_ECI.to_numpy(), catalog.v_magnitude.to_numpy())

    def generate_simulation_data(self) -> pd.DataFrame:
        """
        Generate simulation data:
            unit vectors in body frame
            centroids on image
        """

        # Eqn 34
        self.sim_data["STAR_V_SET"], self.sim_data["STAR_MAGNITUDE"] = zip(
            *self.sim_data[["BORESIGHT_ECI"]].apply(
                lambda row: self.filter_catalog(row.BORESIGHT_ECI),
                axis=1,
            )
        )

        # previous operation stores STAR V SET in awkward format; fix
        self.sim_data["STAR_V_SET"] = self.sim_data[["STAR_V_SET"]].apply(
            lambda row: np.array([np.array([*v]) for v in row.STAR_V_SET]), axis=1
        )

        self.sim_data["DISCERNIBLE"] = self.sim_data[["STAR_MAGNITUDE"]].apply(
            lambda row: row.STAR_MAGNITUDE <= self.mag_thresh, axis=1
        )

        # remove indiscernible stars
        self.sim_data["STAR_V_SET"] = self.sim_data[
            ["STAR_V_SET", "DISCERNIBLE"]
        ].apply(lambda row: row.STAR_V_SET[row.DISCERNIBLE], axis=1)

        self.sim_data["STAR_MAGNITUDE"] = self.sim_data[
            ["STAR_MAGNITUDE", "DISCERNIBLE"]
        ].apply(lambda row: row.STAR_MAGNITUDE[row.DISCERNIBLE], axis=1)

        # Eqn 42
        self.sim_data["STAR_W_SET"] = self.sim_data[["R_BODY_ECI", "STAR_V_SET"]].apply(
            lambda row: np.array([row.R_BODY_ECI @ v for v in row.STAR_V_SET]),
            axis=1,
        )

        # Eqn 19, modified to correct for flip
        self.sim_data["LAMBDA_STAR"] = self.sim_data[["STAR_W_SET"]].apply(
            lambda row: np.array([FOCAL_LENGTH / w[2] for w in row.STAR_W_SET]),
            axis=1,
        )

        # Eqn 18/43
        self.sim_data["IMAGE_CENTROID"] = self.sim_data[
            ["LAMBDA_STAR", "STAR_W_SET"]
        ].apply(
            lambda row: np.array(
                [
                    (lam * w + np.array([0, 0, -FOCAL_LENGTH]))[0:-1]
                    for lam, w in zip(row.LAMBDA_STAR, row.STAR_W_SET)
                ]
            ),
            axis=1,
        )

        # Eqn 44
        self.sim_data["IN_IMAGE"] = self.sim_data[["IMAGE_CENTROID"]].apply(
            lambda row: np.array(
                [
                    (np.abs(c[0]) <= FOCAL_ARR_X / 2)
                    and (np.abs(c[1]) <= FOCAL_ARR_Y / 2)
                    for c in row.IMAGE_CENTROID
                ]
            ),
            axis=1,
        )

        # update all rows with visible stars
        self.sim_data["STAR_V_SET"] = self.sim_data[["STAR_V_SET", "IN_IMAGE"]].apply(
            lambda row: row.STAR_V_SET[row.IN_IMAGE], axis=1
        )

        self.sim_data["STAR_MAGNITUDE"] = self.sim_data[
            ["STAR_MAGNITUDE", "IN_IMAGE"]
        ].apply(lambda row: row.STAR_MAGNITUDE[row.IN_IMAGE], axis=1)

        self.sim_data["STAR_W_SET"] = self.sim_data[["STAR_W_SET", "IN_IMAGE"]].apply(
            lambda row: row.STAR_W_SET[row.IN_IMAGE], axis=1
        )

        self.sim_data["LAMBDA_STAR"] = self.sim_data[["LAMBDA_STAR", "IN_IMAGE"]].apply(
            lambda row: row.LAMBDA_STAR[row.IN_IMAGE], axis=1
        )

        self.sim_data["IMAGE_CENTROID"] = self.sim_data[
            ["IMAGE_CENTROID", "IN_IMAGE"]
        ].apply(lambda row: row.IMAGE_CENTROID[row.IN_IMAGE], axis=1)

        # drop intermediate columns
        self.sim_data = self.sim_data.drop(
            ["DISCERNIBLE", "IN_IMAGE"],
            axis=1,
        )

        return self.sim_data

    def get_image(self, show_flag: bool = True, save_flag: bool = False):
        image = np.zeros((FOCAL_ARR_X, FOCAL_ARR_Y, len(self.sim_data.index)))

        image_index = pd.DataFrame()
        image_index["IMAGE_INDEX"] = self.sim_data[["IMAGE_CENTROID"]].apply(
            lambda row: np.array(
                [
                    np.round(c + np.array([FOCAL_ARR_X / 2, FOCAL_ARR_Y / 2]))
                    for c in row.IMAGE_CENTROID
                ]
            ).astype(int),
            axis=1,
        )

        surrounding_area = np.array([-3, -2, -1, 0, 1, 2, 3])
        star_area_x, star_area_y = np.meshgrid(surrounding_area, surrounding_area)

        for rowidx, row in self.sim_data.iterrows():
            for simidx in range(len(row.STAR_MAGNITUDE)):
                # expand lit area
                ctrx = star_area_x + image_index.iloc[rowidx].IMAGE_INDEX[simidx, 0]
                ctry = star_area_y + image_index.iloc[rowidx].IMAGE_INDEX[simidx, 1]

                # check bounds on expanded area
                ctrx[ctrx < 0] = 0
                ctrx[ctrx >= FOCAL_ARR_X] = FOCAL_ARR_X - 1
                ctry[ctry < 0] = 0
                ctry[ctry >= FOCAL_ARR_Y] = FOCAL_ARR_Y - 1

                # plot star
                image[ctrx, ctry, rowidx] = image[ctrx, ctry, rowidx] + 100 * np.ones(
                    np.shape(image[ctrx, ctry, rowidx])
                )

        if save_flag or show_flag:
            now = datetime.now()
            dtnow = now.strftime("%Y_%m_%d_%H_%M_%S")

            for idx, row in self.sim_data.iterrows():
                fig = plt.figure()
                ax = fig.add_subplot()
                ax.imshow(image[:, :, idx], cmap="gray")
                # ax.axis("equal")
                ax.axis("off")

                if save_flag:
                    # save image and associated data as csv
                    q_true = row.Q_TRUE
                    quatstr = f"{q_true[0]:.3f}_{q_true[1]:.3f}_{q_true[2]:.3f}_{q_true[3]:.3f}"

                    fname = MEDIA_FOLDER + dtnow + "_NUM_" + str(idx) + "_" + quatstr
                    plt.savefig(fname + ".png", bbox_inches="tight", pad_inches=0)

                if show_flag:
                    plt.show()

        if save_flag:
            self.sim_data.to_csv(MEDIA_FOLDER + dtnow + ".csv")


def parse_arguments():
    parser = argparse.ArgumentParser()

    # add batch
    parser.add_argument(
        "--batch",
        metavar="",
        type=int,
        help="Batch create images and data. Default 1.",
        default=1,
    )

    parser.add_argument(
        "-ra",
        metavar="",
        type=float,
        help="Set right ascension [deg] (0, 360). Default random.",
        default=None,
    )

    parser.add_argument(
        "-dec",
        metavar="",
        type=float,
        help="Set declination [deg] (-90, 90). Default random.",
        default=None,
    )

    parser.add_argument(
        "-roll",
        metavar="",
        type=float,
        help="Set roll [deg] (0, 360). Default random.",
        default=None,
    )

    parser.add_argument("--show", help="Show generated image", action="store_true")

    parser.add_argument(
        "--save", help="Save generated image and associated CSV", action="store_true"
    )

    return parser.parse_args()


if __name__ == "__main__":
    # parse cmd arguments
    args = parse_arguments()

    # unpack arguments
    N = args.batch

    # check ra
    if args.ra is None:
        ra = np.random.uniform(0, 2 * np.pi, N)

    else:
        ra = args.ra * np.ones(N) * np.pi / 180

    # check dec
    if args.dec is None:
        dec = np.random.uniform(-np.pi / 2, np.pi / 2, N)

    else:
        dec = args.ra * np.ones(N) * np.pi / 180

    # check roll
    if args.roll is None:
        roll = np.random.uniform(0, 2 * np.pi, N)

    else:
        roll = args.roll * np.ones(N) * np.pi / 180

    # Big dipper attitude
    # ra = np.pi / 180 * (12 + 15 / 60 + 25.56 / 3600 + 5)
    # dec = np.pi / 180 * (57 + 1 / 60 + 57.4156 / 3600 + 2)
    # roll = np.pi / 180 * -30

    """
    run simulator to get simulated data.
    This is what the star tracker would theoretically see!
    """
    sim = Simulator(ra, dec, roll, 4.5)
    sim.get_image(args.show, args.save)
