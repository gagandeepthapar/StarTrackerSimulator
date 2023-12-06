"""
attitude

Class to contain all attitude-associated parameters for a given run.
Includes:
    - Right Ascension
    - Declination
    - Roll Angle
    - Associated Unit Vector Boresight (uvec)
    - Associated Quaternion (qTrue)
    - Associated rotation matrix (R_STAR)

startrackermodel
"""

import logging
import logging.config
from typing import List, Dict
import numpy as np
import pandas as pd

from data import CONSTANTS

logging.config.dictConfig(CONSTANTS.LOGGING_CONFIG)
logger = logging.getLogger(__name__)


class Attitude:
    """
    Attitude class
    """

    @staticmethod
    def ra_dec_to_uvec(right_asc: float, dec: float) -> np.ndarray:
        """
        Convert right ascension, declination into unit vector

        Inputs:
            right_asc (float)  : Right Ascension [rad]
            dec (float) : Declination [rad]

        Returns:
            np.ndarray  : unit vector in same frame as inputs
        """
        return Attitude.unit(
            np.array(
                [
                    np.cos(right_asc) * np.cos(dec),
                    np.sin(right_asc) * np.cos(dec),
                    np.sin(dec),
                ]
            )
        )

    @staticmethod
    def ra_dec_roll_to_rotm(right_asc: float, dec: float, roll: float) -> np.ndarray:
        """
        Converts full attitude in 3-angle system to unique rotation matrix

        Args:
            right_asc (float): Right Ascension [rad]
            dec (float): Declination Angle [rad]
            roll (float): Roll angle [rad]

        Returns:
            np.ndarray: 3x3 Rotation matrix to rotate from ECI into body-frame (assuming Z-axis is along boresight)
        """
        z_hat = Attitude.ra_dec_to_uvec(right_asc, dec)
        x_hat_pre = np.array(
            [np.cos(right_asc - np.pi / 2), np.sin(right_asc - np.pi / 2), 0]
        )
        x_hat = Attitude.axangxform(x_hat_pre, z_hat, roll)
        y_hat = Attitude.unit(np.cross(z_hat, x_hat))
        return np.array([x_hat, y_hat, z_hat])

    @staticmethod
    def unit(u1: np.ndarray) -> np.ndarray:
        """
        Returns unit vector of u1

        Args:
            u1 (np.ndarray): input vector of any length

        Returns:
            np.ndarray: unit vector of u1
        """
        return u1 / np.linalg.norm(u1)

    @staticmethod
    def axangxform(u1: np.ndarray, ax: np.ndarray, ang: float) -> np.ndarray:
        """
        Rodrigues' Rotation Formula to rotate vector u1 about axis ax through angle ang.
        Will enforce unit vector u1 and ax

        Args:
            u1 (np.ndarray): vector to rotate
            ax (np.ndarray): axis to rotate u1 about
            ang (float): angle to rotate u1 through [rad]
        Returns:
            np.ndarray: urot, rotated form of u1 about ax through ang
        """
        urot = (
            u1 * np.cos(ang)
            + (np.cross(ax, u1)) * np.sin(ang)
            + ax * (ax.dot(u1)) * (1 - np.cos(ang))
        )
        return Attitude.unit(urot)

    @staticmethod
    def rotm_to_quat(R: np.ndarray) -> np.ndarray:
        """
        Convert rotation matrix into quaternion

        Inputs:
            R (np.ndarray): Rotation matrix from A to B

        Returns:
            q (np.ndarray): quaternion from A to B
        """
        q12 = 1 / 4 * (1 + R[0, 0] - R[1, 1] - R[2, 2])
        q22 = 1 / 4 * (1 - R[0, 0] + R[1, 1] - R[2, 2])
        q32 = 1 / 4 * (1 - R[0, 0] - R[1, 1] + R[2, 2])
        q42 = 1 / 4 * (1 + R[0, 0] + R[1, 1] + R[2, 2])

        qmax = np.max([q12, q22, q32, q42])

        if qmax == q12:
            qtilde = (
                1
                / (4 * q12)
                * np.array(
                    [
                        4 * q12,
                        R[0, 1] + R[1, 0],
                        R[2, 0] + R[0, 2],
                        R[1, 2] - R[2, 1],
                    ]
                )
            )

        elif qmax == q22:
            qtilde = (
                1
                / (4 * q12)
                * np.array(
                    [
                        R[0, 1] + R[1, 0],
                        4 * q22,
                        R[2, 0] + R[0, 2],
                        R[1, 2] - R[2, 1],
                    ]
                )
            )

        elif qmax == q32:
            qtilde = (
                1
                / (4 * q12)
                * np.array(
                    [
                        R[0, 1] + R[1, 0],
                        R[2, 0] + R[0, 2],
                        4 * q32,
                        R[1, 2] - R[2, 1],
                    ]
                )
            )

        else:
            qtilde = (
                1
                / (4 * q12)
                * np.array(
                    [
                        R[0, 1] - R[1, 0],
                        R[2, 0] - R[0, 2],
                        R[1, 2] - R[2, 1],
                        4 * q42,
                    ]
                )
            )

        return Attitude.unit(qtilde)

    @staticmethod
    def rotm_x(theta: float) -> np.ndarray:
        """
        Return 3D rotation matrix about x axis

        Args:
            theta (float): angle to rotate through

        Returns:
            np.ndarray: 3x3 rotm about x-axis through theta
        """
        return np.array(
            [
                [1, 0, 0],
                [0, np.cos(theta), -np.sin(theta)],
                [0, np.sin(theta), np.cos(theta)],
            ]
        )

    @staticmethod
    def rotm_y(theta: float) -> np.ndarray:
        """
        Return 3D rotation matrix about y axis

        Args:
            theta (float): angle to rotate through

        Returns:
            np.ndarray: 3x3 rotm about y-axis through theta
        """
        return np.array(
            [
                [np.cos(theta), 0, np.sin(theta)],
                [0, 1, 0],
                [-np.sin(theta), 0, np.cos(theta)],
            ]
        )

    @staticmethod
    def rotm_z(theta: float) -> np.ndarray:
        """
        Return 3D rotation matrix about z axis

        Args:
            theta (float): angle to rotate through

        Returns:
            np.ndarray: 3x3 rotm about z-axis through theta
        """
        return np.array(
            [
                [np.cos(theta), -np.sin(theta), 0],
                [np.sin(theta), np.cos(theta), 0],
                [0, 0, 1],
            ]
        )

    @staticmethod
    def SVD_Algorithm(
        w_set: np.ndarray, v_set: np.ndarray, *, eps: float = 1e-6, max_it: int = 50
    ) -> np.ndarray:
        """
        Implementation of SVD algorithm from Markley

        Args:
            w_set (np.ndarray): Set of unit vectors in body-frame
            v_set (np.ndarray): Set of unit vectors in inertial-frame
            eps (float)       : tolerance to stop Newton-Raphson method
            max_it (int)      : max number of iterations for N-R Method

        Returns:
            np.ndarray: quaternion from v_set to w_set to minimize residuals
        """
        a_weights = Attitude.unit(np.ones(len(w_set)))

        # compute lambda params
        B = ((w_set[:, :, None] * v_set[:, None, :]) * a_weights[:, None, None]).sum(0)

        [U, S, V] = np.linalg.svd(B)
        return Attitude.unit(Attitude.rotm_to_quat(U @ V))

    @staticmethod
    def FOAM_algorithm(
        w_set: np.ndarray, v_set: np.ndarray, *, eps: float = 1e-6, max_it: int = 50
    ) -> np.ndarray:
        """
        Implementation of FOAM algorithm from Markley

        !!INCOMPLETE!!

        Args:
            w_set (np.ndarray): Set of unit vectors in body-frame
            v_set (np.ndarray): Set of unit vectors in inertial-frame
            eps (float)       : tolerance to stop Newton-Raphson method
            max_it (int)      : max number of iterations for N-R Method

        Returns:
            np.ndarray: quaternion from v_set to w_set to minimize residuals
        """
        a_weights = Attitude.unit(np.ones(len(w_set)))

        # compute lambda params
        B = ((w_set[:, :, None] * v_set[:, None, :]) * a_weights[:, None, None]).sum(0)
        BT = B.T
        BT_det = np.linalg.det(B.T)
        B_T_adj = (np.linalg.inv(BT) * BT_det).T

        A_opt = (
            (kappa + np.linalg.norm(B)) * B + (opt_lam * B_T_adj) - (B @ B.T @ B)
        ) / zeta

        return Attitude.unit(Attitude.rotm_to_quat(A_opt))

    @staticmethod
    def quest_algorithm(
        w_set: np.ndarray, v_set: np.ndarray, *, eps: float = 1e-6, max_it: int = 50
    ) -> np.ndarray:
        """
        Implementation of QUEST algorithm from "Spacecraft Dynamics and Control", DeRuiter

        Args:
            w_set (np.ndarray): Set of unit vectors in body-frame
            v_set (np.ndarray): Set of unit vectors in inertial-frame
            eps (float)       : tolerance to stop Newton-Raphson method
            max_it (int)      : max number of iterations for N-R Method

        Returns:
            np.ndarray: quaternion from v_set to w_set to minimize residuals
        """
        a_weights = Attitude.unit(np.ones(len(w_set)))

        # compute lambda params
        B = ((w_set[:, :, None] * v_set[:, None, :]) * a_weights[:, None, None]).sum(0)
        S = B + B.T
        k_12 = np.array([B[1][2] - B[2][1], B[2][0] - B[0][2], B[0][1] - B[1][0]]).T
        k_22 = np.trace(B)

        # compute classical adjoint of S
        S_det = np.linalg.det(S)
        S_adj = (np.linalg.inv(S).T * S_det).T

        # compute intermediate values for Newton-Raphson method
        a = k_22**2 - np.trace(S_adj)
        b = k_22**2 + k_12.T @ k_12
        c = np.linalg.det(S) + k_12.T @ S @ k_12
        d = k_12.T @ S @ S @ k_12

        # newton raphson method to compute optimal lambda paramter
        e0 = 1
        e1 = e0 - (
            Attitude.__opt_lam_f(e0, a, b, c, d, k_22)
            / Attitude.__opt_lam_fp(e0, a, b, c)
        )

        err = np.abs(e1 - e0)
        it = 0
        while err > eps and it < max_it:
            e0 = e1
            e1 = e0 - (
                Attitude.__opt_lam_f(e0, a, b, c, d, k_22)
                / Attitude.__opt_lam_fp(e0, a, b, c)
            )
            err = np.abs(e0 - e1)

        opt_lam = e1

        # compute quaternion conversion parameters
        alpha = opt_lam**2 - k_22**2 + np.trace(S_adj)
        beta = opt_lam - k_22
        gamma = (opt_lam + k_22) * alpha - np.linalg.det(S)

        # compute quaternion
        X = (alpha * np.eye(3) + beta * S + (S @ S)) @ k_12
        f = 1 / np.sqrt(gamma**2 + X.T @ X)
        q13 = f * X
        q4 = f * gamma
        return Attitude.unit(np.array([*q13, q4]))

    @staticmethod
    def quat_compare(q1: np.ndarray, q2: np.ndarray) -> float:
        """
        Compare 2 quaternions and return the angular error between them

        Args:
            q1 (np.ndarray): q1
            q2 (np.ndarray): q2

        Returns:
            float: angular error [rad]
        """
        q1_conj = np.array([*(-1 * q1[0:3]), q1[3]])
        q2_q1_13 = (
            (q1_conj[3] * q2[0:3])
            + (q2[3] * q1_conj[0:3])
            + np.cross(q2[0:3], q1_conj[0:3])
        )
        q2_q1_4 = q2[3] * q1_conj[3] - q1_conj[0:3].T @ q2[0:3]
        qdiff = Attitude.unit(np.array([*q2_q1_13, q2_q1_4]))

        # return 2 * np.arccos(np.abs(qdiff[3]))
        return 2 * np.arctan2(np.sqrt(qdiff[0:3].T @ qdiff[0:3]), qdiff[3])

    @staticmethod
    def __opt_lam_f(
        lam: float, a: float, b: float, c: float, d: float, k: float
    ) -> float:
        """
        Newton Raphson Method to compute optimal lambda parameter for QUEST

        Returns:
            float: function return
        """
        A = lam**4
        B = -(a + b) * lam**2
        C = -c * lam
        D = a * b + c * k - d
        return A + B + C + D

    @staticmethod
    def __opt_lam_fp(lam: float, a: float, b: float, c: float) -> float:
        """
        Newton Raphson Method to compute optimal lambda parameter for QUEST

        Returns:
            float: function derivative return
        """
        A = 4 * lam**3
        B = -2 * (a + b) * lam
        C = -c
        return A + B + C
