"""Tests for TyGrit.kinematics.fetch — forward kinematics and camera pose."""

import numpy as np
import pytest

from TyGrit.kinematics.fetch.camera import compute_camera_pose
from TyGrit.kinematics.fetch.fk_numpy import forward_kinematics
from TyGrit.types.geometry import SE2Pose
from TyGrit.types.robot import RobotState
from TyGrit.utils.transforms import se2_to_matrix

torch = pytest.importorskip("torch")


EXPECTED_LINKS = {
    "base_link",
    "torso_fixed_link",
    "torso_lift_link",
    "head_pan_link",
    "head_tilt_link",
    "shoulder_pan_link",
    "shoulder_lift_link",
    "upperarm_roll_link",
    "elbow_flex_link",
    "forearm_roll_link",
    "wrist_flex_link",
    "wrist_roll_link",
    "gripper_link",
    "r_gripper_finger_link",
    "l_gripper_finger_link",
}


class TestForwardKinematics:
    """Test the FK function returns valid transforms for all links."""

    def test_all_links_present_at_zero_config(self):
        joints = np.zeros(10)
        poses = forward_kinematics(joints)
        assert set(poses.keys()) == EXPECTED_LINKS

    def test_all_poses_are_4x4(self):
        joints = np.zeros(10)
        poses = forward_kinematics(joints)
        for name, T in poses.items():
            assert T.shape == (4, 4), f"{name} has shape {T.shape}"

    def test_all_poses_are_valid_transforms(self):
        """Rotation part should be orthonormal, bottom row [0,0,0,1]."""
        joints = np.zeros(10)
        poses = forward_kinematics(joints)
        for name, T in poses.items():
            # Bottom row
            np.testing.assert_allclose(T[3, :], [0, 0, 0, 1], atol=1e-12, err_msg=name)
            # Rotation orthonormality: R^T R ≈ I
            R = T[:3, :3]
            np.testing.assert_allclose(R.T @ R, np.eye(3), atol=1e-10, err_msg=name)

    def test_base_link_is_identity(self):
        poses = forward_kinematics(np.zeros(10))
        np.testing.assert_allclose(poses["base_link"], np.eye(4), atol=1e-12)

    def test_gripper_above_ground_at_zero(self):
        """At zero config, the gripper should be above the ground plane."""
        poses = forward_kinematics(np.zeros(10))
        z = poses["gripper_link"][2, 3]
        assert z > 0.3, f"Gripper z={z} is too low"

    def test_torso_lift_changes_height(self):
        """Raising the torso should raise all arm links."""
        low = forward_kinematics(np.array([0.0, 0, 0, 0, 0, 0, 0, 0, 0, 0]))
        high = forward_kinematics(np.array([0.3, 0, 0, 0, 0, 0, 0, 0, 0, 0]))

        z_low = low["gripper_link"][2, 3]
        z_high = high["gripper_link"][2, 3]
        assert z_high > z_low + 0.2, f"high={z_high}, low={z_low}"

    def test_shoulder_pan_rotates_in_xy(self):
        """Shoulder pan should rotate the arm around Z, changing X/Y but not Z."""
        base = forward_kinematics(np.array([0.2, 0, 0, 0, 0, 0, 0, 0, 0, 0]))
        rotated = forward_kinematics(np.array([0.2, np.pi / 2, 0, 0, 0, 0, 0, 0, 0, 0]))

        ee_base = base["gripper_link"][:3, 3]
        ee_rot = rotated["gripper_link"][:3, 3]

        # Z should be approximately the same
        np.testing.assert_allclose(ee_base[2], ee_rot[2], atol=0.01)
        # XY should differ significantly
        assert not np.allclose(ee_base[:2], ee_rot[:2], atol=0.05)

    def test_head_pan_does_not_affect_arm(self):
        """Head pan should not change any arm link pose."""
        no_head = forward_kinematics(np.array([0.2, 0, 0, 0, 0, 0, 0, 0, 0, 0]))
        with_head = forward_kinematics(np.array([0.2, 0, 0, 0, 0, 0, 0, 0, 0.5, 0]))

        for link in ["shoulder_pan_link", "gripper_link"]:
            np.testing.assert_allclose(
                no_head[link], with_head[link], atol=1e-12, err_msg=link
            )

    def test_head_tilt_changes_head_but_not_arm(self):
        no_tilt = forward_kinematics(np.array([0.2, 0, 0, 0, 0, 0, 0, 0, 0, 0]))
        with_tilt = forward_kinematics(np.array([0.2, 0, 0, 0, 0, 0, 0, 0, 0, 0.5]))

        # Head tilt link should differ
        assert not np.allclose(
            no_tilt["head_tilt_link"], with_tilt["head_tilt_link"], atol=0.01
        )
        # Gripper should be identical
        np.testing.assert_allclose(
            no_tilt["gripper_link"], with_tilt["gripper_link"], atol=1e-12
        )

    def test_gripper_reachable_distance(self):
        """Gripper at zero config should be at a reasonable distance from base."""
        poses = forward_kinematics(np.zeros(10))
        ee_pos = poses["gripper_link"][:3, 3]
        dist = np.linalg.norm(ee_pos[:2])  # XY distance
        # Fetch arm reach is roughly 0.4-1.1m from base
        assert 0.3 < dist < 1.5, f"XY distance from base: {dist}"

    def test_finger_links_symmetric(self):
        """Left and right finger links should be symmetric about gripper Y=0."""
        poses = forward_kinematics(np.zeros(10))
        r_pos = poses["r_gripper_finger_link"][:3, 3]
        l_pos = poses["l_gripper_finger_link"][:3, 3]
        g_pos = poses["gripper_link"][:3, 3]

        # Both should be equidistant from gripper in the gripper's local Y axis
        r_diff = np.linalg.norm(r_pos - g_pos)
        l_diff = np.linalg.norm(l_pos - g_pos)
        np.testing.assert_allclose(r_diff, l_diff, atol=1e-10)


class TestBatchForwardKinematics:
    """Test the batch (torch) FK against the scalar (numpy) FK."""

    @pytest.fixture(autouse=True)
    def _import_batch_fk(self):
        from TyGrit.kinematics.fetch.fk_torch import batch_forward_kinematics

        self.batch_fk = batch_forward_kinematics

    def test_all_15_links_present(self):
        joints = torch.zeros(1, 10)
        poses = self.batch_fk(joints)
        assert set(poses.keys()) == EXPECTED_LINKS

    def test_all_shapes_are_B44(self):
        B = 5
        joints = torch.zeros(B, 10)
        poses = self.batch_fk(joints)
        for name, T in poses.items():
            assert T.shape == (B, 4, 4), f"{name} has shape {T.shape}"

    def test_b1_works(self):
        joints = torch.zeros(1, 10)
        poses = self.batch_fk(joints)
        for name, T in poses.items():
            assert T.shape == (1, 4, 4), f"{name} has shape {T.shape}"

    def test_matches_numpy_fk_zero_config(self):
        joints_np = np.zeros(10)
        joints_torch = torch.zeros(1, 10)

        np_poses = forward_kinematics(joints_np)
        torch_poses = self.batch_fk(joints_torch)

        for name in EXPECTED_LINKS:
            np.testing.assert_allclose(
                torch_poses[name][0].numpy(),
                np_poses[name],
                atol=1e-5,
                err_msg=f"{name} mismatch at zero config",
            )

    def test_matches_numpy_fk_random_configs(self):
        rng = np.random.default_rng(42)
        B = 16
        configs = rng.uniform(-1.0, 1.0, size=(B, 10)).astype(np.float32)
        torch_poses = self.batch_fk(torch.from_numpy(configs))

        for i in range(B):
            np_poses = forward_kinematics(configs[i].astype(np.float64))
            for name in EXPECTED_LINKS:
                np.testing.assert_allclose(
                    torch_poses[name][i].numpy(),
                    np_poses[name].astype(np.float32),
                    atol=1e-5,
                    err_msg=f"{name} mismatch at config {i}",
                )

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="No CUDA GPU")
    def test_gpu(self):
        joints = torch.zeros(4, 10, device="cuda")
        poses = self.batch_fk(joints)
        for name, T in poses.items():
            assert T.device.type == "cuda", f"{name} not on CUDA"
            assert T.shape == (4, 4, 4), f"{name} has shape {T.shape}"


# ── Ground-truth fixtures from ManiSkill3/SAPIEN ────────────────────────────
# Generated by scripts/verify_fk_camera.py: for each joint config, SAPIEN
# world-frame link poses were recorded and converted to base_link frame
# (dividing out the base SE2 pose).  cam2world_cv was derived from
# cam2world_gl @ diag(1, -1, -1, 1).

_MANISKILL_FIXTURES = [
    {
        "name": "zeros",
        "fk_joints": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        "base_pose": [0.0, 0.0, 8.94069671630857e-08],
        "link_poses": {
            "head_tilt_link": [
                [1.0, 0.0, 2.384e-07, 0.10878008604049683],
                [0.0, 1.0, 0.0, 5.984e-08],
                [-2.384e-07, 0.0, 1.0, 1.0384304523468018],
                [0.0, 0.0, 0.0, 1.0],
            ],
            "head_camera_link": [
                [1.0, 0.0, 2.384e-07, 0.1637800931930542],
                [0.0, 1.0, 0.0, 5.960e-08],
                [-2.384e-07, 0.0, 1.0, 1.0609303712844849],
                [0.0, 0.0, 0.0, 1.0],
            ],
            "gripper_link": [
                [1.0, 2.738e-07, 1.919e-07, 1.128100037574768],
                [-2.738e-07, 1.0, -1.192e-07, -6.071e-08],
                [-1.919e-07, 1.192e-07, 1.0, 0.7860098481178284],
                [0.0, 0.0, 0.0, 1.0],
            ],
        },
        "cam2world_cv": [
            [0.0, -2.384e-07, 1.0, 0.1637800931930542],
            [-1.0, 0.0, 0.0, 5.960e-08],
            [0.0, -1.0, -2.384e-07, 1.0609303712844849],
            [0.0, 0.0, 0.0, 1.0],
        ],
    },
    {
        "name": "head_pan_only",
        "fk_joints": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5, 0.0],
        "base_pose": [0.0, 0.0, 8.94069671630857e-08],
        "link_poses": {
            "head_tilt_link": [
                [
                    0.8775825500488281,
                    -0.4794255495071411,
                    7.895e-08,
                    0.09133186936378479,
                ],
                [
                    0.4794255495071411,
                    0.8775825500488281,
                    1.124e-07,
                    0.06833262741565704,
                ],
                [-1.232e-07, -6.082e-08, 1.0, 1.0384306907653809],
                [0.0, 0.0, 0.0, 1.0],
            ],
            "head_camera_link": [
                [
                    0.8775825500488281,
                    -0.4794255495071411,
                    7.895e-08,
                    0.13959893584251404,
                ],
                [
                    0.4794255495071411,
                    0.8775825500488281,
                    1.124e-07,
                    0.09470103681087494,
                ],
                [-1.232e-07, -6.082e-08, 1.0, 1.060930609703064],
                [0.0, 0.0, 0.0, 1.0],
            ],
            "gripper_link": [
                [1.0, 2.738e-07, 1.919e-07, 1.128100037574768],
                [-2.738e-07, 1.0, -1.192e-07, -6.071e-08],
                [-1.919e-07, 1.192e-07, 1.0, 0.7860098481178284],
                [0.0, 0.0, 0.0, 1.0],
            ],
        },
        "cam2world_cv": [
            [0.4794254899024963, -5.960e-08, 0.8775826692581177, 0.13959893584251404],
            [-0.8775826692581177, 0.0, 0.4794256389141083, 0.09470103681087494],
            [5.960e-08, -1.0000001192092896, 0.0, 1.060930609703064],
            [0.0, 0.0, 0.0, 1.0],
        ],
    },
    {
        "name": "pan_and_tilt",
        "fk_joints": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.8, 0.6],
        "base_pose": [0.0, 0.0, 8.94069671630857e-08],
        "link_poses": {
            "head_tilt_link": [
                [
                    0.5750165581703186,
                    0.7173560857772827,
                    0.3933907151222229,
                    0.06555163860321045,
                ],
                [
                    -0.5920593738555908,
                    0.6967067122459412,
                    -0.4050498604774475,
                    -0.10224469751119614,
                ],
                [
                    -0.5646429061889648,
                    -2.831e-07,
                    0.8253353238105774,
                    1.0384304523468018,
                ],
                [0.0, 0.0, 0.0, 1.0],
            ],
            "head_camera_link": [
                [
                    0.5750165581703186,
                    0.7173560857772827,
                    0.3933907151222229,
                    0.10602884739637375,
                ],
                [
                    -0.5920593738555908,
                    0.6967067122459412,
                    -0.4050498604774475,
                    -0.1439215987920761,
                ],
                [
                    -0.5646429061889648,
                    -2.831e-07,
                    0.8253353238105774,
                    1.0259450674057007,
                ],
                [0.0, 0.0, 0.0, 1.0],
            ],
            "gripper_link": [
                [1.0, 2.738e-07, 1.919e-07, 1.128100037574768],
                [-2.738e-07, 1.0, -1.192e-07, -6.071e-08],
                [-1.919e-07, 1.192e-07, 1.0, 0.7860098481178284],
                [0.0, 0.0, 0.0, 1.0],
            ],
        },
        "cam2world_cv": [
            [
                -0.7173562049865723,
                -0.3933906555175781,
                0.5750166177749634,
                0.10602884739637375,
            ],
            [
                -0.6967067718505859,
                0.4050499200820923,
                -0.5920594930648804,
                -0.1439215987920761,
            ],
            [2.980e-07, -0.8253353834152222, -0.5646428465843201, 1.0259450674057007],
            [0.0, 0.0, 0.0, 1.0],
        ],
    },
    {
        "name": "torso_and_head",
        "fk_joints": [0.3, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.3, -0.5],
        "base_pose": [0.0, 0.0, 8.94069671630857e-08],
        "link_poses": {
            "head_tilt_link": [
                [
                    0.8383866548538208,
                    -0.29552018642425537,
                    -0.4580126702785492,
                    0.10241411626338959,
                ],
                [
                    0.25934338569641113,
                    0.9553365111351013,
                    -0.14167991280555725,
                    0.04212062060832977,
                ],
                [0.4794255197048187, 0.0, 0.8775825500488281, 1.338430643081665],
                [0.0, 0.0, 0.0, 1.0],
            ],
            "head_camera_link": [
                [
                    0.8383866548538208,
                    -0.29552018642425537,
                    -0.4580127000808716,
                    0.13822010159492493,
                ],
                [
                    0.25934338569641113,
                    0.9553365111351013,
                    -0.14167991280555725,
                    0.053196705877780914,
                ],
                [0.4794255495071411, 0.0, 0.8775825500488281, 1.3845446109771729],
                [0.0, 0.0, 0.0, 1.0],
            ],
            "gripper_link": [
                [1.0, 2.738e-07, 1.919e-07, 1.128100037574768],
                [-2.738e-07, 1.0, -1.192e-07, -5.326e-08],
                [-1.919e-07, 1.192e-07, 1.0, 1.0860100984573364],
                [0.0, 0.0, 0.0, 1.0],
            ],
        },
        "cam2world_cv": [
            [
                0.29552018642425537,
                0.4580126404762268,
                0.8383866548538208,
                0.13822010159492493,
            ],
            [
                -0.9553365111351013,
                0.14167988300323486,
                0.25934335589408875,
                0.053196705877780914,
            ],
            [0.0, -0.8775825500488281, 0.47942543029785156, 1.3845446109771729],
            [0.0, 0.0, 0.0, 1.0],
        ],
    },
    {
        "name": "all_nonzero",
        "fk_joints": [0.15, 0.5, -0.3, 0.2, 1.0, -0.5, 0.8, -1.0, -0.4, 0.7],
        "base_pose": [0.0, 0.0, 8.94069671630857e-08],
        "link_poses": {
            "head_tilt_link": [
                [
                    0.7044662237167358,
                    0.3894183039665222,
                    0.5933639407157898,
                    0.0975288599729538,
                ],
                [
                    -0.29784348607063293,
                    0.9210610389709473,
                    -0.2508701980113983,
                    -0.05550369247794151,
                ],
                [-0.6442179083824158, 1.490e-08, 0.7648420929908752, 1.188430666923523],
                [0.0, 0.0, 0.0, 1.0],
            ],
            "head_camera_link": [
                [
                    0.7044662237167358,
                    0.3894183039665222,
                    0.5933639407157898,
                    0.14962521195411682,
                ],
                [
                    -0.29784348607063293,
                    0.9210610389709473,
                    -0.2508701980113983,
                    -0.0775296688079834,
                ],
                [
                    -0.6442179083824158,
                    1.490e-08,
                    0.7648420929908752,
                    1.1702076196670532,
                ],
                [0.0, 0.0, 0.0, 1.0],
            ],
            "gripper_link": [
                [
                    0.19801592826843262,
                    -0.9341215491294861,
                    -0.2969960570335388,
                    0.6794529557228088,
                ],
                [
                    -0.06618457287549973,
                    -0.3150460720062256,
                    0.9467658400535583,
                    0.361422598361969,
                ],
                [
                    -0.9779617786407471,
                    -0.1678181141614914,
                    -0.12420856952667236,
                    0.539839506149292,
                ],
                [0.0, 0.0, 0.0, 1.0],
            ],
        },
        "cam2world_cv": [
            [
                -0.3894181251525879,
                -0.593363881111145,
                0.7044662237167358,
                0.14962521195411682,
            ],
            [
                -0.9210610389709473,
                0.25087010860443115,
                -0.29784345626831055,
                -0.0775296688079834,
            ],
            [5.960e-08, -0.7648420333862305, -0.644217848777771, 1.1702076196670532],
            [0.0, 0.0, 0.0, 1.0],
        ],
    },
]


class TestFKAgainstManiSkill:
    """FK and camera pose verified against ManiSkill3/SAPIEN ground truth.

    Fixture data was recorded from SAPIEN link poses for several joint
    configurations (see ``scripts/verify_fk_camera.py``).  All link poses
    are in world frame; our FK outputs base_link frame, so we apply the
    base SE2 transform before comparing.
    """

    @pytest.fixture(params=_MANISKILL_FIXTURES, ids=lambda f: f["name"])
    def fixture(self, request):
        return request.param

    # ── helpers ───────────────────────────────────────────────────────

    @staticmethod
    def _world_pose(T_world_base: np.ndarray, T_base_link: np.ndarray) -> np.ndarray:
        return T_world_base @ T_base_link

    # ── link-pose tests ──────────────────────────────────────────────

    def test_head_tilt_link(self, fixture):
        joints = np.array(fixture["fk_joints"])
        bx, by, bth = fixture["base_pose"]
        T_world_base = se2_to_matrix(bx, by, bth)
        T_expected = np.array(fixture["link_poses"]["head_tilt_link"])

        poses = forward_kinematics(joints)
        T_world = self._world_pose(T_world_base, poses["head_tilt_link"])
        np.testing.assert_allclose(T_world, T_expected, atol=1e-4)

    def test_head_camera_link(self, fixture):
        joints = np.array(fixture["fk_joints"])
        bx, by, bth = fixture["base_pose"]
        T_world_base = se2_to_matrix(bx, by, bth)
        T_expected = np.array(fixture["link_poses"]["head_camera_link"])

        poses = forward_kinematics(joints)
        T_tilt = poses["head_tilt_link"]
        T_cam_offset = np.eye(4)
        T_cam_offset[:3, 3] = [0.055, 0.0, 0.0225]
        T_world = self._world_pose(T_world_base, T_tilt @ T_cam_offset)
        np.testing.assert_allclose(T_world, T_expected, atol=1e-4)

    def test_gripper_link(self, fixture):
        joints = np.array(fixture["fk_joints"])
        bx, by, bth = fixture["base_pose"]
        T_world_base = se2_to_matrix(bx, by, bth)
        T_expected = np.array(fixture["link_poses"]["gripper_link"])

        poses = forward_kinematics(joints)
        T_world = self._world_pose(T_world_base, poses["gripper_link"])
        np.testing.assert_allclose(T_world, T_expected, atol=1e-4)

    # ── camera pose test ─────────────────────────────────────────────

    def test_compute_camera_pose(self, fixture):
        """compute_camera_pose() matches ManiSkill cam2world_cv."""
        joints = fixture["fk_joints"]
        bx, by, bth = fixture["base_pose"]
        planning = tuple(joints[:8])
        head = tuple(joints[8:])

        state = RobotState(
            base_pose=SE2Pose(x=bx, y=by, theta=bth),
            planning_joints=planning,
            head_joints=head,
        )
        cam2world_cv = compute_camera_pose(state)
        expected = np.array(fixture["cam2world_cv"])
        np.testing.assert_allclose(cam2world_cv, expected, atol=1e-4)
