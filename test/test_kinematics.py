"""Tests for TyGrit.kinematics.fetch — forward kinematics."""

import numpy as np
import pytest

from TyGrit.kinematics.fetch.fk_numpy import forward_kinematics

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
