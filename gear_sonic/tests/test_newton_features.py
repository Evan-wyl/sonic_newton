"""Regression checks for selection and physical data conventions."""

from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

from gear_sonic.envs.manager_env.mdp.events import randomize_joint_default_pos, randomize_rigid_body_com
from gear_sonic.envs.physics_adapter import get_linear_jacobian


class NewtonFeatureTests(unittest.TestCase):
    def test_joint_offsets_cartesian_selection_and_repeat(self):
        original = torch.arange(12, dtype=torch.float32).reshape(3,4)
        asset = SimpleNamespace(num_instances=3,num_joints=4,device="cpu",
            joint_names=["a","b","c","d"],data=SimpleNamespace(default_joint_pos=original.clone()))
        action = SimpleNamespace(_joint_names=["d","b","a","c"],_offset=original[:,[3,1,0,2]].clone())
        env = SimpleNamespace(scene={"robot":asset},action_manager=SimpleNamespace(get_term=lambda name:action))
        selected=torch.tensor([2,0]);cfg=SimpleNamespace(name="robot",joint_ids=[3,1])
        for _ in range(2):randomize_joint_default_pos(env,selected,cfg,(.125,.125),"add")
        expected=original.clone();expected[selected[:,None],torch.tensor([3,1])]+=.125
        torch.testing.assert_close(asset.data.default_joint_pos,expected)
        torch.testing.assert_close(action._offset,expected[:,[3,1,0,2]])

    def test_jacobian_column_order_and_offset_point(self):
        full=torch.zeros(2,2,6,8)
        full[:,1,0,6]=1.;full[:,1,5,7]=2.
        asset=SimpleNamespace(num_base_dofs=6,is_fixed_base=False,
            body_names=["root","wrist"],joint_names=["a","b"],
            data=SimpleNamespace(body_link_jacobian_w=full,
                body_quat_w=torch.tensor([[[0.,0.,0.,1.],[0.,0.,0.,1.]]]).expand(2,-1,-1)))
        jac=get_linear_jacobian(asset,"wrist",["b","a"],point_offset=[1.,0.,0.])
        torch.testing.assert_close(jac,torch.tensor([[[0.,1.],[2.,0.],[0.,0.]]]).expand(2,-1,-1))

    def test_com_offsets_subset_and_repeat(self):
        original = torch.arange(24, dtype=torch.float32).reshape(2,4,3)
        asset = SimpleNamespace(num_instances=2, num_bodies=4, device="cpu",
            data=SimpleNamespace(body_com_pos_b=original.clone()))
        env = SimpleNamespace(scene={"robot":asset})
        cfg = SimpleNamespace(name="robot", body_ids=[3,1])
        def write(env, asset, positions, *, body_ids, env_ids):
            asset.data.body_com_pos_b[env_ids[:,None],body_ids] = positions
        with patch("gear_sonic.envs.physics_adapter.set_body_com_startup", side_effect=write):
            for _ in range(2):
                randomize_rigid_body_com(env, torch.tensor([1]), {"x":(.02,.02),"z":(-.03,-.03)}, cfg)
        expected = original.clone()
        expected[1,[3,1]] += torch.tensor([.02,0.,-.03])
        torch.testing.assert_close(asset.data.body_com_pos_b, expected)


if __name__ == "__main__":
    unittest.main()
