"""The observation and model inputs required by the released SONIC v1.1 policy."""

V1_1_TOKENIZER_TERMS = (
    "encoder_index",
    "command_multi_future_nonflat",
    "command_z_multi_future_nonflat",
    "motion_anchor_ori_heading_mf_nonflat",
    "command_multi_future_lower_body",
    "vr_3point_local_target",
    "vr_3point_local_orn_target",
    "motion_anchor_ori_heading",
    "command_z",
    "smpl_joints_multi_future_local_nonflat",
    "smpl_root_ori_heading_multi_future",
    "joint_pos_multi_future_wrist_for_smpl",
)

V1_1_ENCODER_INPUTS = {
    "g1": ("command_multi_future_nonflat", "motion_anchor_ori_heading_mf_nonflat"),
    "teleop": (
        "command_multi_future_lower_body", "vr_3point_local_target",
        "vr_3point_local_orn_target", "motion_anchor_ori_heading",
    ),
    "smpl": (
        "smpl_joints_multi_future_local_nonflat", "smpl_root_ori_heading_multi_future",
        "joint_pos_multi_future_wrist_for_smpl",
    ),
}


def assert_v1_1_contract(config):
    """Fail early if a composed or checkpoint config changes v1.1 policy inputs."""
    tokenizer = config.manager_env.observations.tokenizer
    terms = tuple(name for name in tokenizer if name not in ("_target_", "enable_corruption", "concatenate_terms"))
    if terms != V1_1_TOKENIZER_TERMS:
        raise ValueError(f"SONIC v1.1 tokenizer terms/order differ: {terms!r}")

    actor = config.algo.config.actor.backbone
    for encoder, expected in V1_1_ENCODER_INPUTS.items():
        actual = tuple(actor.encoders[encoder].inputs)
        if actual != expected:
            raise ValueError(f"SONIC v1.1 {encoder} encoder inputs differ: {actual!r}")
    actual = tuple(actor.decoders.g1_kin.outputs)
    if actual != V1_1_ENCODER_INPUTS["g1"]:
        raise ValueError(f"SONIC v1.1 G1 reconstruction outputs differ: {actual!r}")

    motion = config.manager_env.commands.motion
    if (motion.num_future_frames, motion.smpl_num_future_frames) != (10, 10):
        raise ValueError("SONIC v1.1 requires 10 G1 and 10 SMPL future frames")
    if (motion.dt_future_ref_frames, motion.smpl_dt_future_ref_frames) != (0.1, 0.02):
        raise ValueError("SONIC v1.1 future reference spacing differs from the released policy")
    if not motion.randomize_wrist_poses:
        raise ValueError("SONIC v1.1 training requires wrist-pose augmentation")
