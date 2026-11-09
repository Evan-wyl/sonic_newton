"""Validation-only Warp environment for the tracking metrics recorder."""

from isaaclab_experimental.envs import ManagerBasedRLEnvWarp


class SonicManagerBasedRLEnvWarp(ManagerBasedRLEnvWarp):
    """Skip a recorder observation pass that the tracking metrics do not use.

    The experimental environment computes observations before calling recorders.
    TrackingMetricsRecorder reads live task state instead. Its extra observation
    pass would advance the custom Warp history buffers before the policy pass.
    """

    def load_managers(self):
        super().load_managers()
        call_stage = self._manager_call_switch.call_stage

        def call_stage_for_tracking_recorder(*, stage, **kwargs):
            if stage == "ObservationManager_compute_no_history":
                return None
            return call_stage(stage=stage, **kwargs)

        self._manager_call_switch.call_stage = call_stage_for_tracking_recorder
