# MJPC with fixed actions

This is MJPC, but you can specify a fixed action sequence that will replace the actions proposed by the MPC controller during the rollout.


## Selecting the planner

Set the agent_planner to use `Replace CEM`:

```xml
<numeric name="agent_planner" data="7"/>
```

## Sending the fixed action sequence

The number of steps of the rollout is defined by:

```cpp
// planning horizon
horizon_ = GetNumberOrDefault(0.5, model, "agent_horizon");

// time step
timestep_ = GetNumberOrDefault(1.0e-2, model, "agent_timestep");

// planning steps
inline constexpr int kMaxTrajectoryHorizon = 512;
steps_ = mju_max(mju_min(horizon_ / timestep_ + 1, kMaxTrajectoryHorizon), 1);
```

The fixed action sequence is sent through the `userdata` field of the mujoco state. The sequence is a vector of size `steps_ * action_dim`, where `action_dim` is the number of fixed actions.

This `userdata` will replace the actions proposed by the MPC controller during the rollout.

```cpp
// Current time step in the rollout: t
int nf = model->nuserdata / steps_;  // Number of fixed actions per step.

// Copy fixed actions from userdata to the action vector.
for (int j = 0; j < nf; ++j) {
  DataAt(actions, t * nu)[j] = data->userdata[t * nf + j];
}
// Copy the action vector to the control vector.
mju_copy(data->ctrl, DataAt(actions, t * nu), nu);
// Send the control vector to the simulator.
mj_step(model, data);
```

## Attention point
This will override the actions for actuators 0 to `nf-1`, so you should ensure that the actuators are ordered correctly in the model XML file. The remaining actuators will still be controlled by the MPC controller.