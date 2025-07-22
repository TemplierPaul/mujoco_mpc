#include "mjpc/tasks/moving_crane/moving_crane.h"

#include <cmath>
#include <string>

#include <mujoco/mujoco.h>
#include "mjpc/task.h"
#include "mjpc/utilities.h"
#include <absl/random/random.h>
#include <mujoco/mjtnum.h>


namespace mjpc {
std::string Moving_Crane::XmlPath() const {
  return GetModelPath("moving_crane/task.xml");
}
std::string Moving_Crane::Name() const { return "Moving_Crane"; }

void Moving_Crane::ResidualFn::Residual(const mjModel* model, const mjData* data,
                                    double* residual) const {
  int counter = 0;
  // ---------- Residual (0) ----------
  // ---------- Cart Distance to Target ----
  double* payload_pos = SensorByName(model, data, "p_pos");
  mjtNum dif[3] = {parameters_[3]-payload_pos[0], parameters_[4]-payload_pos[1], parameters_[5]-payload_pos[2]};
  residual[0] = mju_sqrt(dif[0]*dif[0] + dif[1]*dif[1] + dif[2]*dif[2]);
  counter++;

  
  // ---------- Residual (1) ----------
  // ---------- Sway of Payload ----------
  const mjtNum C = 3 * mjPI / 180;  // allow only 3 radian sway
  double* tip_pos = SensorByName(model, data, "b_tip_pos");
  payload_pos = SensorByName(model, data, "p_pos");
  mjtNum sigma_rad = mc_payloadSwayAngle(tip_pos, payload_pos);
  //mju_sub(residual + counter, sigma_rad, C, 1);
  residual[counter] = sigma_rad - C;
  counter += 1;

  // ---------- Residual (2) ----------
  // ---------- Kinetic Energy ----------
  residual[counter] = data->energy[1];
  counter += 1;

  // ---------- Residual (3) ----------
  // ---------- Control --------------    
  residual[counter] = data->ctrl[0];
  counter += 1;
  residual[counter] = data->ctrl[1];
  counter += 1;
  // "Control"
  //mju_copy(residual, data->ctrl, model->nu);
  //counter += model->nu; // increment counter
  
    // ---------- Residual (4) ----------
    // Angle between boom tip and paylaod tip
    //double* p_tip_pos = SensorByName(model, data, "p_tip");
    //mjtNum p_b_rad = mc_payloadSwayAngle(tip_pos, p_tip_pos);
    //mju_sub(residual + counter, sigma_rad, C, 1);
    //residual[counter] = p_b_rad - 5 * mjPI / 180;
    //counter += 1;

    // ---------- Linear Acceleration of Payload from userdata----------
   //mju_copy(residual, ptip_acc, 3);
  // double* p_acc = SensorByName(model, data, "p_acc");
   //mju_copy3(residual + counter, p_acc);
   //counter+=3;
  
  // ---------- Linear Velocity of Boom ----------

  //mjtNum dif2[2] = {tip_pos[0]-payload_pos[0], tip_pos[1]-payload_pos[1]};
  //residual[counter] = mju_sqrt(dif2[0]*dif2[0] + dif2[1]*dif2[1]);
  //counter++;
  
  CheckSensorDim(model, counter);

}


void Moving_Crane::ResetLocked(const mjModel* model) {
  //residual_.modify_state_id_ = ParameterIndex(model, "select_Modify State");
  //printf("modify_state = %i",residual_.modify_state_id_);
}

// -------- Transition for target --------
//   Follow GoalX by Target
// ---------------------------------------------
void Moving_Crane::TransitionLocked(mjModel* model, mjData* data) {
    // mocap[1:3] is the vicon tracker now. target is [3:6]
    // track motion platform movement
    //data->mocap_pos[0] = parameters[0];
    //data->mocap_pos[1] = parameters[1];
    //data->mocap_pos[2] = parameters[2];

    // target position update
    data->mocap_pos[3] = parameters[3];
    data->mocap_pos[4] = parameters[4];
    data->mocap_pos[5] = parameters[5];

    mju_copy(qpos, data->userdata, model->nq);
    mju_copy(qvel, data->userdata+model->nq, model->nv);
    mju_copy(residual_.ptip_acc, data->userdata+model->nq+model->nv, 3); 
    
    // test applying external force
    // 1. Get payload mass
    mjtNum mass = model->body_mass[7];
    
    // 2. Calculate force from F = m*a
    mjtNum force[3];
    mju_scl3(force, residual_.ptip_acc, mass);
        
    // 3. Get current payload position
    const mjtNum* payload_pos = data->xipos + 3*7;
        
    // 4. Create a temporary buffer for joint forces
    mjtNum* temp_qfrc = (mjtNum*)mju_malloc(sizeof(mjtNum) * model->nv);
    mju_zero(temp_qfrc, model->nv);
        
    // 5. Convert body force to joint forces (no torque applied)
    mjtNum zero_torque[3] = {0, 0, 0};
    mj_applyFT(model, data, force, zero_torque, payload_pos, 7, temp_qfrc);
        
    // 6. Add computed forces to applied forces
    mju_add(data->qfrc_applied, data->qfrc_applied, temp_qfrc, model->nv);

    //printf("acceleration force: %.4f   %.4f   %.4f  \n", temp_qfrc[0], temp_qfrc[1], temp_qfrc[2]);
    mju_free(temp_qfrc);
}

void Moving_Crane::ModifyState(const mjModel* model, State* state) {
  // sampling token
  absl::BitGen gen_;



  // set state
  state->SetPosition(model, qpos);
  state->SetVelocity(model, qvel);

}


}  // namespace mjpc
