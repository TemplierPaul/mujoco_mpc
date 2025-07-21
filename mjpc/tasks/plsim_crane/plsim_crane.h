#ifndef MJPC_TASKS_PLSIM_CRANE_PLSIM_CRANE_H_
#define MJPC_TASKS_PLSIM_CRANE_PLSIM_CRANE_H_

#include <string>

#include <mujoco/mujoco.h>
#include "mjpc/task.h"

namespace mjpc {

  inline mjtNum plsim_payloadSwayAngle(double *tip_pos, double *payload_pos) {
    mjtNum d[3];
    mju_sub(d, tip_pos, payload_pos, 3);
    mjtNum hypot = mju_sqrt(d[0]*d[0] + d[1]*d[1] + d[2]*d[2]);
    mjtNum opp = mju_sqrt(d[0]*d[0] + d[1]*d[1]); // x-y plane distance, ignoring z
    return asin(opp/hypot);
  }

class PLSim_Crane : public Task {
 public:
  std::string Name() const override;
  std::string XmlPath() const override;

  class ResidualFn : public BaseResidualFn {
   public:
    explicit ResidualFn(const PLSim_Crane* task) : BaseResidualFn(task) {
    }
    // ------- Residuals for ccrane task ------
    //   Number of residuals: 3
    //     Residual (0): distance from payload x pos
    //     Residual (1): distance from payload y pos
    //     Residual (2): distance from payload z pos
    // ------------------------------------------
        
    void Residual(const mjModel* model, const mjData* data,
                  double* residual) const override;

    int modify_state_id_  = -1;
    double ptip_acc[3] = {0, 0, 0};
  };

  PLSim_Crane() : residual_(this) {}
  void TransitionLocked(mjModel* model, mjData* data) override;
  void ModifyState(const mjModel* model, State* state) override;
  void ResetLocked(const mjModel* model) override;
  

 protected:
  std::unique_ptr<mjpc::ResidualFn> ResidualLocked() const override {
    return std::make_unique<ResidualFn>(this);
  }
  ResidualFn* InternalResidual() override { return &residual_; }


 private:
  ResidualFn residual_;

  double qpos[7];
  double qvel[7];
  
  
};
}  // namespace mjpc

#endif  // MJPC_TASKS_PLSIM_CRANE_PLSIM_CRANE_H_