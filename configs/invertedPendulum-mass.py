PERTURBATION_VALUES = [1, 50, 100, 125, 150, 160, 162, 165, 168, 170, 175, 180, 182, 185, 187, 190, 195, 200, 210, 220]
PERTURBATION_PARAM = ["mass"]*len(PERTURBATION_VALUES)

overwrite_args = {
  "env_name": "mujoco_invertedpendulum-v2",
  "perturb_value": PERTURBATION_VALUES,
  "perturb_param": PERTURBATION_PARAM,
}
