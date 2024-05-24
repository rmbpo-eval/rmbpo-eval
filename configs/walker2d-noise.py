# PERTURB_VALUES = [1, 1.5, 2, 2.5, 3, 3.5, 4, 5, 6, 7, 8, 9, 10]
# PERTURB_PARAM = "mass"

PERTURB_VALUES = [1, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55]
PERTURB_PARAM = "noise_scale"

# PERTURB_PARAM = "friction"
# PERTURB_VALUES = [0.1, 0.25, 0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1, 1.05, 1.1, 1.15, 1.2, 1.3, 1.4, 1.5]


overwrite_args = {
  "env_name": "mujoco-walker2d-v3",
  "perturb_param": [PERTURB_PARAM]*len(PERTURB_VALUES),
  "perturb_value": PERTURB_VALUES
}