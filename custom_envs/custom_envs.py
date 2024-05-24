import os
import numpy as np
import gym
import gym.envs
import gym.envs.mujoco
import gym.envs.mujoco.hopper_v3
from gym import ActionWrapper
import xml.etree.ElementTree as ET
from scipy.stats import truncnorm

NOISE = {
    "mujoco_hopper": 0.005,
    "mujoco_walker": 0.005,
    "mujoco_pendulum": 0.2,
}

def get_custom_env(name: str, seed, **kwargs):
    name = name.lower()
    

    #############################
    ######### Hopper ############
    #############################
    if name == "mujoco_hopper-v3":
        if "perturb_param" in kwargs and "perturb_value" in kwargs:
            assert kwargs["perturb_param"] in ["mass", "friction", "noise_scale"]
            value = kwargs["perturb_value"]
            param = kwargs["perturb_param"]
        else:
            env = ActionNoiseTruncatedGaussian(gym.make("Hopper-v3"), NOISE["mujoco_hopper"])
            env.reset(seed=seed)
            env.action_space.seed(seed)
            return env

        current_dir = os.path.dirname(os.path.realpath(__file__))
        relative_path = "hopper/hopper.xml"
        path = os.path.join(current_dir, relative_path)
        with open(path, "r") as f:
            tree = ET.parse(f)
        root = tree.getroot()
        noise_scale = 1.0
        if param == "mass":
            changes = False
            for child in root:
                if child.tag == "worldbody":
                    for child2 in child:
                        if child2.tag == "body":
                            if child2.attrib["name"] == "torso":
                                for child3 in child2:
                                    if child3.tag == "geom":
                                        child3.attrib["mass"] = str(value)
                                        changes = True
            cache_name = "mass_cache"
        elif param == "friction":
            coeff = value
            changes = False
            for geom in root.iter('geom'):
                if 'friction' in geom.attrib:
                    current_friction = float(geom.get('friction'))
                    geom.set('friction', str(current_friction * coeff))
                    changes = True
            cache_name = "friction_cache"
        elif param == "noise_scale":
            changes = True
            noise_scale = value
            cache_name = "noise_cache"
        else:
            raise ValueError(f"Parameter {param} not found")

        if not changes:
            raise ValueError("No changes made to the xml file")

        # Save the updated xml to a new file
        new_path = os.path.join(current_dir, f"hopper/{cache_name}/hopper_{param}_{value}.xml")
        tree.write(new_path)

        env = ActionNoiseTruncatedGaussian(gym.make("Hopper-v3", xml_file=new_path), NOISE["mujoco_hopper"] * noise_scale)
        env.reset(seed=seed)
        env.action_space.seed(seed)
        return env

    #############################
    ######### Walker2d ##########
    #############################
    elif name == "mujoco-walker2d-v3":
        if "perturb_param" in kwargs and "perturb_value" in kwargs:
            assert kwargs["perturb_param"] in ["mass", "friction", "noise_scale"]
            value = kwargs["perturb_value"]
            param = kwargs["perturb_param"]
        else:
            env = ActionNoiseTruncatedGaussian(gym.make("Walker2d-v3"), NOISE["mujoco_walker"])
            env.reset(seed=seed)
            env.action_space.seed(seed)
            return env

        current_dir = os.path.dirname(os.path.realpath(__file__))
        relative_path = "walker2d/walker2d.xml"
        path = os.path.join(current_dir, relative_path)
        with open(path, "r") as f:
            tree = ET.parse(f)
        root = tree.getroot()
        noise_scale = 1.0

        if param == "mass":
            changes = False
            for child in root:
                if child.tag == "worldbody":
                    for child2 in child:
                        if child2.tag == "body":
                            if child2.attrib["name"] == "torso":
                                for child3 in child2:
                                    if child3.tag == "geom":
                                        child3.attrib["mass"] = str(value)
                                        changes = True
            cache_name = "mass_cache"
        elif param == "friction":
            coeff = value
            changes = False
            for geom in root.iter('geom'):
                if 'friction' in geom.attrib:
                    try:
                        current_friction = float(geom.get('friction'))
                    except ValueError:
                        continue
                    geom.set('friction', str(current_friction * coeff))
                    changes = True
            cache_name = "friction_cache"
        elif param == "noise_scale":
            changes = True
            noise_scale = value
            cache_name = "noise_cache"
        else:
            raise ValueError(f"Parameter {param} not found")

        if not changes:
            raise ValueError("No changes made to the xml file")

        # Save the updated xml to a new file
        new_path = os.path.join(current_dir, f"walker2d/{cache_name}/walker2d_{param}_{value}.xml")
        tree.write(new_path)

        env = ActionNoiseTruncatedGaussian(gym.make("Walker2d-v3", xml_file=new_path), NOISE["mujoco_walker"]*noise_scale)
        env.reset(seed=seed)
        env.action_space.seed(seed)
        return env

    #############################
    ######### Pendulum ##########
    #############################
    elif name == "mujoco_invertedpendulum-v2":
        if "perturb_param" in kwargs and "perturb_value" in kwargs:
            assert kwargs["perturb_param"] in ["mass"]
            value = kwargs["perturb_value"]
            param = kwargs["perturb_param"]
        else:
            env = gym.make("InvertedPendulum-v2")
            env.reset(seed=seed)
            env.action_space.seed(seed)
            return ActionNoiseTruncatedGaussian(env, NOISE["mujoco_pendulum"])

        gym.envs.register(
            id="InvertedPendulum-v2_Custom",
            entry_point="unstable_baselines.robust_rl.rmbpo.mujoco_models.invertedPendulum_custom:InvertedPendulumCustom",
            max_episode_steps=1000,
        )
        current_dir = os.path.dirname(os.path.realpath(__file__))
        relative_path = "invertedPendulum/inverted_pendulum.xml"
        path = os.path.join(current_dir, relative_path)
        with open(path, "r") as f:
            tree = ET.parse(f)
        root = tree.getroot()
        changes = False

        if param == "mass":
            # Change the mass of the pole
            for body in root.iter('body'):
                if 'name' in body.attrib and body.get('name') == 'pole':
                    for child in body:
                        if child.tag == "geom":
                            child.attrib["mass"] = str(value)
                            changes = True
            if not changes:
                raise ValueError("No changes made to the xml file")
            cache_name = "mass_cache"
        else:
            raise ValueError(f"Parameter {param} not found")

        new_path = os.path.join(current_dir, f"invertedPendulum/{cache_name}/invertedPendulum_{param}_{value}.xml")
        tree.write(new_path)

        env = gym.make("InvertedPendulum-v2_Custom", xml_file=new_path)
        env.reset(seed=seed)
        env.action_space.seed(seed)
        
        return ActionNoiseTruncatedGaussian(env, NOISE["mujoco_pendulum"])

    else:
        raise ValueError(f"Environment {name} not found")
    

class ActionNoiseTruncatedGaussian(ActionWrapper):
    def __init__(self, env, noise_std: float = 0.2):
        self.noise_std = noise_std
        super().__init__(env)

    def action(self, action):
        ap, bp = ((self.action_space.low - action) / self.noise_std, (self.action_space.high - action) / self.noise_std)
        rv = truncnorm(ap, bp, loc=action, scale=self.noise_std)
        noisy_action = np.array([rv.rvs()])
        if len(noisy_action.shape) > 1:
            noisy_action = noisy_action.squeeze()
        return noisy_action