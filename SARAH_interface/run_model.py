import os
import json
from config import config 


class ObservableComputation:
    """Python wrapper to run SARAH/SPheno to compute observables."""
    def __init__(self, 
                 model_name,
                 cfg = config
                 ):
        self.MODEL_NAME = model_name
        self.MODEL_BASE = os.path.abspath(cfg.MODEL_BASE)
        self.MODEL_PATH = os.path.join(self.MODEL_BASE, model_name)
        self.SARAH_PATH = os.path.abspath(cfg.SARAH_PATH)
        self.SPHENO_PATH = os.path.abspath(cfg.SPHENO_PATH)
        self.OBS_LIST_PATH = os.path.abspath(cfg.OBSERVABLES_PATH)

        # Input and Output Paths
        self.INPUT_PATH = os.path.join(self.MODEL_PATH, "EWSB", "SPheno", "Input_Files", f"LesHouches.in.{self.MODEL_NAME}")
        self.OUTPUT_PATH = os.path.join(self.MODEL_PATH, "Results")
        os.makedirs(self.OUTPUT_PATH, exist_ok = True)

        self.sigma_threshold = cfg.SIGMA_THRESHOLD
        self.timeout = cfg.TIMEOUT
        self.keep_log = cfg.KEEP_LOG
        self.loop_mass = cfg.LOOP_MASS
        self.include_tachyon = cfg.INCLUDE_TACHYON
        self.calc_decays = cfg.CALC_DECAYS
        self.mass_precision = cfg.MASS_PRECISION
        self.three_body_decays = cfg.THREE_BODY_DECAYS
        self.higgs_bounds = cfg.HIGGS_BOUNDS
        self.eft_higgs_coupling = cfg.EFT_HIGGS_COUPLING
        self.diphoton_width = cfg.DIPHOTON_WIDTH

    def pre_check(self):
        # Check if SPheno.m, parameters.m, and particles.m exist in the model path
        exist_spheno_m = os.path.isfile(os.path.join(self.MODEL_PATH, "SPheno.m"))
        exist_parameters_m = os.path.isfile(os.path.join(self.MODEL_PATH, "parameters.m"))
        exist_particles_m = os.path.isfile(os.path.join(self.MODEL_PATH, "particles.m"))
        exist_model_m = os.path.isfile(os.path.join(self.MODEL_PATH, f"{self.MODEL_NAME}.m"))

        missing = []
        if not exist_model_m:
            missing.append(f"{self.MODEL_NAME}.m")
        if not exist_spheno_m:
            missing.append("SPheno.m")
        if not exist_parameters_m:
            missing.append("parameters.m")
        if not exist_particles_m:
            missing.append("particles.m")

        if missing:
            missing_str = ", ".join(missing)
            raise FileNotFoundError(
                f"The following required model files are missing in {self.MODEL_PATH}: {missing_str}"
            )

    def load_observables(self):
        with open(self.OBS_LIST_PATH, "r") as file:
            self.observables = json.load(file)
        
if __name__ == "__main__":
    observable_computation = ObservableComputation(model_name="SM")
    observable_computation.pre_check()
    observable_computation.load_observables()
    print(observable_computation.observables)