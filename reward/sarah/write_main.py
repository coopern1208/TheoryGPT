from dataclasses import dataclass
import os
import json
from datetime import datetime
from PDG import PDG_IDS

@dataclass
class SARAHFile:
    model_name: str
    author: str
    model: dict
    current_time: str
    output_dir: str

    gauge_groups: list[str] = []
    matter_fields: list[str] = []


def write_SARAH(model_name: str,
                author: str,
                model: dict,
                OUTPUT_DIR: str = "dataset/models",
                SM_PARTICLES: str = "sm_particles.json",
                SM_PARAMETERS: str = "sm_params.json"
                ):
    
    pass
    
    



if __name__ == "__main__":

    from sm import SM_full_model
    model_name = "SM"
    author = "AI"
    file = SARAHFile(model_name, author, SM_full_model)

    print(file.model)