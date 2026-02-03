import os
from datetime import datetime
from sarah_utils import write_parameters
from PDG import PDG_IDS
import json


def WRITE_SARAH(model_dict, 
                sm_parameters_json = "reward/SARAH/sm_params.json",
                sm_particles_json = "reward/SARAH/sm_particles.json",
                output_dir = "reward/SARAH"):    
    model_symbol = "SM"
    model_name = "Standard Model"
    author = "AI"
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    model_file = os.path.join(output_dir, f"{model_symbol}.m")
    particle_file = os.path.join(output_dir, f"particles.m")
    parameter_file = os.path.join(output_dir, f"parameters.m")
    SPheno_file = os.path.join(output_dir, f"SPheno.m")

    scalar_idx = 1
    fermion_idx = 1

    # ----------------------- Model.m -----------------------
    with open(model_file, "w") as f:
        f.write("Off[General::spell];\n")
        f.write("\n")
        f.write(f"Model`Name = \"{model_symbol}\";\n")
        f.write(f"Model`NameLaTeX = \"{model_name}\";\n")
        f.write(f"Model`Authors = \"{author}\";\n")
        f.write(f"Model`Date = \"{current_time}\";\n")
        f.write("\n")
        f.write("(*-------------------------------------------*)\n")
        f.write("(*   Particle Content*)\n")
        f.write("(*-------------------------------------------*)\n")
        f.write("\n")
        f.write("(* Gauge Groups *)\n")
        f.write("Gauge[[1]]={B,   U[1], hypercharge, g1,False};\n")
        f.write("Gauge[[2]]={WB, SU[2], left,        g2,True};\n")
        f.write("Gauge[[3]]={G,  SU[3], color,       g3,False};\n")
        f.write("\n")
        f.write("(* Matter Fields *)\n")
        f.write("\n")
        for m_id, mplt in model_dict["multiplets"].items():
            if mplt["type"] in ["CSCALAR", "RSCALAR"]:
                f.write(f"Scalar[[{scalar_idx}]]= {{}};\n")
                scalar_idx += 1
            elif mplt["type"] == "FERMION":
                f.write(f"Fermion[[{fermion_idx}]]= {{}};\n")
                fermion_idx += 1
        f.write("\n")

        
        f.write("(*----------------------------------------------*)\n")
        f.write("(*   DEFINITION                                 *)\n")
        f.write("(*----------------------------------------------*)\n")
        f.write("NameOfStates={GaugeES, EWSB};\n")
        f.write("\n")
        f.write("(* ----- Before EWSB ----- *)\n")
        f.write("\n")
        f.write("DEFINITION[GaugeES][LagrangianInput]= \n")
        f.write("{\n")
        f.write("   {LagHC, {AddHC->True}},\n")
        f.write("   {LagNoHC,{AddHC->False}}\n")
        f.write("};\n")
        f.write("\n")
        f.write(f"LagHC = ;\n")
        f.write(f"LagNoHC = ;\n")

        f.write("\n")
        f.write("(* Gauge Sector *)\n")
        f.write("\n")
        f.write("DEFINITION[EWSB][GaugeSector] = \n")
        f.write("{")
        f.write("   {{VB,VWB[3]},{VP,VZ},ZZ},\n")
        f.write("   {{VWB[1],VWB[2]},{VWp,conj[VWp]},ZW}\n")
        f.write("};\n")
        f.write("\n")
        f.write("(* ----- VEVs ---- *)\n")
        f.write("\n")
        f.write("DEFINITION[EWSB][VEVs]= \n")
        f.write("{\n")
        f.write("    {H0, {v, 1/Sqrt[2]}, {Ah, \[ImaginaryI]/Sqrt[2]},{hh, 1/Sqrt[2]}}\n")
        f.write("};\n")

        f.write("\n")
        f.write("DEFINITION[EWSB][MatterSector]= {\n")
        f.write("\n};\n\n")

        f.write("(*------------------------------------------------------*)\n")
        f.write("(* Dirac-Spinors *)\n")
        f.write("(*------------------------------------------------------*)\n")
        f.write("\n")
        f.write("DEFINITION[EWSB][DiracSpinors]= {\n")

        f.write("\n};\n")
        f.write("\n")

        f.write("DEFINITION[EWSB][GaugeES]= {\n")
        WeylSpinors = []
        
        f.write("\n};\n")
        f.write("\n")

    # ----------------------- Parameters.m -----------------------
    with open(sm_parameters_json, "r") as f:
        all_parameters = json.load(f)

    # ----------------------- Particles.m -----------------------
    with open(sm_particles_json, "r") as f:
        all_particles = json.load(f)

    # ----------------------- SPheno.m -----------------------

if __name__ == "__main__":
    with open("reward/SARAH/sm_full.json", "r") as f:
        model_dict = json.load(f)
    WRITE_SARAH(model_dict, "reward/SARAH")
