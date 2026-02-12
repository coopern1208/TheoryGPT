import os
import json
import getpass
import shutil

from datetime import datetime
from fractions import Fraction
default_parameters = json.load(open(os.path.join(os.path.dirname(__file__), "default_parameters.json")))
default_particles = json.load(open(os.path.join(os.path.dirname(__file__), "default_particles.json")))

class SARAHFile:
    def __init__(self, 
                 model_name: str,
                 author: str,
                 full_model: dict,
                 OUTPUT_DIR: str = "dataset/models",
                 low_energy: bool = True,
                 unitary: bool = True
                 ):
        self.model_name = model_name
        self.author = author or getpass.getuser()
        self.low_energy = low_energy
        self.unitary = unitary
        self.full_model = full_model
        self.OUTPUT_DIR = os.path.join(OUTPUT_DIR, self.model_name)

        if os.path.exists(self.OUTPUT_DIR):
            shutil.rmtree(self.OUTPUT_DIR)
        os.makedirs(self.OUTPUT_DIR, exist_ok=True)


        # =============================== Particle Content ===============================
        self.fermions = {m_id: m for m_id, m in self.full_model['multiplets'].items() if m['type'] == 'FERMION'}
        self.cscalars = {m_id: m for m_id, m in self.full_model['multiplets'].items() if m['type'] == 'CSCALAR'}
        self.rscalars = {m_id: m for m_id, m in self.full_model['multiplets'].items() if m['type'] == 'RSCALAR'}
        self.particles = default_particles.copy()

        # Weyl Spinors
        self.weyl_spinors = {}
        for key, value in self.fermions.items():
            for charge in value['charges']:
                particle_name = value['name_list'][charge][0]
                particle_name = f"F{particle_name}" if particle_name != "ve" else 'Fv'
                if particle_name not in self.weyl_spinors:
                    electric_charge = Fraction(int(charge[7:]), 6)
                    electric_charge = f"{electric_charge.numerator}/{electric_charge.denominator}" if electric_charge.denominator != 1 else electric_charge.numerator
                    
                    self.weyl_spinors[particle_name] = {
                        'Description': particle_name,
                        'LaTeX': particle_name,
                        'OutputName': particle_name,
                        'Mass': value['mass_list'][charge],
                        'PDG': value['pdgi_list'][charge],
                        'ElectricCharge': electric_charge,
                        'Width': value['width_list'][charge],
                        'LEFT': None,
                        'RIGHT': None,
                    }
                chirality = value['chirality']
                self.weyl_spinors[particle_name][chirality] = True
        self.particles['ParticleDefinitions[EWSB]'].update(self.weyl_spinors) 

        # Scalar Particles
        self.scalar_particles = {}
        for key, value in [*self.cscalars.items(), *self.rscalars.items()]:
            if value['name'] == 'H': continue
            for charge in value['charges']:
                particle_name = value['name_list'][charge][0]
                electric_charge = Fraction(int(charge[7:]), 6)
                electric_charge = f"{electric_charge.numerator}/{electric_charge.denominator}" if electric_charge.denominator != 1 else electric_charge.numerator
                self.scalar_particles[particle_name] = {
                    'Description': particle_name,
                    'LaTeX': particle_name,
                    'OutputName': particle_name,
                    'Mass': value['mass_list'][charge],
                    'PDG': value['pdgi_list'][charge],
                    'ElectricCharge': electric_charge,
                    'Width': value['width_list'][charge]
                }
        self.particles['ParticleDefinitions[GaugeES]'].update(self.scalar_particles)
        self.particles['ParticleDefinitions[EWSB]'].update(self.scalar_particles)

        # WeylFermion And Indermediate
        for key, value in self.full_model['multiplets'].items():
            self.particles['WeylFermionAndIndermediate'].update({value['name']: {"LaTeX": value['name']}})
        for key, value in self.weyl_spinors.items():
            name = key[1:]
            if value['LEFT']:
                self.particles['WeylFermionAndIndermediate'].update({f"{name}L": {"LaTeX": name + "_L"}})
            elif value['RIGHT']:
                self.particles['WeylFermionAndIndermediate'].update({f"{name}R": {"LaTeX": name + "_R"}})
            if value['LEFT'] and value['RIGHT']:
                self.particles['WeylFermionAndIndermediate'].update({f"{name.upper()}L": {"LaTeX": name.upper() + "_L"}})
                self.particles['WeylFermionAndIndermediate'].update({f"{name.upper()}R": {"LaTeX": name.upper() + "_R"}})

        # =============================== Parameter Content ===============================
        self.parameters = default_parameters.copy()
        for name, info in self.full_model['internal_params'].items():
            self.parameters[name] = info
        for name, info in self.full_model['external_params'].items():
            self.parameters[name] = info

    # =============================== Write Model File ===============================
    def write_model_file(self, file):
        
        file.write("Off[General::spell];\n")
        file.write("\n")
        file.write(f"Model`Name = \"{self.model_name}\";\n")
        file.write(f"Model`NameLaTeX = \"{self.model_name}\";\n")
        file.write(f"Model`Authors = \"{self.author}\";\n")
        file.write(f"Model`Date = \"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\";\n\n")
        file.write("(*----------------------------------------------*)\n")
        file.write("(*               Particle Content               *)\n")
        file.write("(*----------------------------------------------*)\n")
        file.write("(* Gauge Groups *)\n")
        file.write("Gauge[[1]]={B,   U[1], hypercharge, g1,False};\n")
        file.write("Gauge[[2]]={WB, SU[2], left,        g2,True};\n")
        file.write("Gauge[[3]]={G,  SU[3], color,       g3,False};\n\n")
        file.write("(* Matter Fields *)\n")
        for idx, (m_id, m) in enumerate[tuple](self.fermions.items()):
            file.write(f"FermionFields[[{idx+1}]]= {{{m['multiplet_def']}}};\n")
        file.write("\n")
        for idx, (m_id, m) in enumerate([*self.cscalars.items(), *self.rscalars.items()]):
            file.write(f"ScalarFields[[{idx+1}]]= {{{m['multiplet_def']}}};\n")
        file.write("\n")
        if self.rscalars:
            file.write(f"RealScalars = {{{', '.join([m['name'] for m_id, m in self.rscalars.items()])}}};\n")

        file.write("\n")
        file.write("(*----------------------------------------------*)\n")
        file.write("(*                  DEFINITION                  *)\n")
        file.write("(*----------------------------------------------*)\n")
        file.write("NameOfStates={GaugeES, EWSB};\n")
        file.write("\n")
        file.write("(* ----- Before EWSB ----- *)\n")
        file.write("\n")
        file.write("DEFINITION[GaugeES][LagrangianInput]= \n")
        file.write("{\n")
        file.write("   {LagHC, {AddHC->True}},\n")
        file.write("   {LagNoHC,{AddHC->False}}\n")
        file.write("};\n\n")
        all_LagHC = []
        all_LagNoHC = []
        for _,interaction in self.full_model['interactions'].items():
            if interaction['LagHC']: all_LagHC.append(interaction['LagHC'])
            if interaction['LagNoHC']: all_LagNoHC.append(interaction['LagNoHC'])
        if all_LagHC: file.write(f"LagHC ={''.join(all_LagHC)};\n")
        if all_LagNoHC: file.write(f"LagNoHC ={''.join(all_LagNoHC)};\n")

        file.write("\n")
        file.write("(* Gauge Sector *)\n\n")
        file.write("DEFINITION[EWSB][GaugeSector] = \n")
        file.write("{   {{VB,VWB[3]},{VP,VZ},ZZ},\n")
        file.write("   {{VWB[1],VWB[2]},{VWp,conj[VWp]},ZW}\n")
        file.write("};\n\n")
        file.write("(* ----- VEVs ---- *)\n\n")
        file.write("DEFINITION[EWSB][VEVs]= \n")
        file.write("{\n")
        file.write("    {H0, {v, 1/Sqrt[2]}, {Ah, \\[ImaginaryI]/Sqrt[2]},{hh, 1/Sqrt[2]}}\n")
        file.write("};\n\n")
        
        MatterSector = []
        DiracSpinors = []
        WeylSpinors = []
        for key, value in self.weyl_spinors.items():
            if value['LEFT'] and value['RIGHT']:
                MatterSector.append(f"    {{{{{{{key[1:]}L}}, {{conj[{key[1:]}R]}}}} , {{{{{key[1:].upper()}L, V{key[1:].upper()}}}, {{{key[1:].upper()}R, U{key[1:].upper()}}}}}}}")
                DiracSpinors.append(f"    {key} -> {{{key[1:].upper()}L, conj[{key[1:].upper()}R]}}")
                WeylSpinors.append(f"    {key}1 -> {{{key}L, 0}}")
                WeylSpinors.append(f"    {key}2 -> {{0, {key}R}}")
            elif value['LEFT'] and not value['RIGHT']:
                DiracSpinors.append(f"    {key} -> {{{key[1:]}L, 0}}")
            elif not value['LEFT'] and value['RIGHT']:
                DiracSpinors.append(f"    {key} -> {{0, conj[{key[1:]}R]}}")
        file.write("DEFINITION[EWSB][MatterSector]= {\n")
        file.write(",\n".join(MatterSector))
        file.write("\n};\n\n")
        file.write("DEFINITION[EWSB][DiracSpinors]= {\n")
        file.write(",\n".join(DiracSpinors))
        file.write("\n};\n\n")
        file.write("DEFINITION[EWSB][GaugeES]= {\n")
        file.write(',\n'.join(WeylSpinors))
        file.write("\n};\n\n")

    # =============================== Write Particles File ===============================
    def write_particles(self, file):
        file.write("(*-------------------------------------------*)\n")
        file.write("(*                Particles                  *)\n")
        file.write("(*-------------------------------------------*)\n")
        file.write("\n")
        for section in self.particles:
            particles = []
            for particle in self.particles[section]:
                particle_info = []
                quote_keys = ["Description", "LaTeX", "OutputName"]
                for key, value in self.particles[section][particle].items():
                    if key in ["LEFT", "RIGHT"]: continue
                    if not isinstance(value, list):
                        if key in quote_keys:
                            particle_info.append(f"{key} -> \"{value}\"")
                        else:
                            particle_info.append(f"{key} -> {value}")
                    elif all(item == 'Automatic' for item in value):
                        particle_info.append(f"{key} -> Automatic")
                    else:
                        list_str = []
                        for item in value:
                            if key in quote_keys: item = f"\"{item}\""
                            else: item = str(item)
                            list_str.append(item)
                        math_str = ", ".join(list_str)
                        particle_info.append(f"{key} -> {{{math_str}}}")
                particle_info = '{' + f',\n{" " * 11}'.join(particle_info) + '}'
                particle_name = f"{particle},"
                particles.append(f"    {{{particle_name:4} {particle_info}}}")
            file.write(f"{section} = {{\n")
            file.write(",\n".join(particles))
            file.write("\n")
            file.write("};\n\n")
    
    # =============================== Write Parameters File ===============================
    def write_parameters(self, file, indent=11):
        file.write("(*-------------------------------------------*)\n")
        file.write("(*                Parameters                 *)\n")
        file.write("(*-------------------------------------------*)\n")
        file.write("\n")
        all_parameters = []
        for param in self.parameters:
            parameter_name = f"{param},"
            parameter_info = []
            quote_keys = ["Description", "LaTeX", ]
            for key, value in self.parameters[param].items():
                if key == "Value" and value is None: continue
                elif key in quote_keys: parameter_info.append(f"{key} -> \"{value}\"")
                else: parameter_info.append(f"{key} -> {value}")

            parameter_info = '{' + f',\n{" " * indent}'.join(parameter_info) + '}'
            parameter = f"    {{{parameter_name:4} {parameter_info}}}"
            all_parameters.append(parameter)
        file.write(f"ParameterDefinitions = {{\n")
        file.write(",\n".join(all_parameters))
        file.write("\n")
        file.write("};\n\n")

    # =============================== Write Spheno File ===============================
    def write_spheno(self, file):
        file.write("(*-------------------------------------------*)\n")
        file.write("(*                   Spheno                  *)\n")
        file.write("(*-------------------------------------------*)\n\n")

        # OnlyLowEnergySPheno and AddTreeLevelUnitarityLimits
        if self.low_energy: file.write("OnlyLowEnergySPheno = True;\n")
        if self.unitary: file.write("AddTreeLevelUnitarityLimits=True;\n")
        file.write("\n")
        ext_params = self.full_model['external_params']
        
        # MINPAR
        MINPAR = [f"{{{idx+1}, {param['OutputName']}INPUT}}" for idx, param in enumerate(ext_params.values())]
        file.write(f"MINPAR={{\n")
        file.write(",\n".join(MINPAR))
        file.write("\n")
        file.write("};\n\n")

        # Parameters to solve tadpoles
        file.write(f"ParametersToSolveTadpoles={{{', '.join(self.full_model['tadpole_params'])}}};\n\n")

        # Boundary Low Scale Input
        low_scale_input = [f"{{{param['OutputName']}, {param['OutputName']}INPUT}}" for param in ext_params.values()]
        file.write("BoundaryLowScaleInput={\n")
        file.write(",\n".join(low_scale_input))
        file.write("\n")
        file.write("};\n\n")

        # Matching Conditions
        matching_conditions = [f"{{{condition}}}" for condition in self.full_model['matching_conditions']]
        file.write("MatchingConditions={\n")
        file.write(",\n".join(matching_conditions))
        file.write("\n")
        file.write("};\n\n")

        # List of Decay Particles 
        list_of_decay_particles = ['hh'] + list(self.weyl_spinors.keys())
        file.write("ListOfDecayParticles={")
        file.write(",".join(list_of_decay_particles))
        file.write("};\n\n")

        # Default Values for Input Parameters
        default_values = [f"{param['OutputName']} -> {param['Value']}" for param in ext_params.values()]
        file.write("DefaultInputValues={\n")
        file.write(",\n".join(default_values))
        file.write("\n")
        file.write("};\n\n")

        # Renormalization conditions
        file.write("RenConditionsDecays={\n")
        file.write("{dCosTW, 1/2*Cos[ThetaW] * (PiVWp/(MVWp^2) - PiVZ/(mVZ^2)) },\n")
        file.write("{dSinTW, -dCosTW/Tan[ThetaW]},\n")
        file.write("{dg2, 1/2*g2*(derPiVPheavy0 + PiVPlightMZ/MVZ^2 - (-(PiVWp/MVWp^2) + PiVZ/MVZ^2)/Tan[ThetaW]^2 + (2*PiVZVP*Tan[ThetaW])/MVZ^2)  },\n")
        file.write("{dg1, dg2*Tan[ThetaW]+g2*dSinTW/Cos[ThetaW]- dCosTW*g2*Tan[ThetaW]/Cos[ThetaW]}\n")
        file.write("};\n\n")
            
    # =============================== Output SARAH ===============================
    def output_sarah(self):
        model_file = os.path.join(self.OUTPUT_DIR, f"{self.model_name}.m")
        particle_file = os.path.join(self.OUTPUT_DIR, f"particles.m")
        parameter_file = os.path.join(self.OUTPUT_DIR, f"parameters.m")
        spheno_file = os.path.join(self.OUTPUT_DIR, f"SPheno.m")

        with open(model_file, "w") as file: 
            self.write_model_file(file)
        with open(particle_file, "w") as file: 
            self.write_particles(file)
        with open(parameter_file, "w") as file: 
            self.write_parameters(file)
        with open(spheno_file, "w") as file: 
            self.write_spheno(file)

    def get_free_parameters(self):
        for param_id, param in self.full_model['external_params'].items():
            return {param_id: {
                'OutputName': param['OutputName'],
                'Value': param['Value']
            }}

# if __name__ == "__main__":
#     full_model = json.load(open(os.path.join(os.path.dirname(__file__), "sm_model.json")))
#     standard_model = SARAHFile(model_name="SM", author="AI", full_model=full_model)
#     standard_model.output_sarah()
