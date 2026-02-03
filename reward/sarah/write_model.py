from write_main import SARAHFile
import os


def write_model(SARAHFile: SARAHFile, file):

    file.write("Off[General::spell];\n")
    file.write("\n")
    file.write(f"Model`Name = \"{SARAHFile.model_name}\";\n")
    file.write(f"Model`NameLaTeX = \"{SARAHFile.model_name}\";\n")
    file.write(f"Model`Authors = \"{SARAHFile.author}\";\n")
    file.write(f"Model`Date = \"{SARAHFile.current_time}\";\n")
    file.write("\n")
    file.write("(*-------------------------------------------*)\n")
    file.write("(*   Particle Content*)\n")
    file.write("(*-------------------------------------------*)\n")
    file.write("\n")
    file.write("(* Gauge Groups *)\n")
    file.write("Gauge[[1]]={B,   U[1], hypercharge, g1,False};\n")
    file.write("Gauge[[2]]={WB, SU[2], left,        g2,True};\n")
    file.write("Gauge[[3]]={G,  SU[3], color,       g3,False};\n")
    file.write("\n")
    file.write("(* Matter Fields *)\n")
    file.write("\n")
    


    file.write("(*----------------------------------------------*)\n")
    file.write("(*   DEFINITION                                 *)\n")
    file.write("(*----------------------------------------------*)\n")
    file.write("NameOfStates={GaugeES, EWSB};\n")
    file.write("\n")
    file.write("(* ----- Before EWSB ----- *)\n")
    file.write("\n")
    file.write("DEFINITION[GaugeES][LagrangianInput]= \n")
    file.write("{\n")
    file.write("   {LagHC, {AddHC->True}},\n")
    file.write("   {LagNoHC,{AddHC->False}}\n")
    file.write("};\n")
    file.write("\n")
