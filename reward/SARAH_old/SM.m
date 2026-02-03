Off[General::spell];

Model`Name = "SM";
Model`NameLaTeX = "Standard Model";
Model`Authors = "AI";
Model`Date = "2026-01-27 02:27:06";

(*-------------------------------------------*)
(*   Particle Content*)
(*-------------------------------------------*)

(* Gauge Groups *)
Gauge[[1]]={B,   U[1], hypercharge, g1,False};
Gauge[[2]]={WB, SU[2], left,        g2,True};
Gauge[[3]]={G,  SU[3], color,       g3,False};

(* Matter Fields *)

Scalar[[1]]= {};
Fermion[[1]]= {};
Fermion[[2]]= {};
Fermion[[3]]= {};
Fermion[[4]]= {};
Fermion[[5]]= {};

(*----------------------------------------------*)
(*   DEFINITION                                 *)
(*----------------------------------------------*)
NameOfStates={GaugeES, EWSB};

(* ----- Before EWSB ----- *)

DEFINITION[GaugeES][LagrangianInput]= 
{
   {LagHC, {AddHC->True}},
   {LagNoHC,{AddHC->False}}
};

LagHC = ;
LagNoHC = ;

(* Gauge Sector *)

DEFINITION[EWSB][GaugeSector] = 
{   {{VB,VWB[3]},{VP,VZ},ZZ},
   {{VWB[1],VWB[2]},{VWp,conj[VWp]},ZW}
};

(* ----- VEVs ---- *)

DEFINITION[EWSB][VEVs]= 
{
    {H0, {v, 1/Sqrt[2]}, {Ah, \[ImaginaryI]/Sqrt[2]},{hh, 1/Sqrt[2]}}
};

DEFINITION[EWSB][MatterSector]= {
