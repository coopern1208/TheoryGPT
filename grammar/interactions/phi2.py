def allowed_mplts(multiplets: dict) -> list[str]:
    allowed_list = []
    for key, value in multiplets.items():
        if value["type"] in ["CSCALAR", "RSCALAR"]:
            allowed_list.append([key])
    return allowed_list

def get_Lag(scalar):
    if scalar['type'] == "CSCALAR":
        return f" - m{scalar['name']}2 conj[{scalar['name']}].{scalar['name']}"
    elif scalar['type'] == "RSCALAR":
        return f" - m{scalar['name']}2 {scalar['name']}.{scalar['name']}"

def get_param(multiplets, interaction, LesHouches_idx):
    scalar = multiplets[interaction['mplt_list'][0]]
    vev_id = scalar["vev_id"]
    ext_param = {}
    int_param = {}
    matching_conditions = []
    tadpole_params = []
    if vev_id: 
        int_param['mu2'] = {
            'Description': "SM Mu Parameter",
            'OutputName': "m2SM",
            'Block': "SM",
            'Dependence': None,
            'DependenceNum': None,
            'DependenceSPheno': None,
            'DependenceOptional': None,
            'Real': False,
            'LesHouches': "{HMIX, 2}",
            'LaTeX': "\\\\mu^2"
        }
        tadpole_params.append('mu2')
    else: 
        ext_param[f'm2{scalar["name"]}'] = {
            'Description': f"{scalar['name']} Mass Squared",
            'OutputName': f"m2{scalar['name']}",
            'Block': "BSM",
            'Dependence': None,
            'DependenceNum': None,
            'DependenceSPheno': None,
            'DependenceOptional': None,
            'Real': False,
            'Value': float(interaction['param_list'][0].split("_")[-1]),
            'LesHouches': f"{{BSM, {LesHouches_idx}}}",
            'LaTeX': f"\\\\mu_{scalar['name']}^2"
        }
        LesHouches_idx += 1
    return {"ext_param": ext_param, 
            "int_param": int_param, 
            "LesHouches_idx": LesHouches_idx,
            "matching_conditions": matching_conditions,
            "tadpole_params": tadpole_params
            }
