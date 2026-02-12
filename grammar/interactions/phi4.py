def allowed_mplts(multiplets: dict) -> list[str]:
    allowed_list = []
    for key, value in multiplets.items():
        if value["type"] in ["CSCALAR", "RSCALAR"]:
            allowed_list.append([key])
    return allowed_list

def params(multiplets: dict, mplt_list: list[str]) -> list[str]:
    pass
    return None, 1

def get_Lag(scalar):
    if scalar['name'] == "H": 
        lambda4 = "\[Lambda]"
    else:
        lambda4 = f"\[Lambda]{scalar['name']}"
    if scalar['type'] == "CSCALAR":
        return f" - 1/2 {lambda4} conj[{scalar['name']}].{scalar['name']}.conj[{scalar['name']}].{scalar['name']}"
    elif scalar['type'] == "RSCALAR":
        return f" - 1/2 {lambda4} {scalar['name']}.{scalar['name']}.{scalar['name']}.{scalar['name']}"

def get_param(multiplets, interaction, LesHouches_idx):
    scalar = multiplets[interaction['mplt_list'][0]]
    vev_id = scalar["vev_id"]
    ext_param = {}
    int_param = {}
    matching_conditions = []
    tadpole_params = []
    if vev_id:
        ext_param['\[Lambda]'] = {
            'Description': f"Higgs Self-coupling",
            'OutputName': f'Lambda',
            'Block': "SM",
            'Dependence': None,
            'DependenceNum': "Mass[hh]^2/v^2",
            'DependenceSPheno': None,
            'DependenceOptional': None,
            'Real': True,
            'Value': float(interaction['param_list'][0].split("_")[-1]),
            'LesHouches': "{SM, 1}",
            'LaTeX': f"\\\\lambda"
        }
    else:
        ext_param[f"\[Lambda]{scalar['name']}"] = {
            'Description': f"{scalar['name']} Quartic Self-coupling",
            'OutputName': f"Lambda4{scalar['name']}",
            'Block': "BSM",
            'Dependence': None,
            'DependenceNum': None,
            'DependenceSPheno': None,
            'DependenceOptional': None,
            'Real': True,
            'Value': float(interaction['param_list'][0].split("_")[-1]),
            'LesHouches': f"{{BSM, {LesHouches_idx}}}",
            'LaTeX': f"\\\\lambda_4{scalar['name']}"
        }
        LesHouches_idx += 1
    return {"ext_param": ext_param, 
            "int_param": int_param, 
            "LesHouches_idx": LesHouches_idx,
            "matching_conditions": matching_conditions,
            "tadpole_params": tadpole_params
            }