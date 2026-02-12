def allowed_mplts(multiplets: dict) -> list[str]:
    allowed_list = []
    for key, value in multiplets.items():
        if value["type"] == "RSCALAR":
            allowed_list.append([key])
    return allowed_list

def get_Lag(scalar):
    if scalar['type'] == "RSCALAR":
        return f" - 1/2 \[Lambda]{scalar['name']}3 {scalar['name']}.{scalar['name']}.{scalar['name']}"
    else:
        raise ValueError(f"Scalar type {scalar['type']} not supported")

def get_param(multiplets, interaction, LesHouches_idx):
    scalar = multiplets[interaction['mplt_list'][0]]
    ext_param = {}
    int_param = {}
    matching_conditions = []
    tadpole_params = []
    ext_param[f"\[Lambda]3{scalar['name']}"] = {
        'Description': f"{scalar['name']} Cubic Selfcoupling",
        'OutputName': f"Lambda3{scalar['name']}",
        'Block': "BSM",
        'Dependence': None,
        'DependenceNum': None,
        'DependenceSPheno': None,
        'DependenceOptional': None,
        'Real': True,
        'Value': float(interaction['param_list'][0].split("_")[-1]),
        'LesHouches': f"{{BSM, {LesHouches_idx}}}",
        'LaTeX': f"\\\\lambda_3{scalar['name']}"
    }
    LesHouches_idx += 1
    return {"ext_param": ext_param, 
            "int_param": int_param, 
            "LesHouches_idx": LesHouches_idx,
            "matching_conditions": matching_conditions,
            "tadpole_params": tadpole_params
            }