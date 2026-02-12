import grammar.groups.lie_algebra as lie

# [U(1)] [SU(3)]^2 anomaly
def U1_SU3_anomaly(multiplets):
    coeff = 0 
    for key, multiplet in multiplets.items():
        if multiplet['type'] != 'FERMION': continue
        chiral = 1 if multiplet['chirality'] == "LEFT" else -1
        SU3_rep = multiplet['rep_list'][0]
        color_rep = multiplet['rep_list'][0]
        color_num = lie.SU_N_dim(3, color_rep)
        gen_num = multiplet['gen']
        dim_num = multiplet['dim']
        Y = int(multiplet['rep_list'][2][12:])    
        dynkin_idx = lie.SU_N_Dynkin_index(3, SU3_rep)
        coeff += chiral * gen_num * dim_num * dynkin_idx * color_num * Y
    return int(coeff)

# [U(1)] [SU(2)]^2 anomaly
def U1_SU2_anomaly(multiplets):
    coeff = 0 
    for key, multiplet in multiplets.items():
        if multiplet['type'] != 'FERMION': continue
        chiral = 1 if multiplet['chirality'] == "LEFT" else -1
        SU2_rep = multiplet['rep_list'][1]
        color_rep = multiplet['rep_list'][0]
        color_num = lie.SU_N_dim(3, color_rep)
        gen_num = multiplet['gen']
        dim_num = multiplet['dim']
        Y = int(multiplet['rep_list'][2][12:])    
        dynkin_idx = lie.SU_N_Dynkin_index(2, SU2_rep)
        coeff += chiral * gen_num * dim_num * dynkin_idx * color_num * Y
    return int(coeff)

# [U(1)]^3 anomaly
def U1_anomaly(multiplets):
    coeff = 0 
    for key, multiplet in multiplets.items():
        if multiplet['type'] != 'FERMION': continue
        chiral = 1 if multiplet['chirality'] == "LEFT" else -1
        color_rep = multiplet['rep_list'][0]
        color_num = lie.SU_N_dim(3, color_rep)
        gen_num = multiplet['gen']
        dim_num = multiplet['dim']
        Y = int(multiplet['rep_list'][2][12:])    
        coeff += chiral * gen_num * dim_num * color_num * Y**3
    return coeff

# [U(1)] [grav]^2 anomaly
def grav_anomaly(multiplets):
    coeff = 0 
    for key, multiplet in multiplets.items():
        if multiplet['type'] != 'FERMION': continue
        chiral = 1 if multiplet['chirality'] == "LEFT" else -1
        color_rep = multiplet['rep_list'][0]
        color_num = lie.SU_N_dim(3, color_rep)
        gen_num = multiplet['gen']
        dim_num = multiplet['dim']
        Y = int(multiplet['rep_list'][2][12:])    
        coeff += chiral * gen_num * dim_num * color_num * Y
    return coeff

# Witten anomaly
def witten_anomaly(multiplets):
    num_SU2_doublets = 0 
    for key, multiplet in multiplets.items():
        if multiplet['type'] != 'FERMION': continue
        if multiplet['rep_list'][1] == 'fnd':
            color_rep = multiplet['rep_list'][0]
            color_num = lie.SU_N_dim(3, color_rep)
            num_SU2_doublets += color_num
            
    return num_SU2_doublets % 2 

# tokenize the anomaly coefficient
def anomaly_range(anomaly_coeff: int) -> str:
    if anomaly_coeff > 0: 
        return "POS_SMALL" if anomaly_coeff < 50 else "POS_BIG"
    elif anomaly_coeff < 0: 
        return "NEG_SMALL" if anomaly_coeff > -50 else "NEG_BIG"
    else: return "ZERO"
