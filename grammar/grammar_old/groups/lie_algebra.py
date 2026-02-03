def is_abelian(type, rank):
    return type == "GAUGE_U" and rank == "rank_1"

# Dimension of the SU(N) reps
def SU_N_dim(N, rep = "fnd"):
    if rep == "singlet": return 1
    elif rep in ("fnd", "anti_fnd"): return N
    elif rep == "adj": return N**2 - 1
    else: return None

# Dynkin label of the SU(N) reps
def SU_N_Dynkin_label(N, rep = "fnd"):
    if rep == "singlet": return [0] * (N - 1)
    elif rep == "fnd": return [1] + [0] * (N - 2)
    elif rep == "anti_fnd": return [0] * (N - 2) + [1]
    elif rep == "adj":
        adjoint = [0] * (N - 1)
        adjoint[0] += 1
        adjoint[-1] += 1
        return adjoint
    else: return None

# Quadratic Casimir of the SU(N) reps
def SU_N_quad_casimir(N, dynkin_labels):
    c2 = 0.0
    for i in range(1, N):
        for j in range(1, N):
            inverse_cartan_matrix = min(i, j) - (i * j) / N
            term = dynkin_labels[i - 1] * inverse_cartan_matrix * (dynkin_labels[j - 1] + 2)
            c2 += term
    return c2/2

# Dynkin index of the SU(N) reps
def SU_N_Dynkin_index(N, rep):
    label = SU_N_Dynkin_label(N, rep)
    dimR = SU_N_dim(N, rep)
    dimG = N**2 - 1
    c2 = SU_N_quad_casimir(N, label)
    T = (dimR * c2) / dimG
    return T

# Cubic anomaly of the SU(N) reps.
def SU_N_cubic_anomaly(N, rep):
    if rep == "singlet": return 0
    elif rep == "fnd": return 1 if N >= 3 else 0    
    elif rep == "anti_fnd": return -1 if N >= 3 else 0
    elif rep == "adj": return 0
    else: return None
