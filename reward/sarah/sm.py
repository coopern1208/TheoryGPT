SM_full_model = {
    'too_long': False,
    'gauge_groups': {
        'g_1': {'id': 'g_1', 'type': 'GAUGE_U', 'rank': 'rank_1'},
        'g_2': {'id': 'g_2', 'type': 'GAUGE_SU', 'rank': 'rank_2'},
        'g_3': {'id': 'g_3', 'type': 'GAUGE_SU', 'rank': 'rank_3'}
    },
    'vevs': {
        'v_1': {
            'id': 'v_1',
            'vector': [0, 1],
            'rep_list': ['singlet', 'fnd', 'hypercharge_3'],
            'charge_list': ['charge_6', 'charge_0'],
            'dim': 2,
            'multiplets': 'm_1'
        }
    },
    'particles': {
        'RSCALAR': {'NO_COLOR': {}, 'COLOR': {}},
        'CSCALAR': {'NO_COLOR': {}, 'COLOR': {}},
        'FERMION': {
            'LEFT': {
                'NO_COLOR': {
                    'charge_-6': {'ptcls': ['SM_E', 'SM_MU', 'SM_TAU'], 'num_ptcls': 3},
                    'charge_0': {'ptcls': ['SM_VE', 'SM_VM', 'SM_VT'], 'num_ptcls': 3}
                },
                'COLOR': {
                    'charge_4': {'ptcls': ['SM_U', 'SM_C'], 'num_ptcls': 2},
                    'charge_-2': {'ptcls': ['SM_D', 'SM_S', 'SM_B'], 'num_ptcls': 3}
                }
            },
            'RIGHT': {
                'NO_COLOR': {
                    'charge_-6': {'ptcls': ['SM_E', 'SM_MU', 'SM_TAU'], 'num_ptcls': 3}
                },
                'COLOR': {
                    'charge_4': {'ptcls': ['SM_U', 'SM_C'], 'num_ptcls': 2},
                    'charge_-2': {'ptcls': ['SM_D', 'SM_S', 'SM_B'], 'num_ptcls': 3}
                }
            }
        }
    },
    'multiplets': {
        'm_1': {'id': 'm_1', 'type': 'CSCALAR', 'chirality': 'NULL', 'rep_list': ['singlet', 'fnd', 'hypercharge_3'], 'gen': 'gen_1', 'dim': 2, 'charges': ['charge_6', 'charge_0'], 'tagged_ptcls': {}, 'mplt_mass_list': {'mass_0': 77.79203044014213}, 'vev_id': 'v_1'},
        'm_2': {'id': 'm_2', 'type': 'FERMION', 'chirality': 'LEFT', 'rep_list': ['singlet', 'fnd', 'hypercharge_-3'], 'gen': 'gen_3', 'dim': 2, 'charges': ['charge_0', 'charge_-6'], 'tagged_ptcls': {'charge_0': ['SM_VE', 'SM_VM', 'SM_VT'], 'charge_-6': ['SM_E', 'SM_MU', 'SM_TAU']}, 'mplt_mass_list': {'charge_0': ['mass_0', 'mass_0', 'mass_0'], 'charge_-6': ['SM_E', 'SM_MU', 'SM_TAU']}, 'vev_id': None},
        'm_3': {'id': 'm_3', 'type': 'FERMION', 'chirality': 'RIGHT', 'rep_list': ['singlet', 'singlet', 'hypercharge_-6'], 'gen': 'gen_3', 'dim': 1, 'charges': ['charge_-6'], 'tagged_ptcls': {'charge_-6': ['SM_E', 'SM_MU', 'SM_TAU']}, 'mplt_mass_list': {'charge_-6': ['SM_E', 'SM_MU', 'SM_TAU']}, 'vev_id': None},
        'm_4': {'id': 'm_4', 'type': 'FERMION', 'chirality': 'LEFT', 'rep_list': ['fnd', 'fnd', 'hypercharge_1'], 'gen': 'gen_3', 'dim': 2, 'charges': ['charge_4', 'charge_-2'], 'tagged_ptcls': {'charge_4': ['SM_U', 'SM_C'], 'charge_-2': ['SM_D', 'SM_S', 'SM_B']}, 'mplt_mass_list': {'charge_4': ['SM_U', 'SM_C', 'p_2'], 'charge_-2': ['SM_D', 'SM_S', 'SM_B']}, 'vev_id': None},
        'm_5': {'id': 'm_5', 'type': 'FERMION', 'chirality': 'RIGHT', 'rep_list': ['fnd', 'singlet', 'hypercharge_4'], 'gen': 'gen_3', 'dim': 1, 'charges': ['charge_4'], 'tagged_ptcls': {'charge_4': ['SM_U', 'SM_C']}, 'mplt_mass_list': {'charge_4': ['SM_U', 'SM_C', 'p_2']}, 'vev_id': None},
        'm_6': {'id': 'm_6', 'type': 'FERMION', 'chirality': 'RIGHT', 'rep_list': ['fnd', 'singlet', 'hypercharge_-2'], 'gen': 'gen_3', 'dim': 1, 'charges': ['charge_-2'], 'tagged_ptcls': {'charge_-2': ['SM_D', 'SM_S', 'SM_B']}, 'mplt_mass_list': {'charge_-2': ['SM_D', 'SM_S', 'SM_B']}, 'vev_id': None}
    },
    'interactions': {
        'i_1': {'id': 'i_1', 'type': 'TERM_PHI2', 'param_list': ['v_1'], 'mplt_list': ['m_1']},
        'i_2': {'id': 'i_2', 'type': 'TERM_PHI4', 'param_list': ['p_1'], 'mplt_list': ['m_1']},
        'i_3': {'id': 'i_3', 'type': 'TERM_YUKAWA', 'param_list': ['SM_E', 'SM_MU', 'SM_TAU'], 'mplt_list': ['m_1', 'm_2', 'm_3']},
        'i_4': {'id': 'i_4', 'type': 'TERM_YUKAWA', 'param_list': ['SM_U', 'SM_C', 'p_2'], 'mplt_list': ['m_1', 'm_4', 'm_5']},
        'i_5': {'id': 'i_5', 'type': 'TERM_YUKAWA', 'param_list': ['SM_D', 'SM_S', 'SM_B'], 'mplt_list': ['m_1', 'm_4', 'm_6']}
    },
    'anomalies': {
        'U1_SU3_anomaly': 0,
        'U1_SU2_anomaly': 0,
        'U1_anomaly': 0,
        'grav_anomaly': 0,
        'witten_anomaly': 0
    },
    'params': {
        'v_1': 246.0,
        'p_1': 'param_1e-1',
        'p_2': 'mass_1e2'
    }
}


if __name__ == "__main__":
    from name import fermion
    multiplets = SM_full_model['multiplets']
    
    fermion_multiplets = {m_id: m for m_id, m in multiplets.items() if m['type'] == 'FERMION'}
    scalar_multiplets = {m_id: m for m_id, m in multiplets.items() if m['type'] in ('CSCALAR', 'RSCALAR')}
    
    for idx, (m_id, m) in enumerate(fermion_multiplets.items()):
        print(f"FermionFields[[{idx+1}]]= {{   }}")
    for idx, (m_id, m) in enumerate(scalar_multiplets.items()):
        print(f"ScalarFields[[{idx+1}]]= {{}}")
