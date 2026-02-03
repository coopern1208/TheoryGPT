from dataclasses import dataclass, field
from config import config
import grammar.vocab as vocab

@dataclass
class GrammarState:
    last_token: str = None
    length: int = 1
    current_block: str = None

    # model dicts
    gauge_groups: dict = field(default_factory=dict)
    vevs: dict = field(default_factory=dict)
    particles: dict = field(default_factory=lambda: {
        "RSCALAR": {"NO_COLOR": {}, "COLOR": {}},
        "CSCALAR": {"NO_COLOR": {}, "COLOR": {}},
        "FERMION": {
            "LEFT": {"NO_COLOR": {}, "COLOR": {}},
            "RIGHT": {"NO_COLOR": {}, "COLOR": {}}
        }
    })
    multiplets: dict = field(default_factory=dict)
    interactions: dict = field(default_factory=dict)

    tags: dict = field(default_factory=lambda: vocab.SM_TAGS_DICT.copy())
    vev_opts: list[str] = field(default_factory=list)

    ptcl_counts: dict = field(default_factory=lambda: {"CSCALAR": 0, "RSCALAR": 0, "FERMION": 0})
    mplt_counts: dict = field(default_factory=lambda: {"CSCALAR": 0, "RSCALAR": 0, "FERMION": 0})

    # gauge block
    group_id: str = None
    group_type: str = None
    group_rank: str = None

    # ssb block
    vev_id: str = None
    vev_vector: list[int] = field(default_factory=list)

    # particle block
    particle_type: str = None
    particle_count: int = 0
    color: str = None
    charge: str = None
    num_ptcls: int = 0
    charge_opts: dict = field(default_factory=lambda: {
        "NULL": vocab.CHARGES.copy(),
        "LEFT": vocab.CHARGES.copy(),
        "RIGHT": vocab.CHARGES.copy()
    })
    tag_opts: list[str] = field(default_factory=list)
    tag_list: list[str] = field(default_factory=list)
    
    #multiplet block
    multiplet_id: str = None
    multiplet_type: str = None
    chirality: str = None
    charge_list: list[int] = field(default_factory=list)
    rep_list: list[str] = field(default_factory=list)
    dim: int = 0
    gen: int = 0
    req_fulfilled: bool = False
    tagged_ptcls: list[str] = field(default_factory=list)
    anomalies: dict = field(default_factory=dict)

    # interaction block
    allowed_mplts: dict = field(default_factory=dict)
    interaction_id: str = None
    interaction_type: str = None
    param_list: list[str] = field(default_factory=list)
    num_params: int = 0
    mplt_list: list[str] = field(default_factory=list)
    param_opts: list[str] = field(default_factory=list)

