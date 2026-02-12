from dataclasses import dataclass, field
from typing import Literal, List, Dict, Tuple

@dataclass
class Config:

    # ========================================================================
    # Random Seed
    # ========================================================================
    random_seed: int = 42
    
    # ========================================================================
    # Model Parameters
    # ========================================================================
    max_gauge_groups: int = 3
    max_group_rank: int = 3
    max_vevs: int = 1
    max_interactions: int = 20
    max_multiplets: int = 15
    max_dim: int = 3
    max_gen: int = 4
    max_charge: int = 6
    max_hypercharge: int = 9

    max_free_param_num: int = 35
    max_mass_exp: int = 4
    min_mass_exp: int = -6
    max_value_exp: int = 1
    min_value_exp: int = -2
    base = ["1", "2", "5"]
    
    max_ptcl_count: dict = field(default_factory=lambda: {
        "CSCALAR": 1,
        "RSCALAR": 1,
        "FERMION": 8
    })

    max_mplt_count: dict = field(default_factory=lambda: {
        "CSCALAR": 4,
        "RSCALAR": 1,
        "FERMION": 10
    })

    PAD_TOKEN_ID: int = 2

    # ========================================================================
    # Debugging
    # ========================================================================
    GRAMMAR_DEBUG: bool = False
    DEBUG_MODE: bool = False

    # ========================================================================
    # Network Parameters
    # ========================================================================
    D_MODEL: int = 512
    NHEAD: int = 8
    NUM_LAYERS: int = 4
    DIM_FEEDFORWARD: int = 1024
    DROPOUT: float = 0.1
    MAX_LENGTH: int = 512

    TORCH_DTYPE: Literal["bfloat16", "float16", "float32"] = "bfloat16"
    USE_QUANTIZATION: bool = True  # Use 4-bit quantization for memory efficiency
    TRUST_REMOTE_CODE: bool = True

    # Checkpoint parameters
    CHECKPOINT_DIR: str = "checkpoints"
    CHECKPOINT_INTERVAL: int = 1000
    CHECKPOINT_MAX_TO_KEEP: int = 3
    
    # ========================================================================
    # Data Parameters
    # ========================================================================
    VAL_SPLIT_SEED: int = 1337
    VAL_SPLIT_SIZE: float = 0.1

    # ========================================================================
    # Pretraining Parameters
    # ========================================================================
    PRETRAIN_LEARNING_RATE: float = 1e-4
    PRETRAIN_BATCH_SIZE: int = 64
    PRETRAIN_NUM_EPOCHS: int = 1 #5
    PRETRAIN_GRAD_ACCUMULATION_STEPS: int = 4
    PRETRAIN_LOG_INTERVAL: int = 10

    # ========================================================================
    # Supervised Fine-Tuning Parameters
    # ========================================================================
    SFT_LEARNING_RATE: float = 5e-5
    SFT_BATCH_SIZE: int = 8 #32
    SFT_NUM_EPOCHS: int = 5
    SFT_GRAD_ACCUMULATION_STEPS: int = 2
    SFT_LOG_INTERVAL: int = 10
    SFT_TEMPERATURE: float = 1.0
    SFT_WARMUP_RATIO: float = 0.1

    # ========================================================================
    # RL Training Parameters
    # ========================================================================
    REPLAY_BUFFER_SIZE: int = 1000
    
    RL_RANDOM_SEED: int = 42  # Random seed for GRPO training reproducibility
    RL_LEARNING_RATE: float = 1e-4
    RL_BATCH_SIZE: int = 64
    RL_MAX_STEPS: int = 1000
    GRPO_GROUP_SIZE: int = 64
    GRPO_GENERATION_BATCH_SIZE: int = 64  
    RL_KL_COEF: float = 0.05
    RL_CLIP_RANGE: float = 0.2
    
    # Entropy coefficient with decay
    RL_ENTROPY_COEF_INITIAL: float = 0.01   
    RL_ENTROPY_COEF_FINAL: float = 0.01   
    RL_ENTROPY_DECAY_SCHEDULE: Literal["linear", "exponential", "cosine"] = "linear"
    
    RL_NUM_EPISODES: int = 100
    RL_VAL_EPISODES: int = 10
    RL_LOG_INTERVAL: int = 5
    RL_SAVE_INTERVAL: int = 10
    RL_TEMPERATURE: float = 1.0
    RL_USE_CURRICULUM: bool = False

    # ========================================================================
    # SARAH Interface Parameters
    # ========================================================================
    SARAH_PATH: str = "../SARAH-4.15.4"
    SPHENO_PATH: str = "../SPheno-4.0.5"
    MODEL_BASE: str = "dataset/models"
    OBSERVABLES_PATH: str = "observables.json"
    TIMEOUT: int = 1
    SIGMA_THRESHOLD: int = 3
    KEEP_LOG: bool = True
    LOOP_MASS: bool = True
    INCLUDE_TACHYON: bool = True
    CALC_DECAYS: bool = False
    MASS_PRECISION: float = 1e-6
    THREE_BODY_DECAYS: bool = False
    HIGGS_BOUNDS: bool = False
    EFT_HIGGS_COUPLING: bool = False
    DIPHOTON_WIDTH: bool = False

config = Config()