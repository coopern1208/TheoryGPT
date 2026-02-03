import random
from config import config 
from grammar.masker import GrammarMasker
from grammar.parser import read_model_txt, render_sequence
import grammar.vocab as vocab


def generate_theory(grammar_masker: GrammarMasker, 
                    seed: int = None,
                    prompt: list[str] = None,
                    max_length: int = config.MAX_LENGTH,
                    print_tokens: bool = False
                    ) -> list[str]:
    if seed is not None: random.seed(seed)
    state = grammar_masker.init_state()
    sequence = ["BOS"]
    valid_token_list = [[0]]
    prev_token = "BOS"
    search_space = 1
    theory_too_long = False
    
    break_gen = False
    if prompt:
        if prompt[0] == "BOS": prompt = prompt[1:]
        if prompt[-1] == "EOS": prompt = prompt[:-1]

        for token in prompt:
            valid_tokens = grammar_masker.get_valid_tokens(state, prev_token)
            
            valid_tokens_encoded = vocab.encode(valid_tokens)
            valid_token_list.append(valid_tokens_encoded)

            # ----------------------- CHECK VALID TOKENS -----------------------
            if not valid_tokens:
                print(f"Warning: No valid tokens after {prev_token} (in prompt)")
                break_gen = True
                break
            if token not in valid_tokens: 
                print(f"Warning: Token '{token}' is not valid after {prev_token} (in prompt)")
                print(f"Valid tokens: {valid_tokens}")
                break_gen = True
                break
            # ----------------------- END CHECK VALID TOKENS -----------------------

            sequence.append(token)
            search_space *= len(valid_tokens)
            state = grammar_masker.step(state, token)
            if print_tokens: print(state.current_block, token)
            prev_token = token
    
    while state.length < max_length and not break_gen: 
        if prev_token == "EOS": break
        valid_tokens = grammar_masker.get_valid_tokens(state, prev_token)
        if "THEORY_TOO_LONG" in valid_tokens: theory_too_long = True
        elif "TOO_MANY_INTERACTIONS" in valid_tokens: theory_too_long = True
        elif "TOO_MANY_PARAMS" in valid_tokens: theory_too_long = True
        valid_token_list.append(vocab.encode(valid_tokens))

        # ----------------------- CHECK VALID TOKENS --------------------------------
        if "THEORY_TOO_LONG" in valid_tokens:
            theory_too_long = True
            break
        if not valid_tokens:
            print(f"Warning: No valid tokens after {prev_token} (in generation)")
            break
        # ----------------------- Token Selection Preference -----------------------
        if valid_tokens == vocab.CHIRALITIES:
            chiral_weights = [70, 15, 15] # prefer NULL over LEFT and RIGHT
            token = random.choices(valid_tokens, weights=chiral_weights, k=1)[0]
        else:
            token = random.choice(valid_tokens)

        
        search_space *= len(valid_tokens)
        sequence.append(token)
        state = grammar_masker.step(state, token)
        if print_tokens: print(state.current_block, token)
        prev_token = token

    full_model = {"too_long": theory_too_long,
                  "gauge_groups": state.gauge_groups,
                  "vevs": state.vevs,
                  "particles": state.particles,
                  "multiplets": state.multiplets,
                  "interactions": state.interactions,
                  "anomalies": state.anomalies,
                  "params": state.params,
                  "sequence": sequence,
                  "valid_tokens": valid_token_list
                  }
    return full_model

def get_model_dict(grammar_masker: GrammarMasker, prompt: list[str]):
    state = grammar_masker.init_state()
    sequence = ["BOS"]
    prev_token = "BOS"
    search_space = 1

    if prompt[0] == "BOS": prompt = prompt[1:]
    if prompt[-1] == "EOS": prompt = prompt[:-1]

    for token in prompt:
        valid_tokens = grammar_masker.get_valid_tokens(state, prev_token)

        # ----------------------- CHECK VALID TOKENS -----------------------
        if not valid_tokens:
            print(f"Warning: No valid tokens after {prev_token} (in prompt)")
            break
        if token not in valid_tokens: 
            print(f"Warning: Token '{token}' is not valid after {prev_token} (in prompt)")
            print(f"Valid tokens: {valid_tokens}")
            break
        # ----------------------- END CHECK VALID TOKENS -----------------------

        sequence.append(token)
        search_space *= len(valid_tokens)
        state = grammar_masker.step(state, token)
        prev_token = token
    
    full_model = {"gauge_groups": state.gauge_groups,
                  "vevs": state.vevs,
                  "multiplets": state.multiplets,
                  "interactions": state.interactions,
                  "anomalies": state.anomalies,
                  "params": state.params
                  }

    return full_model


def gen(shard_id, num_shards): 
    import numpy as np
    prompt = read_model_txt("dataset/prompts/SM_prompt.txt")

    if isinstance(shard_id, list):
        shard_id = shard_id[0]
    if isinstance(num_shards, list):
        num_shards = num_shards[0]
    print(f"Shard {shard_id} of {num_shards}")

    for i in range(shard_id*num_shards, (shard_id+1)*num_shards):
        grammar_masker = GrammarMasker()
        result = generate_theory(grammar_masker, prompt=prompt, print_tokens=False, seed=i)
        sequence = result["sequence"]
        valid_token_list = result["valid_tokens"]
        id_string = np.array(vocab.encode(sequence), dtype=np.uint8)
        yield {"inputs": id_string, 
               "valid_tokens": valid_token_list,
               "theory_id": i}

def main(TOTAL: int = 100, NUM_PROC: int = 1, save_path: str = "dataset/theory_test"):
    from datasets import Dataset
    num_shards = TOTAL // NUM_PROC
    dataset = Dataset.from_generator(
        gen,
        num_proc=NUM_PROC,
        gen_kwargs={
            "shard_id": list(range(NUM_PROC)),
            "num_shards": num_shards
        }
    )
    dataset.save_to_disk(save_path)
        

if __name__ == "__main__":
    main(TOTAL = 100_000, NUM_PROC = 10, save_path = "dataset/pretrain_data_100k")

    # dataset = Dataset.load_from_disk("dataset/pretrain_data")
    # print(len(dataset))
    # print(dataset[100])