from grammar.gen_theory import generate_theory
from grammar.parser import read_model_txt, render_sequence, render_model
from grammar.masker import GrammarMasker

import grammar.reward as reward
import matplotlib.pyplot as plt

from SARAH_interface.write_model import SARAHFile
prompt = read_model_txt("dataset/prompts/SM_full.txt")
#prompt = read_model_txt("dataset/prompts/SM_gauge_prompt.txt")
#prompt = read_model_txt("dataset/prompts/debug.txt")
all_rewards = []
perfect_case = 0

grammar_masker = GrammarMasker()
full_model = generate_theory(grammar_masker, 
                             prompt=prompt, 
                             print_tokens=False, 
                             seed=None)
#print(render_model(full_model))

standard_model = SARAHFile(model_name="SM", author="AI", full_model=full_model)
standard_model.output_sarah()
standard_model.get_free_parameters()