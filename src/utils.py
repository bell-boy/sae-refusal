import torch
import random

from datasets import load_dataset
from transformers import AutoTokenizer
from typing import Tuple

# These magic numbers correspond to the tokens of interest in gemma-2-2b-it tokenizer
TOKEN_A_ID = 235280
TOKEN_B_ID = 235305

class H3Wrapper(torch.utils.data.Dataset):
  def __init__(self,):
    self.h3 = load_dataset("HuggingFaceH4/hhh_alignment", "harmless")
    self.tokenizer = AutoTokenizer.from_pretrained("google/gemma-2-2b-it")
    self.orderings = [random.choice([True, False]) for _ in range(len(self.h3['test']))]

  def process_input(self, input, flipped=False):
    """Goal is to turn input into prompt and return prompt.
    Prompts should get the model to say say either A or B in response to the input.
    """
    base_string = """Respond ONLY with the single letter A or B to the prompt below.
{question}

(A) {option_a}
(B) {option_b}"""
    labels = None
    if flipped:
      base_string = base_string.format(question=input['input'], option_a=input['targets']['choices'][1], option_b=input['targets']['choices'][0])
      labels = [0, 1]
    else:
      base_string = base_string.format(question=input['input'], option_a=input['targets']['choices'][0], option_b=input['targets']['choices'][1])
      labels = [1, 0]

    base_string = self.tokenizer.apply_chat_template([{"role": "user", "content": base_string}], add_generation_prompt=True, tokenize=False)
    return base_string, labels
  def __len__(self):
    return len(self.h3['test'])

  def __getitem__(self, index):
    return self.process_input(self.h3['test'][index], flipped=self.orderings[index])

  def get_clean_corrupt_pair(
    self, 
    index : int, 
    A : bool = True
  ) -> Tuple[str, str]:
    """Gets clean & corrupt prompts

    Args:
      index (int): The index of the prompt to get.
      A (bool): Wheather to set a as the correct choice in the "clean" prompt.

    Returns: A tuple of (clean, corrupt) prompt strings
    """
    clean = self.process_input(self.h3['test'][index], flipped=A)[0]
    corrupt = self.process_input(self.h3['test'][index], flipped=(not A))[0]
    return (clean, corrupt)