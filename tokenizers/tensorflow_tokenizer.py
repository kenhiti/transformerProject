import random
import time

import tensorflow_datasets as tfds
from tensorflow_datasets.core.deprecated.text import SubwordTextEncoder

from transformer_architecture.utils.inspect_utils import get_variable_name

'''
Target vocabulary size maximum value is 2**13 (8192) 
'''
def tokenize(str_list: list[str]):
    start = time.perf_counter()
    subword_text_encoder = tfds.deprecated.text.SubwordTextEncoder.build_from_corpus(str_list, target_vocab_size=2 ** 13)
    end = time.perf_counter()
    print(f'Vocab size is: {subword_text_encoder.vocab_size} to dataset: {get_variable_name(str_list)}, duration: {end - start} seconds' )
    return set_sos_eos_in_tokenized_list(subword_text_encoder, str_list)
tokenize.__doc__= "A function to tokenize a list of the phrases (convert from text to number) "

'''
The start of the sentence receive vocabulary size 
'''
def set_sos_eos_in_tokenized_list(encoded_object: SubwordTextEncoder, str_list: list[str]):
    vocab_size = encoded_object.vocab_size
    text_encoding = [[vocab_size] + encoded_object.encode(sentence) + [vocab_size + 1] for sentence in str_list]
    __print_encoded_list__(text_encoding)
    return text_encoding
set_sos_eos_in_tokenized_list.__doc__ = "A function to set the start of the sentence and the end of the sentence to each phrase"

def __print_encoded_list__(str_list):
    # for _ in range(5):
    #     print(f'The random line: {str_list[random.randint(0, len(str_list) - 1)]}')
    for idx in range(5):
        print(f'The fixed line: {str_list[idx]}')
__print_encoded_list__.__doc__ = "A simple function to print data to check tokenizers"

