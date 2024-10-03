from transformerProject.tokenizers.tensorflow_tokenizer import tokenize
from transformerProject.utils.data_utils import load_data, split_data, compare_length, print_data_to_check_it, \
    clean_data, check_load_dataset

__max_length__ = 15

from transformerProject.utils.developer_utils import remove_long_sentences

english = load_data('utils/files/english.txt')
portuguese = load_data('utils/files/portuguese.txt')

# check_load_dataset(english, portuguese, log_it=False)

cleaned_en_data, cleaned_pt_data = clean_data(english, portuguese)

input_en_data_tokenized = tokenize(cleaned_en_data)
output_pt_data_tokenized = tokenize(cleaned_pt_data)

#Removing long sentences to execute this rapidly for development tests
remove_long_sentences(input_en_data_tokenized, output_pt_data_tokenized, __max_length__)

len(input_en_data_tokenized)
len(output_pt_data_tokenized)