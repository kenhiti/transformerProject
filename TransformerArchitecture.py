from transformer_architecture.batchs.tensorflow_batch import create_tensor_slices_load_to_cache_and_do_prefetch
from transformer_architecture.paddings.tensorflow_paddings import set_paddings
from transformer_architecture.tokenizers.tensorflow_tokenizer import tokenize
from transformer_architecture.utils.data_utils import load_data, clean_data, check_load_dataset
from transformer_architecture.utils.developer_utils import remove_long_sentences

__max_length__ = 15

english = load_data('utils/files/english.txt')
portuguese = load_data('utils/files/portuguese.txt')
check_load_dataset(english, portuguese, log_it=False)

cleaned_en_data, cleaned_pt_data = clean_data(english, portuguese)

input_en_data_tokenized = tokenize(cleaned_en_data)
output_pt_data_tokenized = tokenize(cleaned_pt_data)

#Removing long sentences to execute this rapidly for development tests
remove_long_sentences(input_en_data_tokenized, output_pt_data_tokenized, __max_length__)

print(f'Length of the english data tokenized is:{len(input_en_data_tokenized)}')
print(f'Length of the portuguese data tokenized is:{len(output_pt_data_tokenized)}')

#paddings
inputs = set_paddings(input_en_data_tokenized, __max_length__)
outputs = set_paddings(output_pt_data_tokenized, __max_length__)

#Create dataset object by tensorSlices , load this object to cache and do prefetch
dataset = create_tensor_slices_load_to_cache_and_do_prefetch(inputs, outputs)


