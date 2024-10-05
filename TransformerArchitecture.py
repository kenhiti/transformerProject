from keras import backend as backend

from transformer_architecture.batchs.tensorflow_batch import create_tensor_slices_load_to_cache_and_do_prefetch
from transformer_architecture.paddings.tensorflow_paddings import set_paddings
from transformer_architecture.tokenizers.tensorflow_tokenizer import tokenize
from transformer_architecture.transformer.TensorFlowTransformer import TensorFlowTransformer
from transformer_architecture.utils.data_utils import load_data, clean_data, check_load_dataset
from transformer_architecture.utils.developer_utils import remove_long_sentences


def main():
    pass

if __name__ == '__main__':
    main()

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

#Training.....
backend.clear_session()

'''
For d_model variable, the recommended value is 512. We are set 128.
For nb_layers variable, the recommended value is 6 layers. We are set 4.
For ffn_units variable, the recommended value is 2048. We are set 512.
All the information above is to run in a local environment and quickly.
'''

d_model = 128
nb_layers = 4
ffn_units = 512
nb_proj = 8
dropout_rate = 0.1

transformer = TensorFlowTransformer(vocab_size_enc=input_en_data_tokenized,
                                    vocab_size_dec=output_pt_data_tokenized,
                                    d_model=d_model,
                                    nb_layers=nb_layers,
                                    FFN_units   =ffn_units,
                                    nb_proj=nb_proj,
                                    dropout_rate=dropout_rate)
