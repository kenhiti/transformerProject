from contextlib import asynccontextmanager
from time import sleep

import uvicorn as uvicorn
from fastapi import FastAPI, Query
from transformer_architecture.optimzers.tensorflow_batch import create_tensor_slices_load_to_cache_and_do_prefetch
from transformer_architecture.paddings.tensorflow_paddings import set_paddings
from transformer_architecture.tokenizers.tensorflow_tokenizer import tokenize
from transformer_architecture.transformer.Transformer import Transformer
from transformer_architecture.utils.data_utils import load_data, check_load_datafiles, clean_data
from transformer_architecture.utils.developer_utils import remove_long_sentences

__max_length__ = 15

inputs = None
outputs = None
dataset = None

if __name__ == "__main__":
    #Open files and load data
    print("Open files and load data")
    english = load_data('utils/files/input/english.txt')
    portuguese = load_data('utils/files/input/portuguese.txt')
    check_load_datafiles(english, portuguese, log_it=False)

    #Remove dots and double spaces
    print("Remove dots and double spaces")
    cleaned_en_data, cleaned_pt_data = clean_data(english, portuguese)

    '''
        DATA TOKENIZING
        input_en_data_tokenized:SubwordTextEncoder(inputs of the Neural Networks)
        output_pt_data_tokenized:SubwordTextEncoder(outputs of the Neural Networks - Predictions)
    '''
    print("Data Tokenizing")
    input_en_data_tokenized = tokenize(cleaned_en_data)
    output_pt_data_tokenized = tokenize(cleaned_pt_data)

    '''
        FOR DEVELOP ENVIRONMENT
        Removing long sentences to execute this rapidly for development tests. 
        FOR PRODUCTION ENVIRONMENT
        Comment this code.
    '''
    print("Removing long sentences to execute this rapidly for development tests")
    remove_long_sentences(input_en_data_tokenized, output_pt_data_tokenized, __max_length__)

    print(f'Length of the english data tokenized is:{len(input_en_data_tokenized)}')
    print(f'Length of the portuguese data tokenized is:{len(output_pt_data_tokenized)}')

    '''
        PADDINGS
        Insert zeroes into the end of multi-dimensional arrays for all lines have to equals size
    '''
    inputs = set_paddings(input_en_data_tokenized, __max_length__)
    outputs = set_paddings(output_pt_data_tokenized, __max_length__)

    '''
        Create TensorFlow Dataset object by tensorSlices , load this object to cache and do prefetch
    '''
    dataset = create_tensor_slices_load_to_cache_and_do_prefetch(inputs, outputs)
    sleep(120)
    uvicorn.run("main:app", port=8000, log_level="info", reload=True)

@asynccontextmanager
async def lifespan(app: FastAPI):
    print('init lifespan method')
    yield
app = FastAPI(lifespan=lifespan)

@app.put("/training")
async def training_model():
    print('Start training model...')
    Transformer().training_model(inputs, outputs, dataset)
    return "test a training model"


@app.get("/translate")
def translate(sentence: str = Query(None)):
    return "test at the translate"


