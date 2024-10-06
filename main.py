from contextlib import asynccontextmanager
import uvicorn as uvicorn
from fastapi import FastAPI, Query
import logging

from transformer_architecture.batchs.tensorflow_batch import create_tensor_slices_load_to_cache_and_do_prefetch
from transformer_architecture.paddings.tensorflow_paddings import set_paddings
from transformer_architecture.tokenizers.tensorflow_tokenizer import tokenize
from transformer_architecture.transformer.Transformer import Transformer
from transformer_architecture.utils.data_utils import load_data, check_load_datafiles, clean_data

__max_length__ = 15

from transformer_architecture.utils.developer_utils import remove_long_sentences

if __name__ == "__main__":
    #Open files and load data
    logging.info("Open files and load data")
    english = load_data('utils/files/input/english.txt')
    portuguese = load_data('utils/files/input/portuguese.txt')
    check_load_datafiles(english, portuguese, log_it=False)

    #Remove dots and double spaces
    logging.info("Remove dots and double spaces")
    cleaned_en_data, cleaned_pt_data = clean_data(english, portuguese)

    '''
        DATA TOKENIZING
        input_en_data_tokenized:SubwordTextEncoder(inputs of the Neural Networks)
        output_pt_data_tokenized:SubwordTextEncoder(outputs of the Neural Networks - Predictions)
    '''
    logging.info("Data Tokenizing")
    input_en_data_tokenized = tokenize(cleaned_en_data)
    output_pt_data_tokenized = tokenize(cleaned_pt_data)

    '''
        FOR DEVELOP ENVIRONMENT
        Removing long sentences to execute this rapidly for development tests. 
        FOR PRODUCTION ENVIRONMENT
        Comment this code.
    '''
    logging.info("Removing long sentences to execute this rapidly for development tests")
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

    uvicorn.run("main:app", port=8000, log_level="info", reload=True)

@asynccontextmanager
async def lifespan(app: FastAPI):
    print('init method')
    yield
app = FastAPI(lifespan=lifespan)

@app.put("/training")
async def training_model():
    Transformer().training_model(inputs, outputs, dataset)
    return "test a training model"


@app.get("/translate")
def translate(sentence: str = Query(None)):
    return "test at the translate"


