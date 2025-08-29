import os
from dotenv import load_dotenv
from huggingface_hub import login
from transformers import AutoModel, AutoTokenizer
from sentence_transformers import SentenceTransformer

load_dotenv()

hf_token = os.getenv("HF_TOKEN")
if hf_token:
    login(token=hf_token)

def download_hf_model(model_name, model_dir):
    '''
    Downloads model from Huggingface

    Parameters:
    - model_name(str): Name of model on huggingface
    - model_dir(os.path): Directory for where model will be saved

    '''
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name)
        # Save to a local path 
        tokenizer.save_pretrained(model_dir)
        model.save_pretrained(model_dir)
    except GatedRepoError as e:
        print(f'No access to gated model: {e}')
    except Exception as e:
        print(f'Unable to download model: {e}')


# To add models from huggingface
if __name__ == "__main__":
    model_name = "Snowflake/snowflake-arctic-embed-m"
    model_dir = "../agents/local_models/arctic-embed-m"
    download_hf_model(model_name, model_dir)

