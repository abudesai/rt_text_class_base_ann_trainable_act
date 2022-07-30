
from pydoc import doc
import numpy as np, pandas as pd, random
import sys, os
import glob
import json



def set_seeds(seed_value=42):
    if type(seed_value) == int or type(seed_value) == float:          
        os.environ['PYTHONHASHSEED']=str(seed_value)
        random.seed(seed_value)
        np.random.seed(seed_value)
    else: 
        print(f"Invalid seed value: {seed_value}. Cannot set seeds.")


def read_documents_into_df(dir_path, class_ = None): 
    file_list = glob.glob(dir_path + "/*.txt")
    documents = []
    for f in file_list: 
        with open(f,mode='r', encoding="utf-8") as fh:
            documents.append([os.path.basename(f), fh.read()])    
    df = pd.DataFrame(documents, columns=["id", "text"])
    if class_ is not None:  df["class"] = class_        
    return df


def get_data(data_path, data_type):   
    print("Reading data...")  
    data = []
    if data_type == "training": 
        class_names = os.listdir(data_path)        
        for class_ in class_names:           
            data.append(read_documents_into_df(os.path.join(data_path, class_), class_= class_) )
    else: 
        data.append(read_documents_into_df(data_path) )    
    data = pd.concat(data, ignore_index=True)
    return data


def get_data_schema(data_schema_path): 
    try: 
        json_files = list(filter(lambda f: f.endswith('.json'), os.listdir(data_schema_path) )) 
        if len(json_files) > 1: raise Exception(f'Multiple json files found in {data_schema_path}. Expecting only one schema file.')
        full_fpath = os.path.join(data_schema_path, json_files[0])
        with open(full_fpath, 'r') as f:
            data_schema = json.load(f)
            return data_schema  
    except: 
        raise Exception(f"Error reading data_schema file at: {data_schema_path}")  
      

def get_json_file(file_path, file_type): 
    try:
        json_data = json.load(open(file_path)) 
        return json_data
    except: 
        raise Exception(f"Error reading {file_type} file at: {file_path}")   


def get_hyperparameters(hyper_param_path): 
    hyperparameters_path = os.path.join(hyper_param_path, 'hyperparameters.json')
    return get_json_file(hyperparameters_path, "hyperparameters")


def get_model_config():
    model_cfg_path = os.path.join(os.path.dirname(__file__), 'config', 'model_config.json')
    return get_json_file(model_cfg_path, "model config")


def get_hpt_specs():
    hpt_params_path = os.path.join(os.path.dirname(__file__), 'config', 'hpt_params.json')
    return get_json_file(hpt_params_path, "HPT config")
    

def save_json(file_path_and_name, data):
    """Save json to a path (directory + filename)"""
    with open(file_path_and_name, 'w') as f:
        json.dump( data,  f, 
                  default=lambda o: make_serializable(o), 
                  sort_keys=True, 
                  indent=4, 
                  separators=(',', ': ') 
                  )

def make_serializable(obj): 
    if isinstance(obj, (int, np.integer)):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    else:
        return json.JSONEncoder.default(None, obj)


def print_json(result):
    """Pretty-print a jsonable structure"""
    print(json.dumps(
        result,
        default=lambda o: make_serializable(o), 
        sort_keys=True,
        indent=4, separators=(',', ': ')
    ))
    

def save_dataframe(df, save_path, file_name): 
    df.to_csv(os.path.join(save_path, file_name), index=False)
    