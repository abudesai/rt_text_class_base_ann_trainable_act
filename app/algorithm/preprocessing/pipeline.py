
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
import numpy as np, pandas as pd 
from sklearn.pipeline import Pipeline, FeatureUnion, _fit_transform_one, _transform_one
from scipy import sparse
import sys, os
import joblib
from joblib import Parallel, delayed


import algorithm.preprocessing.preprocessors as preprocessors


PREPROCESSOR_FNAME = "preprocessor.save"



'''

PRE-POCESSING STEPS =====>


=========== for text (document column) ========
- 

=========== for target variable ========
- 
===============================================
'''

def get_preprocess_pipeline(pp_params, model_cfg): 
    pp_step_names = model_cfg["pp_params"]["pp_step_names"]        
    text_pipeline = get_text_pipeline(pp_params = pp_params, model_cfg=model_cfg)
    target_pipeline = get_target_pipeline(pp_params = pp_params, model_cfg=model_cfg)
    
    main_pipeline = Pipeline(
        [
            (
                pp_step_names["TARGET_FEATURE_ADDER"],
                preprocessors.TargetFeatureAdder(
                    target_col=pp_params['target_field'],
                    fill_value = model_cfg['target_dummy_val']
                    ),
            ),
            (
                pp_step_names["FEATURE_UNION"], PandasFeatureUnion(
                    [
                        ( pp_step_names["TEXT_PIPELINE"], text_pipeline ),
                        ( pp_step_names["TARGET_PIPELINE"], target_pipeline ),
                        (
                            pp_step_names["ID_SELECTOR"], 
                            preprocessors.ColumnsSelector(
                                columns=pp_params['id_field'],
                                selector_type='keep'
                            ) 
                        )
                    ]
                )
            ),
            (
                pp_step_names["XYSPLITTER"], 
                preprocessors.XYSplitter(
                    target_col=pp_params['target_field'],
                    id_col=pp_params['id_field'],
                    ),
            )
        ]
    )
    
    return main_pipeline
    


def get_text_pipeline(pp_params, model_cfg):     
    pp_step_names = model_cfg["pp_params"]["pp_step_names"]    
    pipe_steps = []     
    
    # select the text column
    pipe_steps.append(
        (
            pp_step_names["CUSTOM_TOKENIZER"], 
            preprocessors.CustomTokenizerWithLimitedVocab(
                text_col = pp_params['document_field'],
                vocab_size = 5000,
                keep_words=[], 
                start_token=None, 
                end_token=None            
            )
        )
    )   
    # select the text column
    pipe_steps.append(
        (
            pp_step_names["TEXT_SELECTOR"], 
            preprocessors.ColumnSelector(pp_params['document_field'])
        )
    )    
    # tf-idf vectorize
    pipe_steps.append(
        (
            pp_step_names["TF_IDF"], 
            TfidfVectorizer(
                # tokenizer=preprocessors.Tokenizer, # handled separately in step above
                min_df=.0025, 
                max_df=0.9, 
                ngram_range=(1,1)
            )
        )
    )    
    # apply truncated svd
    pipe_steps.append(
        (
            pp_step_names["SVD"], 
            TruncatedSVD(
                algorithm='randomized', 
                n_components=300
            )
        )
    )    
    # convert back to df
    pipe_steps.append(
        (
            pp_step_names["ARRAY_TO_DF_CONVERTER"], 
            preprocessors.ArrayToDataFrameConverter()
        )
    )           
    text_pipeline = Pipeline( pipe_steps )    
    return text_pipeline


def get_target_pipeline(pp_params, model_cfg):     
    pp_step_names = model_cfg["pp_params"]["pp_step_names"]    
    pipe_steps = []       
    # select the text column
    pipe_steps.append(
        (
            pp_step_names["TARGET_SELECTOR"], 
            preprocessors.ColumnsSelector(
                columns=pp_params['target_field'],
                selector_type='keep'
                )
        )
    )         
    # label binarizer
    pipe_steps.append(
        (
            pp_step_names["LABEL_ENCODER"],
            preprocessors.CustomLabelEncoder( 
                target_col=pp_params['target_field'],
                dummy_label=model_cfg['target_dummy_val'],
                ),
        )
    )  
           
    target_pipeline = Pipeline( pipe_steps )    
    return target_pipeline



class PandasFeatureUnion(FeatureUnion):    
    def fit_transform(self, X, y=None, **fit_params):
        self._validate_transformers()

        result = Parallel(n_jobs=self.n_jobs)(
            delayed(_fit_transform_one)( transformer = trans, X=X,  y=y, weight=weight, **fit_params)
            for name, trans, weight in self._iter())
                        
        if not result:
            # All transformers are None
            return np.zeros((X.shape[0], 0))
        Xs, transformers = zip(*result)
        self._update_transformer_list(transformers)
        if any(sparse.issparse(f) for f in Xs):
            Xs = sparse.hstack(Xs).tocsr()
        else:
            Xs = self.merge_dataframes_by_column(Xs)
        return Xs

    def merge_dataframes_by_column(self, Xs):
        Xs = [ X.reset_index(drop=True) for X in Xs]
        merged = pd.concat(Xs, axis="columns", copy=False)
        return merged

    def transform(self, X):        
        Xs = Parallel(n_jobs=self.n_jobs)(
            delayed(_transform_one)(transformer=trans, X=X,  y=None, weight=weight)
            for name, trans, weight in self._iter())
        if not Xs:
            # All transformers are None
            return np.zeros((X.shape[0], 0))
        if any(sparse.issparse(f) for f in Xs):
            Xs = sparse.hstack(Xs).tocsr()
        else:
            Xs = self.merge_dataframes_by_column(Xs)
        return Xs



def get_class_names(pipeline, model_cfg):
    pp_step_names = model_cfg["pp_params"]["pp_step_names"]   
    lbl_binarizer = None
    for t in pipeline[pp_step_names['FEATURE_UNION']].transformer_list:
        if t[0] == pp_step_names["TARGET_PIPELINE"]:
            target_pipeline = t[1]
            for step_tup in target_pipeline.steps:
                if step_tup[0] == pp_step_names['LABEL_ENCODER']:
                    lbl_binarizer = step_tup[1]
                    class_names = lbl_binarizer.classes_
                    break
    
    if lbl_binarizer is None: 
        raise Exception("Error: Cannot find lbl_binarizer in pipeline.")  
    return class_names

 
    

def save_preprocessor(preprocess_pipe, file_path):
    file_path_and_name = os.path.join(file_path, PREPROCESSOR_FNAME)
    try: 
        joblib.dump(preprocess_pipe, file_path_and_name)   
    except: 
        raise Exception(f'''
            Error saving the preprocessor. 
            Does the file path exist {file_path}?''')  
    return    
    

def load_preprocessor(file_path):
    file_path_and_name = os.path.join(file_path, PREPROCESSOR_FNAME)
    if not os.path.exists(file_path_and_name):
        raise Exception(f'''Error: No trained preprocessor found. 
        Expected to find model files in path: {file_path_and_name}''')
        
    try: 
        preprocess_pipe = joblib.load(file_path_and_name)     
    except: 
        raise Exception(f'''
            Error loading the preprocessor. 
            Do you have the right trained preprocessor at {file_path_and_name}?''')
    
    return preprocess_pipe 
    