
import os, warnings
import joblib
from sklearn.preprocessing import LabelEncoder
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}
warnings.filterwarnings('ignore') 


import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences



PREPROCESSOR_FNAME = "preprocessor.save"

class DataPreProcessor(): 
    def __init__(self, max_vocab_size, max_len=None, 
                 padding_pre=True, truncating_pre=True, pad_value=0.) -> None:
        self.max_vocab_size = max_vocab_size
        self.max_len = max_len
        self.padding = 'pre' if padding_pre else 'post'
        self.truncating = 'pre' if truncating_pre else 'post'
        self.pad_value = pad_value
        self.tokenizer = Tokenizer(num_words=self.max_vocab_size)
        self.label_encoder = LabelEncoder()
        self.classes_ = None
        
        
    def fit(self, texts, labels ): 
        self._fit_on_texts(texts)
        self._fit_labels(labels)     
        self.classes_ = self.label_encoder.classes_
        
        
    def transform(self, texts, labels = None ):
        sequences = self._texts_to_sequences(text_as_series=texts)
        if labels is not None:  labels = self._transform_labels(labels=labels)
        return sequences, labels
    
    
    def fit_transform(self, texts, labels): 
        self.fit(texts, labels)
        return self.transform(texts, labels)
    
        
    def _fit_on_texts(self, text_as_series):
        self.tokenizer.fit_on_texts(text_as_series)
        # print("vocab size", len(self.tokenizer.word_index))
        
        
    def _texts_to_sequences(self, text_as_series):
        sequences = self.tokenizer.texts_to_sequences(text_as_series)
        sequences = pad_sequences(sequences, 
                maxlen = self.max_len,
                padding = self.padding,
                truncating = self.truncating,
                value=self.pad_value
            )
        return sequences
        
        
    def _fit_labels(self, labels): 
        self.label_encoder.fit(labels)
        
    def _transform_labels(self, labels): 
        encoded_labels = self.label_encoder.transform(labels)
        return encoded_labels
       
    
    def save(self, model_path): 
        try: 
            joblib.dump(self, os.path.join(model_path, PREPROCESSOR_FNAME))   
        except: 
            raise Exception(f'''
                Error saving the preprocessor. 
                Does the file path exist {model_path}?''')  
        return   
    
    
    @classmethod
    def load(cls, model_path):
        file_path_and_name = os.path.join(model_path, PREPROCESSOR_FNAME)
        if not os.path.exists(file_path_and_name):
            raise Exception(f'''Error: No trained preprocessor found. 
            Expected to find file in path: {model_path}''')            
        try: 
            preprocessor = joblib.load(file_path_and_name)     
        except: 
            raise Exception(f'''
                Error loading the preprocessor. 
                Do you have the right trained preprocessor at {file_path_and_name}?''')
        
        return preprocessor 
    
    
    
def load_preprocessor(file_path):
    file_path_and_name = os.path.join(file_path, PREPROCESSOR_FNAME)
    if not os.path.exists(file_path_and_name):
        raise Exception(f'''Error: No trained preprocessor found. 
        Expected to find model files in path: {file_path_and_name}''')        
    try: 
        preprocessor = joblib.load(file_path_and_name)     
    except: 
        raise Exception(f'''
            Error loading the preprocessor. 
            Do you have the right trained preprocessor at {file_path_and_name}?''')
    
    return preprocessor 
    