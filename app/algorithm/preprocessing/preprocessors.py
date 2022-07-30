
import numpy as np, pandas as pd
import nltk
import re
import sys , os
import operator
from sklearn.preprocessing import LabelEncoder
from sklearn.base import BaseEstimator, TransformerMixin

stop_words_path = os.path.join(os.path.dirname(__file__), 'stop_words.txt')
stopwords = set(w.rstrip() for w in open(stop_words_path))
porter_stemmer=nltk.PorterStemmer()


def Tokenizer(str_input):
    tokens = tokenize(str_input)
    return tokens


def tokenize(document):
    tokens = re.sub(r"[^A-Za-z0-9\-]", " ", document).lower().split()
    tokens = [t for t in tokens if len(t) > 2] # remove short words, they're probably not useful
    tokens = [t for t in tokens if t not in stopwords] # remove stopwords
    porter_stemmer=nltk.PorterStemmer()
    tokens = [porter_stemmer.stem(t) for t in tokens]
    return tokens


class CustomTokenizerWithLimitedVocab(BaseEstimator, TransformerMixin):
    '''
    Tokenizes the text column in given dataframe. 
    Can be used to limit the vocabulary size. 
    text_col: name of field with text in dataframe
    vocab_size: max vocab size to use
    keep_words: words to keep regardless of their frequency
    start_token: token used to indicate start of text (if any)
    end_token: token used to indicate end of text (if any)    
    
    Original code referenced from here: 
    https://github.com/lazyprogrammer/machine_learning_examples/blob/master/rnn_class/brown.py
    
    '''   
    
    def __init__(self, text_col, vocab_size=5000, keep_words=[], start_token=None, end_token=None):
        self.text_col = text_col
        self.vocab_size = vocab_size
        self.keep_words = keep_words
        self.START_TOKEN = start_token
        self.END_TOKEN = end_token
        
        
    def fit(self, data):
        sentences = list(data[self.text_col])
        
        word2idx = {}
        word_idx_count = {}
        idx2word = []
        i = 0
        if self.START_TOKEN is not None: 
            word2idx[self.START_TOKEN] = 0
            idx2word.append(self.START_TOKEN)
            word_idx_count[0] = float('inf')
            i += 1
            
        if self.END_TOKEN is not None: 
            word2idx[self.END_TOKEN] = 1
            idx2word.append(self.END_TOKEN)
            word_idx_count[1] = float('inf')
            i += 2     
  
        for sentence in sentences:
            tokens = tokenize(sentence)
            for token in tokens:
                if token not in word2idx:
                    idx2word.append(token)
                    word2idx[token] = i
                    i += 1 
                
                # keep track of counts for later sorting
                idx = word2idx[token]
                word_idx_count[idx] = word_idx_count.get(idx, 0) + 1
        
        # set all the words we want to keep to infinity
        # so that they are included when we pick the most
        # common words
        for word in self.keep_words:
            word_idx_count[word2idx[word]] = float('inf')
        
        # sort words in decreasing order of counts
        sorted_word_idx_count = sorted(word_idx_count.items(), key=operator.itemgetter(1), reverse=True)
        
        word2idx_small = {}
        new_idx = 0
        idx_new_idx_map = {}
        for idx, count in sorted_word_idx_count[:self.vocab_size]:
            word = idx2word[idx]
            word2idx_small[word] = new_idx
            idx_new_idx_map[idx] = new_idx
            new_idx += 1
        # let 'unknown' be the last token
        word2idx_small['UNKNOWN'] = new_idx 
        
        self.word2idx_small = word2idx_small
        self.unknown = new_idx        
        return self
    
        
    def transform(self, data): 
        sentences = list(data[self.text_col])
        sentences_small = []
        for sentence in sentences:
            tokens = tokenize(sentence)
            new_sentence = [ token #str(self.word2idx_small[token])
                                if token in self.word2idx_small else 'UNKNOWN'
                                for token in tokens]
            new_sentence = " ".join(new_sentence)
            sentences_small.append(new_sentence)
        
        data[self.text_col] = sentences_small        
        return data


class TargetFeatureAdder(BaseEstimator, TransformerMixin): 
    def __init__(self, target_col, fill_value) -> None:
        super().__init__()
        self.target_col = target_col
        self.fill_value = fill_value
    
    def fit(self, data): return self
    
    def transform(self, data): 
        if self.target_col not in data.columns: 
            data[self.target_col] = self.fill_value
        return data



class ColumnSelector(BaseEstimator, TransformerMixin):
    def __init__(self, col):
        self.col = col
        
    def fit(self, X, y=None):  return self
    
    def transform(self, X): return X[self.col]
    
    
class TypeCaster(BaseEstimator, TransformerMixin):  
    def __init__(self, vars, cast_type):
        super().__init__()
        self.vars = vars
        self.cast_type = cast_type
        
    def fit(self, X, y=None): return self
        

    def transform(self, data):  
        data = data.copy()
        applied_cols = [col for col in self.vars if col in data.columns] 
        for var in applied_cols: 
            data[var] = data[var].apply(self.cast_type)
        return data


class StringTypeCaster(TypeCaster):  
    ''' Casts categorical features as object type if they are not already so.
    This is needed when some categorical features have values that can inferred as numerical.
    This causes an error when doing categorical feature engineering. 
    '''
    def __init__(self, cat_vars): 
        super(StringTypeCaster, self).__init__(cat_vars, str)


class FloatTypeCaster(TypeCaster):  
    ''' Casts float features as object type if they are not already so.
    This is needed when some categorical features have values that can inferred as numerical.
    This causes an error when doing categorical feature engineering. 
    '''
    def __init__(self, num_vars):
        super(FloatTypeCaster, self).__init__(num_vars, float)


class ColumnsSelector(BaseEstimator, TransformerMixin):
    """Select only specified columns."""
    def __init__(self, columns, selector_type='keep'):
        self.columns = columns
        self.selector_type = selector_type
        
        
    def fit(self, X, y=None):
        return self
    
    
    def transform(self, X):  
        if self.selector_type == 'keep':
            retained_cols = [col for col in X.columns if col in self.columns]
            X = X[retained_cols].copy()
        elif self.selector_type == 'drop':
            dropped_cols = [col for col in X.columns if col in self.columns]  
            X = X.drop(dropped_cols, axis=1)      
        else: 
            raise Exception(f'''
                Error: Invalid selector_type. 
                Allowed values ['keep', 'drop']
                Given type = {self.selector_type} ''')   
        return X
    
 
class ArrayToDataFrameConverter(BaseEstimator, TransformerMixin): 
    def __init__(self):
        self.cols = None
    
    def fit(self, data): 
        _, D = data.shape
        self.columns = [f"_input_{i}" for i in range(D)]              
        return self    
    
    def transform(self, data): 
        N, D = data.shape         
        if len(self.columns) != D: 
            raise Exception("Error. Data has more columns ({D}) than in prior fitted data ({len(self.columns)}). ")
        df = pd.DataFrame(data, columns=self.columns)
        return df 
    


class CustomLabelEncoder(BaseEstimator, TransformerMixin): 
    def __init__(self, target_col, dummy_label) -> None:
        super().__init__()
        self.target_col = target_col
        self.dummy_label = dummy_label
        self.lb = LabelEncoder()


    def fit(self, data):                
        self.lb.fit(data[self.target_col])             
        self.classes_ = self.lb.classes_ 
        return self 
    
    
    def transform(self, data): 
        check_val_if_pred = data.loc[0, self.target_col]
        if self.target_col in data.columns and check_val_if_pred != self.dummy_label: 
            data[self.target_col] = self.lb.transform(data[self.target_col])
        return data



class XYSplitter(BaseEstimator, TransformerMixin): 
    def __init__(self, target_col, id_col):
        self.target_col = target_col
        self.id_col = id_col
    
    def fit(self, data): return self
    
    def transform(self, data): 
        if self.target_col in data.columns: 
            y = data[self.target_col].values
        else: 
            y = None
        
        not_X_cols = [ self.id_col, self.target_col ] 
        X_cols = [ col for col in data.columns if col not in not_X_cols ]        
        X = data[X_cols].values           
        return { 'X': X, 'y': y  }
    
        
    