import nltk
import sys
import torch
import os
import math
import re
import string
import warnings
from torch import nn
import numpy as np
import pandas as pd
from tqdm import tqdm
from dataset import Dataset
from mamba import MambaClassifier
from torch.optim import Adam,AdamW,SGD
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from textblob import TextBlob
from sklearn.decomposition import PCA
from collections import defaultdict
from nltk.stem import PorterStemmer
from collections import defaultdict
from sklearn.preprocessing import normalize
from sklearn.cluster import SpectralClustering
from sklearn.impute import SimpleImputer
from sklearn.metrics import pairwise_distances
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import cosine_similarity
from peft import get_peft_model,LoraConfig, TaskType,PromptEncoderConfig
from transformers importAutoTokenizer, MambaModel,get_linear_schedule_with_warmup,BertModel


torch.manual_seed(123)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
os.environ["TOKENIZERS_PARALLELISM"] = "false"
torch.backends.cudnn.benchmark = True
nltk.download('punkt')
nltk.download('stopwords')

labels = {'negative':0,'positive':1}
clusters = defaultdict(list)
centroids={} 
model= MambaClassifier()
LR =2e-5
EPOCHS =3
model_name="mamba_clustering_PEFT_100"



# Load the Mamba model from a pretrained model.
model = MambaModel.from_pretrained("state-spaces/mamba-130m-hf")#.to(device)

#PEFT configuration
peft_config = LoraConfig(task_type=TaskType.SEQ_CLS, inference_mode=False, r=8, lora_alpha=32, lora_dropout=0.1,target_modules=["x_proj", "embeddings", "in_proj", "out_proj"])

# Load the tokenizer of the Mamba model from "mamba-130m-hf"
tokenizer = AutoTokenizer.from_pretrained("state-spaces/mamba-130m-hf",ignore_mismatched_sizes=True)# EleutherAI/gpt-neox-20b,mamba2-130m

#Reduce Mamba model parameters using LoRa
mambamodel = get_peft_model(model, peft_config)

model.to(device)

#load dataset from cvs files

df_train = pd.read_csv('labeled_reviews_120_b.csv')
df_train = df_train.dropna()

df_test = pd.read_csv('labled_reviews_60_a.csv')
df_test=df_test.dropna()


def remove_urls(text):
    url_pattern = re.compile(r'https?://\S+')
    return url_pattern.sub('', text)

#removing Punctuation
def remove_punct(text):
     exclude = string.punctuation
     table = str.maketrans(dict.fromkeys(string.punctuation))
     return text.translate(table)

#Removing Stop Words
def remove_stopwords_sentence(sentence):
       stop_words = set(stopwords.words('english'))
       word_tokens = word_tokenize(sentence)
       filtered_sentence = []
       for w in word_tokens:
          if w not in stop_words:
             filtered_sentence.append(w)

       return " ".join(word_tokens)
     

def preprocess(df):
    
    df = df.map(lambda x: x.lower() if isinstance(x, str) else x)
    df['Text']=df['Text'].apply(remove_urls)
    df['Text']=df['Text'].apply(remove_punct)
    
    #remove digits
    df = df.replace(to_replace=r'\d', value='', regex=True)
    
    df['Text']=df['Text'].apply(remove_stopwords)
    df['Text']=df['Text'].apply(remove_stopwords) 
    
    return df

df_train=preprocess(df_train)
df_test=preprocess(df_test)


def train(model, train_data,val_data,learning_rate, epochs, mode_name):

    print("begining of the training : ")   
    
    train = Dataset(train_data)
    val=Dataset(val_data)
    train_dataloader = torch.utils.data.DataLoader(train, batch_size=2, shuffle=True,num_workers=2,pin_memory=True)
    val_dataloader = torch.utils.data.DataLoader(val, batch_size=2,shuffle=True,num_workers=2,pin_memory=True)
    
    criterion = nn.CrossEntropyLoss()    
    optimizer = AdamW(model.parameters(), lr= learning_rate, weight_decay=0.5)   
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=learning_rate, steps_per_epoch=len(train_dataloader), epochs=epochs)
    
    
    for epoch_num in range(epochs):          
         

            total_acc_train = 0
            total_loss_train = 0
            
            with torch.no_grad():
                
                for train_text,train_sentence, train_label in tqdm(train_dataloader):                   

                    text_train_ids = train_text["input_ids"].squeeze(1)
                    text_train_ids=torch.from_numpy(np.asarray(text_train_ids.detach().cpu()))

                    sentence_train_ids = train_sentence["input_ids"].squeeze(1)
                    sentence_train_ids=torch.from_numpy(np.asarray(sentence_train_ids.detach().cpu()))
                    
                    output=model(text_train_ids,sentence_train_ids)
                    
                    with torch.cuda.amp.autocast():
                       
                        batch_loss = criterion(output, train_label.long())
                        total_loss_train += batch_loss.item() 
                    

                    acc = (output.argmax(dim=1) == train_label).sum().item()                
                    total_acc_train += acc
                    
                    batch_loss.requires_grad = True                    
                    batch_loss.backward()
                    for param in model.parameters():
                         param.grad = None
                    optimizer.step()                    
                    scheduler.step()                 
     
            total_acc_val = 0
            total_loss_val = 0

            with torch.no_grad():

                   for val_text,val_sentence, val_label in tqdm(val_dataloader):

                        text_ids = val_text["input_ids"].squeeze(1)#.to(device)
                        text_ids=torch.from_numpy(np.asarray(text_ids.detach().cpu()))

                        sentence_ids = val_sentence["input_ids"].squeeze(1)#.to(device)
                        sentence_ids=torch.from_numpy(np.asarray(sentence_ids.detach().cpu()))

                        output=model(text_ids,sentence_ids)#.to(device)

                        batch_loss = criterion(output, val_label.long())
                        total_loss_val += batch_loss.item()                      
                        

                        acc = (output.argmax(dim=1) == val_label).sum().item()
                        total_acc_val += acc
            
           
            print(
                f'Epochs: {epoch_num + 1} | Train Loss: {total_loss_train / len(train_data): .3f} \
                | Train Accuracy: {total_acc_train / len(train_data): .3f} \
                | Val Loss: {total_loss_val / len(val_data): .3f} \
               | Val Accuracy: {total_acc_val / len(val_data): .3f}')
            
            
    torch.save(model.state_dict(),model_name)

#training
train(model, df_train[:792],df_test[:58], LR, EPOCHS,model_name)


                
    
    
    
    

