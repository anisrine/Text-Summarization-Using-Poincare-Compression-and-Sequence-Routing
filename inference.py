from tqdm import tqdm
from mamba import MambaClassifier
from rouge_score import rouge_scorer
from transformers import  BartTokenize, BartForConditionalGeneration

model = MambaClassifier()
model.load_state_dict(torch.load("mamba_clusterig_PEFT_100.pt")) 

bartmodel = BartForConditionalGeneration.from_pretrained('eugenesiow/bart-paraphrase')
bartmodel = bartmodelmodel.to(device)
tokenizer = BartTokenizer.from_pretrained('eugenesiow/bart-paraphrase')

scorer2 = rouge_scorer.RougeScorer(['rouge2'], use_stemmer=True)
scorer2 = rouge_scorer.RougeScorer(['rouge1'], use_stemmer=True)
scorerL = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)


warnings.filterwarnings("ignore")
tqdm.pandas()



#Generate summaries of the test data
def sentences_labels(model,test_data):
    
    predicted_labels=[]   
    test = Dataset(test_data)
    test_dataloader = torch.utils.data.DataLoader(test, batch_size=2)

    with torch.no_grad():

        for test_text,test_sentence, test_label in tqdm(test_dataloader):
            
              test_label = test_label
              test_text=test_text
              test_sentence=test_sentence            

              text_id = test_text['input_ids'].squeeze(1)
              text_id= torch.from_numpy(np.asarray(text_id.detach().cpu()))
            
              sentence_id = test_sentence['input_ids'].squeeze(1)
              sentence_id= torch.from_numpy(np.asarray(sentence_id.detach().cpu()))

              output = model(text_id, sentence_id)
              
              predicted_labels.append( output.argmax(dim=1) )

    return predicted_labels


def paraphrase ( input_sentence):      
      batch = tokenizer(input_sentence, return_tensors='pt')
      generated_ids = bartmodel.generate(batch['input_ids'])
      generated_sentence = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
      return generated_sentence[0]
    
def indexes(lst,value):
     inds=[]
     for index, element in enumerate(lst):    
        if element == value:
            inds.append(index)
     return inds

def get_summary(sentences,relevance):         
    inds=indexes(relevance,1)         
    paraphrase_sentences =""
    for index in inds:                
        paraphrase_sentences +=paraphrase(sentences[index])+" "          
    return paraphrase_sentences


#Evaluation
def evaluate(model, df_test):

    predicted_labels=sentences_labels(model,df_test)
    predicted_label=[]
    
    for label in predicted_labels:
         predicted_label.append(label[0].item())
         predicted_label.append(label[1].item())
    
    df_test["relevance"]=predicted_label    
    df_test_1= (df_test.groupby(['Summary','Text']).agg({'label':list,'relevance':list,'sentences':list}).reset_index())     
    df_test_1['text_summary']=df_test_1.progress_apply(lambda x:get_summary(x['sentences'],x['relevance']) , axis=1)
    return df_test_1


df_evaluation=evaluate(model, df_test[:380])

df_evaluation["rouge_score_1"]=df_tes.apply(lambda x:scorer.score(x['text_summary'],x['Summary'])['rouge1'].fmeasure,axis=1)
df_evaluation["rouge_score_2"]=df_tes.apply(lambda x:scorer.score(x['text_summary'],x['Summary'])['rouge2'].fmeasure,axis=1)
df_evaluation["rouge_score_L"]=df_tes.apply(lambda x:scorer.score(x['text_summary'],x['Summary'])['rougeL'].fmeasure,axis=1)


pd.set_option('display.max_colwidth', None)
print(df_evaluation['rouge_score_1'].mean())
print(df_evaluation['rouge_score_2'].mean())
print(df_evaluation['rouge_score_L'].mean())