class Dataset(torch.utils.data.Dataset):

        def __init__(self, df,tokenizer):
            
           self.tokenizer=tokenizer
           self.labels = self.labels = [labels[label] for label in df['label']]
           self.texts = [tokenizer(text,padding='max_length', max_length = 128, truncation=True, return_tensors="pt") for text in df['Text']]
           self.sentences = [tokenizer(sentence ,padding='max_length', max_length = 128, truncation=True, return_tensors="pt") for sentence in df['sentences']]
            
        def classes(self):
            return self.labels

        def __len__(self):
            return len(self.labels)

        def get_batch_labels(self, idx):
             # Fetch a batch of labels
             return np.array(self.labels[idx])

        def get_batch_texts(self, idx):
            # Fetch a batch of inputs
            return self.texts[idx]

        def get_batch_sentences(self, idx):
            # Fetch a batch of inputs
             return self.sentences[idx]

        def __getitem__(self, idx):

            batch_texts = self.get_batch_texts(idx)
            batch_sentences=self.get_batch_sentences(idx)
            batch_y = self.get_batch_labels(idx)
            
            return batch_texts,batch_sentences,batch_y