import pandas as pd
import numpy as np
from transformers import AutoTokenizer, BertForMaskedLM, BertTokenizer
from transformers import AdamW
import torch
import nltk

tokenizer = AutoTokenizer.from_pretrained("./AutoTokenizer")
#tokenizer = BertTokenizer("/home/lxh/Documents/Ontology/all_vocab.txt")
model = BertForMaskedLM.from_pretrained("bert-base-uncased")
#model = BertForMaskedLM.from_pretrained("./model_best")
model.eval()
model.cuda()

optimizer_grouped_parameters = []
for key, value in dict(model.named_parameters()).items():
    optimizer_grouped_parameters += [{'params': [value], 'lr': 1e-4,'weight_decay': 0.0005}]
optimizer = AdamW(optimizer_grouped_parameters)

"""load data"""
file = open('/home/lxh/Documents/Ontology/extract_all.txt', "r") #"/home/lxh/Documents/Ontology/extract_all.txt"
text_list = file.readlines()
file.close()

# domain vocab
file_vocab = open('./AutoTokenizer/vocab.txt','r')
vocab = file_vocab.readlines()
file_vocab.close()
domain_vocab = [vocab[i][:-2] for i in range(862,999)]

epochs_total = 100
th_words_length = 3
for epoch in range(epochs_total):
    for i in range(len(text_list)):
        parapraph = text_list[i]
        if len(parapraph) < 100 or len(parapraph) > 2000:
            continue
        sentence_list = nltk.sent_tokenize(parapraph)
        sentence_list_input = []
        sentence_list_label = []
        word_masked_list = []
        for j in range(len(sentence_list)):
            words = nltk.word_tokenize(sentence_list[j])
            words = [w.lower() for w in words]
            sentence_label = " ".join(words)
            words_length = np.array([len(w) for w in words])
            mask_long_words = words_length>th_words_length
            idx = np.where(words_length>th_words_length)[0]
            if idx.shape[0] < 2:
                continue
            
            idx_mask = -1
            for k in range(len(words)):
                w = words[k]
                if w in domain_vocab:
                    idx_mask = k
                    break
            if idx_mask < 0:
                continue
                
            word_masked_list.append(words[idx_mask])
            words[idx_mask] = '[MASK]'
            sentence_input = " ".join(words)
            sentence_list_input.append(sentence_input)
            sentence_list_label.append(sentence_label)                        
        if not len(sentence_list_input):
            continue
        
        # add mask
        inputs = tokenizer(sentence_list_input,
                            truncation=True,
                            padding="max_length",
                            max_length=128,
                            return_tensors="pt")
        input_ids = inputs['input_ids'].cuda()
        attention_mask = inputs['attention_mask'].cuda()
        token_type_ids = inputs['token_type_ids'].cuda()        
        # testing
        if False:
            logits = model(input_ids,attention_mask,token_type_ids).logits
            # retrieve index of [MASK]
            mask_token_index = (inputs.input_ids == tokenizer.mask_token_id).nonzero(as_tuple=True)
            predicted_token_id = logits[mask_token_index[0], mask_token_index[1], :].argmax(axis=-1)
            print(tokenizer.decode(predicted_token_id))
            print(word_masked_list)
        else:
            # mask labels of non-[MASK] tokens
            labels = tokenizer(sentence_list_label,
                                truncation=True,
                                padding="max_length",
                                max_length=128,
                                return_tensors="pt")
            labels_ids = labels['input_ids'].cuda()
            labels = torch.where(input_ids==tokenizer.mask_token_id, labels_ids, -100)
            outputs = model(input_ids,attention_mask,token_type_ids, labels=labels)
            loss = outputs.loss
            loss.backward()
            optimizer.step()      
            if i%10==0:
                print([epoch, i, len(text_list), loss.item()])
            if i%2000==0:
                model.save_pretrained('./model_best')