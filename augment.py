import numpy as np
import pandas as pd
from pythainlp.augment import WordNetAug
import nlpaug.augmenter.word as naw
import nlpaug.augmenter.sentence as nas
import nltk
#nltk.download('punkt')
import nlpaug.augmenter.char as nac
#nltk.download('omw-1.4')

"""pip install pythainlp
pip install numpy git+https://github.com/makcedward/nlpaug.git
pip install transformers
pip install sacremoses"""

"""Load dataset"""
original_data = pd.read_csv('Eutrophication_A.csv')
data=original_data["Response"].tolist()
score=original_data["D"].tolist()
df_original = {}
df_original['Response'] = data
df_original['D'] = score
df_original = pd.DataFrame(df_original)

def augment_text(text):
    aug = naw.ContextualWordEmbsAug(model_path='bert-base-uncased', action="insert")
    augmented_text = aug.augment(text, n=3)
    return augmented_text

for idx in range(len(data)):
  print(idx)
  score_temp = score[idx]
  augment_response = augment_text(data[idx])
  data_aug = {}
  data_aug['Response']=augment_response
  data_aug['D']=[score_temp for i in range(len(augment_response))]
  df_temp = pd.DataFrame(data_aug)
  df_original = pd.concat([df_original, df_temp],ignore_index=True)

df_original.to_csv('./D_data_augment.csv')

