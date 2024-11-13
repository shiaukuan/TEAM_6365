#!/usr/bin/env python
# coding: utf-8

# In[ ]:





# # 使用 openbmb/MiniCPM-Embedding

# In[271]:


from transformers import AutoModel, AutoTokenizer
import torch
import torch.nn.functional as F

model_name = "openbmb/MiniCPM-Embedding"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model_emb = AutoModel.from_pretrained(model_name, trust_remote_code=True,  torch_dtype=torch.float16).to("cuda")
model_emb.eval()

# 由于在 `model.forward` 中缩放了最终隐层表示，此处的 mean pooling 实际上起到了 weighted mean pooling 的作用
# As we scale hidden states in `model.forward`, mean pooling here actually works as weighted mean pooling
def mean_pooling(hidden, attention_mask):
    s = torch.sum(hidden * attention_mask.unsqueeze(-1).float(), dim=1)
    d = attention_mask.sum(dim=1, keepdim=True).float()
    reps = s / d
    return reps

@torch.no_grad()
def encode(input_texts):
    batch_dict = tokenizer(input_texts, max_length=512, padding=True, truncation=True, return_tensors='pt', return_attention_mask=True).to("cuda")
    
    outputs = model_emb(**batch_dict)
    attention_mask = batch_dict["attention_mask"]
    hidden = outputs.last_hidden_state

    reps = mean_pooling(hidden, attention_mask)   
    embeddings = F.normalize(reps, p=2, dim=1).detach().cpu().numpy()
    return embeddings

# queries = ["中国的首都是哪里？"]
# passages = ["beijing", "shanghai"]

# INSTRUCTION = "Query: "
# queries = [INSTRUCTION + query for query in queries]

# embeddings_query = encode(queries)
# embeddings_doc = encode(passages)

# scores = (embeddings_query @ embeddings_doc.T)
# print(scores.tolist())  # [[0.3535913825035095, 0.18596848845481873]]


# # 使用 BAAI/bge-m3

# In[273]:


from FlagEmbedding import BGEM3FlagModel

model = BGEM3FlagModel('BAAI/bge-m3',  use_fp16=True) 


# In[274]:


import os
import pickle

import opencc
t2s = opencc.OpenCC('tw2sp.json')
t2s.convert('隨身碟')


# In[ ]:





# In[280]:


import pandas as pd
import json

ans_path="競賽資料集/dataset/preliminary/ground_truths_example.json"
with open(ans_path, 'rb') as f:
    ans = json.load(f)  # 讀取問題檔案

with open("競賽資料集/dataset/preliminary/questions_example.json", 'rb') as f:
    que = json.load(f) 
    
questions = pd.DataFrame( que['questions'] )

y=pd.DataFrame(ans['ground_truths'])
output=pd.merge(y,questions,on=['qid','category'])
output.loc[output[output['qid']==99].index[0],'retrieve']=693
output.loc[output[output['qid']==97].index[0],'retrieve']=579
output.loc[output[output['qid']==50].index[0],'source'].append(78)
output.loc[output[output['qid']==109].index[0],'source'].append(283)
output.loc[output[output['qid']==135].index[0],'source'].append(28)


# In[ ]:





# # 訓練資料

# In[291]:


faq=output[output['category'] == 'faq']

insurance=output[output['category'] == 'insurance']

finance=output[output['category'] == 'finance']


# In[ ]:





# # 預測資料

# In[287]:


faq=questions[questions['category'] == 'faq']

insurance=questions[questions['category'] == 'insurance']

finance=questions[questions['category'] == 'finance']


# In[ ]:





# In[ ]:





# In[285]:


fag_key={}
with open(os.path.join('競賽資料集/reference', 'faq/pid_map_content.json'), 'rb') as f_s:
    key_to_source_dict = json.load(f_s)  # 讀取參考資料文件
    key_to_source_dict = {int(key): value for key, value in key_to_source_dict.items()}
    for key, value in key_to_source_dict.items():
        fag_key[int(key)]=[t2s.convert(v['question']) for v in value]


# In[292]:


get_ipython().run_cell_magic('time', '', '\nans_list=[]\nfor row in faq.itertuples():\n    sentences_1=[ t2s.convert(row.query) ]\n\n    c_max=[]\n    c_mean=[]\n    for y_id in row.source:    \n        sentences_2=fag_key[y_id]\n        sentence_pairs = [[i,j] for i in sentences_1 for j in sentences_2]\n        score_list=model.compute_score(sentence_pairs, \n                          max_passage_length=250,\n                          weights_for_different_modes=[0.35, 0.3, 0.35])\n   \n        c_max.append(max(score_list[\'colbert+sparse+dense\']))\n        c_mean.append(sum(score_list[\'colbert+sparse+dense\'])/len(score_list[\'colbert+sparse+dense\']))\n\n\n    c_max_e=[]\n    c_mean_e=[]\n    sentences_1=["Query: "+t2s.convert(row.query) ]\n    for y_id in row.source:    \n        sentences_2=fag_key[y_id]\n        embeddings_1 = encode(sentences_1)\n        embeddings_2 = encode(sentences_2)\n        similarity = embeddings_1 @ embeddings_2.T\n        emb=similarity.tolist()\n        \n        c_max_e.append(max(emb[0]))\n        c_mean_e.append(sum(emb[0])/len(emb[0]))\n\n    \n    ans_list.append({\'qid\':row.qid, \'category\':row.category ,\'retrieve\':row.retrieve,\'source\':row.source,\'c_max\':c_max, \'c_mean\':c_mean,\'c_max_e\':c_max_e, \'c_mean_e\':c_mean_e})\n    \n    # ans_list.append({\'qid\':row.qid, \'category\':row.category ,\'source\':row.source,\'c_max\':c_max, \'c_mean\':c_mean,\'c_max_e\':c_max_e, \'c_mean_e\':c_mean_e})\n\n    \n        # break\n    # break\n')


# In[330]:


ans_df=pd.DataFrame(ans_list)


# In[ ]:





# In[334]:


ans_df['c_m']= ans_df.apply( lambda x: [a*0.7 + b*0.3 for a, b in zip(x['c_max'], x['c_mean'])] ,axis=1 )
ans_df['ans']=ans_df.apply( lambda x:  x['source'][x['c_m'].index(max(x['c_m']))] ,axis=1)


# In[335]:


ans_df['score']=ans_df.apply(lambda x:1 if x['retrieve']==x['ans'] else 0,axis=1)
ans_df['score'].sum()


# In[ ]:





# In[338]:


ans_df['c_m']= ans_df.apply( lambda x: [a*0 + b*1 for a, b in zip(x['c_max_e'], x['c_mean_e'])] ,axis=1 )
ans_df['ans']=ans_df.apply( lambda x:  x['source'][x['c_m'].index(max(x['c_m']))] ,axis=1)


# In[339]:


ans_df['score']=ans_df.apply(lambda x:1 if x['retrieve']==x['ans'] else 0,axis=1)
ans_df['score'].sum()


# In[ ]:





# In[340]:


ans_df['c_m']= ans_df.apply( lambda x: [a*0.4 + b*0.3 + c*0.2 + d*0.1 for a, b, c, d in zip(x['c_max'], x['c_max_e'],x['c_mean'],x['c_mean_e'])] ,axis=1 )
ans_df['ans']=ans_df.apply( lambda x:  x['source'][x['c_m'].index(max(x['c_m']))] ,axis=1)


# In[341]:


ans_df['score']=ans_df.apply(lambda x:1 if x['retrieve']==x['ans'] else 0,axis=1)
ans_df['score'].sum()


# In[ ]:





# In[ ]:





# In[342]:


with open("llm_s_ans_insurance.pkl", "rb") as f:
    llm_s_ans=pickle.load(f)
with open("llm_ans_insurance.pkl", "rb") as f:
    llm_ans=pickle.load(f)


# In[295]:


def parse_list(text):
    match = re.search(r'\[.*?\]', text, re.DOTALL)
    json_str = match.group(0)
    return eval(json_str)


# In[296]:


get_ipython().run_cell_magic('time', '', 'import re\ninsurance_key={}\ninsurance_s_key={}\n\nfor key in llm_ans.keys():\n    tmp1=[]\n    tmp2=[]\n    for v1, v2 in zip(llm_ans[key], llm_s_ans[key]):\n        v1=t2s.convert(v1[1])\n        v2=t2s.convert(v2[1])\n        try:\n            tmp1.extend(parse_list(v1))\n        except:\n            v1=v1.replace(\'未到期保险费会如何处理？\',\'未到期保险费会如何处理？"]\').replace(\'？」]\',\'？"]\')\n            tmp1.extend(parse_list(v1))\n            \n        tmp2.extend(parse_list(v2))\n\n    new_key=int(key.split(\'/\')[-1].strip(\'.pdf\'))\n    insurance_key[new_key]=tmp1\n    insurance_s_key[new_key]=tmp2\n')


# In[ ]:





# In[297]:


get_ipython().run_cell_magic('time', '', 'ans_list2=[]\nfor row in insurance.itertuples():\n    sentences_1=[ t2s.convert(row.query) ]\n\n    c_max=[]\n    c_mean=[]\n    for y_id in row.source:    \n        sentences_2=insurance_s_key[y_id]+insurance_key[y_id]\n        sentence_pairs = [[i,j] for i in sentences_1 for j in sentences_2]\n        \n        score_list=model.compute_score(sentence_pairs, \n                          max_passage_length=1024,\n                          # weights_for_different_modes=[0.4, 0.3, 0.3],\n                          weights_for_different_modes=[0, 0, 1])\n        c_max.append(max(score_list[\'colbert+sparse+dense\']))\n        c_mean.append(sum(score_list[\'colbert+sparse+dense\'])/len(score_list[\'colbert+sparse+dense\']))\n\n    c_max_e=[]\n    c_mean_e=[]\n    sentences_1=["Query: "+t2s.convert(row.query) ]\n    for y_id in row.source:    \n        sentences_2= insurance_s_key[y_id]+insurance_key[y_id]\n        embeddings_1 = encode(sentences_1)\n        embeddings_2 = encode(sentences_2)\n        similarity = embeddings_1 @ embeddings_2.T\n        emb=similarity.tolist()\n        \n        c_max_e.append(max(emb[0]))\n        c_mean_e.append(sum(emb[0])/len(emb[0]))\n    \n    ans_list2.append({\'qid\':row.qid, \'category\':row.category, \'retrieve\':row.retrieve,\'source\':row.source,\'c_max\':c_max, \'c_mean\':c_mean,\'c_max_e\':c_max_e, \'c_mean_e\':c_mean_e})\n    # ans_list2.append({\'qid\':row.qid, \'category\':row.category ,\'source\':row.source,\'c_max\':c_max, \'c_mean\':c_mean,\'c_max_e\':c_max_e, \'c_mean_e\':c_mean_e})\n')


# In[343]:


ans_df=pd.DataFrame(ans_list2)


# In[ ]:





# In[366]:


ans_df['c_m']= ans_df.apply( lambda x: [a*0.7 + b*0.3 for a, b in zip(x['c_max'], x['c_mean'])] ,axis=1 )
ans_df['ans']=ans_df.apply( lambda x:  x['source'][x['c_m'].index(max(x['c_m']))] ,axis=1)


# In[367]:


ans_df['score']=ans_df.apply(lambda x:1 if x['retrieve']==x['ans'] else 0,axis=1)
ans_df['score'].sum()


# In[ ]:





# In[358]:


ans_df['c_m']= ans_df.apply( lambda x: [a*1 + b*0 for a, b in zip(x['c_max_e'], x['c_mean_e'])] ,axis=1 )
ans_df['ans']=ans_df.apply( lambda x:  x['source'][x['c_m'].index(max(x['c_m']))] ,axis=1)


# In[359]:


ans_df['score']=ans_df.apply(lambda x:1 if x['retrieve']==x['ans'] else 0,axis=1)
ans_df['score'].sum()


# In[ ]:





# In[364]:


ans_df['c_m']= ans_df.apply( lambda x: [a*0.4 + b*0.4 + c*0.1 + d*0.1 for a, b, c, d in zip(x['c_max'], x['c_max_e'],x['c_mean'],x['c_mean_e'])] ,axis=1 )
ans_df['ans']=ans_df.apply( lambda x:  x['source'][x['c_m'].index(max(x['c_m']))] ,axis=1)


# In[365]:


ans_df['score']=ans_df.apply(lambda x:1 if x['retrieve']==x['ans'] else 0,axis=1)
ans_df['score'].sum()


# In[ ]:





# In[20]:


# !pip install langchain


# In[300]:


with open("finance_new_q.pkl", "rb") as f:
    finance_new_q=pickle.load(f)
finance_new_q=finance_new_q.set_index('qid')['new_query'].to_dict()


# In[301]:


with open("corpus_dict_finance.pkl", "rb") as f:
    corpus_dict_finance=pickle.load(f)

with open("corpus_dict_finance_img.pkl", "rb") as f:
    corpus_dict_finance_img=pickle.load(f)

corpus_dict_finance.update(corpus_dict_finance_img)


# In[ ]:





# In[409]:


from langchain_text_splitters import RecursiveCharacterTextSplitter
text_splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=30)


# In[484]:


get_ipython().run_cell_magic('time', '', 'ans_list3=[]\nfor row in finance.itertuples():\n    \n    sentences_1=[ t2s.convert(finance_new_q[row.qid]) ]\n    c_max=[]\n    for y_id in row.source:\n        texts = text_splitter.create_documents([t2s.convert(corpus_dict_finance[y_id][:8000] )])\n        sentences_2= [text.page_content for text in texts]\n\n        sentence_pairs = [[i,j] for i in sentences_1 for j in sentences_2]\n        score_list=model.compute_score(sentence_pairs, \n                          max_passage_length=600,\n                          weights_for_different_modes=[0.2, 0.5, 0.3])\n        c_max.append(max(score_list[\'colbert+sparse+dense\']))\n\n    c_mean=[]\n    for y_id in row.source:\n        sentences_2=[t2s.convert(corpus_dict_finance[y_id][:8000])]\n        sentence_pairs = [[i,j] for i in sentences_1 for j in sentences_2]\n        \n        score_list=model.compute_score(sentence_pairs, \n                          max_passage_length=8000,\n                          weights_for_different_modes=[0.2, 0.3, 0.5])\n        c_mean.append(score_list[\'colbert+sparse+dense\'][0])\n\n    \n    c_max_e=[]\n    c_mean_e=[]\n    sentences_1=["Query: "+t2s.convert(row.query) ]\n    for y_id in row.source:    \n        texts = text_splitter.create_documents([t2s.convert(corpus_dict_finance[y_id][:8000] )])\n        sentences_2= [text.page_content for text in texts]\n        \n        embeddings_1 = encode(sentences_1)\n        embeddings_2 = encode(sentences_2)\n        similarity = embeddings_1 @ embeddings_2.T\n        emb=similarity.tolist()\n        \n        c_max_e.append(max(emb[0]))\n        c_mean_e.append(sum(emb[0])/len(emb[0]))\n    \n\n    ans_list3.append({\'qid\':row.qid, \'category\':row.category, \'retrieve\':row.retrieve,\'source\':row.source,\'c_max\':c_max, \'c_mean\':c_mean,\'c_max_e\':c_max_e, \'c_mean_e\':c_mean_e})\n    \n    # ans_list3.append({\'qid\':row.qid, \'category\':row.category ,\'source\':row.source,\'c_max\':c_max, \'c_mean\':c_mean,\'c_max_e\':c_max_e, \'c_mean_e\':c_mean_e})\n    \n    # break\n')


# In[485]:


ans_df=pd.DataFrame(ans_list3)


# In[ ]:





# In[486]:


ans_df['c_m']= ans_df.apply( lambda x: [a*0.4 + b*0.6 for a, b in zip(x['c_max'], x['c_mean'])] ,axis=1 )
ans_df['ans']=ans_df.apply( lambda x:  x['source'][x['c_m'].index(max(x['c_m']))] ,axis=1)


# In[487]:


ans_df['score']=ans_df.apply(lambda x:1 if x['retrieve']==x['ans'] else 0,axis=1)
ans_df['score'].sum()


# In[ ]:





# In[463]:


ans_df['c_m']= ans_df.apply( lambda x: [a*1 + b*0 for a, b in zip(x['c_max_e'], x['c_mean_e'])] ,axis=1 )
ans_df['ans']=ans_df.apply( lambda x:  x['source'][x['c_m'].index(max(x['c_m']))] ,axis=1)


# In[464]:


ans_df['score']=ans_df.apply(lambda x:1 if x['retrieve']==x['ans'] else 0,axis=1)
ans_df['score'].sum()


# In[ ]:





# In[467]:


ans_df['c_m']= ans_df.apply( lambda x: [a*0 + b*1 for a, b in zip(x['c_max'], x['c_mean'])] ,axis=1 )
ans_df['ans']=ans_df.apply( lambda x:  x['source'][x['c_m'].index(max(x['c_m']))] ,axis=1)


# In[468]:


ans_df['score']=ans_df.apply(lambda x:1 if x['retrieve']==x['ans'] else 0,axis=1)
ans_df['score'].sum()


# In[ ]:





# In[469]:


ans_df['c_m']= ans_df.apply( lambda x: [a*0.3 + b*0.2 + c*0.4 + d*0.1 for a, b, c, d in zip(x['c_max'], x['c_max_e'],x['c_mean'],x['c_mean_e'])] ,axis=1 )
ans_df['ans']=ans_df.apply( lambda x:  x['source'][x['c_m'].index(max(x['c_m']))] ,axis=1)


# In[470]:


ans_df['score']=ans_df.apply(lambda x:1 if x['retrieve']==x['ans'] else 0,axis=1)
ans_df['score'].sum()


# In[ ]:





# In[471]:


# ans_df[['c_max', 'c_mean','c_max_e', 'c_mean_e']] = ans_df[['c_max', 'c_mean','c_max_e', 'c_mean_e']].applymap(lambda lst: [round(num, 2) for num in lst])
# ans_df[ans_df['score']!=1]


# In[ ]:





# In[473]:


ans1=pd.DataFrame(ans_list2)
ans2=pd.DataFrame(ans_list3)
ans3=pd.DataFrame(ans_list)


# In[475]:


all_ans=pd.concat([ans1,ans2,ans3])


# In[501]:


import numpy as np

def one_model(category,x1,x2):
    x1=np.array(x1)
    x2=np.array(x2)
    if category=='insurance':
        return x1*0.7+x2*0.3
    if category=='finance':
        return x1*0.4+x2*0.6
    if category=='faq':
        return x1*0.7+x2*0.3


# In[502]:


# (x['c_max'], x['c_max_e'],x['c_mean'],x['c_mean_e']
def two_model(category,x1,x2,x3,x4):
    x1=np.array(x1)
    x2=np.array(x2)
    x3=np.array(x3)
    x4=np.array(x4)
    if category=='insurance':
        return x1*0.4+x2*0.4+x3*0.1+x4*0.1
    if category=='finance':
        return x1*0.3+x2*0.2+x3*0.4+x4*0.1
    if category=='faq':
        return x1*0.3+x2*0.2+x3*0.4+x4*0.1


# In[509]:


all_ans['c_m']=all_ans.apply(lambda x: one_model(x['category'],x['c_max'], x['c_mean']).tolist(),axis=1)
all_ans['ans']=all_ans.apply( lambda x:  x['source'][x['c_m'].index(max(x['c_m']))] ,axis=1)


# In[511]:


all_ans['score']=all_ans.apply(lambda x:1 if x['retrieve']==x['ans'] else 0,axis=1)
all_ans['score'].sum()


# In[513]:


all_ans['c_m']=all_ans.apply(lambda x: two_model(x['category'],x['c_max'], x['c_max_e'],x['c_mean'],x['c_mean_e']).tolist(),axis=1)
all_ans['ans2']=all_ans.apply( lambda x:  x['source'][x['c_m'].index(max(x['c_m']))] ,axis=1)


# In[514]:


all_ans['score']=all_ans.apply(lambda x:1 if x['retrieve']==x['ans2'] else 0,axis=1)
all_ans['score'].sum()


# In[516]:


all_ans['retrieve']=all_ans.apply( lambda x:  x['source'][x['c_m'].index(max(x['c_m']))] ,axis=1)


# In[523]:


ans_output={'answers':all_ans[['qid','retrieve']].to_dict('records')}


# In[524]:


with open('pred_retrieve.json', 'w', encoding='utf8') as f:
    json.dump(ans_output, f, ensure_ascii=False, indent=4)  # 儲存檔案，確保格式和非ASCII字符


# In[ ]:




