#!/usr/bin/env python
# coding: utf-8

# In[6]:


import pandas as pd
import json
import re

pd.set_option('display.max_rows', 310)

ans_path="競賽資料集/dataset/preliminary/ground_truths_example.json"
with open(ans_path, 'rb') as f:
    ans = json.load(f)  # 讀取問題檔案

with open("競賽資料集/dataset/preliminary/questions_example.json", 'rb') as f:
    que = json.load(f) 
    

y=pd.DataFrame(ans['ground_truths'])

questions = pd.DataFrame( que['questions'] )

output=pd.merge(y,questions,on=['qid','category'])

output.loc[output[output['qid']==99].index[0],'retrieve']=693
output.loc[output[output['qid']==97].index[0],'retrieve']=579
output.loc[output[output['qid']==50].index[0],'source'].append(78)
output.loc[output[output['qid']==109].index[0],'source'].append(283)
output.loc[output[output['qid']==135].index[0],'source'].append(28)


finance=output[output['category'] == 'finance']


# In[2]:


from accelerate import init_empty_weights, load_checkpoint_and_dispatch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch

model_name = "Qwen/Qwen2.5-14B-Instruct"

bnb_config = BitsAndBytesConfig(
    load_in_8bit=True,          
    llm_int8_threshold=6.0,
    llm_int8_has_fp16_weight=False
)

# Initialize weights
with init_empty_weights():
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        torch_dtype="auto"
    )

model = torch.compile(model)

tokenizer = AutoTokenizer.from_pretrained(model_name)


# In[9]:


import opencc
t2s = opencc.OpenCC('tw2sp.json')
t2s.convert('隨身碟')


# In[25]:


get_ipython().run_cell_magic('time', '', 'responses=[]\n\nfor r in finance.itertuples():\n    ans={}\n    \n    prompt_temp=f"""\n    請直接輸出跟[問題]一樣的內容只刪除公司名跟時間資訊，輸出格式[修正]...\n    範例:\n    [問題]: 2022年第3季，联电公司及子公司因进口机器设备开立但未使用的信用状约为多少亿元？\n    [修正]: 因进口机器设备开立但未使用的信用状约为多少亿元？\n    \n    [問題]: {r.query}\n    [修正]: \n    """\n    \n    messages = [\n        {"role": "user", "content": t2s.convert(prompt_temp) }\n    ]\n    text = tokenizer.apply_chat_template(\n        messages,\n        tokenize=False,\n        add_generation_prompt=True\n    )\n    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)\n    \n    generated_ids = model.generate(\n        **model_inputs,\n        max_new_tokens=512\n    )\n    generated_ids = [\n        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)\n    ]\n    \n    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]\n\n    responses.append({ \'qid\':r.qid , \'query\':r.query , \'new_query\':response} )\n\n')


# In[38]:


finance_new_q=pd.DataFrame(responses)


# In[39]:


finance_new_q


# In[40]:


finance_new_q['new_query2']=finance_new_q['new_query'].apply(lambda x: re.search(r'\[修正]:\s*(.*?）?\?)', x.replace('？','?')).group(1)   )


# In[41]:


finance_new_q


# In[42]:


finance_new_q['new_query2']=''


# In[43]:


for _i,r in enumerate(finance_new_q['new_query']):
    try:
        output=re.search(r'\[修正]:\s*(.*?）?\?)', r.replace('？','?')).group(1)
    except:
        print(r)
    finance_new_q.loc[_i,'new_query2']=output


# In[ ]:





# In[ ]:


finance_new_q['new_query']=finance_new_q['new_query2']


# In[49]:


# import pandas as pd
# finance_new_q=pd.DataFrame(responses)

import pickle
with open('finance_new_q.pkl', 'wb') as f:
    pickle.dump(finance_new_q, f)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




