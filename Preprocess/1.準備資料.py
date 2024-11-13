#!/usr/bin/env python
# coding: utf-8

# In[1]:


question_path="競賽資料集/dataset/preliminary/questions_example.json"
source_path="競賽資料集/reference"


# In[ ]:





# In[3]:


import os
import json
import argparse

from tqdm import tqdm
import pdfplumber  # 用於從PDF文件中提取文字的工具


# 載入參考資料，返回一個字典，key為檔案名稱，value為PDF檔內容的文本
def load_data(source_path):
    masked_file_ls = os.listdir(source_path)  # 獲取資料夾中的檔案列表
    corpus_dict = {int(file.replace('.pdf', '')): read_pdf(os.path.join(source_path, file)) for file in tqdm(masked_file_ls) if file.endswith('.pdf') }  # 讀取每個PDF文件的文本，並以檔案名作為鍵，文本內容作為值存入字典
    return corpus_dict


# 讀取單個PDF文件並返回其文本內容
def read_pdf(pdf_loc, page_infos: list = None):
    pdf = pdfplumber.open(pdf_loc)  # 打開指定的PDF文件

    # TODO: 可自行用其他方法讀入資料，或是對pdf中多模態資料（表格,圖片等）進行處理

    # 如果指定了頁面範圍，則只提取該範圍的頁面，否則提取所有頁面
    pages = pdf.pages[page_infos[0]:page_infos[1]] if page_infos else pdf.pages
    pdf_text = ''
    for _, page in enumerate(pages):  # 迴圈遍歷每一頁
        text = page.extract_text()  # 提取頁面的文本內容
        if text:
            pdf_text += text
    pdf.close()  # 關閉PDF文件

    return pdf_text  # 返回萃取出的文本


# In[4]:


get_ipython().run_cell_magic('time', '', "source_path_finance = os.path.join(source_path, 'finance')  # 設定參考資料路徑\ncorpus_dict_finance = load_data(source_path_finance)\n")


# In[5]:


import pickle
with open("corpus_dict_finance.pkl", "wb") as f:
    pickle.dump(corpus_dict_finance, f)


# # 選轉圖片

# In[14]:


# !pip install pymupdf


# In[9]:


import fitz  # PyMuPDF

# 開啟 PDF 文件
def rotation_pdf(angle,input_pdf_path,output_pdf_path):
    pdf_document = fitz.open(input_pdf_path)
    
    # 旋轉每一頁
    for page_number in range(len(pdf_document)):
        page = pdf_document[page_number]
        page.set_rotation(angle)# 將頁面旋轉 90 度
    
    # 儲存旋轉後的 PDF
    pdf_document.save(output_pdf_path)
    pdf_document.close()


# In[10]:


# 開啟 PDF
input_pdf_path = "競賽資料集/reference/finance/979.pdf"
output_pdf_path = "979.pdf"
rotation_pdf(270,input_pdf_path,output_pdf_path)

input_pdf_path = "競賽資料集/reference/finance/360.pdf"
output_pdf_path = "360.pdf"
rotation_pdf(360,input_pdf_path,output_pdf_path)

input_pdf_path = "競賽資料集/reference/finance/753.pdf"
output_pdf_path = "753.pdf"
rotation_pdf(90,input_pdf_path,output_pdf_path)

input_pdf_path = "競賽資料集/reference/finance/652.pdf"
output_pdf_path = "652.pdf"
rotation_pdf(90,input_pdf_path,output_pdf_path)


# In[11]:


get_ipython().system('cp 979.pdf 競賽資料集/reference/finance/979.pdf')
get_ipython().system('cp 753.pdf 競賽資料集/reference/finance/753.pdf')
get_ipython().system('cp 360.pdf 競賽資料集/reference/finance/360.pdf')
get_ipython().system('cp 652.pdf 競賽資料集/reference/finance/652.pdf')


# In[ ]:





# In[13]:


# !pip install pytesseract
# apt update
# apt install -y tesseract-ocr
# apt install -y tesseract-ocr-chi-tra
# apt install poppler-utils
# !pip install pdf2image


# In[24]:


get_ipython().run_cell_magic('time', '', "from pdf2image import convert_from_path\nimport pytesseract\nfrom PIL import Image\nimport numpy as np\n\n\n# 自定義 Tesseract 配置\ncustom_config = r'--oem 3 --psm 6'\n\n# 設定紅色範圍\ndef remove_red_stamp(image):\n    # 將圖片轉為 numpy 陣列\n    image_np = np.array(image)\n    \n    # 定義紅色的 HSV 範圍，根據需要調整範圍\n    lower_red = np.array([100, 0, 0])  # 紅色範圍的下限\n    upper_red = np.array([255, 150, 150])  # 紅色範圍的上限\n\n    # 創建遮罩，篩選出紅色範圍內的像素\n    red_mask = np.all((image_np >= lower_red) & (image_np <= upper_red), axis=-1)\n\n    # 將紅色像素轉為白色\n    image_np[red_mask] = [255, 255, 255]\n\n    # 將 numpy 陣列轉回 PIL 圖片\n    return Image.fromarray(image_np)\n\n\nerror_pdf= []\ncorpus_dict_finance_img={}\nfor k in corpus_dict_finance.keys():\n    corpus=corpus_dict_finance[k]\n    if len(corpus)<100:\n        error_pdf.append([k,corpus])\n        \n        pdf_path = source_path+f'/finance/{k}.pdf'\n        images = convert_from_path(pdf_path, dpi=600)\n        \n        # 處理每一頁\n        all_text=''\n        for i, image in enumerate(images):\n            # 移除紅色印章\n            image_no_stamp = remove_red_stamp(image)\n            # OCR 文字識別\n            text = pytesseract.image_to_string(image_no_stamp, config=custom_config, lang='chi_tra')\n            all_text+=text\n        \n        corpus_dict_finance_img[k]=all_text.replace('  ',' ').replace('  ',' ').replace('  ',' ')\n        # break\n")


# In[25]:


with open('corpus_dict_finance_img.pkl', 'wb') as f:
    pickle.dump(corpus_dict_finance_img, f)


# In[ ]:





# In[26]:


get_ipython().run_cell_magic('time', '', "source_path_insurance = os.path.join(source_path, 'insurance')  # 設定參考資料路徑\ncorpus_dict_insurance = load_data(source_path_insurance)\n")


# In[32]:


tmp=[]
for k in corpus_dict_insurance.keys():
    tmp.append(len(corpus_dict_insurance[k]))
    


# In[33]:


import pandas as pd


# In[35]:


tmp=pd.DataFrame(tmp)


# In[40]:


tmp.sort_values(0)


# In[ ]:





# In[ ]:





# In[27]:


with open('corpus_dict_insurance.pkl', 'wb') as f:
    pickle.dump(corpus_dict_insurance, f)


# In[28]:


import opencc
t2s = opencc.OpenCC('tw2sp.json')
t2s.convert('隨身碟')


#     失敗的 finance 生成資料
#             prompt=f"""
#     對下面文件，提取以下關鍵:
#     公司名稱: 公司全名
#     報告期間: 西元年
#     財務報表類型或文件: 如損益表、資產負債表、現金流量表、合併權益變動表、附註等
#     關鍵字: 詳細列出盡量列出越多資訊不要遺漏
#     
#     文件:
#     {corpus}
#             """

# In[ ]:





# # 要跑 13h 44min 6s

# In[ ]:


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


# In[ ]:


get_ipython().run_cell_magic('time', '', 'from collections import defaultdict\nimport re\nllm_ans=defaultdict(list)\n\nfor path in glob.glob(\'競賽資料集/reference/insurance/*\'):\n    print(path)\n    try:\n        text=read_pdf(path)\n    \n        sections = re.split(r"\\n(?=第.\\S+.條\\s)", text)\n    \n        # 打印分段結果\n        for chunk in sections:\n            chunk=re.sub(r"\\n(【.*】)","",chunk).strip()\n            # print(chunk)\n            if len(chunk)>=100:\n                # print(chunk[:20])\n                prompt = f"""\n    請根據以下段落產生多個相關問題，問題需包含細節並有助於釐清條文中的條款要求及細項規範。請以列表格式輸出問題，確保 Python 程式能夠擷取為問題的列表。\n    \n    範例問題：["如果金融卡不見了，還可以使用刷臉提款嗎？", "掛失金融卡後需要做什麼才能繼續使用刷臉提款？"]\n    \n    段落：{chunk}\n    \n    請以 ["", "", ...] 的格式輸出問題，盡量問越多問題，確保覆蓋所有重要細節與規定。\n    """\n                messages = [\n                    {"role": "user", "content": prompt}\n                ]\n                text = tokenizer.apply_chat_template(\n                    messages,\n                    tokenize=False,\n                    add_generation_prompt=True\n                )\n                model_inputs = tokenizer([text], return_tensors="pt").to(model.device)\n                \n                generated_ids = model.generate(\n                    **model_inputs,\n                    max_new_tokens=1024\n                )\n                generated_ids = [\n                    output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)\n                ]\n                \n                response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]\n                llm_ans[path].append([chunk,response])\n                # break\n                # print("=" * 30)  # 分隔線\n    except:\n        continue\n    \n    # break\n\nimport pickle\nwith open(\'llm_ans_insurance.pkl\', \'wb\') as f:\n    pickle.dump(llm_ans, f)\n')


# ## 一樣程式再跑簡體問答
#                 
#                 messages = [
#                     {"role": "user", "content": t2s.convert(prompt)}
#                 ]
# 存 llm_s_ans_insurance.pkl

# In[ ]:





# In[ ]:




