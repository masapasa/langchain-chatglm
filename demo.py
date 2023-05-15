#%%
from IPython.display import display, Markdown, clear_output

def display_answer(agent, query, vs_path, history=[]):
    for resp, history in local_doc_qa.get_knowledge_based_answer(query=query,
                                                                 vs_path=vs_path,
                                                                 chat_history=history,
                                                                 streaming=True):
        clear_output(wait=True)
        display(Markdown(resp["result"]))
    return resp, history
# %%
import torch.cuda
import torch.backends

from configs import model_config

# 全局参数，修改后请重新初始化
model_config.embedding_model_dict = {
    "ernie-tiny": "nghuyong/ernie-3.0-nano-zh",
    "ernie-base": "nghuyong/ernie-3.0-base-zh",
    "text2vec-base": "shibing624/text2vec-base-chinese",
    "text2vec": "/home/mw/input/text2vec2538",
}
model_config.llm_model_dict = {
    "chatyuan": "ClueAI/ChatYuan-large-v2",
    "chatglm-6b-int4-qe": "THUDM/chatglm-6b-int4-qe",
    "chatglm-6b-int4": "THUDM/chatglm-6b-int4",
    "chatglm-6b-int8": "THUDM/chatglm-6b-int8",
    "chatglm-6b": "/home/mw/input/ChatGLM6B6449",
}
model_config.VS_ROOT_PATH = "/home/mw/temp"

from chains.local_doc_qa import LocalDocQA

EMBEDDING_MODEL = "text2vec" # embedding 模型，对应 embedding_model_dict
VECTOR_SEARCH_TOP_K = 6
LLM_MODEL = "chatglm-6b"     # LLM 模型名，对应 llm_model_dict
LLM_HISTORY_LEN = 3
DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

local_doc_qa = LocalDocQA()

local_doc_qa.init_cfg(llm_model=LLM_MODEL,
                          embedding_model=EMBEDDING_MODEL,
                          llm_history_len=LLM_HISTORY_LEN,
                          top_k=VECTOR_SEARCH_TOP_K)
# %%
