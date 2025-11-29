import os
from openai import OpenAI
from sentence_transformers import SentenceTransformer

API_KEY = os.getenv("PiLLar_API_KEY", "YOUR_API_KEY_HERE")
BASE_URL = os.getenv("PiLLar_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")

client = OpenAI(
    api_key=API_KEY,
    base_url=BASE_URL,
)

args = None

total_time_wasted = 0

model = SentenceTransformer("all-distilroberta-v1")
bert_similarity_cache = {}

source_attributes = []
target_attributes = []
df_source = None
df_target = None
source_types = {}
target_types = {}
column_explanations = None

similarity_cache = {}
similarity_sum_cache = {}
calculating = {}

start_time = None
end_time = None
