from dotenv import load_dotenv
import os, asyncio
from src.classes.Parser import Parser
from src.classes.Preprocessor import Preprocessor
from src.classes.Model import Model
from sentence_transformers import SentenceTransformer

from utils import *

dotenv_path = os.path.join(os.path.dirname(__file__), "src", "utils", ".env")
raw_data_path = os.path.join(os.path.dirname(__file__), "data", "raw_dialog", "data.csv")
load_dotenv(dotenv_path=dotenv_path)

# Data
api_id = int(os.getenv(key='API_ID'))
api_hash = os.getenv(key='API_HASH')
chat_id = os.getenv(key='CHAT_ID')

# Parsing dialog
parser = Parser(api_id=api_id, api_hash=api_hash)
asyncio.get_event_loop().run_until_complete(parse_data(parser=parser, chat_id=chat_id))

# Preprocessing
embedding_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

data = pd.read_csv(raw_data_path)
preprocessor = Preprocessor(data=data, model=embedding_model)
preprocessor.preprocess_data(chat_id=chat_id, sim_instr=False)

model = Model(model_name="yandex/YandexGPT-5-Lite-8B-pretrain",
              path_to_data="../data/processed", output_directory="../trained/")

model.train()

