# import dependencies
from langchain_chroma import Chroma
import pandas as pd
from uuid import uuid4
import os
import chromadb
import pathlib
from langchain_together import ChatTogether
from langchain_together import TogetherEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

load_dotenv()

# Data preprocessing and chunking


