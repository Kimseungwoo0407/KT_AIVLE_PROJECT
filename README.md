# 챗봇 시스템

- 이 프로젝트는 다양한 문서를 로드하고, 해당 문서에서 질문에 대한 답변을 생성하는 챗봇 시스템입니다.
- 주로 PDF, CSV 형식의 문서를 처리하며, 문서의 메타데이터(출처, 페이지, 카테고리 등)를 기반으로 적절한 답변을 제공합니다.

## 설치 및 환경 설정

### 1. 필수 라이브러리 설치
먼저, 필요한 라이브러리를 설치합니다.

```bash
pip install load_dotenv
pip install langchain
pip install -U langchain-community
pip install pymupdf
pip install openai
pip install openai==0.28
pip install tiktoken
pip install faiss-cpu
```

### 2. API 키 설정

OpenAI API 키와 Langchain API 키를 환경 변수로 설정합니다. `.env` 파일을 생성하여 아래와 같이 작성합니다.

```plaintext
OPENAI_API_KEY=본인 키 입력
LANGCHAIN_TRACING_V2=true
LANGCHAIN_ENDPOINT=https://api.smith.langchain.com
LANGCHAIN_API_KEY=본인 키 입력
LANGCHAIN_PROJECT=Projects
```

- 이후, `.env` 파일을 로드하여 환경 변수를 설정할 수 있습니다. 
- 아래의 Python 코드를 사용하여 `.env` 파일을 로드하고 API 키를 사용할 수 있도록 설정합니다.

```python
from dotenv import load_dotenv
import os
```
- env 파일 경로 설정
```python
env_path = '/content/drive/MyDrive/env'  # .env 파일의 경로를 설정해주세요.
load_dotenv(dotenv_path=env_path)
```

- API 키 로드
```python
api_key = os.getenv('OPENAI_API_KEY')
```

### 3. 문서 로드 및 분할
문서를 로드하고 텍스트를 분할하는 부분을 설정합니다. PDF 및 CSV 파일을 처리하며, 텍스트를 분할하여 검색을 최적화합니다.

```python
from langchain.document_loaders import PyMuPDFLoader, CSVLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

def load_and_classify_documents():
    all_documents = []
    # 문서 로드 및 메타 데이터 추가
    # 예: PDF 및 CSV 파일 로드 후 메타데이터 삽입
    # ...
    return all_documents

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
split_documents = text_splitter.split_documents(all_documents)
```
이 코드는 PDF 및 CSV 파일을 로드한 후, 텍스트를 지정된 크기로 분할하여 검색 성능을 최적화합니다. chunk_size와 chunk_overlap은 문서 텍스트의 분할 방식을 조정합니다.

### 4. 벡터스토어 및 검색기 생성
문서의 임베딩을 생성하고, 이를 FAISS 벡터스토어에 저장하여 빠르게 검색할 수 있도록 합니다.

```python
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS

# OpenAI 임베딩 모델을 사용하여 문서 임베딩 생성
embeddings = OpenAIEmbeddings()

# FAISS 벡터스토어에 임베딩 저장
vectorstore = FAISS.from_documents(documents=split_documents, embedding=embeddings)

# 벡터스토어에서 검색할 수 있는 retriever 생성
retriever = vectorstore.as_retriever()
```

