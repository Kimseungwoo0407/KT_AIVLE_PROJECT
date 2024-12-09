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

## 3.env 파일 경로 설정
```python
env_path = '/content/drive/MyDrive/env'  # .env 파일의 경로를 설정해주세요.
load_dotenv(dotenv_path=env_path)
```

## 4.API 키 로드
```python
api_key = os.getenv('OPENAI_API_KEY')
```
