# 챗봇 시스템

이 프로젝트는 다양한 문서를 로드하고, 해당 문서에서 질문에 대한 답변을 생성하는 챗봇 시스템입니다. 주로 PDF, CSV 형식의 문서를 처리하며, 문서의 메타데이터(출처, 페이지, 카테고리 등)를 기반으로 적절한 답변을 제공합니다.

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
