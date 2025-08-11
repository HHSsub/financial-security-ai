# 금융보안 특화 AI 모델

## 프로젝트 개요
이 프로젝트는 금융보안 분야에 특화된 AI 모델을 개발하여 FSKU 평가지표에서 우수한 성능을 보이는 것을 목표로 합니다. 객관식 및 주관식 문항에 모두 대응할 수 있는 단일 LLM 모델을 사용하여 금융보안 지식을 평가합니다.

## 주요 기능
1. 데이터 분석: 테스트 데이터셋 분석 및 시각화
2. 모델 선택: 최적의 금융보안 특화 LLM 모델 선정
3. 모델 평가: 다양한 모델의 성능 및 추론 시간 비교
4. 모델 추론: 선택된 모델을 사용하여 테스트 데이터에 대한 추론 수행
5. 대시보드: 모델 성능 및 분석 결과 시각화

## 요구사항
- Python 3.10
- CUDA 11.8
- PyTorch 2.1.0
- 4bit 양자화를 지원하는 GPU (최소 24GB VRAM 권장)

## 설치 방법
```bash
# 저장소 클론
git clone https://github.com/your-username/financial-security-ai.git
cd financial-security-ai

# 의존성 설치
pip install -r requirements.txt
```

## 사용 방법
### 데이터 분석
```bash
python data_analysis.py
```

### 모델 선택
```bash
python model_selection.py
```

### 모델 평가
```bash
python evaluate_models.py --test_file /path/to/test.csv --num_samples 10
```

### 모델 추론
```bash
python inference.py --model_path "upstage/SOLAR-10.7B-Instruct-v1.0" --test_file /path/to/test.csv --output_file /path/to/output.csv
```

### 대시보드 실행
```bash
python main.py --model "upstage/SOLAR-10.7B-Instruct-v1.0" --test_file /path/to/test.csv --run_inference --num_samples 5
```

## 모델 선택 및 성능
금융보안 특화 모델로 다음 모델들을 평가했습니다:

1. SOLAR-10.7B-Instruct-v1.0 (Upstage)
2. Mistral-7B-Instruct-v0.2 (MistralAI)
3. Gemma-7B-it (Google)
4. Yi-6B-Chat (01.AI)

평가 결과, 추론 속도와 성능 면에서 SOLAR-10.7B-Instruct-v1.0 모델이 가장 우수한 성능을 보였습니다. 이 모델은 4bit 양자화를 적용하여 RTX 4090에서 제한 시간(270분) 내에 전체 평가 데이터셋에 대한 추론이 가능합니다.

## 디렉토리 구조
```
financial_security_ai/
│
├── README.md               # 프로젝트 문서
├── requirements.txt        # 필요 패키지 목록
├── data_analysis.py        # 데이터 분석 모듈
├── model_selection.py      # 모델 선택 모듈
├── model_training.py       # 모델 훈련 모듈
├── evaluate_models.py      # 모델 평가 모듈
├── inference.py            # 모델 추론 모듈
├── dashboard_integration.py # 대시보드 통합 모듈
└── main.py                 # 메인 실행 스크립트
```

## 라이선스
이 프로젝트는 MIT 라이선스 하에 배포됩니다. 자세한 내용은 LICENSE 파일을 참조하세요.