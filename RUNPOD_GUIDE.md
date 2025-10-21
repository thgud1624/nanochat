# RunPod nanochat Training Guide

RunPod에서 nanochat "Best ChatGPT clone that $100 can buy"를 실행하는 완전한 가이드입니다.

## 🚀 RunPod 설정

### 1. RunPod 인스턴스 생성
- **GPU**: `8x RTX H100 SXM` (8x H100) 선택
- **Container Image**: `runpod/pytorch:2.1.0-py3.10-cuda12.1.1-devel-ubuntu22.04`
- **Container Disk**: 최소 `200GB` (권장 300GB+)
- **Volume**: 선택사항 (장기 저장용)

### 2. 예상 비용 (2024년 기준)
- **8x H100**: ~$24-32/시간
- **4시간 실행**: ~$96-128
- **총 비용**: $100 전후 (RunPod 시장 가격에 따라 변동)

## 📋 실행 단계

### Step 1: 인스턴스 시작 및 준비

```bash
# RunPod 웹 터미널에 접속 후
cd /workspace

# 코드 클론
git clone https://github.com/karpathy/nanochat.git
cd nanochat

# 스크립트 실행 권한 부여
chmod +x runpod_speedrun.sh runpod_storage.sh
```

### Step 2: 전체 훈련 파이프라인 실행

```bash
# 간단 실행 (4시간 걸림)
bash runpod_speedrun.sh

# 또는 screen 세션에서 실행 (권장)
screen -L -Logfile speedrun.log -S nanochat bash runpod_speedrun.sh
```

**Screen 세션 관리:**
```bash
# 세션에서 나오기: Ctrl+A, 그 다음 D (macOS 포함 모든 OS 동일)
# 세션 목록 보기: screen -ls
# 세션 다시 들어가기: screen -r nanochat  
# 세션 종료: Ctrl+A, 그 다음 K
# 로그 확인: tail -f speedrun.log
```

### Step 3: 진행 상황 모니터링

```bash
# 저장 상태 확인
bash runpod_storage.sh status

# GPU 사용률 확인
nvidia-smi

# 로그 실시간 확인
tail -f speedrun.log
```

## 🕐 예상 실행 시간

| 단계 | 소요 시간 | 주요 작업 |
|------|-----------|-----------|
| 토크나이저 훈련 | 30-60분 | Rust 빌드, 데이터 다운로드, 토크나이저 훈련 |
| 베이스 모델 사전훈련 | 2-3시간 | d20 모델 (561M 파라미터) 훈련 |
| 중간훈련 | 30-60분 | 대화 토큰, 도구 사용 학습 |
| 지도 미세조정 | 30-60분 | SFT 훈련 |
| 강화학습 (선택) | 30-60분 | GSM8K RL 훈련 |
| **총 시간** | **4-6시간** | |

## 💾 데이터 저장 및 백업

### 자동 저장 위치
- **모든 아티팩트**: `/workspace/nanochat_artifacts/` (약 30-40GB)
- **최종 리포트**: `/workspace/nanochat_final_report.md`
- **훈련 로그**: `speedrun.log`

### 백업 생성
```bash
# 백업 생성
bash runpod_storage.sh backup my_chatgpt_model

# 상태 확인
bash runpod_storage.sh status

# 백업 복원 (필요시)
bash runpod_storage.sh restore my_chatgpt_model.tar.gz
```

### 로컬 다운로드
```bash
# 다운로드 방법 확인
bash runpod_storage.sh download

# RunPod SSH를 통한 다운로드 (로컬에서 실행)
scp -P [SSH_PORT] root@[RUNPOD_IP]:/workspace/backups/my_chatgpt_model.tar.gz .
```

## 🔧 클라우드 저장소 연동

### AWS S3 연동
```bash
# AWS CLI 설치
apt update && apt install -y awscli

# 자격 증명 설정
aws configure

# 백업 업로드
aws s3 cp /workspace/backups/my_chatgpt_model.tar.gz s3://your-bucket/
```

### Google Cloud Storage 연동
```bash
# gcloud CLI 설치
curl https://sdk.cloud.google.com | bash
exec -l $SHELL

# 인증
gcloud init

# 백업 업로드
gsutil cp /workspace/backups/my_chatgpt_model.tar.gz gs://your-bucket/
```

## 💬 훈련된 모델 사용하기

### CLI로 채팅
```bash
cd /workspace/nanochat
source .venv/bin/activate

# 간단한 질문
python -m scripts.chat_cli -p "Why is the sky blue?"

# 대화형 모드
python -m scripts.chat_cli
```

### 웹 UI로 채팅
```bash
# 웹 서버 시작
python -m scripts.chat_web

# RunPod 포트 포워딩으로 접속
# RunPod 웹 인터페이스에서 'Connect' → 'HTTP Service [8000]' 클릭
```

## 📊 결과 분석

훈련 완료 후 `/workspace/nanochat_final_report.md`에서 다음 정보를 확인할 수 있습니다:

### 예상 성능 지표 (d20 모델)
| Metric | BASE | MID | SFT | RL |
|--------|------|-----|-----|-----|
| CORE | 0.2219 | - | - | - |
| ARC-Challenge | - | 0.2875 | 0.2807 | - |
| ARC-Easy | - | 0.3561 | 0.3876 | - |
| GSM8K | - | 0.0250 | 0.0455 | 0.0758 |
| HumanEval | - | 0.0671 | 0.0854 | - |
| MMLU | - | 0.3111 | 0.3151 | - |

## ⚠️ 주의사항

### 비용 관리
- **인스턴스 자동 종료**: 훈련 완료 후 즉시 종료하여 추가 비용 방지
- **스토리지 비용**: `/workspace` 디렉토리는 인스턴스 실행 중 계속 과금됨
- **백업 후 정리**: 중요한 모델은 외부 저장소로 백업 후 인스턴스 정리

### 실행 환경
- **GPU 메모리**: H100 80GB x 8 = 640GB 총 VRAM
- **시스템 RAM**: 최소 200GB+ 권장
- **디스크 공간**: 훈련 중 최대 50-60GB 사용

### 문제 해결
```bash
# GPU 메모리 부족 시
# 스크립트가 자동으로 배치 크기 조정하지만, 
# 필요시 수동으로 --device_batch_size 줄이기

# 디스크 공간 부족 시
df -h /workspace
bash runpod_storage.sh cleanup

# 중단된 훈련 재시작
# 체크포인트에서 자동 재시작됨 (대부분의 경우)
```

## 🎯 최적화 팁

### 비용 최적화
1. **Spot 인스턴스 사용**: 가격이 저렴하지만 중단될 수 있음
2. **정확한 시간 예측**: 4시간 후 자동 종료 설정
3. **불필요한 단계 생략**: RL 단계는 선택사항

### 성능 최적화
1. **SSD 스토리지**: 빠른 데이터 로딩을 위해 SSD 사용
2. **네트워크 대역폭**: 데이터 다운로드 속도 확인
3. **동시 실행**: 여러 평가 작업을 병렬로 실행

## 📈 확장 옵션

### 더 큰 모델 훈련 (d26, d32)
```bash
# d26 모델 (GPT-2 수준)
# runpod_speedrun.sh 에서 다음 수정:
# --depth=26 --device_batch_size=16

# d32 모델 (더 큰 모델)
# 더 많은 GPU 메모리와 시간 필요
# 예상 비용: $300-1000
```

### 커스텀 데이터셋
```bash
# 자신만의 데이터셋으로 훈련
# nanochat/dataset.py 수정하여 커스텀 데이터 로드
```

---

**💡 핵심 포인트**: RunPod는 유연한 GPU 클라우드 서비스로, nanochat을 원래 설계된 대로 8xH100에서 4시간 만에 완전한 ChatGPT 클론을 훈련할 수 있습니다. `/workspace` 디렉토리의 지속성을 활용하여 모든 결과물을 안전하게 보관하고, 필요시 외부 클라우드 저장소로 백업하는 것이 중요합니다.