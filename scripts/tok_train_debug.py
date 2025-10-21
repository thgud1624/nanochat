"""
학습용 토크나이저 트레이닝 - 로그 추가 버전
"""
import os
import time
import argparse
import torch
from nanochat.tokenizer import RustBPETokenizer
from nanochat.common import get_base_dir
from nanochat.dataset import parquets_iter_batched

print("🚀 tok_train.py 시작!")
print("=" * 50)

# -----------------------------------------------------------------------------
# Parse command line arguments

parser = argparse.ArgumentParser(description='Train a BPE tokenizer')
parser.add_argument('--max_chars', type=int, default=1_000_000, help='Maximum characters (학습용으로 작게)')
parser.add_argument('--doc_cap', type=int, default=10_000, help='Maximum characters per document')
parser.add_argument('--vocab_size', type=int, default=1000, help='Vocabulary size (학습용으로 작게)')
args = parser.parse_args()

print(f"📊 설정값들:")
print(f"  - max_chars: {args.max_chars:,}")
print(f"  - doc_cap: {args.doc_cap:,}")
print(f"  - vocab_size: {args.vocab_size:,}")
print()

# -----------------------------------------------------------------------------
# Text iterator

def text_iterator():
    """
    1) Flatten the batches into a single iterator
    2) Cap each document to args.doc_cap characters
    3) Stop after args.max_chars characters total
    """
    print("📁 데이터 로딩 중...")
    chars_processed = 0
    doc_count = 0
    
    for batch in parquets_iter_batched():
        print(f"  배치 처리 중... (현재까지 {chars_processed:,} 문자 처리됨)")
        
        for row in batch:
            if chars_processed >= args.max_chars:
                print(f"✅ 목표 문자 수 달성! ({args.max_chars:,} 문자)")
                return
                
            text = row["text"]
            if args.doc_cap > 0:
                text = text[:args.doc_cap]
            
            chars_processed += len(text)
            doc_count += 1
            
            if doc_count % 100 == 0:
                print(f"    📄 문서 {doc_count}개 처리됨")
            
            yield text

print("🔄 텍스트 이터레이터 생성...")
text_iter = text_iterator()

# -----------------------------------------------------------------------------
# Train the tokenizer
print("🧠 토크나이저 훈련 시작!")
print("=" * 30)

t0 = time.time()
print("  → RustBPETokenizer.train_from_iterator 호출...")

tokenizer = RustBPETokenizer.train_from_iterator(text_iter, args.vocab_size)

t1 = time.time()
train_time = t1 - t0

print(f"✅ 훈련 완료!")
print(f"  ⏱️  소요 시간: {train_time:.2f}초")
print(f"  📝 어휘 크기: {args.vocab_size:,}개")

# -----------------------------------------------------------------------------
# 간단한 테스트
print("\n🧪 토크나이저 테스트:")
test_text = "Hello, world! This is a test."
encoded = tokenizer.encode(test_text)
decoded = tokenizer.decode(encoded)

print(f"  원본: {test_text}")
print(f"  토큰화: {encoded}")
print(f"  복원: {decoded}")
print(f"  토큰 수: {len(encoded)}개")

# -----------------------------------------------------------------------------
# Save the tokenizer to disk
print("\n💾 토크나이저 저장 중...")
base_dir = get_base_dir()
tokenizer_dir = os.path.join(base_dir, "tokenizer")
tokenizer.save(tokenizer_dir)
print(f"✅ 저장 완료: {tokenizer_dir}")

print("\n🎉 모든 작업 완료!")
print("=" * 50)