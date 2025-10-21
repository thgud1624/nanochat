"""
🔰 초보자용 BPE 토크나이저 이해하기
단계별로 나누어서 천천히 따라해보세요!
"""

def step1_understand_bpe_concept():
    """Step 1: BPE가 뭔지 간단히 이해하기"""
    print("=" * 60)
    print("🔤 Step 1: BPE (Byte Pair Encoding) 개념 이해")
    print("=" * 60)
    
    # 간단한 예시
    text = "hello hello world world"
    print(f"원본 텍스트: '{text}'")
    
    # 1단계: 문자별로 분해
    chars = list(text.replace(" ", "▁"))  # 공백을 특수 문자로
    print(f"문자 분해: {chars}")
    
    # 2단계: 가장 많이 나오는 쌍 찾기
    pairs = {}
    for i in range(len(chars) - 1):
        pair = (chars[i], chars[i+1])
        pairs[pair] = pairs.get(pair, 0) + 1
    
    most_frequent = max(pairs.items(), key=lambda x: x[1])
    print(f"가장 빈번한 쌍: {most_frequent[0]} (횟수: {most_frequent[1]})")
    
    print("💡 BPE는 이런 식으로 자주 나오는 문자 쌍을 하나의 토큰으로 합치는 알고리즘입니다!")
    print()

def step2_explore_nanochat_structure():
    """Step 2: nanochat 구조 살펴보기"""
    print("=" * 60)
    print("🏗️  Step 2: nanochat 코드 구조 탐색")
    print("=" * 60)
    
    structure = """
    nanochat/
    ├── 🐍 scripts/tok_train.py          ← 시작점 (Python)
    ├── 🐍 nanochat/tokenizer.py         ← Python 래퍼
    └── 🦀 rustbpe/src/lib.rs           ← 핵심 알고리즘 (Rust)
    
    실행 흐름:
    tok_train.py → RustBPETokenizer → rustbpe.Tokenizer → BPE 알고리즘
    """
    print(structure)

def step3_trace_function_calls():
    """Step 3: 함수 호출 추적해보기"""
    print("=" * 60)
    print("🔍 Step 3: 함수 호출 추적")
    print("=" * 60)
    
    calls = [
        "1. python -m scripts.tok_train",
        "2. RustBPETokenizer.train_from_iterator()",
        "3. rustbpe.Tokenizer().train_from_iterator()",
        "4. train_core_incremental()",
        "5. count_pairs_parallel()",
        "6. heap.pop() → merge_pair()",
        "7. tiktoken.Encoding() 생성",
        "8. 디스크에 저장"
    ]
    
    for call in calls:
        print(f"   {call}")
    
    print("\n💡 각 단계를 하나씩 따라가면서 print문을 추가해보세요!")

def step4_hands_on_learning():
    """Step 4: 실습 방법"""
    print("=" * 60)  
    print("🛠️  Step 4: 실습 학습 방법")
    print("=" * 60)
    
    methods = [
        "방법 1: 아주 작은 데이터로 실행",
        "  → python tok_train_debug.py --max_chars=10000 --vocab_size=500",
        "",
        "방법 2: 각 함수에 print문 추가", 
        "  → 어디서 뭐가 호출되는지 눈으로 확인",
        "",
        "방법 3: 단순한 버전부터 만들어보기",
        "  → Python으로 간단한 BPE 구현해보기",
        "",
        "방법 4: 에러 메시지로 배우기",
        "  → 일부러 코드를 수정해서 에러 보기",
        "",
        "방법 5: 테스트 코드 실행",
        "  → python -m pytest tests/test_rustbpe.py -v -s"
    ]
    
    for method in methods:
        if method.startswith("  →"):
            print(f"    💻 {method}")
        elif method == "":
            print()
        else:
            print(f"📚 {method}")

def step5_create_simple_version():
    """Step 5: 간단한 버전 만들어보기"""
    print("=" * 60)
    print("🧪 Step 5: 초간단 BPE 만들어보기")
    print("=" * 60)
    
    simple_bpe_code = '''
# 10줄로 만드는 초간단 BPE
def simple_bpe_demo():
    text = "hello world hello"
    vocab = {chr(i): i for i in range(256)}  # 기본 ASCII
    
    # 1. 텍스트를 바이트로
    tokens = list(text.encode('utf-8'))
    
    # 2. 가장 빈번한 쌍 찾기  
    pairs = {}
    for i in range(len(tokens)-1):
        pair = (tokens[i], tokens[i+1])
        pairs[pair] = pairs.get(pair, 0) + 1
    
    # 3. 가장 빈번한 쌍 합치기
    if pairs:
        best_pair = max(pairs.items(), key=lambda x: x[1])
        print(f"병합: {best_pair[0]} -> 새토큰{len(vocab)}")
    
    return tokens
'''
    
    print(simple_bpe_code)
    print("💡 이런 식으로 원리부터 차근차근 이해해보세요!")

if __name__ == "__main__":
    print("🎓 nanochat BPE 토크나이저 학습 가이드")
    print("천천히 따라하면서 이해해보세요!\n")
    
    step1_understand_bpe_concept()
    step2_explore_nanochat_structure() 
    step3_trace_function_calls()
    step4_hands_on_learning()
    step5_create_simple_version()
    
    print("=" * 60)
    print("🚀 다음 단계: tok_train_debug.py 실행해보기!")
    print("python tok_train_debug.py --max_chars=10000 --vocab_size=500")
    print("=" * 60)