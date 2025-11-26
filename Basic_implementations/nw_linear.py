import random
import time
import numpy as np
import os
import multiprocessing as mp

MATCH = 1
MISMATCH = -1
GAP_OPEN = -10
GAP_EXTEND = -1
INF = float('-inf')

def validate_alignment(alignedA, alignedB, expected_score, rel_tol=1e-4):
    score = 0
    gapA = gapB = False
    #gap이 이전에도 있던 건지 새로 열리는건지 확인하는 용도

    #zip으로 두 문자열을 한 글자씩 쌍으로 처리
    for a, b in zip(alignedA, alignedB):
       # 양쪽 다 gap이면 잘못된 정렬
        if a == '_' and b == '_':
            return False, "Invalid: both are gaps"

        elif a == '_':
            if gapA:
                score += GAP_EXTEND
            else:
                score += GAP_OPEN + GAP_EXTEND
                gapA = True
            gapB = False

        elif b == '_':
            if gapB:
                score += GAP_EXTEND
            else:
                score += GAP_OPEN + GAP_EXTEND
                gapB = True
            gapA = False

        else:
            score += MATCH if a == b else MISMATCH
            gapA = gapB = False

    if expected_score != 0:
        relative_error = abs(score - expected_score) / abs(expected_score)
    else:
        relative_error = abs(score - expected_score)  # fallback

    return relative_error <= rel_tol, score

def score(a, b):
    return MATCH if a == b else MISMATCH

def do_traceback(DP, Dx, Dy, trace, traceDx, traceDy, A, B):
    alignedA = []
    alignedB = []
    i, j = len(A), len(B)
    state = 'M'

    while i > 0 or j > 0:
        if state == 'M':
            prev_state = trace[i][j]
            if prev_state == 'M':
                alignedA.append(A[i - 1])
                alignedB.append(B[j - 1])
                i -= 1
                j -= 1
            elif prev_state == 'Dx':
                state = 'Dx'
            elif prev_state == 'Dy':
                state = 'Dy'
        elif state == 'Dx':
            prev_state = traceDx[i][j]
            alignedA.append(A[i - 1])
            alignedB.append('_')
            i -= 1
            state = prev_state
        elif state == 'Dy':
            prev_state = traceDy[i][j]
            alignedA.append('_')
            alignedB.append(B[j - 1])
            j -= 1
            state = prev_state

    return ''.join(reversed(alignedA)), ''.join(reversed(alignedB))





def needleman_cpu(A, B):
    lenA, lenB = len(A), len(B)

    DP = np.full((lenA + 1, lenB + 1), INF)
    Dx = np.full((lenA + 1, lenB + 1), INF)
    Dy = np.full((lenA + 1, lenB + 1), INF)

    trace = [[None] * (lenB + 1) for _ in range(lenA + 1)]
    traceDx = [[None] * (lenB + 1) for _ in range(lenA + 1)]
    traceDy = [[None] * (lenB + 1) for _ in range(lenA + 1)]

    DP[0][0] = 0

    for i in range(1, lenA + 1):
        Dx[i][0] = GAP_OPEN + (i - 1) * GAP_EXTEND
        DP[i][0] = Dx[i][0]
        traceDx[i][0] = 'Dx'
        trace[i][0] = 'Dx'

    for j in range(1, lenB + 1):
        Dy[0][j] = GAP_OPEN + (j - 1) * GAP_EXTEND
        DP[0][j] = Dy[0][j]
        traceDy[0][j] = 'Dy'
        trace[0][j] = 'Dy'

    for i in range(1, lenA + 1):
        for j in range(1, lenB + 1):
            up_ext = Dx[i - 1][j] + GAP_EXTEND
            up_open = DP[i - 1][j] + GAP_OPEN + GAP_EXTEND
            if up_ext >= up_open:
                Dx[i][j] = up_ext
                traceDx[i][j] = 'Dx'
            else:
                Dx[i][j] = up_open
                traceDx[i][j] = 'M'

            left_ext = Dy[i][j - 1] + GAP_EXTEND
            left_open = DP[i][j - 1] + GAP_OPEN + GAP_EXTEND
            if left_ext >= left_open:
                Dy[i][j] = left_ext
                traceDy[i][j] = 'Dy'
            else:
                Dy[i][j] = left_open
                traceDy[i][j] = 'M'

            match = DP[i - 1][j - 1] + score(A[i - 1], B[j - 1])
            if match >= Dx[i][j] and match >= Dy[i][j]:
                DP[i][j] = match
                trace[i][j] = 'M'
            elif Dx[i][j] >= Dy[i][j]:
                DP[i][j] = Dx[i][j]
                trace[i][j] = 'Dx'
            else:
                DP[i][j] = Dy[i][j]
                trace[i][j] = 'Dy'

    return DP, Dx, Dy, trace, traceDx, traceDy

def generate_random_sequence(length):
    return ''.join(random.choices(['A', 'C', 'G', 'T'], k=length))


if __name__=="__main__":
    for run in range(1, 11):
        print(f"\n 테스트 {run} : Needleman-Wunsch 정렬 시작")

        A = generate_random_sequence(10000)
        B = generate_random_sequence(10000)

        start = time.time()
        DP, Dx, Dy, trace, traceDx, traceDy = needleman_cpu(A, B)
        alignedA, alignedB = do_traceback(DP, Dx, Dy, trace, traceDx, traceDy, A, B)
        end = time.time()

        alignment_score = DP[len(A)][len(B)]
        duration = end - start

        print(f"완료 | 점수: {alignment_score} | 시간: {duration:.2f}초")

        # 결과 파일 저장
        filename = f"aligned_result_{run}_twice.txt"
        with open(filename, "w") as f:
            f.write(f"[Run {run}]\n")
            f.write(f"Alignment Score: {alignment_score}\n")
            f.write(f"Execution Time: {duration:.2f} seconds\n\n")
            f.write("Aligned A:\n")
            f.write(alignedA + "\n\n")
            f.write("Aligned B:\n")
            f.write(alignedB + "\n")

        print(f"{filename} 저장 완료")




        for i in range(1, 11):
            filename = f"aligned_result_{i}_twice.txt"
            if not os.path.exists(filename):
                print(f"[{i}]  파일 없음: {filename}")
                continue

            with open(filename, "r") as f:
                lines = f.readlines()

            try:
                expected_score = float(lines[1].strip().split(":")[1])
                a_index = lines.index("Aligned A:\n")
                b_index = lines.index("Aligned B:\n")
            except (ValueError, IndexError):
                print(f"[{i}]  파일 형식 오류")
                continue

            alignedA = lines[a_index + 1].strip()
            alignedB = lines[b_index + 1].strip()

            is_valid, recomputed = validate_alignment(alignedA, alignedB, expected_score, rel_tol=1e-3)
            result = " PASS" if is_valid else f" FAIL (Recalculated={recomputed})"
            print(f"[{i}] 검증 결과: {result}")



