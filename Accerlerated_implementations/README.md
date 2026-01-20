# Needleman-Wunsch CUDA Implementation


### 전체 흐름

```
[Host (CPU)]                           [Device (GPU)]
    |
    ├─ FASTA 파일 읽기
    ├─ 메모리 할당 (DP matrix, sequences)
    ├─ 데이터 전송 ────────────────────→  GPU 메모리
    |
    ├─ Wavefront Loop (대각선 순차 처리)
    |     for k = 1 to m+n:
    |       커널 실행 ──────────────────→  compute_diagonal<<<>>>
    |                                        ├─ 대각선 상의 셀들 병렬 계산
    |                                        └─ DP + Traceback 저장
    |       동기화 대기 ←──────────────────
    |
    ├─ 결과 복사 ←────────────────────── GPU 메모리
    ├─ Traceback (CPU에서 순차 수행)
    └─ 출력 및 저장
```
