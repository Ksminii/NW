#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <libgen.h>
#include <cuda_runtime.h>

// 점수 체계 정의 (일치, 불일치, 갭 패널티)
#define MATCH 1
#define MISMATCH -1
#define GAP -1

// CUDA 매크로 정의
#define MATCH_SCORE 1
#define MISMATCH_PENALTY -1
#define GAP_PENALTY -1

// CUDA 에러 처리 헬퍼 함수
#define CUDA_CHECK(call) \
    do { \
        cudaError_t error = call; \
        if (error != cudaSuccess) { \
            fprintf(stderr, "CUDA Error: %s at %s:%d\n", \
                    cudaGetErrorString(error), __FILE__, __LINE__); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

// -------------------------------------------------------------------------
// CUDA 커널 함수 (GPU에서 실행될 로직)
// -------------------------------------------------------------------------
// 두 문자가 같으면 매치 점수, 다르면 불일치 벌점 반환
__device__ int score_func(char a, char b) {
    return (a == b) ? MATCH_SCORE : MISMATCH_PENALTY;
}

// 대각선 계산을 위한 커널 함수
__global__ void compute_diagonal(
    const char* seq_a,
    const char* seq_b,
    int* dp_matrix,
    char* traceback_matrix,
    const int seq_a_len,
    const int seq_b_len,
    const int diagonal_sum,  // 현재 처리 중인 대각선의 인덱스 합 (row + col)
    const int start_row,
    const int end_row)
{
    int thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    int row = start_row + thread_id;

    if (row <= end_row) {
        // 현재 대각선 합(diagonal_sum)에서 row를 빼면 col이 나옴
        int col = diagonal_sum - row;

        if (col >= 1 && col <= seq_b_len) {
            // 1차원 배열로 펼쳐진 행렬의 인덱스 계산
            int current_idx = row * (seq_b_len + 1) + col;

            // 이전 셀들의 인덱스 (대각선 위, 위, 왼쪽)
            int diagonal_idx = (row - 1) * (seq_b_len + 1) + (col - 1);
            int upper_idx = (row - 1) * (seq_b_len + 1) + col;
            int left_idx = row * (seq_b_len + 1) + (col - 1);

            // 점수 계산
            int match_score = dp_matrix[diagonal_idx] + score_func(seq_a[row - 1], seq_b[col - 1]);
            int delete_score = dp_matrix[upper_idx] + GAP_PENALTY;
            int insert_score = dp_matrix[left_idx] + GAP_PENALTY;

            // 최적 점수 선택 및 저장
            int optimal_score = max(match_score, max(delete_score, insert_score));
            dp_matrix[current_idx] = optimal_score;

            // 역추적(Traceback)을 위한 방향 기록 (D: 대각선, U: 위, L: 왼쪽)
            if (optimal_score == match_score) {
                traceback_matrix[current_idx] = 'D';
            } else if (optimal_score == delete_score) {
                traceback_matrix[current_idx] = 'U';
            } else {
                traceback_matrix[current_idx] = 'L';
            }
        }
    }
}

// 문자열 뒤집기 함수 (역추적 후 결과를 바로잡기 위함)
void rev(char *str) {
    int len = strlen(str);
    for (int i = 0; i < len / 2; i++) {
        char tmp = str[i];
        str[i] = str[len - 1 - i];
        str[len - 1 - i] = tmp;
    }
}

// FASTA 파일 읽기 함수
char* read_fasta(const char* filename) {
    FILE *file = fopen(filename, "r");
    if (!file) {
        printf("Cannot open file: %s\n", filename);
        return NULL;
    }

    fseek(file, 0, SEEK_END);
    long fsize = ftell(file);
    rewind(file);

    char *sequence = (char*)malloc(fsize + 1);
    if (!sequence) {
        printf("Memory allocation failed\n");
        fclose(file);
        return NULL;
    }

    char line[1024];
    int pos = 0;

    while (fgets(line, sizeof(line), file)) {
        if (line[0] == '>') continue; // 헤더 라인 건너뛰기

        for (int i = 0; line[i]; i++) {
            if (line[i] >= 'A' && line[i] <= 'Z') {
                sequence[pos++] = line[i];
            }
        }
    }
    sequence[pos] = '\0';
    fclose(file);
    return sequence;
}

// 파일 경로에서 확장자를 제외한 파일명만 추출
char* get_basename_without_ext(const char* path) {
    char* path_copy = strdup(path);
    char* base = basename(path_copy);
    char* dot = strrchr(base, '.');
    if (dot) *dot = '\0';

    char* result = strdup(base);
    free(path_copy);
    return result;
}

// 정렬 결과를 담을 구조체
typedef struct {
    int score;          // 최종 정렬 점수
    int length;         // 정렬된 길이
    int matches;        // 일치 개수
    int mismatches;     // 불일치 개수
    int gaps;           // 갭 개수
    double similarity;  // 유사도 (%)
    char* alignedA;     // 정렬된 서열 A
    char* alignedB;     // 정렬된 서열 B
} AlignmentResult;

// -------------------------------------------------------------------------
// Needleman-Wunsch 알고리즘 메인 함수 (CUDA 호스트 코드)
// -------------------------------------------------------------------------
AlignmentResult needleman_wunsch_cuda(char *a, char *b) {
    int lenA = strlen(a);
    int lenB = strlen(b);

    // DP 행렬 및 역추적 행렬 크기 설정
    int dp_matrix_size = (lenA + 1) * (lenB + 1);
    int *dp_matrix = (int *)malloc(sizeof(int) * dp_matrix_size);
    char *traceback_matrix = (char *)malloc(sizeof(char) * dp_matrix_size);

    // 행렬 초기화 (첫 행과 첫 열에 갭 패널티 누적)
    for (int i = 0; i <= lenA; i++) dp_matrix[i * (lenB + 1)] = i * GAP;
    for (int j = 0; j <= lenB; j++) dp_matrix[j] = j * GAP;

    // CUDA 메모리 할당
    char *d_seq_a, *d_seq_b;
    int *d_dp_matrix;
    char *d_traceback_matrix;

    CUDA_CHECK(cudaMalloc(&d_seq_a, sizeof(char) * lenA));
    CUDA_CHECK(cudaMalloc(&d_seq_b, sizeof(char) * lenB));
    CUDA_CHECK(cudaMalloc(&d_dp_matrix, sizeof(int) * dp_matrix_size));
    CUDA_CHECK(cudaMalloc(&d_traceback_matrix, sizeof(char) * dp_matrix_size));

    // 호스트에서 디바이스로 데이터 복사
    CUDA_CHECK(cudaMemcpy(d_seq_a, a, sizeof(char) * lenA, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_seq_b, b, sizeof(char) * lenB, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_dp_matrix, dp_matrix, sizeof(int) * dp_matrix_size, cudaMemcpyHostToDevice));

    // 대각선(Wavefront) 루프
    // DP 테이블 채우기는 데이터 의존성 때문에 한 번에 병렬화할 수 없음
    // 하지만 대각선(k) 상의 셀들은 서로 의존성이 없으므로 동시에 계산 가능
    for (int k = 1; k <= lenA + lenB; k++) {
        // 현재 대각선 k에서 계산해야 할 행(row)의 범위 계산
        int start_row = (k > lenB) ? k - lenB : 1;
        int end_row = (k > lenA) ? lenA : k - 1;

        int num_threads = end_row - start_row + 1;
        if (num_threads > 0) {
            // 블록 크기 설정 (일반적으로 256 또는 512)
            int block_size = 256;
            int grid_size = (num_threads + block_size - 1) / block_size;

            // 커널 실행
            compute_diagonal<<<grid_size, block_size>>>(
                d_seq_a, d_seq_b, d_dp_matrix, d_traceback_matrix,
                lenA, lenB, k, start_row, end_row
            );

            // 커널 실행 후 에러 체크
            CUDA_CHECK(cudaGetLastError());
        }
    }

    // 커널 실행 완료 대기
    CUDA_CHECK(cudaDeviceSynchronize());

    // 계산 완료 후 결과 행렬을 디바이스에서 호스트로 읽어옴
    CUDA_CHECK(cudaMemcpy(dp_matrix, d_dp_matrix, sizeof(int) * dp_matrix_size, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(traceback_matrix, d_traceback_matrix, sizeof(char) * dp_matrix_size, cudaMemcpyDeviceToHost));

    // ---------------------------------------------------------------------
    // 역추적 (Traceback) 단계 - CPU에서 수행
    // 행렬의 우하단 끝에서부터 좌상단(0,0)으로 이동하며 경로 복원
    // ---------------------------------------------------------------------
    char *alignedA = (char*)malloc(lenA + lenB + 1);
    char *alignedB = (char*)malloc(lenA + lenB + 1);
    int ai = 0, bi = 0;
    int i = lenA, j = lenB;

    while (i > 0 || j > 0) {
        if (i > 0 && j > 0 && traceback_matrix[i * (lenB + 1) + j] == 'D') {
            // 대각선 이동: 매치 또는 미스매치
            alignedA[ai++] = a[i - 1];
            alignedB[bi++] = b[j - 1];
            i--; j--;
        } else if (i > 0 && traceback_matrix[i * (lenB + 1) + j] == 'U') {
            // 위쪽 이동: 서열 B에 갭(_) 추가
            alignedA[ai++] = a[i - 1];
            alignedB[bi++] = '_';
            i--;
        } else if (j > 0 && traceback_matrix[i * (lenB + 1) + j] == 'L') {
            // 왼쪽 이동: 서열 A에 갭(_) 추가
            alignedA[ai++] = '_';
            alignedB[bi++] = b[j - 1];
            j--;
        } else break; // 오류 방지용 탈출
    }

    // 문자열 끝 처리 및 뒤집기 (역추적했으므로 순서가 반대임)
    alignedA[ai] = '\0';
    alignedB[bi] = '\0';
    rev(alignedA);
    rev(alignedB);

    // 결과 통계 계산
    int matches = 0, mismatches = 0, gaps = 0;
    for (int k = 0; alignedA[k] && alignedB[k]; k++) {
        if (alignedA[k] == '_' || alignedB[k] == '_') {
            gaps++;
        } else if (alignedA[k] == alignedB[k]) {
            matches++;
        } else {
            mismatches++;
        }
    }

    double similarity = (double)matches / (matches + mismatches + gaps) * 100.0;

    // 결과 구조체 생성
    AlignmentResult result;
    result.score = dp_matrix[dp_matrix_size - 1]; // 마지막 셀의 값이 최종 점수
    result.length = ai;
    result.matches = matches;
    result.mismatches = mismatches;
    result.gaps = gaps;
    result.similarity = similarity;
    result.alignedA = alignedA;
    result.alignedB = alignedB;

    // 메모리 해제
    free(dp_matrix);
    free(traceback_matrix);
    CUDA_CHECK(cudaFree(d_seq_a));
    CUDA_CHECK(cudaFree(d_seq_b));
    CUDA_CHECK(cudaFree(d_dp_matrix));
    CUDA_CHECK(cudaFree(d_traceback_matrix));

    return result;
}

int main(int argc, char* argv[]) {
    // 인자 확인
    if (argc != 3) {
        printf("사용법: %s <fasta_file1> <fasta_file2>\n", argv[0]);
        printf("예시: %s seq1.fasta seq2.fasta\n", argv[0]);
        return 1;
    }

    printf("=== Needleman-Wunsch CUDA - 일반 버전 ===\n\n");

    // CUDA 디바이스 정보 출력
    int device_count;
    CUDA_CHECK(cudaGetDeviceCount(&device_count));
    if (device_count == 0) {
        fprintf(stderr, "사용 가능한 CUDA 디바이스가 없습니다.\n");
        return 1;
    }

    // GPU 0번 사용 설정 (기본 GPU)
    int device_id = 0;
    if (device_id >= device_count) {
        fprintf(stderr, "GPU %d를 사용할 수 없습니다. (사용 가능한 GPU: 0-%d)\n", device_id, device_count - 1);
        return 1;
    }
    CUDA_CHECK(cudaSetDevice(device_id));

    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device_id));
    printf("사용 중인 GPU: %s (Device %d)\n", prop.name, device_id);
    printf("컴퓨트 성능: %d.%d\n\n", prop.major, prop.minor);

    // 입력 파일에서 서열 읽기
    char* seq1 = read_fasta(argv[1]);
    char* seq2 = read_fasta(argv[2]);

    if (!seq1 || !seq2) {
        printf("서열을 읽는데 실패했습니다.\n");
        if (seq1) free(seq1);
        if (seq2) free(seq2);
        return 1;
    }

    char* name1 = get_basename_without_ext(argv[1]);
    char* name2 = get_basename_without_ext(argv[2]);

    printf("서열 1 (%s): %d bp\n", name1, (int)strlen(seq1));
    printf("서열 2 (%s): %d bp\n\n", name2, (int)strlen(seq2));

    // 실행 시간 측정 및 알고리즘 실행
    clock_t start = clock();
    AlignmentResult result = needleman_wunsch_cuda(seq1, seq2);
    clock_t end = clock();

    double duration = (double)(end - start) / CLOCKS_PER_SEC;

    // 결과 콘솔 출력
    printf("===== CUDA 정렬 결과 =====\n");
    printf("실행 시간: %.4f 초\n", duration);
    printf("정렬 점수: %d\n", result.score);
    printf("정렬 길이: %d\n", result.length);
    printf("일치: %d, 불일치: %d, 갭: %d\n", result.matches, result.mismatches, result.gaps);
    printf("유사도: %.2f%%\n\n", result.similarity);

    // 결과를 파일로 저장
    char output_filename[512];
    snprintf(output_filename, sizeof(output_filename), "%s_vs_%s_cuda_alignment.txt", name1, name2);

    FILE* fout = fopen(output_filename, "w");
    if (fout) {
        fprintf(fout, "%s vs %s - CUDA Alignment\n", name1, name2);
        fprintf(fout, "Execution Time: %.4f seconds\n", duration);
        fprintf(fout, "Alignment Score: %d\n", result.score);
        fprintf(fout, "Aligned Length: %d\n", result.length);
        fprintf(fout, "Matches: %d, Mismatches: %d, Gaps: %d\n", result.matches, result.mismatches, result.gaps);
        fprintf(fout, "Similarity: %.2f%%\n\n", result.similarity);
        fprintf(fout, "Aligned %s:\n%s\n\n", name1, result.alignedA);
        fprintf(fout, "Aligned %s:\n%s\n", name2, result.alignedB);
        fclose(fout);
        printf("결과 저장됨: %s\n", output_filename);
    } else {
        printf("결과 파일 저장 실패\n");
    }

    // 자원 해제
    free(name1);
    free(name2);
    free(seq1);
    free(seq2);
    free(result.alignedA);
    free(result.alignedB);

    return 0;
}
