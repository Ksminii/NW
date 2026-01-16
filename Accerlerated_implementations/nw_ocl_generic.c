#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <libgen.h>

#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif

#define MATCH 1
#define MISMATCH -1
#define GAP -1

// -------------------------------------------------------------------------
// GPU에서 실행될 커널 소스코드
// -------------------------------------------------------------------------
// 
const char *kernel_source = "\
#define MATCH_SCORE 1\n\
#define MISMATCH_PENALTY -1\n\
#define GAP_PENALTY -1\n\
\n\
// 세 값 중 최댓값을 반환하는 헬퍼 함수\n\
int max3(int a, int b, int c) {\n\
    return (a >= b && a >= c) ? a : (b >= c ? b : c);\n\
}\n\
\n\
// 두 문자가 같으면 매치 점수, 다르면 불일치 벌점 반환\n\
int score_func(char a, char b) {\n\
    return (a == b) ? MATCH_SCORE : MISMATCH_PENALTY;\n\
}\n\
\n\
// 대각선 계산을 위한 커널 함수\n\
__kernel void compute_diagonal(\n\
    __global const char* seq_a,\n\
    __global const char* seq_b,\n\
    __global int* dp_matrix,\n\
    __global char* traceback_matrix,\n\
    const int seq_a_len,\n\
    const int seq_b_len,\n\
    const int diagonal_sum,  // 현재 처리 중인 대각선의 인덱스 합 (row + col)\n\
    const int start_row,\n\
    const int end_row)\n\
{\n\
    int thread_id = get_global_id(0);\n\
    int row = start_row + thread_id;\n\
    \n\
    if (row <= end_row) {\n\
        // 현재 대각선 합(diagonal_sum)에서 row를 빼면 col이 나옴\n\
        int col = diagonal_sum - row;\n\
        \n\
        if (col >= 1 && col <= seq_b_len) {\n\
            // 1차원 배열로 펼쳐진 행렬의 인덱스 계산\n\
            int current_idx = row * (seq_b_len + 1) + col;\n\
            \n\
            // 이전 셀들의 인덱스 (대각선 위, 위, 왼쪽)\n\
            int diagonal_idx = (row - 1) * (seq_b_len + 1) + (col - 1);\n\
            int upper_idx = (row - 1) * (seq_b_len + 1) + col;\n\
            int left_idx = row * (seq_b_len + 1) + (col - 1);\n\
            \n\
            // 점수 계산\n\
            int match_score = dp_matrix[diagonal_idx] + score_func(seq_a[row - 1], seq_b[col - 1]);\n\
            int delete_score = dp_matrix[upper_idx] + GAP_PENALTY;\n\
            int insert_score = dp_matrix[left_idx] + GAP_PENALTY;\n\
            \n\
            // 최적 점수 선택 및 저장\n\
            int optimal_score = max3(match_score, delete_score, insert_score);\n\
            dp_matrix[current_idx] = optimal_score;\n\
            \n\
            // 역추적(Traceback)을 위한 방향 기록 (D: 대각선, U: 위, L: 왼쪽)\n\
            if (optimal_score == match_score) {\n\
                traceback_matrix[current_idx] = 'D';\n\
            } else if (optimal_score == delete_score) {\n\
                traceback_matrix[current_idx] = 'U';\n\
            } else {\n\
                traceback_matrix[current_idx] = 'L';\n\
            }\n\
        }\n\
    }\n\
}";

// OpenCL 에러 처리 헬퍼 함수
void handle_opencl_error(cl_int error_code, const char *operation) {
    if (error_code != CL_SUCCESS) {
        fprintf(stderr, "OpenCL Error %d during: %s\n", error_code, operation);
        exit(EXIT_FAILURE);
    }
}

// 문자열 뒤집기 함수 (역추적 한 결과를 뒤집어야 정렬결과)
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
    
    // 헤더 라인 건너뛰기 
    while (fgets(line, sizeof(line), file)) {
        if (line[0] == '>') continue; 

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
// Needleman-Wunsch 알고리즘 메인 함수 (OpenCL 호스트 코드)
// -------------------------------------------------------------------------
AlignmentResult needleman_wunsch_ocl(char *a, char *b, cl_context context, cl_command_queue queue, cl_kernel kernel) {
    int lenA = strlen(a);
    int lenB = strlen(b);
    cl_int err;

    // DP 행렬 및 역추적 행렬 크기 설정
    int dp_matrix_size = (lenA + 1) * (lenB + 1);
    int *dp_matrix = (int *)malloc(sizeof(int) * dp_matrix_size);
    char *traceback_matrix = (char *)malloc(sizeof(char) * dp_matrix_size);

    // 행렬 초기화 (첫 행과 첫 열에 갭 패널티 누적)
    for (int i = 0; i <= lenA; i++) dp_matrix[i * (lenB + 1)] = i * GAP;
    for (int j = 0; j <= lenB; j++) dp_matrix[j] = j * GAP;

    // [중요] OpenCL 메모리 버퍼 생성 (호스트 -> 디바이스)
    cl_mem buf_seq_a = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(char) * lenA, a, &err);
    cl_mem buf_seq_b = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(char) * lenB, b, &err);
    cl_mem buf_dp_matrix = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(int) * dp_matrix_size, dp_matrix, &err);
    cl_mem buf_traceback_matrix = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(char) * dp_matrix_size, NULL, &err);

    // 커널 인자 설정 (변하지 않는 값들 먼저 설정)
    clSetKernelArg(kernel, 0, sizeof(cl_mem), &buf_seq_a);
    clSetKernelArg(kernel, 1, sizeof(cl_mem), &buf_seq_b);
    clSetKernelArg(kernel, 2, sizeof(cl_mem), &buf_dp_matrix);
    clSetKernelArg(kernel, 3, sizeof(cl_mem), &buf_traceback_matrix);
    clSetKernelArg(kernel, 4, sizeof(int), &lenA);
    clSetKernelArg(kernel, 5, sizeof(int), &lenB);

    // [핵심] 대각선(Wavefront) 루프
    // DP 테이블 채우기는 데이터 의존성 때문에 한 번에 병렬화할 수 없습니다.
    // 하지만 대각선(k) 상의 셀들은 서로 의존성이 없으므로 동시에 계산 가능합니다.
    for (int k = 1; k <= lenA + lenB; k++) {
        // 현재 대각선 k에서 계산해야 할 행(row)의 범위 계산
        int start_row = (k > lenB) ? k - lenB : 1;
        int end_row = (k > lenA) ? lenA : k - 1;

        // 대각선마다 변하는 인자 설정
        clSetKernelArg(kernel, 6, sizeof(int), &k); // k = diagonal_sum (row + col)
        clSetKernelArg(kernel, 7, sizeof(int), &start_row);
        clSetKernelArg(kernel, 8, sizeof(int), &end_row);

        // 작업 항목 개수(Global Work Size) 설정 및 커널 실행 요청
        size_t global_work_size = end_row - start_row + 1;
        if (global_work_size > 0) {
            err = clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &global_work_size, NULL, 0, NULL, NULL);
            handle_opencl_error(err, "clEnqueueNDRangeKernel");
        }
    }
    
    // 계산 완료 후 결과 행렬을 디바이스에서 호스트로 읽어옴
    clEnqueueReadBuffer(queue, buf_dp_matrix, CL_TRUE, 0, sizeof(int) * dp_matrix_size, dp_matrix, 0, NULL, NULL);
    clEnqueueReadBuffer(queue, buf_traceback_matrix, CL_TRUE, 0, sizeof(char) * dp_matrix_size, traceback_matrix, 0, NULL, NULL);

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
    clReleaseMemObject(buf_seq_a);
    clReleaseMemObject(buf_seq_b);
    clReleaseMemObject(buf_dp_matrix);
    clReleaseMemObject(buf_traceback_matrix);

    return result;
}

int main(int argc, char* argv[]) {
    // 인자 확인
    if (argc != 3) {
        printf("사용법: %s <fasta_file1> <fasta_file2>\n", argv[0]);
        printf("예시: %s seq1.fasta seq2.fasta\n", argv[0]);
        return 1;
    }

    printf("=== Needleman-Wunsch OpenCL - 일반 버전 ===\n\n");

    // ---------------------------------------------------------------------
    // OpenCL 초기화 (플랫폼, 디바이스, 컨텍스트, 커맨드 큐 설정)
    // ---------------------------------------------------------------------
    cl_platform_id platform;
    cl_device_id device;
    cl_context context;
    cl_command_queue queue;
    cl_program program;
    cl_kernel kernel;
    cl_int err;

    // 플랫폼 가져오기
    err = clGetPlatformIDs(1, &platform, NULL);
    handle_opencl_error(err, "clGetPlatformIDs");

    // GPU 디바이스 시도, 실패 시 CPU 사용
    err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, NULL);
    if (err != CL_SUCCESS) {
        fprintf(stderr, "GPU 디바이스를 찾을 수 없어 CPU를 사용합니다.\n");
        err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_CPU, 1, &device, NULL);
        handle_opencl_error(err, "clGetDeviceIDs CPU");
    }

    // 컨텍스트 생성
    context = clCreateContext(NULL, 1, &device, NULL, NULL, &err);
    handle_opencl_error(err, "clCreateContext");

    // 커맨드 큐 생성
    queue = clCreateCommandQueue(context, device, 0, &err);
    handle_opencl_error(err, "clCreateCommandQueue");

    // 프로그램 객체 생성 (소스 코드 로드)
    program = clCreateProgramWithSource(context, 1, &kernel_source, NULL, &err);
    handle_opencl_error(err, "clCreateProgramWithSource");

    // 프로그램 빌드 (컴파일)
    err = clBuildProgram(program, 1, &device, NULL, NULL, NULL);
    if (err != CL_SUCCESS) {
        // 빌드 실패 시 로그 출력
        size_t log_size;
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);
        char *log = (char *)malloc(log_size);
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, log_size, log, NULL);
        fprintf(stderr, "커널 빌드 에러:\n%s\n", log);
        free(log);
        exit(EXIT_FAILURE);
    }

    // 커널 객체 생성
    kernel = clCreateKernel(program, "compute_diagonal", &err);
    handle_opencl_error(err, "clCreateKernel");

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
    AlignmentResult result = needleman_wunsch_ocl(seq1, seq2, context, queue, kernel);
    clock_t end = clock();

    double duration = (double)(end - start) / CLOCKS_PER_SEC;

    // 결과 콘솔 출력
    printf("===== OpenCL 정렬 결과 =====\n");
    printf("실행 시간: %.4f 초\n", duration);
    printf("정렬 점수: %d\n", result.score);
    printf("정렬 길이: %d\n", result.length);
    printf("일치: %d, 불일치: %d, 갭: %d\n", result.matches, result.mismatches, result.gaps);
    printf("유사도: %.2f%%\n\n", result.similarity);

    // 결과를 파일로 저장
    char output_filename[512];
    snprintf(output_filename, sizeof(output_filename), "%s_vs_%s_ocl_alignment.txt", name1, name2);

    FILE* fout = fopen(output_filename, "w");
    if (fout) {
        fprintf(fout, "%s vs %s - OpenCL Alignment\n", name1, name2);
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

    clReleaseKernel(kernel);
    clReleaseProgram(program);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);

    return 0;
}