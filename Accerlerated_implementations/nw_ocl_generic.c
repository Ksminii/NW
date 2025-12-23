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

const char *kernel_source = "\
#define MATCH_SCORE 1\n\
#define MISMATCH_PENALTY -1\n\
#define GAP_PENALTY -1\n\
\n\
int max3(int a, int b, int c) {\n\
    return (a >= b && a >= c) ? a : (b >= c ? b : c);\n\
}\n\
\n\
int score_func(char a, char b) {\n\
    return (a == b) ? MATCH_SCORE : MISMATCH_PENALTY;\n\
}\n\
\n\
__kernel void compute_diagonal(\n\
    __global const char* seq_a,\n\
    __global const char* seq_b,\n\
    __global int* dp_matrix,\n\
    __global char* traceback_matrix,\n\
    const int seq_a_len,\n\
    const int seq_b_len,\n\
    const int diagonal_sum,\n\
    const int start_row,\n\
    const int end_row)\n\
{\n\
    int thread_id = get_global_id(0);\n\
    int row = start_row + thread_id;\n\
    \n\
    if (row <= end_row) {\n\
        int col = diagonal_sum - row;\n\
        \n\
        if (col >= 1 && col <= seq_b_len) {\n\
            int current_idx = row * (seq_b_len + 1) + col;\n\
            \n\
            int diagonal_idx = (row - 1) * (seq_b_len + 1) + (col - 1);\n\
            int upper_idx = (row - 1) * (seq_b_len + 1) + col;\n\
            int left_idx = row * (seq_b_len + 1) + (col - 1);\n\
            \n\
            int match_score = dp_matrix[diagonal_idx] + score_func(seq_a[row - 1], seq_b[col - 1]);\n\
            int delete_score = dp_matrix[upper_idx] + GAP_PENALTY;\n\
            int insert_score = dp_matrix[left_idx] + GAP_PENALTY;\n\
            \n\
            int optimal_score = max3(match_score, delete_score, insert_score);\n\
            dp_matrix[current_idx] = optimal_score;\n\
            \n\
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

void handle_opencl_error(cl_int error_code, const char *operation) {
    if (error_code != CL_SUCCESS) {
        fprintf(stderr, "OpenCL Error %d during: %s\n", error_code, operation);
        exit(EXIT_FAILURE);
    }
}

void rev(char *str) {
    int len = strlen(str);
    for (int i = 0; i < len / 2; i++) {
        char tmp = str[i];
        str[i] = str[len - 1 - i];
        str[len - 1 - i] = tmp;
    }
}

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

char* get_basename_without_ext(const char* path) {
    char* path_copy = strdup(path);
    char* base = basename(path_copy);
    char* dot = strrchr(base, '.');
    if (dot) *dot = '\0';

    char* result = strdup(base);
    free(path_copy);
    return result;
}

typedef struct {
    int score;
    int length;
    int matches;
    int mismatches;
    int gaps;
    double similarity;
    char* alignedA;
    char* alignedB;
} AlignmentResult;

AlignmentResult needleman_wunsch_ocl(char *a, char *b, cl_context context, cl_command_queue queue, cl_kernel kernel) {
    int lenA = strlen(a);
    int lenB = strlen(b);
    cl_int err;

    int dp_matrix_size = (lenA + 1) * (lenB + 1);
    int *dp_matrix = (int *)malloc(sizeof(int) * dp_matrix_size);
    char *traceback_matrix = (char *)malloc(sizeof(char) * dp_matrix_size);

    for (int i = 0; i <= lenA; i++) dp_matrix[i * (lenB + 1)] = i * GAP;
    for (int j = 0; j <= lenB; j++) dp_matrix[j] = j * GAP;

    cl_mem buf_seq_a = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(char) * lenA, a, &err);
    cl_mem buf_seq_b = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(char) * lenB, b, &err);
    cl_mem buf_dp_matrix = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(int) * dp_matrix_size, dp_matrix, &err);
    cl_mem buf_traceback_matrix = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(char) * dp_matrix_size, NULL, &err);

    clSetKernelArg(kernel, 0, sizeof(cl_mem), &buf_seq_a);
    clSetKernelArg(kernel, 1, sizeof(cl_mem), &buf_seq_b);
    clSetKernelArg(kernel, 2, sizeof(cl_mem), &buf_dp_matrix);
    clSetKernelArg(kernel, 3, sizeof(cl_mem), &buf_traceback_matrix);
    clSetKernelArg(kernel, 4, sizeof(int), &lenA);
    clSetKernelArg(kernel, 5, sizeof(int), &lenB);

    for (int k = 1; k <= lenA + lenB; k++) {
        int start_row = (k > lenB) ? k - lenB : 1;
        int end_row = (k > lenA) ? lenA : k - 1;

        clSetKernelArg(kernel, 6, sizeof(int), &k);
        clSetKernelArg(kernel, 7, sizeof(int), &start_row);
        clSetKernelArg(kernel, 8, sizeof(int), &end_row);

        size_t global_work_size = end_row - start_row + 1;
        if (global_work_size > 0) {
            err = clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &global_work_size, NULL, 0, NULL, NULL);
            handle_opencl_error(err, "clEnqueueNDRangeKernel");
        }
    }

    clEnqueueReadBuffer(queue, buf_dp_matrix, CL_TRUE, 0, sizeof(int) * dp_matrix_size, dp_matrix, 0, NULL, NULL);
    clEnqueueReadBuffer(queue, buf_traceback_matrix, CL_TRUE, 0, sizeof(char) * dp_matrix_size, traceback_matrix, 0, NULL, NULL);

    char *alignedA = (char*)malloc(lenA + lenB + 1);
    char *alignedB = (char*)malloc(lenA + lenB + 1);
    int ai = 0, bi = 0;
    int i = lenA, j = lenB;

    while (i > 0 || j > 0) {
        if (i > 0 && j > 0 && traceback_matrix[i * (lenB + 1) + j] == 'D') {
            alignedA[ai++] = a[i - 1];
            alignedB[bi++] = b[j - 1];
            i--; j--;
        } else if (i > 0 && traceback_matrix[i * (lenB + 1) + j] == 'U') {
            alignedA[ai++] = a[i - 1];
            alignedB[bi++] = '_';
            i--;
        } else if (j > 0 && traceback_matrix[i * (lenB + 1) + j] == 'L') {
            alignedA[ai++] = '_';
            alignedB[bi++] = b[j - 1];
            j--;
        } else break;
    }

    alignedA[ai] = '\0';
    alignedB[bi] = '\0';
    rev(alignedA);
    rev(alignedB);

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

    AlignmentResult result;
    result.score = dp_matrix[dp_matrix_size - 1];
    result.length = ai;
    result.matches = matches;
    result.mismatches = mismatches;
    result.gaps = gaps;
    result.similarity = similarity;
    result.alignedA = alignedA;
    result.alignedB = alignedB;

    free(dp_matrix);
    free(traceback_matrix);
    clReleaseMemObject(buf_seq_a);
    clReleaseMemObject(buf_seq_b);
    clReleaseMemObject(buf_dp_matrix);
    clReleaseMemObject(buf_traceback_matrix);

    return result;
}

int main(int argc, char* argv[]) {
    if (argc != 3) {
        printf("Usage: %s <fasta_file1> <fasta_file2>\n", argv[0]);
        printf("Example: %s seq1.fasta seq2.fasta\n", argv[0]);
        return 1;
    }

    printf("=== Needleman-Wunsch OpenCL - Generic Version ===\n\n");

    // Initialize OpenCL
    cl_platform_id platform;
    cl_device_id device;
    cl_context context;
    cl_command_queue queue;
    cl_program program;
    cl_kernel kernel;
    cl_int err;

    err = clGetPlatformIDs(1, &platform, NULL);
    handle_opencl_error(err, "clGetPlatformIDs");

    err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, NULL);
    if (err != CL_SUCCESS) {
        fprintf(stderr, "GPU device not found, falling back to CPU.\n");
        err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_CPU, 1, &device, NULL);
        handle_opencl_error(err, "clGetDeviceIDs CPU");
    }

    context = clCreateContext(NULL, 1, &device, NULL, NULL, &err);
    handle_opencl_error(err, "clCreateContext");

    queue = clCreateCommandQueue(context, device, 0, &err);
    handle_opencl_error(err, "clCreateCommandQueue");

    program = clCreateProgramWithSource(context, 1, &kernel_source, NULL, &err);
    handle_opencl_error(err, "clCreateProgramWithSource");

    err = clBuildProgram(program, 1, &device, NULL, NULL, NULL);
    if (err != CL_SUCCESS) {
        size_t log_size;
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);
        char *log = (char *)malloc(log_size);
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, log_size, log, NULL);
        fprintf(stderr, "Kernel build error:\n%s\n", log);
        free(log);
        exit(EXIT_FAILURE);
    }

    kernel = clCreateKernel(program, "compute_diagonal", &err);
    handle_opencl_error(err, "clCreateKernel");

    // Read sequences
    char* seq1 = read_fasta(argv[1]);
    char* seq2 = read_fasta(argv[2]);

    if (!seq1 || !seq2) {
        printf("Failed to read sequences\n");
        if (seq1) free(seq1);
        if (seq2) free(seq2);
        return 1;
    }

    char* name1 = get_basename_without_ext(argv[1]);
    char* name2 = get_basename_without_ext(argv[2]);

    printf("Sequence 1 (%s): %d bp\n", name1, (int)strlen(seq1));
    printf("Sequence 2 (%s): %d bp\n\n", name2, (int)strlen(seq2));

    clock_t start = clock();
    AlignmentResult result = needleman_wunsch_ocl(seq1, seq2, context, queue, kernel);
    clock_t end = clock();

    double duration = (double)(end - start) / CLOCKS_PER_SEC;

    printf("===== OpenCL Alignment Result =====\n");
    printf("Execution Time: %.4f seconds\n", duration);
    printf("Alignment Score: %d\n", result.score);
    printf("Aligned Length: %d\n", result.length);
    printf("Matches: %d, Mismatches: %d, Gaps: %d\n", result.matches, result.mismatches, result.gaps);
    printf("Similarity: %.2f%%\n\n", result.similarity);

    // Save result to file
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
        printf("Result saved to: %s\n", output_filename);
    } else {
        printf("Failed to save result file\n");
    }

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
