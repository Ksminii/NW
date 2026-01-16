#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <memory>
#include <chrono>
#include <algorithm>
#include <iomanip>
#include <cstdlib>
#include <cstring>

#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif

namespace NW {

const char* KERNEL_SOURCE = R"(
#define MATCH_SCORE 1
#define MISMATCH_PENALTY -1
#define GAP_PENALTY -1

int max3(int a, int b, int c) {
    return (a >= b && a >= c) ? a : (b >= c ? b : c);
}

int score_func(char a, char b) {
    return (a == b) ? MATCH_SCORE : MISMATCH_PENALTY;
}

__kernel void compute_diagonal(__global const char* seq_a, __global const char* seq_b, __global int* dp_matrix, __global char* traceback_matrix, const int seq_a_len, const int seq_b_len, const int diagonal_sum, const int start_row, const int end_row) {
    int thread_id = get_global_id(0);
    int row = start_row + thread_id;

    if (row <= end_row) {
        int col = diagonal_sum - row;

        if (col >= 1 && col <= seq_b_len) {
            int current_idx = row * (seq_b_len + 1) + col;

            int diagonal_idx = (row - 1) * (seq_b_len + 1) + (col - 1);
            int upper_idx = (row - 1) * (seq_b_len + 1) + col;
            int left_idx = row * (seq_b_len + 1) + (col - 1);

            int match_score = dp_matrix[diagonal_idx] + score_func(seq_a[row - 1], seq_b[col - 1]);
            int delete_score = dp_matrix[upper_idx] + GAP_PENALTY;
            int insert_score = dp_matrix[left_idx] + GAP_PENALTY;

            int optimal_score = max3(match_score, delete_score, insert_score);
            dp_matrix[current_idx] = optimal_score;

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
)";


class OpenCLContext {
public:
    cl_platform_id platform = nullptr;
    cl_device_id device = nullptr;
    cl_context context = nullptr;
    cl_command_queue queue = nullptr;
    cl_program program = nullptr;
    cl_kernel kernel = nullptr;

    OpenCLContext() {
        initialize();
    }

    ~OpenCLContext() {
        cleanup();
    }

    
    OpenCLContext(const OpenCLContext&) = delete;
    OpenCLContext& operator=(const OpenCLContext&) = delete;

private:
    void initialize() {
        cl_int err;

        err = clGetPlatformIDs(1, &platform, nullptr);
        handleError(err, "clGetPlatformIDs");

        err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, nullptr);
        if (err != CL_SUCCESS) {
            std::cerr << "GPU device not found, falling back to CPU.\n";
            err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_CPU, 1, &device, nullptr);
            handleError(err, "clGetDeviceIDs CPU");
        }

        context = clCreateContext(nullptr, 1, &device, nullptr, nullptr, &err);
        handleError(err, "clCreateContext");

        queue = clCreateCommandQueue(context, device, 0, &err);
        handleError(err, "clCreateCommandQueue");

        size_t kernel_length = std::strlen(KERNEL_SOURCE);
        program = clCreateProgramWithSource(context, 1, &KERNEL_SOURCE, &kernel_length, &err);
        handleError(err, "clCreateProgramWithSource");

        err = clBuildProgram(program, 1, &device, nullptr, nullptr, nullptr);
        if (err != CL_SUCCESS) {
            size_t log_size;
            clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, nullptr, &log_size);
            std::vector<char> log(log_size);
            clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, log_size, log.data(), nullptr);
            std::cerr << "Kernel build error:\n" << log.data() << "\n";
            throw std::runtime_error("Kernel build failed");
        }

        kernel = clCreateKernel(program, "compute_diagonal", &err);
        handleError(err, "clCreateKernel");
    }

    void cleanup() {
        if (kernel) clReleaseKernel(kernel);
        if (program) clReleaseProgram(program);
        if (queue) clReleaseCommandQueue(queue);
        if (context) clReleaseContext(context);
    }

    static void handleError(cl_int error_code, const std::string& operation) {
        if (error_code != CL_SUCCESS) {
            throw std::runtime_error("OpenCL Error " + std::to_string(error_code) + " during: " + operation);
        }
    }
};


class OpenCLBuffer {
public:
    OpenCLBuffer(cl_context context, cl_mem_flags flags, size_t size, void* host_ptr = nullptr) {
        cl_int err;
        buffer = clCreateBuffer(context, flags, size, host_ptr, &err);
        if (err != CL_SUCCESS) {
            throw std::runtime_error("Failed to create OpenCL buffer: " + std::to_string(err));
        }
    }

    ~OpenCLBuffer() {
        if (buffer) {
            clReleaseMemObject(buffer);
        }
    }

    OpenCLBuffer(const OpenCLBuffer&) = delete;
    OpenCLBuffer& operator=(const OpenCLBuffer&) = delete;

    cl_mem get() const { return buffer; }

private:
    cl_mem buffer = nullptr;
};

class SequenceAligner {
public:
    SequenceAligner(OpenCLContext& ctx) : ctx(ctx) {}

    struct AlignmentResult {
        std::string alignedA;
        std::string alignedB;
        int score;
        int matches;
        int mismatches;
        int gaps;
        double similarity;
    };

    AlignmentResult align(const std::string& seqA, const std::string& seqB) {
        auto t0 = std::chrono::high_resolution_clock::now();

        const int lenA = seqA.length();
        const int lenB = seqB.length();
        const int matrixSize = (lenA + 1) * (lenB + 1);

        std::vector<int> dpMatrix(matrixSize);
        std::vector<char> tracebackMatrix(matrixSize);


        for (int i = 0; i <= lenA; i++) {
            dpMatrix[i * (lenB + 1)] = i * -1;
        }
        for (int j = 0; j <= lenB; j++) {
            dpMatrix[j] = j * -1;
        }

        auto t1 = std::chrono::high_resolution_clock::now();
        std::cout << "  [1] 벡터 초기화: " << std::chrono::duration<double>(t1-t0).count() << "초\n";


        OpenCLBuffer bufSeqA(ctx.context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                             seqA.length(), const_cast<char*>(seqA.c_str()));
        OpenCLBuffer bufSeqB(ctx.context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                             seqB.length(), const_cast<char*>(seqB.c_str()));
        OpenCLBuffer bufDpMatrix(ctx.context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                                 sizeof(int) * matrixSize, dpMatrix.data());
        OpenCLBuffer bufTraceback(ctx.context, CL_MEM_WRITE_ONLY,
                                  sizeof(char) * matrixSize);

        auto t2 = std::chrono::high_resolution_clock::now();
        std::cout << "  [2] OpenCL 버퍼 생성: " << std::chrono::duration<double>(t2-t1).count() << "초\n";

        // Set kernel arguments
        cl_mem memSeqA = bufSeqA.get();
        cl_mem memSeqB = bufSeqB.get();
        cl_mem memDpMatrix = bufDpMatrix.get();
        cl_mem memTraceback = bufTraceback.get();

        clSetKernelArg(ctx.kernel, 0, sizeof(cl_mem), &memSeqA);
        clSetKernelArg(ctx.kernel, 1, sizeof(cl_mem), &memSeqB);
        clSetKernelArg(ctx.kernel, 2, sizeof(cl_mem), &memDpMatrix);
        clSetKernelArg(ctx.kernel, 3, sizeof(cl_mem), &memTraceback);
        clSetKernelArg(ctx.kernel, 4, sizeof(int), &lenA);
        clSetKernelArg(ctx.kernel, 5, sizeof(int), &lenB);

        auto t3 = std::chrono::high_resolution_clock::now();
        std::cout << "  [3] 커널 인자 설정: " << std::chrono::duration<double>(t3-t2).count() << "초\n";

        // Execute kernel for each diagonal
        for (int k = 1; k <= lenA + lenB; k++) {
            int startRow = (k > lenB) ? k - lenB : 1;
            int endRow = (k > lenA) ? lenA : k - 1;

            clSetKernelArg(ctx.kernel, 6, sizeof(int), &k);
            clSetKernelArg(ctx.kernel, 7, sizeof(int), &startRow);
            clSetKernelArg(ctx.kernel, 8, sizeof(int), &endRow);

            size_t globalWorkSize = endRow - startRow + 1;
            if (globalWorkSize > 0) {
                cl_int err = clEnqueueNDRangeKernel(ctx.queue, ctx.kernel, 1, nullptr,
                                                     &globalWorkSize, nullptr, 0, nullptr, nullptr);
                if (err != CL_SUCCESS) {
                    throw std::runtime_error("Kernel execution failed");
                }
            }
        }

        auto t4 = std::chrono::high_resolution_clock::now();
        std::cout << "  [4] 커널 실행 (대각선 루프): " << std::chrono::duration<double>(t4-t3).count() << "초\n";

        // Read results back
        clEnqueueReadBuffer(ctx.queue, bufDpMatrix.get(), CL_TRUE, 0,
                           sizeof(int) * matrixSize, dpMatrix.data(), 0, nullptr, nullptr);
        clEnqueueReadBuffer(ctx.queue, bufTraceback.get(), CL_TRUE, 0,
                           sizeof(char) * matrixSize, tracebackMatrix.data(), 0, nullptr, nullptr);

        auto t5 = std::chrono::high_resolution_clock::now();
        std::cout << "  [5] GPU → CPU 결과 읽기: " << std::chrono::duration<double>(t5-t4).count() << "초\n";

        // Traceback
        std::string alignedA, alignedB;
        int i = lenA, j = lenB;

        while (i > 0 || j > 0) {
            char direction = tracebackMatrix[i * (lenB + 1) + j];
            if (i > 0 && j > 0 && direction == 'D') {
                alignedA += seqA[i - 1];
                alignedB += seqB[j - 1];
                i--; j--;
            } else if (i > 0 && direction == 'U') {
                alignedA += seqA[i - 1];
                alignedB += '_';
                i--;
            } else if (j > 0 && direction == 'L') {
                alignedA += '_';
                alignedB += seqB[j - 1];
                j--;
            } else {
                break;
            }
        }

        auto t6 = std::chrono::high_resolution_clock::now();
        std::cout << "  [6] 역추적 (Traceback): " << std::chrono::duration<double>(t6-t5).count() << "초\n";

        std::reverse(alignedA.begin(), alignedA.end());
        std::reverse(alignedB.begin(), alignedB.end());

        auto t7 = std::chrono::high_resolution_clock::now();
        std::cout << "  [7] 문자열 뒤집기: " << std::chrono::duration<double>(t7-t6).count() << "초\n";

        // Calculate statistics
        AlignmentResult result;
        result.alignedA = alignedA;
        result.alignedB = alignedB;
        result.score = dpMatrix[matrixSize - 1];
        result.matches = 0;
        result.mismatches = 0;
        result.gaps = 0;

        for (size_t k = 0; k < alignedA.length() && k < alignedB.length(); k++) {
            if (alignedA[k] == '_' || alignedB[k] == '_') {
                result.gaps++;
            } else if (alignedA[k] == alignedB[k]) {
                result.matches++;
            } else {
                result.mismatches++;
            }
        }

        int total = result.matches + result.mismatches + result.gaps;
        result.similarity = total > 0 ? (static_cast<double>(result.matches) / total) * 100.0 : 0.0;

        auto t8 = std::chrono::high_resolution_clock::now();
        std::cout << "  [8] 통계 계산: " << std::chrono::duration<double>(t8-t7).count() << "초\n";
        std::cout << "  [총합]: " << std::chrono::duration<double>(t8-t0).count() << "초\n\n";

        return result;
    }

private:
    OpenCLContext& ctx;
};

std::string readFasta(const std::string& filename) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        throw std::runtime_error("파일을 열 수 없습니다: " + filename);
    }

    std::string sequence;
    std::string line;

    while (std::getline(file, line)) {
        if (line.empty() || line[0] == '>') continue;

        for (char c : line) {
            if (c >= 'A' && c <= 'Z') {
                sequence += c;
            }
        }
    }

    return sequence;
}

std::string generateRandomSequence(int length) {
    static const char alphabet[] = "ACGT";
    std::string seq;
    seq.reserve(length);

    for (int i = 0; i < length; i++) {
        seq += alphabet[rand() % 4];
    }

    return seq;
}

void saveAlignment(const std::string& filename, const std::string& speciesName,
                   const SequenceAligner::AlignmentResult& result) {
    std::ofstream fout(filename);
    if (!fout.is_open()) {
        throw std::runtime_error("파일 저장 실패: " + filename);
    }

    fout << "Human vs " << speciesName << " Mitochondrial Genome Alignment (OpenCL - Modern C++)\n";
    fout << "Alignment Score: " << result.score << "\n";
    fout << "Aligned length: " << result.alignedA.length() << "\n";
    fout << "Matches: " << result.matches << ", Mismatches: " << result.mismatches
         << ", Gaps: " << result.gaps << "\n";
    fout << "Similarity: " << std::fixed << std::setprecision(2) << result.similarity << "%\n\n";
    fout << "Aligned Human:\n" << result.alignedA << "\n\n";
    fout << "Aligned " << speciesName << ":\n" << result.alignedB << "\n";
}

} // namespace NW

int main() {
    using namespace NW;

    std::srand(std::time(nullptr));

    try {
        OpenCLContext ctx;
        SequenceAligner aligner(ctx);

        // Warm-up 
        std::cout << "\n=== Performing a warm-up run with random sequences to initialize GPU ===\n";
        int warmupLen = 16500;
        auto dummyA = generateRandomSequence(warmupLen);
        auto dummyB = generateRandomSequence(warmupLen);

        auto warmupResult = aligner.align(dummyA, dummyB);
        std::cout << "Warm-up alignment score: " << warmupResult.score << "\n";
        std::cout << "=== Warm-up complete. Starting actual measurements. ===\n\n";

        // Species (테스트용으로 Gorilla만)
        const std::vector<std::pair<std::string, std::string>> species = {
            {"Gorilla", "Gorilla"}
        };

        
        std::cout << "Reading mitochondrial genome FASTA files...\n";
        std::string human = readFasta("/Users/gimseongmin/Desktop/URP/3-S/Needleman-wunsch/DATASETS/mito/Homosapiens_mitochondrion.fasta");
        std::cout << "Human mitochondrial genome length: " << human.length() << "\n";
        std::cout << "Human mitochondrial genome loaded: " << human.substr(0, 50) << "...\n";
        std::cout << "Starting alignments with " << species.size() << " species...\n\n";

        auto totalStart = std::chrono::high_resolution_clock::now();

        for (const auto& [filename, displayName] : species) {
            std::cout << "=== Aligning Human vs " << displayName << " ===\n";

            std::string filepath = "/Users/gimseongmin/Desktop/URP/3-S/Needleman-wunsch/DATASETS/mito/"
                                  + filename + "_mitochondrion.fasta";

            std::string otherSpecies = readFasta(filepath);
            std::cout << displayName << " mitochondrial genome length: " << otherSpecies.length() << "\n";
            std::cout << displayName << " mitochondrial genome: " << otherSpecies.substr(0, 50) << "...\n";

            auto start = std::chrono::high_resolution_clock::now();
            auto result = aligner.align(human, otherSpecies);
            auto end = std::chrono::high_resolution_clock::now();

            std::chrono::duration<double> duration = end - start;

            std::cout << "\nAlignment Score: " << result.score << "\n";
            std::cout << "Aligned length: " << result.alignedA.length() << "\n";
            std::cout << "Matches: " << result.matches << ", Mismatches: " << result.mismatches
                      << ", Gaps: " << result.gaps << "\n";
            std::cout << "Similarity: " << std::fixed << std::setprecision(2)
                      << result.similarity << "%\n";

            std::string outputFilename = "human_" + displayName + "_mito_alignment_ocl_modern.txt";
            saveAlignment(outputFilename, displayName, result);
            std::cout << "결과 파일 저장 완료: " << outputFilename << "\n";
            std::cout << "수행 시간: " << duration.count() << "초\n\n";
        }

        auto totalEnd = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> totalDuration = totalEnd - totalStart;
        std::cout << "전체 수행 시간: " << totalDuration.count() << "초\n";

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
        
    }

    return 0;
}
