#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <libgen.h>

#define MATCH 1
#define MISMATCH -1
#define GAP -1
#define HIRSCHBERG_THRESHOLD 10

int max3(int a, int b, int c) {
    if (a >= b && a >= c) return a;
    if (b >= a && b >= c) return b;
    return c;
}

int score_match(char a, char b) {
    return a == b ? MATCH : MISMATCH;
}

void rev(char* str) {
    int len = strlen(str);
    for (int i = 0; i < len / 2; i++) {
        char tmp = str[i];
        str[i] = str[len - 1 - i];
        str[len - 1 - i] = tmp;
    }
}

typedef struct {
    char* alignedA;
    char* alignedB;
    int length;
} Alignment;

int* nw_score(char* seqA, char* seqB) {
    int lenA = strlen(seqA);
    int lenB = strlen(seqB);

    int* prev_row = (int*)malloc((lenB + 1) * sizeof(int));
    int* curr_row = (int*)malloc((lenB + 1) * sizeof(int));

    for (int j = 0; j <= lenB; j++) {
        prev_row[j] = j * GAP;
    }

    for (int i = 1; i <= lenA; i++) {
        curr_row[0] = i * GAP;
        for (int j = 1; j <= lenB; j++) {
            int diag = prev_row[j-1] + score_match(seqA[i-1], seqB[j-1]);
            int up = prev_row[j] + GAP;
            int left = curr_row[j-1] + GAP;
            curr_row[j] = max3(diag, up, left);
        }
        int* temp = prev_row;
        prev_row = curr_row;
        curr_row = temp;
    }

    int* last_row = (int*)malloc((lenB + 1) * sizeof(int));
    for (int j = 0; j <= lenB; j++) {
        last_row[j] = prev_row[j];
    }

    free(prev_row);
    free(curr_row);
    return last_row;
}

Alignment nw_full(char* seqA, char* seqB) {
    int lenA = strlen(seqA);
    int lenB = strlen(seqB);

    int** dp = (int**)malloc((lenA + 1) * sizeof(int*));
    for (int i = 0; i <= lenA; i++) {
        dp[i] = (int*)malloc((lenB + 1) * sizeof(int));
    }

    dp[0][0] = 0;
    for (int i = 1; i <= lenA; i++) dp[i][0] = i * GAP;
    for (int j = 1; j <= lenB; j++) dp[0][j] = j * GAP;

    for (int i = 1; i <= lenA; i++) {
        for (int j = 1; j <= lenB; j++) {
            int diag = dp[i-1][j-1] + score_match(seqA[i - 1], seqB[j - 1]);
            int up = dp[i-1][j] + GAP;
            int left = dp[i][j-1] + GAP;
            dp[i][j] = max3(diag, up, left);
        }
    }

    char* alignedA = (char*)malloc(lenA + lenB + 1);
    char* alignedB = (char*)malloc(lenA + lenB + 1);
    int ai = 0, bi = 0;
    int i = lenA, j = lenB;

    while (i > 0 || j > 0) {
        if (i > 0 && j > 0 && dp[i][j] == dp[i-1][j-1] + score_match(seqA[i - 1], seqB[j - 1])) {
            alignedA[ai++] = seqA[i - 1];
            alignedB[bi++] = seqB[j - 1];
            i--; j--;
        } else if (i > 0 && dp[i][j] == dp[i-1][j] + GAP) {
            alignedA[ai++] = seqA[i - 1];
            alignedB[bi++] = '_';
            i--;
        } else if (j > 0) {
            alignedA[ai++] = '_';
            alignedB[bi++] = seqB[j - 1];
            j--;
        } else break;
    }

    alignedA[ai] = '\0';
    alignedB[bi] = '\0';
    rev(alignedA);
    rev(alignedB);

    Alignment result;
    result.alignedA = alignedA;
    result.alignedB = alignedB;
    result.length = ai;

    for (int i = 0; i <= lenA; i++) free(dp[i]);
    free(dp);

    return result;
}

Alignment hirschberg_align(char* seqA, char* seqB, int depth) {
    int lenA = strlen(seqA);
    int lenB = strlen(seqB);

    Alignment result;
    result.alignedA = NULL;
    result.alignedB = NULL;
    result.length = 0;

    if (lenA == 0) {
        result.alignedA = (char*)malloc(lenB + 1);
        result.alignedB = (char*)malloc(lenB + 1);
        for (int i = 0; i < lenB; i++) {
            result.alignedA[i] = '_';
            result.alignedB[i] = seqB[i];
        }
        result.alignedA[lenB] = '\0';
        result.alignedB[lenB] = '\0';
        result.length = lenB;
        return result;
    }

    if (lenB == 0) {
        result.alignedA = (char*)malloc(lenA + 1);
        result.alignedB = (char*)malloc(lenA + 1);
        for (int i = 0; i < lenA; i++) {
            result.alignedA[i] = seqA[i];
            result.alignedB[i] = '_';
        }
        result.alignedA[lenA] = '\0';
        result.alignedB[lenA] = '\0';
        result.length = lenA;
        return result;
    }

    if (lenA <= HIRSCHBERG_THRESHOLD || lenB <= HIRSCHBERG_THRESHOLD) {
        return nw_full(seqA, seqB);
    }

    int midA = lenA / 2;

    char* seqA_left = (char*)malloc(midA + 1);
    strncpy(seqA_left, seqA, midA);
    seqA_left[midA] = '\0';
    int* scoreL = nw_score(seqA_left, seqB);
    free(seqA_left);

    char* seqA_right = (char*)malloc((lenA - midA) + 1);
    char* seqB_rev = (char*)malloc(lenB + 1);

    for (int i = 0; i < lenA - midA; i++) {
        seqA_right[i] = seqA[lenA - 1 - i];
    }
    seqA_right[lenA - midA] = '\0';

    for (int i = 0; i < lenB; i++) {
        seqB_rev[i] = seqB[lenB - 1 - i];
    }
    seqB_rev[lenB] = '\0';

    int* scoreR = nw_score(seqA_right, seqB_rev);
    free(seqA_right);
    free(seqB_rev);

    int midB = 0;
    int max_score = scoreL[0] + scoreR[lenB];
    for (int j = 1; j <= lenB; j++) {
        int score = scoreL[j] + scoreR[lenB - j];
        if (score > max_score) {
            max_score = score;
            midB = j;
        }
    }

    free(scoreL);
    free(scoreR);

    char* seqA_L = (char*)malloc(midA + 1);
    char* seqB_L = (char*)malloc(midB + 1);
    char* seqA_R = (char*)malloc((lenA - midA) + 1);
    char* seqB_R = (char*)malloc((lenB - midB) + 1);

    strncpy(seqA_L, seqA, midA);
    seqA_L[midA] = '\0';
    strncpy(seqB_L, seqB, midB);
    seqB_L[midB] = '\0';
    strncpy(seqA_R, seqA + midA, lenA - midA);
    seqA_R[lenA - midA] = '\0';
    strncpy(seqB_R, seqB + midB, lenB - midB);
    seqB_R[lenB - midB] = '\0';

    Alignment left = hirschberg_align(seqA_L, seqB_L, depth + 1);
    Alignment right = hirschberg_align(seqA_R, seqB_R, depth + 1);

    free(seqA_L);
    free(seqB_L);
    free(seqA_R);
    free(seqB_R);

    result.length = left.length + right.length;
    result.alignedA = (char*)malloc(result.length + 1);
    result.alignedB = (char*)malloc(result.length + 1);

    memcpy(result.alignedA, left.alignedA, left.length);
    memcpy(result.alignedA + left.length, right.alignedA, right.length);
    result.alignedA[result.length] = '\0';

    memcpy(result.alignedB, left.alignedB, left.length);
    memcpy(result.alignedB + left.length, right.alignedB, right.length);
    result.alignedB[result.length] = '\0';

    free(left.alignedA);
    free(left.alignedB);
    free(right.alignedA);
    free(right.alignedB);

    return result;
}

char* read_fasta(const char* filename) {
    FILE* file = fopen(filename, "r");
    if (!file) {
        printf("Cannot open file: %s\n", filename);
        return NULL;
    }

    fseek(file, 0, SEEK_END);
    long fsize = ftell(file);
    rewind(file);

    char* sequence = (char*)malloc(fsize + 1);
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

int main(int argc, char* argv[]) {
    if (argc != 3) {
        printf("Usage: %s <fasta_file1> <fasta_file2>\n", argv[0]);
        printf("Example: %s seq1.fasta seq2.fasta\n", argv[0]);
        return 1;
    }

    printf("=== Hirschberg Algorithm - Generic Version ===\n\n");

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
    Alignment result = hirschberg_align(seq1, seq2, 0);
    clock_t end = clock();

    double duration = (double)(end - start) / CLOCKS_PER_SEC;

    int matches = 0, mismatches = 0, gaps = 0, score = 0;
    for (int i = 0; i < result.length; i++) {
        if (result.alignedA[i] == '_' || result.alignedB[i] == '_') {
            gaps++;
            score += GAP;
        } else if (result.alignedA[i] == result.alignedB[i]) {
            matches++;
            score += MATCH;
        } else {
            mismatches++;
            score += MISMATCH;
        }
    }

    double similarity = (double)matches / (matches + mismatches + gaps) * 100.0;

    printf("===== Hirschberg Alignment Result =====\n");
    printf("Execution Time: %.4f seconds\n", duration);
    printf("Alignment Score: %d\n", score);
    printf("Aligned Length: %d\n", result.length);
    printf("Matches: %d, Mismatches: %d, Gaps: %d\n", matches, mismatches, gaps);
    printf("Similarity: %.2f%%\n\n", similarity);

    // Save result to file
    char output_filename[512];
    snprintf(output_filename, sizeof(output_filename), "%s_vs_%s_hirschberg_alignment.txt", name1, name2);

    FILE* fout = fopen(output_filename, "w");
    if (fout) {
        fprintf(fout, "%s vs %s - Hirschberg Alignment\n", name1, name2);
        fprintf(fout, "Execution Time: %.4f seconds\n", duration);
        fprintf(fout, "Alignment Score: %d\n", score);
        fprintf(fout, "Aligned Length: %d\n", result.length);
        fprintf(fout, "Matches: %d, Mismatches: %d, Gaps: %d\n", matches, mismatches, gaps);
        fprintf(fout, "Similarity: %.2f%%\n\n", similarity);
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

    return 0;
}
