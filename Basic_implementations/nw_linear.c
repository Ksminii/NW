
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define MATCH 1
#define MISMATCH -1
#define GAP -1
#define SEQ_LEN 10000
#define TEST_CASES 25

int max_of_three(int a, int b, int c) {
    if (a >= b && a >= c) return a;
    if (b >= a && b >= c) return b;
    return c;
}

int score(char a, char b) {
    return a == b ? MATCH : MISMATCH;
}

void rev(char *str) {
    int len = strlen(str);
    for (int i = 0; i < len / 2; i++) {
        char tmp = str[i];
        str[i] = str[len - 1 - i];
        str[len - 1 - i] = tmp;
    }
}

char *generate_random_sequence(int len) {
    char *seq = malloc(len + 1);
    char bases[] = {'A', 'C', 'G', 'T'};
    for (int i = 0; i < len; i++) {
        seq[i] = bases[rand() % 4];
    }
    seq[len] = '\0';
    return seq;
}

int validate_alignment(const char *a, const char *b, int expected_score, int *recomputed) {
    int score_ = 0;
    for (int i = 0; a[i] && b[i]; i++) {
        if (a[i] == '_' && b[i] == '_') 
            return 0;
        else if (a[i] == '_' || b[i] == '_') 
            score_ += GAP;
        else 
            score_ += score(a[i], b[i]);
    }
    *recomputed = score_;
    return score_ == expected_score;
}

void needleman_wunsch(char *a, char *b, int test_index) {
    int lenA = strlen(a);
    int lenB = strlen(b);

    int **dp = malloc((lenA + 1) * sizeof(int *));
    char **trace = malloc((lenA + 1) * sizeof(char *));
    for (int i = 0; i <= lenA; i++) {
        dp[i] = malloc((lenB + 1) * sizeof(int));
        trace[i] = malloc((lenB + 1) * sizeof(char));
    }
    /*
    dp[0] → ┌──────────────┐
            │ int, int, ...│ (lenB+1개)
            └──────────────┘
    
    dp[1] → ┌──────────────┐
            │ int, int, ...│
        ...
    */

    dp[0][0] = 0;
    trace[0][0] = 'O';  // origin

    for (int i = 1; i <= lenA; i++) {
        dp[i][0] = i * GAP;
        trace[i][0] = 'U';
    }

    for (int j = 1; j <= lenB; j++) {
        dp[0][j] = j * GAP;
        trace[0][j] = 'L';
    }

    /*
            B  ""   B₁   B₂   B₃   B₄
    A      0   -1   -2   -3   -4
    A₁    -1
    A₂    -2
    A₃    -3


            B  ""   B₁   B₂   B₃   B₄
    A      O    L    L    L    L
    A₁     U
    A₂     U
    A₃     U
    
    */

    for (int i = 1; i <= lenA; i++) {
        for (int j = 1; j <= lenB; j++) {
            int diag = dp[i - 1][j - 1] + score(a[i - 1], b[j - 1]);
            int up = dp[i - 1][j] + GAP;
            int left = dp[i][j - 1] + GAP;

            dp[i][j] = max_of_three(diag, up, left);
            if (dp[i][j] == diag) trace[i][j] = 'D';
            else if (dp[i][j] == up) trace[i][j] = 'U';
            else trace[i][j] = 'L';
        }
    }

    // Traceback
    char *alignedA = malloc(lenA + lenB + 1);
    char *alignedB = malloc(lenA + lenB + 1);
    int ai = 0, bi = 0;
    int i = lenA, j = lenB;

    while (i > 0 || j > 0) {
        if (i > 0 && j > 0 && trace[i][j] == 'D') {
            alignedA[ai++] = a[i - 1];
            alignedB[bi++] = b[j - 1];
            i--; j--;
        } else if (i > 0 && trace[i][j] == 'U') {
            alignedA[ai++] = a[i - 1];
            alignedB[bi++] = '_';
            i--;
        } else if (j > 0 && trace[i][j] == 'L') {
            alignedA[ai++] = '_';
            alignedB[bi++] = b[j - 1];
            j--;
        } else break;
        
    }

    alignedA[ai] = '\0';
    alignedB[bi] = '\0';
    //문자열이므로 끝에 null 삽입
    // 정렬 결과는 뒤집어져 있으니까 다시 rev
    rev(alignedA);
    rev(alignedB);

    // Save result
    char filename[64];
    sprintf(filename, "aligned_result_%d_linear.txt", test_index);
    FILE *fout = fopen(filename, "w");
    fprintf(fout, "[Run %d]\n", test_index);
    fprintf(fout, "Alignment Score: %d\n", dp[lenA][lenB]);
    fprintf(fout, "Aligned A:\n%s\n\n", alignedA);
    fprintf(fout, "Aligned B:\n%s\n", alignedB);
    fclose(fout);
    printf("파일 저장 완료: %s\n", filename);

    for (int i = 0; i <= lenA; i++) {
        free(dp[i]);
        free(trace[i]);
    }
    free(dp); free(trace);
    free(alignedA); free(alignedB);
}

int main() {
    srand(time(NULL));
    for (int t = 1; t <= TEST_CASES; t++) {
        printf("\n==== 테스트 %d ====\n", t);
        char *A = generate_random_sequence(SEQ_LEN);
        char *B = generate_random_sequence(SEQ_LEN);

        clock_t start = clock();
        needleman_wunsch(A, B, t);
        clock_t end = clock();

        double duration = (double)(end - start) / CLOCKS_PER_SEC;
        printf("수행 시간: %.4f초\n", duration);

        free(A); free(B);
    }

    // 검증
    for (int i = 1; i <= TEST_CASES; i++) {
        char filename[64];
        sprintf(filename, "aligned_result_%d_linear.txt", i);
        FILE *f = fopen(filename, "r");
        if (!f) {
            printf("[%d] 파일 없음\n", i);
            continue;
        }

        char line[SEQ_LEN];
        int expected_score = 0;
        char alignedA[SEQ_LEN * 2], alignedB[SEQ_LEN * 2];
        alignedA[0] = alignedB[0] = '\0';

        while (fgets(line, sizeof(line), f)) {
            if (strncmp(line, "Alignment Score:", 16) == 0) {
                sscanf(line + 16, "%d", &expected_score);
            } else if (strncmp(line, "Aligned A:", 10) == 0) {
                fgets(alignedA, sizeof(alignedA), f);
            } else if (strncmp(line, "Aligned B:", 10) == 0) {
                fgets(alignedB, sizeof(alignedB), f);
            }
        }
        fclose(f);
        alignedA[strcspn(alignedA, "\r\n")] = 0;
        alignedB[strcspn(alignedB, "\r\n")] = 0;

        int recomputed = 0;
        int valid = validate_alignment(alignedA, alignedB, expected_score, &recomputed);
        if (valid)
            printf("[%d] 검증 결과: PASS(Recomputed=%d, Expected=%d)\n",i,recomputed,expected_score);
        else
            printf("[%d] 검증 결과: FAIL (Recomputed=%d, Expected=%d)\n", i, recomputed, expected_score);
    }

    return 0;
}