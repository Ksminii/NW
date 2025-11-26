#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define MATCH 1
#define MISMATCH -1
#define GAP_OPEN -10
#define GAP_EXTEND -1
#define INF -1000000000

typedef enum { STATE_M, STATE_DX, STATE_DY } State;

int score(char a, char b) {
    return (a == b) ? MATCH : MISMATCH;
}

char *generate_random_sequence(int length) {
    char *seq = malloc(length + 1);
    char bases[] = {'A', 'C', 'G', 'T'};
    for (int i = 0; i < length; i++) {
        seq[i] = bases[rand() % 4];
    }
    seq[length] = '\0';
    return seq;
}

void traceback(int **DP, int **Dx, int **Dy, State **trace, State **traceDx, State **traceDy, char *A, char *B, char **outA, char **outB) {
    int i = strlen(A), j = strlen(B);
    State state = STATE_M;

    char *alignedA = malloc(i + j + 1);
    char *alignedB = malloc(i + j + 1);
    int idx = 0;

    while (i > 0 || j > 0) {
        if (state == STATE_M) {
            State prev = trace[i][j];
            if (prev == STATE_M) {
                alignedA[idx] = A[i - 1];
                alignedB[idx] = B[j - 1];
                i--; j--;
            } else if (prev == STATE_DX) {
                state = STATE_DX;
            } else {
                state = STATE_DY;
            }
        } else if (state == STATE_DX) {
            State prev = traceDx[i][j];
            alignedA[idx] = A[i - 1];
            alignedB[idx] = '_';
            i--;
            state = prev;
        } else if (state == STATE_DY) {
            State prev = traceDy[i][j];
            alignedA[idx] = '_';
            alignedB[idx] = B[j - 1];
            j--;
            state = prev;
        }
        idx++;
    }

    alignedA[idx] = '\0';
    alignedB[idx] = '\0';

    *outA = malloc(idx + 1);
    *outB = malloc(idx + 1);
    for (int k = 0; k < idx; k++) {
        (*outA)[k] = alignedA[idx - k - 1];
        (*outB)[k] = alignedB[idx - k - 1];
    }
    (*outA)[idx] = '\0';
    (*outB)[idx] = '\0';

    free(alignedA);
    free(alignedB);
}

int main() {
    srand(time(NULL));
    const int TESTS = 10;
    const int LEN = 10000;

    for (int run = 1; run <= TESTS; run++) {
        printf("\n[Run %d] Needleman-Wunsch 정렬 시작...\n", run);

        char *A = generate_random_sequence(LEN);
        char *B = generate_random_sequence(LEN);
        int lenA = strlen(A), lenB = strlen(B);

        int **DP = malloc((lenA + 1) * sizeof(int *));
        int **Dx = malloc((lenA + 1) * sizeof(int *));
        int **Dy = malloc((lenA + 1) * sizeof(int *));
        State **trace = malloc((lenA + 1) * sizeof(State *));
        State **traceDx = malloc((lenA + 1) * sizeof(State *));
        State **traceDy = malloc((lenA + 1) * sizeof(State *));

        for (int i = 0; i <= lenA; i++) {
            DP[i] = malloc((lenB + 1) * sizeof(int));
            Dx[i] = malloc((lenB + 1) * sizeof(int));
            Dy[i] = malloc((lenB + 1) * sizeof(int));
            trace[i] = malloc((lenB + 1) * sizeof(State));
            traceDx[i] = malloc((lenB + 1) * sizeof(State));
            traceDy[i] = malloc((lenB + 1) * sizeof(State));
        }

        for (int i = 0; i <= lenA; i++) {
            for (int j = 0; j <= lenB; j++) {
                DP[i][j] = Dx[i][j] = Dy[i][j] = INF;
            }
        }

        DP[0][0] = 0;

        for (int i = 1; i <= lenA; i++) {
            Dx[i][0] = GAP_OPEN + (i - 1) * GAP_EXTEND;
            DP[i][0] = Dx[i][0];
            traceDx[i][0] = STATE_DX;
            trace[i][0] = STATE_DX;
        }
        for (int j = 1; j <= lenB; j++) {
            Dy[0][j] = GAP_OPEN + (j - 1) * GAP_EXTEND;
            DP[0][j] = Dy[0][j];
            traceDy[0][j] = STATE_DY;
            trace[0][j] = STATE_DY;
        }

        clock_t start = clock();

        for (int i = 1; i <= lenA; i++) {
            for (int j = 1; j <= lenB; j++) {
                int up_ext = Dx[i - 1][j] + GAP_EXTEND;
                int up_open = DP[i - 1][j] + GAP_OPEN + GAP_EXTEND;
                if (up_ext >= up_open) {
                    Dx[i][j] = up_ext;
                    traceDx[i][j] = STATE_DX;
                } else {
                    Dx[i][j] = up_open;
                    traceDx[i][j] = STATE_M;
                }

                int left_ext = Dy[i][j - 1] + GAP_EXTEND;
                int left_open = DP[i][j - 1] + GAP_OPEN + GAP_EXTEND;
                if (left_ext >= left_open) {
                    Dy[i][j] = left_ext;
                    traceDy[i][j] = STATE_DY;
                } else {
                    Dy[i][j] = left_open;
                    traceDy[i][j] = STATE_M;
                }

                int m = DP[i - 1][j - 1] + score(A[i - 1], B[j - 1]);

                if (m >= Dx[i][j] && m >= Dy[i][j]) {
                    DP[i][j] = m;
                    trace[i][j] = STATE_M;
                } else if (Dx[i][j] >= Dy[i][j]) {
                    DP[i][j] = Dx[i][j];
                    trace[i][j] = STATE_DX;
                } else {
                    DP[i][j] = Dy[i][j];
                    trace[i][j] = STATE_DY;
                }
            }
        }

        char *alignedA, *alignedB;
        traceback(DP, Dx, Dy, trace, traceDx, traceDy, A, B, &alignedA, &alignedB);
        clock_t end = clock();
        double time_spent = (double)(end - start) / CLOCKS_PER_SEC;

        printf("정렬 완료 | 점수: %d | 시간: %.2f초\n", DP[lenA][lenB], time_spent);

        char filename[50];
        sprintf(filename, "aligned_result_%d.txt", run);
        FILE *f = fopen(filename, "w");
        fprintf(f, "[Run %d]\n", run);
        fprintf(f, "Alignment Score: %d\n", DP[lenA][lenB]);
        fprintf(f, "Execution Time: %.2f seconds\n\n", time_spent);
        fprintf(f, "Aligned A:\n%s\n\n", alignedA);
        fprintf(f, "Aligned B:\n%s\n", alignedB);
        fclose(f);

        printf("%s 저장 완료\n", filename);

        free(A); free(B); free(alignedA); free(alignedB);
        for (int i = 0; i <= lenA; i++) {
            free(DP[i]); free(Dx[i]); free(Dy[i]);
            free(trace[i]); free(traceDx[i]); free(traceDy[i]);
        }
        free(DP); free(Dx); free(Dy);
        free(trace); free(traceDx); free(traceDy);
    }

    return 0;
}