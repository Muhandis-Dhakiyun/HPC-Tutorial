#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>

#define DEFAULT_SIZE 18250  /* Approximately 50 years of daily records (50 * 365) */

/* Swap two floating point temperature values */
void swap(float *a, float *b) {
    float temp = *a;
    *a = *b;
    *b = temp;
}

/* Partition function for QuickSort (Lomuto partition scheme) */
int partition(float arr[], int low, int high) {
    float pivot = arr[high];
    int i = (low - 1);

    for (int j = low; j < high; j++) {
        if (arr[j] <= pivot) {
            i++;
            swap(&arr[i], &arr[j]);
        }
    }
    swap(&arr[i + 1], &arr[high]);
    return (i + 1);
}

/* QuickSort recursive implementation */
void quick_sort(float arr[], int low, int high) {
    if (low < high) {
        int pi = partition(arr, low, high);
        quick_sort(arr, low, pi - 1);
        quick_sort(arr, pi + 1, high);
    }
}

/* Bubble Sort implementation (Baseline) */
void bubble_sort(float arr[], int n) {
    for (int i = 0; i < n - 1; i++) {
        int swapped = 0;
        for (int j = 0; j < n - i - 1; j++) {
            if (arr[j] > arr[j + 1]) {
                swap(&arr[j], &arr[j + 1]);
                swapped = 1;
            }
        }
        if (!swapped) break;
    }
}

/* Generate realistic simulated daily temperatures (e.g. -10.0 C to 45.0 C) */
void generate_temperatures(float arr[], int n, unsigned int seed) {
    srand(seed);
    for (int i = 0; i < n; i++) {
        /* Generate float between -10.0 and +45.0 degrees Celsius */
        arr[i] = -10.0f + ((float)rand() / (float)RAND_MAX) * 55.0f;
    }
}

/* Copy array helper */
void copy_array(const float src[], float dest[], int n) {
    for (int i = 0; i < n; i++) {
        dest[i] = src[i];
    }
}

/* Verify if an array is sorted in ascending order */
int verify_sorted(const float arr[], int n) {
    for (int i = 0; i < n - 1; i++) {
        if (arr[i] > arr[i + 1]) {
            return 0;
        }
    }
    return 1;
}

int main(int argc, char *argv[]) {
    int n = DEFAULT_SIZE;
    int run_bubble = 1;
    int run_quick = 1;

    if (argc >= 2) {
        n = atoi(argv[1]);
        if (n <= 0) {
            fprintf(stderr, "Invalid size: %s. Using default %d.\n", argv[1], DEFAULT_SIZE);
            n = DEFAULT_SIZE;
        }
    }

    if (argc >= 3) {
        if (strcmp(argv[2], "bubble") == 0) {
            run_quick = 0;
        } else if (strcmp(argv[2], "quick") == 0) {
            run_bubble = 0;
        }
    } else {
        /* Safety check: if n > 100,000, bubble sort takes hours (O(n^2)). Skip unless requested */
        if (n > 100000) {
            printf("[NOTE] Dataset size N=%d is very large. Bubble Sort O(N^2) would require ~10^12 operations.\n", n);
            printf("[NOTE] Running QuickSort only. To force Bubble Sort, run: %s %d bubble\n\n", argv[0], n);
            run_bubble = 0;
        }
    }

    printf("=================================================================\n");
    printf(" Weather Analytics: Daily Temperature Sorting Benchmark\n");
    printf(" Dataset Size: %d records (~%.1f years of daily data)\n", n, (float)n / 365.25f);
    printf("=================================================================\n\n");

    /* Allocate memory for datasets */
    float *original_data = (float *)malloc(n * sizeof(float));
    float *work_data = (float *)malloc(n * sizeof(float));

    if (!original_data || !work_data) {
        fprintf(stderr, "Memory allocation failed for %d elements.\n", n);
        return 1;
    }

    /* Seed with fixed number for reproducible benchmark comparisons */
    generate_temperatures(original_data, n, 42);

    /* ------------------------------------------------------------- */
    /* 1. Quick Sort Benchmark                                        */
    /* ------------------------------------------------------------- */
    if (run_quick) {
        copy_array(original_data, work_data, n);
        clock_t start_q = clock();
        quick_sort(work_data, 0, n - 1);
        clock_t end_q = clock();
        double time_q = ((double)(end_q - start_q)) / CLOCKS_PER_SEC;

        int q_ok = verify_sorted(work_data, n);
        printf("[Quick Sort]\n");
        printf("  Status: %s\n", q_ok ? "PASSED (Array sorted correctly)" : "FAILED");
        printf("  Execution Time: %.6f seconds (%.3f ms)\n\n", time_q, time_q * 1000.0);
    }

    /* ------------------------------------------------------------- */
    /* 2. Bubble Sort Benchmark                                       */
    /* ------------------------------------------------------------- */
    if (run_bubble) {
        copy_array(original_data, work_data, n);
        clock_t start_b = clock();
        bubble_sort(work_data, n);
        clock_t end_b = clock();
        double time_b = ((double)(end_b - start_b)) / CLOCKS_PER_SEC;

        int b_ok = verify_sorted(work_data, n);
        printf("[Bubble Sort (Baseline)]\n");
        printf("  Status: %s\n", b_ok ? "PASSED (Array sorted correctly)" : "FAILED");
        printf("  Execution Time: %.6f seconds (%.3f ms)\n\n", time_b, time_b * 1000.0);
    }

    printf("=================================================================\n");

    free(original_data);
    free(work_data);
    return 0;
}

