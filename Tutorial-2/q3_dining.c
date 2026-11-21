/*
 * q3_dining.c - Dining Philosophers Problem Using Pthreads
 *
 * Implements two solutions:
 *   Solution 1 (DEADLOCK-PRONE): Each philosopher picks up LEFT chopstick
 *              first, then RIGHT. If all pick up left simultaneously → deadlock.
 *   Solution 2 (DEADLOCK-FREE):  Odd-numbered philosophers pick up LEFT first
 *              then RIGHT; even-numbered pick up RIGHT first then LEFT.
 *              This breaks the circular wait condition.
 *
 * Compile: gcc -o q3_dining q3_dining.c -lpthread
 * Run:     ./q3_dining
 *
 * The program runs Solution 1 briefly (with a timeout to detect deadlock),
 * then runs Solution 2 to demonstrate deadlock-free operation.
 */

#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <unistd.h>

#ifdef _WIN32
#include <windows.h>
#define sleep_ms(ms) Sleep(ms)
#else
#define sleep_ms(ms) usleep((ms) * 1000)
#endif

#define NUM_PHILOSOPHERS 5
#define NUM_MEALS 3    /* Each philosopher eats this many times */

/* Chopsticks (mutexes) */
pthread_mutex_t chopstick[NUM_PHILOSOPHERS];

/* Control flag for deadlock-prone solution */
volatile int deadlock_detected = 0;

/* =====================================================
 *  SOLUTION 1: Deadlock-Prone (all pick left first)
 * =====================================================*/

void *philosopher_deadlock(void *arg) {
    int id = *(int *)arg;
    int left = id;
    int right = (id + 1) % NUM_PHILOSOPHERS;

    for (int meal = 0; meal < NUM_MEALS && !deadlock_detected; meal++) {
        /* Thinking */
        printf("[DEADLOCK] Philosopher %d is thinking.\n", id);
        sleep_ms(100 + (rand() % 100));

        /* Pick up LEFT chopstick first */
        printf("[DEADLOCK] Philosopher %d picks up LEFT chopstick %d.\n", id, left);
        pthread_mutex_lock(&chopstick[left]);

        /* Small delay to increase chance of deadlock */
        sleep_ms(50);

        /* Try to pick up RIGHT chopstick */
        printf("[DEADLOCK] Philosopher %d tries to pick up RIGHT chopstick %d.\n", id, right);

        /*
         * Use trylock to detect deadlock (in a real deadlock scenario with
         * regular lock, the program would hang forever).
         */
        int result = pthread_mutex_trylock(&chopstick[right]);
        if (result != 0) {
            printf("[DEADLOCK] Philosopher %d CANNOT get RIGHT chopstick %d "
                   "— potential DEADLOCK!\n", id, right);
            pthread_mutex_unlock(&chopstick[left]);
            deadlock_detected = 1;
            break;
        }

        /* Eating */
        printf("[DEADLOCK] Philosopher %d is EATING (meal %d).\n", id, meal + 1);
        sleep_ms(100 + (rand() % 100));

        /* Put down chopsticks */
        printf("[DEADLOCK] Philosopher %d puts down RIGHT chopstick %d.\n", id, right);
        pthread_mutex_unlock(&chopstick[right]);
        printf("[DEADLOCK] Philosopher %d puts down LEFT chopstick %d.\n", id, left);
        pthread_mutex_unlock(&chopstick[left]);
    }

    return NULL;
}

/* =====================================================
 *  SOLUTION 2: Deadlock-Free (odd/even ordering)
 * =====================================================
 *
 * Strategy: Break the circular wait condition.
 *   - Odd philosophers (1, 3): pick up LEFT first, then RIGHT
 *   - Even philosophers (0, 2, 4): pick up RIGHT first, then LEFT
 *
 * This ensures at least one philosopher can always acquire both
 * chopsticks, preventing circular wait.
 */

void *philosopher_safe(void *arg) {
    int id = *(int *)arg;
    int left = id;
    int right = (id + 1) % NUM_PHILOSOPHERS;

    /* Determine pickup order based on odd/even */
    int first, second;
    if (id % 2 == 0) {
        /* Even: pick right first, then left */
        first = right;
        second = left;
    } else {
        /* Odd: pick left first, then right */
        first = left;
        second = right;
    }

    for (int meal = 0; meal < NUM_MEALS; meal++) {
        /* Thinking */
        printf("[SAFE] Philosopher %d is thinking.\n", id);
        sleep_ms(100 + (rand() % 150));

        /* Pick up first chopstick */
        printf("[SAFE] Philosopher %d picks up chopstick %d (first).\n", id, first);
        pthread_mutex_lock(&chopstick[first]);

        /* Pick up second chopstick */
        printf("[SAFE] Philosopher %d picks up chopstick %d (second).\n", id, second);
        pthread_mutex_lock(&chopstick[second]);

        /* Eating */
        printf("[SAFE] Philosopher %d is EATING (meal %d/%d).\n", id, meal + 1, NUM_MEALS);
        sleep_ms(100 + (rand() % 150));

        /* Put down chopsticks */
        printf("[SAFE] Philosopher %d puts down chopstick %d.\n", id, second);
        pthread_mutex_unlock(&chopstick[second]);
        printf("[SAFE] Philosopher %d puts down chopstick %d.\n", id, first);
        pthread_mutex_unlock(&chopstick[first]);
    }

    printf("[SAFE] Philosopher %d has finished all meals.\n", id);
    return NULL;
}

int main(void) {
    pthread_t threads[NUM_PHILOSOPHERS];
    int ids[NUM_PHILOSOPHERS];

    srand(42);

    /* Initialize chopstick mutexes */
    for (int i = 0; i < NUM_PHILOSOPHERS; i++) {
        pthread_mutex_init(&chopstick[i], NULL);
        ids[i] = i;
    }

    /* ===== SOLUTION 1: Deadlock-Prone ===== */
    printf("=============================================================\n");
    printf("  SOLUTION 1: Deadlock-Prone (all pick left first)\n");
    printf("  (Using trylock to detect deadlock rather than hang)\n");
    printf("=============================================================\n\n");

    deadlock_detected = 0;

    for (int i = 0; i < NUM_PHILOSOPHERS; i++) {
        pthread_create(&threads[i], NULL, philosopher_deadlock, &ids[i]);
    }

    for (int i = 0; i < NUM_PHILOSOPHERS; i++) {
        pthread_join(threads[i], NULL);
    }

    if (deadlock_detected) {
        printf("\n*** Deadlock was detected! All philosophers tried to pick up\n");
        printf("*** their left chopstick simultaneously, creating a circular wait.\n");
    }

    /* Re-initialize mutexes for Solution 2 */
    for (int i = 0; i < NUM_PHILOSOPHERS; i++) {
        pthread_mutex_destroy(&chopstick[i]);
        pthread_mutex_init(&chopstick[i], NULL);
    }

    /* ===== SOLUTION 2: Deadlock-Free ===== */
    printf("\n=============================================================\n");
    printf("  SOLUTION 2: Deadlock-Free (odd/even ordering)\n");
    printf("  Odd philosophers: left first. Even: right first.\n");
    printf("=============================================================\n\n");

    for (int i = 0; i < NUM_PHILOSOPHERS; i++) {
        pthread_create(&threads[i], NULL, philosopher_safe, &ids[i]);
    }

    for (int i = 0; i < NUM_PHILOSOPHERS; i++) {
        pthread_join(threads[i], NULL);
    }

    printf("\n*** All philosophers finished eating without deadlock!\n");

    /* Cleanup */
    for (int i = 0; i < NUM_PHILOSOPHERS; i++) {
        pthread_mutex_destroy(&chopstick[i]);
    }

    printf("=============================================================\n");

    return 0;
}
