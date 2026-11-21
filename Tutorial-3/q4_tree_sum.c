/*
 * Question 4: Parallel Binary Tree Sum using OpenMP Tasks
 * ========================================================
 * Build a binary tree with random integer values, then compute
 * the sum of all node values using OpenMP tasks.
 *
 * Each recursive call spawns a task for the left subtree and
 * processes the right subtree in the current task. The parent
 * waits for child tasks using #pragma omp taskwait.
 *
 * Compile: gcc -fopenmp -O2 -o q4_tree_sum q4_tree_sum.c
 * Run:     ./q4_tree_sum
 */

#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

/* Binary tree node */
typedef struct TreeNode {
    int value;
    struct TreeNode *left;
    struct TreeNode *right;
} TreeNode;

/*
 * Build a balanced binary tree of given depth.
 * Values are assigned sequentially for easy verification.
 * counter is used to assign unique values to each node.
 */
TreeNode* build_tree(int depth, int *counter) {
    if (depth <= 0) return NULL;

    TreeNode *node = (TreeNode *)malloc(sizeof(TreeNode));
    if (!node) {
        fprintf(stderr, "Memory allocation failed\n");
        exit(1);
    }

    node->value = ++(*counter);  /* Assign sequential value */
    node->left  = build_tree(depth - 1, counter);
    node->right = build_tree(depth - 1, counter);

    return node;
}

/*
 * Compute sum of all nodes in the tree — SERIAL version.
 * Used as reference for correctness checking.
 */
long long serial_tree_sum(TreeNode *node) {
    if (node == NULL) return 0;
    return node->value + serial_tree_sum(node->left) + serial_tree_sum(node->right);
}

/*
 * Compute sum of all nodes in the tree — PARALLEL version using OpenMP tasks.
 *
 * Strategy:
 * - For each node, spawn a task for the left subtree.
 * - Compute the right subtree sum in the current task (avoids overhead).
 * - Use taskwait to synchronize before combining results.
 * - Use a depth cutoff: below a certain depth, switch to serial
 *   computation to avoid excessive task creation overhead.
 */
long long parallel_tree_sum(TreeNode *node, int depth) {
    if (node == NULL) return 0;

    long long left_sum = 0, right_sum = 0;

    /* Cutoff: for shallow subtrees, compute serially to avoid task overhead */
    if (depth < 5) {
        return serial_tree_sum(node);
    }

    /* Spawn a task for the left subtree */
    #pragma omp task shared(left_sum) firstprivate(node, depth)
    {
        left_sum = parallel_tree_sum(node->left, depth - 1);
    }

    /* Compute right subtree in current task (no extra task overhead) */
    right_sum = parallel_tree_sum(node->right, depth - 1);

    /* Wait for the left subtree task to complete */
    #pragma omp taskwait

    return node->value + left_sum + right_sum;
}

/* Free all nodes in the tree */
void free_tree(TreeNode *node) {
    if (node == NULL) return;
    free_tree(node->left);
    free_tree(node->right);
    free(node);
}

/* Count nodes in tree */
int count_nodes(TreeNode *node) {
    if (node == NULL) return 0;
    return 1 + count_nodes(node->left) + count_nodes(node->right);
}

int main() {
    int tree_depth = 20;  /* Creates 2^20 - 1 = 1,048,575 nodes */
    int counter = 0;

    printf("============================================\n");
    printf("  Parallel Binary Tree Sum (OpenMP Tasks)\n");
    printf("  Threads: %d\n", omp_get_max_threads());
    printf("============================================\n\n");

    /* Build tree */
    printf("Building binary tree of depth %d...\n", tree_depth);
    double t0 = omp_get_wtime();
    TreeNode *root = build_tree(tree_depth, &counter);
    double t1 = omp_get_wtime();

    int total_nodes = counter;
    printf("Tree built: %d nodes in %.4f s\n\n", total_nodes, t1 - t0);

    /* Serial sum */
    printf("Computing serial sum...\n");
    t0 = omp_get_wtime();
    long long serial_sum = serial_tree_sum(root);
    t1 = omp_get_wtime();
    double serial_time = t1 - t0;
    printf("  Serial sum   = %lld\n", serial_sum);
    printf("  Serial time  = %.6f s\n\n", serial_time);

    /* Parallel sum */
    printf("Computing parallel sum (OpenMP tasks)...\n");
    long long par_sum = 0;
    t0 = omp_get_wtime();

    #pragma omp parallel
    {
        /* Only one thread creates the initial task */
        #pragma omp single
        {
            par_sum = parallel_tree_sum(root, tree_depth);
        }
    }

    t1 = omp_get_wtime();
    double parallel_time = t1 - t0;
    printf("  Parallel sum = %lld\n", par_sum);
    printf("  Parallel time= %.6f s\n\n", parallel_time);

    /* Verification */
    /* For nodes valued 1..N, sum = N*(N+1)/2 */
    long long expected = (long long)total_nodes * (total_nodes + 1) / 2;
    printf("Verification:\n");
    printf("  Expected sum (N*(N+1)/2) = %lld\n", expected);
    printf("  Serial  result matches: %s\n", serial_sum == expected ? "YES" : "NO");
    printf("  Parallel result matches: %s\n", par_sum == expected ? "YES" : "NO");
    printf("  Speedup: %.2fx\n", serial_time / parallel_time);

    /* Cleanup */
    free_tree(root);

    return 0;
}
