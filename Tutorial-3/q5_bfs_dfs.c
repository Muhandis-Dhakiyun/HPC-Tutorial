/*
 * Question 5: Parallel BFS and DFS using OpenMP
 * ===============================================
 * - BFS: Uses #pragma omp parallel for on the frontier (level-synchronous BFS)
 * - DFS: Uses #pragma omp task for recursive calls
 *
 * Graph is represented as an adjacency list.
 *
 * Compile: gcc -fopenmp -O2 -o q5_bfs_dfs q5_bfs_dfs.c
 * Run:     ./q5_bfs_dfs
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>

/* ========================== Graph Data Structures ========================== */

/* Adjacency list node */
typedef struct AdjNode {
    int vertex;
    struct AdjNode *next;
} AdjNode;

/* Graph structure */
typedef struct {
    int num_vertices;
    AdjNode **adj;  /* Array of adjacency list heads */
} Graph;

/* Create a graph with n vertices */
Graph* create_graph(int n) {
    Graph *g = (Graph *)malloc(sizeof(Graph));
    g->num_vertices = n;
    g->adj = (AdjNode **)calloc(n, sizeof(AdjNode *));
    return g;
}

/* Add an undirected edge */
void add_edge(Graph *g, int u, int v) {
    /* Add v to u's list */
    AdjNode *node = (AdjNode *)malloc(sizeof(AdjNode));
    node->vertex = v;
    node->next = g->adj[u];
    g->adj[u] = node;

    /* Add u to v's list (undirected) */
    node = (AdjNode *)malloc(sizeof(AdjNode));
    node->vertex = u;
    node->next = g->adj[v];
    g->adj[v] = node;
}

/* Free graph memory */
void free_graph(Graph *g) {
    for (int i = 0; i < g->num_vertices; i++) {
        AdjNode *curr = g->adj[i];
        while (curr) {
            AdjNode *tmp = curr;
            curr = curr->next;
            free(tmp);
        }
    }
    free(g->adj);
    free(g);
}

/* Build a sample graph for demonstration */
Graph* build_sample_graph() {
    /*
     * Graph with 12 vertices:
     *
     *       0 --- 1 --- 2
     *       |     |     |
     *       3 --- 4 --- 5
     *       |     |     |
     *       6 --- 7 --- 8
     *             |
     *             9 --- 10 --- 11
     */
    int n = 12;
    Graph *g = create_graph(n);
    add_edge(g, 0, 1);  add_edge(g, 1, 2);
    add_edge(g, 0, 3);  add_edge(g, 1, 4);  add_edge(g, 2, 5);
    add_edge(g, 3, 4);  add_edge(g, 4, 5);
    add_edge(g, 3, 6);  add_edge(g, 4, 7);  add_edge(g, 5, 8);
    add_edge(g, 6, 7);  add_edge(g, 7, 8);
    add_edge(g, 7, 9);  add_edge(g, 9, 10); add_edge(g, 10, 11);
    return g;
}

/* ============================== Parallel BFS ============================== */

/*
 * Level-synchronous BFS using OpenMP.
 * At each level, we process all vertices in the current frontier in parallel.
 * Next-frontier construction is protected with critical sections.
 */
void parallel_bfs(Graph *g, int source) {
    int n = g->num_vertices;
    int *visited = (int *)calloc(n, sizeof(int));
    int *distance = (int *)malloc(n * sizeof(int));
    for (int i = 0; i < n; i++) distance[i] = -1;

    /* Frontier arrays */
    int *current_frontier = (int *)malloc(n * sizeof(int));
    int *next_frontier    = (int *)malloc(n * sizeof(int));
    int current_size = 0, next_size = 0;

    /* Start BFS from source */
    visited[source] = 1;
    distance[source] = 0;
    current_frontier[0] = source;
    current_size = 1;

    int level = 0;

    printf("  BFS traversal from vertex %d:\n", source);
    printf("  Level %d: ", level);
    for (int i = 0; i < current_size; i++) printf("%d ", current_frontier[i]);
    printf("\n");

    while (current_size > 0) {
        next_size = 0;

        /*
         * Process all vertices in the current frontier in parallel.
         * Each thread explores neighbors of its assigned frontier vertices.
         */
        #pragma omp parallel for schedule(dynamic)
        for (int i = 0; i < current_size; i++) {
            int u = current_frontier[i];
            AdjNode *neighbor = g->adj[u];

            while (neighbor) {
                int v = neighbor->vertex;

                /*
                 * Use atomic compare to avoid duplicate visits.
                 * __sync_bool_compare_and_swap atomically sets visited[v]=1
                 * only if it was 0, returning true on success.
                 */
                if (!visited[v]) {
                    int old = __sync_bool_compare_and_swap(&visited[v], 0, 1);
                    if (old) {
                        distance[v] = distance[u] + 1;

                        /* Add to next frontier — protected by critical */
                        #pragma omp critical
                        {
                            next_frontier[next_size++] = v;
                        }
                    }
                }
                neighbor = neighbor->next;
            }
        }

        level++;
        if (next_size > 0) {
            printf("  Level %d: ", level);
            for (int i = 0; i < next_size; i++) printf("%d ", next_frontier[i]);
            printf("\n");
        }

        /* Swap frontiers */
        int *tmp = current_frontier;
        current_frontier = next_frontier;
        next_frontier = tmp;
        current_size = next_size;
    }

    /* Print distances */
    printf("  Distances from source %d:\n    ", source);
    for (int i = 0; i < n; i++) printf("v%d=%d ", i, distance[i]);
    printf("\n");

    free(visited);
    free(distance);
    free(current_frontier);
    free(next_frontier);
}

/* ============================== Parallel DFS ============================== */

/*
 * Recursive DFS using OpenMP tasks.
 * Each unvisited neighbor spawns a new task for parallel exploration.
 * visited[] array is protected with atomic operations.
 */
void dfs_task(Graph *g, int u, int *visited, int *order, int *order_idx) {
    /* Record visit order */
    int idx;
    #pragma omp atomic capture
    idx = (*order_idx)++;

    order[idx] = u;

    /* Explore neighbors */
    AdjNode *neighbor = g->adj[u];
    while (neighbor) {
        int v = neighbor->vertex;

        /* Atomically check and mark as visited */
        int already_visited;
        #pragma omp critical(dfs_visit)
        {
            already_visited = visited[v];
            if (!already_visited) {
                visited[v] = 1;
            }
        }

        if (!already_visited) {
            /* Spawn a task for each unvisited neighbor */
            #pragma omp task firstprivate(v)
            {
                dfs_task(g, v, visited, order, order_idx);
            }
        }
        neighbor = neighbor->next;
    }

    /* Wait for all child tasks before returning */
    #pragma omp taskwait
}

void parallel_dfs(Graph *g, int source) {
    int n = g->num_vertices;
    int *visited = (int *)calloc(n, sizeof(int));
    int *order = (int *)malloc(n * sizeof(int));
    int order_idx = 0;

    visited[source] = 1;

    printf("  DFS traversal from vertex %d:\n", source);

    #pragma omp parallel
    {
        #pragma omp single
        {
            dfs_task(g, source, visited, order, &order_idx);
        }
    }

    printf("  Visit order: ");
    for (int i = 0; i < order_idx; i++) {
        printf("%d ", order[i]);
    }
    printf("\n");
    printf("  Total vertices visited: %d\n", order_idx);

    free(visited);
    free(order);
}

/* ================================= Main ================================= */

int main() {
    printf("============================================\n");
    printf("  Parallel BFS and DFS (OpenMP)\n");
    printf("  Threads: %d\n", omp_get_max_threads());
    printf("============================================\n\n");

    Graph *g = build_sample_graph();
    printf("Graph: 12 vertices, grid + branch structure\n\n");

    /* BFS */
    printf("--- Parallel BFS (level-synchronous) ---\n");
    double t0 = omp_get_wtime();
    parallel_bfs(g, 0);
    double t1 = omp_get_wtime();
    printf("  Time: %.6f s\n\n", t1 - t0);

    /* DFS */
    printf("--- Parallel DFS (task-based) ---\n");
    t0 = omp_get_wtime();
    parallel_dfs(g, 0);
    t1 = omp_get_wtime();
    printf("  Time: %.6f s\n\n", t1 - t0);

    printf("Note: Parallel DFS visit order may differ from serial DFS\n");
    printf("due to non-deterministic task scheduling, but all reachable\n");
    printf("vertices are visited exactly once.\n");

    free_graph(g);

    return 0;
}
