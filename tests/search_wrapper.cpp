// search_wrapper.cpp - UNIFY query execution wrapper for FANNS benchmarking
// This wrapper performs range-filtered ANN queries using UNIFY HSIG index

#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <queue>
#include <set>
#include <chrono>
#include <thread>
#include <atomic>
#include <omp.h>

#include "../hannlib/api.h"
#include "../include/fanns_survey_helpers.cpp"

// Global atomic for peak thread count
std::atomic<int> peak_threads(0);

using namespace std;
using namespace std::chrono;

const int QUERY_K = 10;

int main(int argc, char** argv) {
    if (argc != 11) {
        cerr << "Usage: " << argv[0] << " --query_path <query.fvecs> "
             << "--query_ranges_file <query_ranges.csv> "
             << "--groundtruth_file <groundtruth.ivecs> "
             << "--index_file <index_path> "
             << "--ef_search <ef_search>\n";
        cerr << "\n";
        cerr << "Arguments:\n";
        cerr << "  --query_path         - Query vectors in .fvecs format\n";
        cerr << "  --query_ranges_file  - Query ranges (low-high per line, CSV)\n";
        cerr << "  --groundtruth_file   - Groundtruth in .ivecs format\n";
        cerr << "  --index_file         - Path to the saved index\n";
        cerr << "  --ef_search          - Search ef parameter\n";
        return 1;
    }

    // Parse command-line arguments
    string query_path, query_ranges_file, groundtruth_file, index_file;
    int ef_search = -1;

    for (int i = 1; i < argc; i += 2) {
        string arg = argv[i];
        if (arg == "--query_path") query_path = argv[i + 1];
        else if (arg == "--query_ranges_file") query_ranges_file = argv[i + 1];
        else if (arg == "--groundtruth_file") groundtruth_file = argv[i + 1];
        else if (arg == "--index_file") index_file = argv[i + 1];
        else if (arg == "--ef_search") ef_search = stoi(argv[i + 1]);
    }

    // Validate inputs
    if (query_path.empty() || query_ranges_file.empty() || 
        groundtruth_file.empty() || index_file.empty()) {
        cerr << "Error: Missing required arguments\n";
        return 1;
    }
    if (ef_search <= 0) {
        cerr << "Error: ef_search must be a positive integer\n";
        return 1;
    }

    // Restrict to single thread for query execution
    omp_set_num_threads(1);

    cout << "=== UNIFY Query Execution ===" << endl;
    cout << "Query file: " << query_path << endl;
    cout << "Query ranges: " << query_ranges_file << endl;
    cout << "Groundtruth: " << groundtruth_file << endl;
    cout << "Index: " << index_file << endl;
    cout << "ef_search: " << ef_search << endl;

    // ========== DATA LOADING (NOT TIMED) ==========
    cout << "\nLoading queries..." << endl;
    vector<vector<float>> queries = read_fvecs(query_path);
    int num_queries = queries.size();
    int dim = queries.empty() ? 0 : queries[0].size();
    cout << "Loaded " << num_queries << " queries of dimension " << dim << endl;

    // Load query ranges (format: "low-high" per line, e.g., "10-50")
    vector<pair<int, int>> query_ranges = read_two_ints_per_line(query_ranges_file);
    if (query_ranges.size() != num_queries) {
        cerr << "Error: Number of query ranges (" << query_ranges.size() 
             << ") != number of queries (" << num_queries << ")\n";
        return 1;
    }
    cout << "Loaded " << query_ranges.size() << " query ranges" << endl;

    // Load groundtruth
    vector<vector<int>> groundtruth = read_ivecs(groundtruth_file);
    if (groundtruth.size() != num_queries) {
        cerr << "Error: Number of groundtruth entries (" << groundtruth.size() 
             << ") != number of queries (" << num_queries << ")\n";
        return 1;
    }
    cout << "Loaded groundtruth with " << groundtruth.size() << " entries" << endl;

    // Load the index
    cout << "\nLoading index..." << endl;
    hannlib::L2Space space(dim);
    // Load with max_elements = 0 means auto-detect from index file
    hannlib::ScalarHSIG<float> index(&space, index_file, false, 0);
    cout << "Index loaded successfully" << endl;
    cout << "Index size: " << index.get_current_count() << " points" << endl;

    // Set search parameters
    index.set_ef(ef_search);

    // ========== QUERY EXECUTION (TIMED) ==========
    cout << "\n--- Starting query execution (TIMED) ---" << endl;
    
    // Start thread monitoring
    atomic<bool> done_monitoring(false);
    thread monitor_thread(monitor_thread_count, ref(done_monitoring));

    // Store results for later recall calculation
    vector<vector<int>> query_results(num_queries);

    auto start_time = high_resolution_clock::now();

    // Execute queries
    for (int i = 0; i < num_queries; i++) {
        int64_t low = query_ranges[i].first;
        int64_t high = query_ranges[i].second;
        
        // Perform hybrid search (range-filtered ANN)
        auto result = index.OptimizedHybridSearch(
            queries[i].data(), 
            QUERY_K, 
            make_pair(low, high)
        );
        
        // Extract IDs from result priority queue
        query_results[i].reserve(QUERY_K);
        while (!result.empty()) {
            query_results[i].push_back(result.top().second);
            result.pop();
        }
        
        if ((i + 1) % 1000 == 0) {
            cout << "  Processed " << (i + 1) << " / " << num_queries << " queries" << endl;
        }
    }

    auto end_time = high_resolution_clock::now();
    
    // Stop thread monitoring
    done_monitoring = true;
    monitor_thread.join();
    
    cout << "--- Query execution complete ---\n" << endl;

    // ========== TIMING OUTPUT ==========
    double query_time_sec = duration_cast<duration<double>>(end_time - start_time).count();
    double qps = num_queries / query_time_sec;

    // ========== RECALL CALCULATION (NOT TIMED) ==========
    int total_true_positives = 0;
    for (int i = 0; i < num_queries; i++) {
        // Convert query results to set for faster lookup
        set<int> result_set(query_results[i].begin(), query_results[i].end());
        
        // Count true positives
        for (int gt_id : groundtruth[i]) {
            if (result_set.count(gt_id)) {
                total_true_positives++;
            }
        }
    }

    float recall = static_cast<float>(total_true_positives) / (num_queries * QUERY_K);

    // ========== OUTPUT RESULTS ==========
    cout << "Query time (s): " << query_time_sec << endl;
    cout << "Peak thread count: " << peak_threads.load() << endl;
    cout << "QPS: " << qps << endl;
    cout << "Recall: " << recall << endl;
    
    // Memory footprint
    peak_memory_footprint();

    return 0;
}
