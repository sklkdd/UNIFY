// buildindex_wrapper.cpp - UNIFY index construction wrapper for FANNS benchmarking
// This wrapper builds a UNIFY HSIG index with slot-based partitioning for range filtering

#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <algorithm>
#include <cmath>
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

// Read .bin format (num_points, dim, then flat vectors)
pair<vector<vector<float>>, int> read_bin(const string& filename) {
    ifstream file(filename, ios::binary);
    if (!file) {
        cerr << "Error: Unable to open file " << filename << "\n";
        exit(1);
    }
    
    int num_points, dim;
    file.read(reinterpret_cast<char*>(&num_points), sizeof(int));
    file.read(reinterpret_cast<char*>(&dim), sizeof(int));
    
    vector<vector<float>> data(num_points, vector<float>(dim));
    for (int i = 0; i < num_points; i++) {
        file.read(reinterpret_cast<char*>(data[i].data()), dim * sizeof(float));
    }
    
    file.close();
    return {data, dim};
}

// Compute slot ranges using equal-frequency partitioning
// Ports the Python utils.compute_slot_ranges() function
vector<pair<int64_t, int64_t>> compute_slot_ranges(const vector<int>& scalars, int num_slots) {
    if (scalars.empty()) {
        throw runtime_error("Cannot compute slot ranges: scalars is empty");
    }
    
    // Sort scalars to compute percentiles
    vector<int> sorted_scalars = scalars;
    sort(sorted_scalars.begin(), sorted_scalars.end());
    
    int n = sorted_scalars.size();
    vector<pair<int64_t, int64_t>> slot_ranges(num_slots);
    
    // Compute percentile values
    vector<int64_t> percentile_values;
    double step = 100.0 / num_slots;
    
    for (int i = 1; i < num_slots; i++) {
        double percentile = step * i;
        // Linear interpolation for percentile
        double pos = percentile / 100.0 * (n - 1);
        int lower_idx = static_cast<int>(floor(pos));
        int upper_idx = static_cast<int>(ceil(pos));
        
        int64_t value;
        if (lower_idx == upper_idx) {
            value = sorted_scalars[lower_idx];
        } else {
            double frac = pos - lower_idx;
            value = static_cast<int64_t>(
                sorted_scalars[lower_idx] * (1 - frac) + sorted_scalars[upper_idx] * frac
            );
        }
        percentile_values.push_back(value);
    }
    
    // Construct slot ranges
    int64_t min_val = sorted_scalars.front();
    int64_t max_val = sorted_scalars.back();
    
    for (int i = 0; i < num_slots; i++) {
        if (i == 0) {
            slot_ranges[i].first = min_val;
        } else {
            slot_ranges[i].first = percentile_values[i - 1];
        }
        
        if (i == num_slots - 1) {
            slot_ranges[i].second = max_val;
        } else {
            slot_ranges[i].second = percentile_values[i];
        }
    }
    
    return slot_ranges;
}

int main(int argc, char** argv) {
    if (argc != 8) {
        cerr << "Usage: " << argv[0] << " <data.bin> <attribute_values.txt> "
             << "<output_index> <M> <ef_construction> <num_slots> <random_seed>\n";
        cerr << "\n";
        cerr << "Arguments:\n";
        cerr << "  data.bin            - Input vectors in .bin format\n";
        cerr << "  attribute_values.txt - One attribute value per line (integer)\n";
        cerr << "  output_index        - Path to save the index\n";
        cerr << "  M                   - Max links per slot (UNIFY parameter)\n";
        cerr << "  ef_construction     - Construction ef parameter\n";
        cerr << "  num_slots           - Number of slots for partitioning\n";
        cerr << "  random_seed         - Random seed for index construction\n";
        return 1;
    }

    string data_bin = argv[1];
    string attr_file = argv[2];
    string output_index = argv[3];
    size_t M = stoull(argv[4]);
    size_t ef_construction = stoull(argv[5]);
    size_t num_slots = stoull(argv[6]);
    size_t random_seed = stoull(argv[7]);

    // Use all available threads for index construction
    int num_threads = thread::hardware_concurrency();
    omp_set_num_threads(num_threads);
    
    cout << "=== UNIFY Index Construction ===" << endl;
    cout << "Data: " << data_bin << endl;
    cout << "Attributes: " << attr_file << endl;
    cout << "Output index: " << output_index << endl;
    cout << "Parameters: M=" << M << ", ef_construction=" << ef_construction 
         << ", num_slots=" << num_slots << ", seed=" << random_seed << endl;
    cout << "Threads: " << num_threads << endl;

    // ========== DATA LOADING (NOT TIMED) ==========
    cout << "\nLoading data..." << endl;
    auto [data, dim] = read_bin(data_bin);
    size_t num_points = data.size();
    cout << "Loaded " << num_points << " vectors of dimension " << dim << endl;

    // Load attribute values
    vector<int> attributes = read_one_int_per_line(attr_file);
    if (attributes.size() != num_points) {
        cerr << "Error: Mismatch between data size (" << num_points 
             << ") and attribute size (" << attributes.size() << ")\n";
        return 1;
    }
    cout << "Loaded " << attributes.size() << " attribute values" << endl;

    // Compute slot ranges (NOT TIMED - preprocessing)
    cout << "\nComputing slot ranges..." << endl;
    vector<pair<int64_t, int64_t>> slot_ranges = compute_slot_ranges(attributes, num_slots);
    
    cout << "Slot ranges:" << endl;
    for (size_t i = 0; i < slot_ranges.size(); i++) {
        cout << "  Slot " << i << ": [" << slot_ranges[i].first 
             << ", " << slot_ranges[i].second << "]" << endl;
    }

    // ========== INDEX CONSTRUCTION (TIMED) ==========
    cout << "\n--- Starting index construction (TIMED) ---" << endl;
    
    // Start thread monitoring
    atomic<bool> done_monitoring(false);
    thread monitor_thread(monitor_thread_count, ref(done_monitoring));
    
    auto start_time = high_resolution_clock::now();

    // Initialize UNIFY index
    hannlib::L2Space space(dim);
    hannlib::ScalarHSIG<float> index(&space, slot_ranges, num_points, M, ef_construction, random_seed);
    
    // Insert all points with their attributes
    for (size_t i = 0; i < num_points; i++) {
        index.Insert(data[i].data(), i, attributes[i]);
        
        if ((i + 1) % 10000 == 0) {
            cout << "  Inserted " << (i + 1) << " / " << num_points << " points" << endl;
        }
    }
    
    // Save index
    index.SaveIndex(output_index);
    
    auto end_time = high_resolution_clock::now();
    
    // Stop thread monitoring
    done_monitoring = true;
    monitor_thread.join();
    
    cout << "--- Index construction complete ---\n" << endl;

    // ========== TIMING OUTPUT ==========
    double build_time_sec = duration_cast<duration<double>>(end_time - start_time).count();
    
    cout << "BUILD_TIME_SECONDS: " << build_time_sec << endl;
    cout << "PEAK_THREADS: " << peak_threads.load() << endl;
    
    // Memory footprint
    peak_memory_footprint();
    
    cout << "\nIndex saved to: " << output_index << endl;
    
    return 0;
}
