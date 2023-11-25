#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <omp.h>
#include <algorithm>

// Constants for experiment parameters
const std::vector<int> Ms = {10000, 100000, 1000000};
const std::vector<int> chunkSizes = {1, 10, 100, 1000};
const std::vector<int> threadCounts = {2, 4, 8};
const std::vector<std::string> schedMethods = {"static", "dynamic", "guided"};

// Function to check if a number is prime
bool isPrime(int number, const std::vector<int>& primes) {
    for (int prime : primes) {
        if (number % prime == 0) return false;
        if (prime * prime > number) break;  // Optimized break condition
    }
    return true;
}

// Function to calculate primes using OpenMP parallelization
double calculatePrimes(int M, int numThreads, int chunkSize, const std::string& schedMethod) {
    std::vector<int> primes;
    primes.push_back(2);
    int until = sqrt(M);

    // Calculate initial set of primes
    for (int n = 3; n <= until; n += 2) {
        if (isPrime(n, primes)) {
            primes.push_back(n);
        }
    }

    omp_set_num_threads(numThreads); // Set the number of threads for OpenMP

    double startTime = omp_get_wtime();
    std::vector<std::vector<int>> threadPrimes(omp_get_max_threads());

    // Parallel section using OpenMP
    #pragma omp parallel
    {
        std::vector<int> localPrimes = primes; // Local copy of primes for each thread
        // Different scheduling methods
        if (schedMethod == "static") {
            #pragma omp for schedule(static, chunkSize) nowait
            for (int n = until + 1; n <= M; n += 2) {
                if (isPrime(n, localPrimes)) {
                    localPrimes.push_back(n);
                }
            }
        } else if (schedMethod == "dynamic") {
            #pragma omp for schedule(dynamic, chunkSize) nowait
            for (int n = until + 1; n <= M; n += 2) {
                if (isPrime(n, localPrimes)) {
                    localPrimes.push_back(n);
                }
            }
        } else if (schedMethod == "guided") {
            #pragma omp for schedule(guided, chunkSize) nowait
            for (int n = until + 1; n <= M; n += 2) {
                if (isPrime(n, localPrimes)) {
                    localPrimes.push_back(n);
                }
            }
        }

        // Combine results from all threads
        for (auto& tp : threadPrimes) {
            primes.insert(primes.end(), tp.begin(), tp.end());
        }
    }

    // Sort and remove duplicates from the final list of primes
    std::sort(primes.begin(), primes.end());
    primes.erase(std::unique(primes.begin(), primes.end()), primes.end());

    return omp_get_wtime() - startTime; // Return the time taken
}

int main() {
    std::ofstream outFile("results.csv");
    outFile << "M,Openmp Loop Scheduling Method,Chunk Size,T1,T2,T4,T8,S2,S4,S8\n";

    // Warm-up runs to stabilize performance
    for (int i = 0; i < 5; i++) {
        calculatePrimes(1000, 1, 10, "static");
    }

    // Main benchmark loop
    for (int M : Ms) {
        for (const auto& schedMethod : schedMethods) {
            for (int chunkSize : chunkSizes) {
                std::vector<double> speedups;
                double timeSingleThread = calculatePrimes(M, 1, chunkSize, schedMethod);
                outFile << M << "," << schedMethod << "," << chunkSize << "," << timeSingleThread;
                
                // Calculate and record times for different thread counts
                for (int numThreads : threadCounts) {
                    double timeMultipleThreads = calculatePrimes(M, numThreads, chunkSize, schedMethod);
                    outFile << "," << timeMultipleThreads;

                    // Calculate speedup for multiple threads
                    if (numThreads > 1) {
                        speedups.push_back(timeSingleThread / timeMultipleThreads);
                    }
                }

                // Write speedup results to the file
                for(double speedup : speedups)
                    outFile << "," << speedup;

                outFile << "\n";
            }
        }
    }

    outFile.close();
    return 0;
}
