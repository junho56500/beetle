#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <map>
#include <algorithm>
#include <cctype> // For std::tolower

// --- PROBLEM DESCRIPTION ---
// Write a C++ program that reads a text file, counts the frequency of each word,
// and prints the top N most frequent words. The program should be case-insensitive.
// Punctuation should be ignored. For simplicity, we will consider words as
// sequences of alphabetic characters.
//
// Input:
// 1. A text file named "input.txt".
// 2. An integer N, representing the number of top words to display.
//
// Output:
// A list of the top N words and their counts, sorted in descending order of frequency.
//
// Example "input.txt":
// Hello world. This is a simple test. Hello again, world.

// Function to convert a string to lowercase
std::string toLower(const std::string& str) {
    std::string lower_str = str;
    std::transform(lower_str.begin(), lower_str.end(), lower_str.begin(),
                   [](unsigned char c){ return std::tolower(c); });
    return lower_str;
}

// Function to clean a word (remove non-alphabetic characters)
std::string cleanWord(const std::string& str) {
    std::string cleaned_str;
    for (char c : str) {
        if (std::isalpha(c)) {
            cleaned_str += c;
        }
    }
    return cleaned_str;
}

// Function to count word frequencies in a file
std::map<std::string, int> countWords(const std::string& filename) {
    std::map<std::string, int> word_counts;
    std::ifstream file(filename);

    if (!file.is_open()) {
        std::cerr << "Error: Could not open file " << filename << std::endl;
        return word_counts;
    }

    std::string word;
    while (file >> word) {
        // Clean and normalize the word before counting
        std::string cleaned = cleanWord(word);
        std::string lower_case_word = toLower(cleaned);
        
        // Only count the word if it's not empty after cleaning
        if (!lower_case_word.empty()) {
            word_counts[lower_case_word]++;
        }
    }

    file.close();
    return word_counts;
}

// Function to find the top N words
std::vector<std::pair<std::string, int>> getTopNWords(const std::map<std::string, int>& counts, int n) {
    // Create a vector of pairs to store words and their counts
    std::vector<std::pair<std::string, int>> top_words;
    for (const auto& pair : counts) {
        top_words.push_back(pair);
    }

    // Sort the vector by frequency in descending order
    std::sort(top_words.begin(), top_words.end(), [](const auto& a, const auto& b) {
        return a.second > b.second;
    });

    // Resize the vector to contain only the top N elements
    if (top_words.size() > n) {
        top_words.resize(n);
    }

    return top_words;
}

int main() {
    // We'll create a dummy file for the demonstration
    std::ofstream dummy_file("input.txt");
    dummy_file << "C++ is a powerful language. A a A. a.";
    dummy_file.close();

    const std::string filename = "input.txt";
    const int top_n = 3;

    // Count the words
    std::map<std::string, int> word_counts = countWords(filename);

    if (word_counts.empty()) {
        std::cout << "No words found or file error." << std::endl;
        return 1;
    }

    // Get the top N words
    std::vector<std::pair<std::string, int>> top_words = getTopNWords(word_counts, top_n);

    // Print the results
    std::cout << "Top " << top_n << " most frequent words:" << std::endl;
    for (const auto& pair : top_words) {
        std::cout << "\"" << pair.first << "\" : " << pair.second << std::endl;
    }

    return 0;
}
