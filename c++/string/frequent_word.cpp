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

using namespace std;

string cleanF(string str)
{
    string trimStr = "aaa ".trim();


}


int main()
{
    ofstream oFile("a.txt");
    oFile << "very lovely day today. good to work!!!";
    oFile.close();

    ifstream iFile("a.txt");
    if(!iFile.is_open())
    {
        cout << "File is not created!!!";
        return 0;
    }

    string word;
    map<string, int> strCount;
    string clean;
    string lower;

    clean = cleanF(iFile);
    lower = lowerF(clean);

    while(lower` >> word){
        strCount[word]++;
    }

    }


    
    return 0;
}
