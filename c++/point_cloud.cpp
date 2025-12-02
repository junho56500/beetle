#include <iostream>
#include <vector>
#include <memory>
#include <iomanip>
#include <string>
#include <stdexcept>

using namespace std;

// --- 1. Base Class and Structures (Polymorphic Data Handling) ---
class DataPoint {
public:
    // Mandatory virtual destructor for proper cleanup
    virtual ~DataPoint() = default; 
    
    // Pure virtual function for polymorphic printing
    virtual void print() const = 0; 
};

// --- Point Data (double x, y, z) ---
class PointData : public DataPoint {

public:
    PointData(double x_val, double y_val, double z_val) 
        : x(x_val), y(y_val), z(z_val) {}

    // Implementation of the print method for base coordinates
    void print() const override {
        cout << fixed << setprecision(2) 
                  << "[Point] X: " << x << ", Y: " << y << ", Z: " << z;
    }

private:
    double x, y, z;
};

// --- Intensity Data (Point + int intensity) ---
class IntensityData : public PointData {
public:
    int intensity;

    IntensityData(double x_val, double y_val, double z_val, int intensity_val) 
        : PointData(x_val, y_val, z_val), intensity(intensity_val) {}

    // Implementation of the print method (calls base print and adds intensity)
    void print() const override {
        PointData::print(); // Print X, Y, Z
        cout << ", Intensity: " << intensity;
    }
};

// --- Orientation Data (Point + double yaw, pitch, roll) ---
class OrientationData : public PointData {
public:
    double yaw, pitch, roll;

    OrientationData(double x_val, double y_val, double z_val, double yaw_val, double pitch_val, double roll_val) 
        : PointData(x_val, y_val, z_val), yaw(yaw_val), pitch(pitch_val), roll(roll_val) {}

    // Implementation of the print method (calls base print and adds orientation)
    void print() const override {
        PointData::print(); // Print X, Y, Z
        cout << ", Yaw: " << yaw << ", Pitch: " << pitch << ", Roll: " << roll;
    }
};


// --- 2. Main API Class: DataPointManager ---
class DataPointManager {
private:
    vector<unique_ptr<DataPoint>> collection;

public:
    const vector<unique_ptr<DataPoint>>& get_all_data() const {
        return collection;
    }

    void print_all() const {
        if (collection.empty()) {
            cout << "\n--- Collection is empty. ---" << endl;
            return;
        }

        cout << "\n--- Current Data Collection (" << collection.size() << " items) ---" << endl;
        for (const auto& data : collection) {
            // cout << "[" << i << "] ";
            // Polymorphic call: calls the correct derived class's print()
            data->print();
            cout << endl;
        }
        cout << "---------------------------------------------------" << endl;
    }

    // --- 3. Add Data Functionality (Overloaded for different versions) ---

    /**
     * @brief Adds a new simple PointData object to the collection (x, y, z only).
     */
    size_t add_data(double x, double y, double z) {
        collection.push_back(make_unique<PointData>(x, y, z));
        return collection.size() - 1; // Return last index
    }

    /**
     * @brief 5. Handles data with position (x,y,z) and int intensity.
     */
    size_t add_data(double x, double y, double z, int intensity) {
        collection.push_back(make_unique<IntensityData>(x, y, z, intensity));
        return collection.size() - 1; // Return last index
    }

    /**
     * @brief 6. Handles data with position (x,y,z) and orientation (yaw, pitch, roll).
     */
    size_t add_data(double x, double y, double z, double yaw, double pitch, double roll) {
        collection.push_back(make_unique<OrientationData>(x, y, z, yaw, pitch, roll));
        return collection.size() - 1; // Return last index
    }


    // --- 4. Delete Functionality ---

    /**
     * @brief Deletes a data point at a specific index.
     */
    bool delete_data(size_t index) {
        if (index >= collection.size()) {
            cerr << "Error: Index " << index << " is out of bounds (Size: " << collection.size() << "). Deletion failed." << endl;
            return false;
        }

        // Use vector's erase function to remove the unique_ptr and free memory
        collection.erase(collection.begin() + index);
        cout << "Successfully deleted item at index " << index << "." << endl;
        return true;
    }
};

/**
 * @brief Main function demonstrating the usage of the DataPointManager API.
 */
int main() {
    DataPointManager manager;

    cout << "Starting DataPointManager API Demo (All points now include X, Y, Z)..." << endl;

    // Add Simple Point Data (x, y, z)
    size_t idx0 = manager.add_data(10.5, -20.3, 5.0);
    cout << "Added Simple Point Data at Index: " << idx0 << endl;

    // Add Point + Intensity Data (x, y, z, intensity)
    size_t idx1 = manager.add_data(1.1, 2.2, 3.3, 4096);
    cout << "Added Intensity Data at Index: " << idx1 << endl;

    // Add Point + Orientation Data (x, y, z, yaw, pitch, roll)
    size_t idx2 = manager.add_data(100.0, 50.0, 10.0, 90.1, 45.0, 0.5);
    cout << "Added Orientation Data at Index: " << idx2 << endl;
    
    // Add another Simple Point Data
    size_t idx3 = manager.add_data(1.0, 1.0, 1.0);
    cout << "Added Simple Point Data at Index: " << idx3 << endl;

    // 2. Print all data
    manager.print_all();

    // 4. Delete Data - Delete the element at index 1 (Intensity Data)
    cout << "\nAttempting to delete item at index 1..." << endl;
    if (manager.delete_data(1)) {
        cout << "Deletion successful." << endl;
    } else {
        cout << "Deletion failed." << endl;
    }

    // Print all data after deletion (notice the indexes shift)
    manager.print_all();

    // 1. Get the raw collection and inspect the size
    const auto& raw_data = manager.get_all_data();
    cout << "Verifying collection size via get_all_data(): " << raw_data.size() << endl;
    
    // Example access of element at index 0 (now the original Simple Point)
    if (!raw_data.empty()) {
        cout << "Example access of element at index 0: ";
        raw_data[0]->print();
        cout << endl;
    }

    return 0;
}