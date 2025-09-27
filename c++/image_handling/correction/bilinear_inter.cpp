#include <iostream>
#include <vector>

// Helper function to get a pixel value, with bounds checking
// This assumes an image is represented as a 2D vector
double getPixelValue(const std::vector<std::vector<double>>& image, int x, int y) {
    if (x >= 0 && x < image[0].size() && y >= 0 && y < image.size()) {
        return image[y][x];
    }
    return 0.0; // Return a default value for out-of-bounds access
}

// Function to perform bilinear interpolation
double bilinearInterpolation(const std::vector<std::vector<double>>& image, double x, double y) {
    // Get the coordinates of the four surrounding pixels
    int x1 = static_cast<int>(x);
    int y1 = static_cast<int>(y);
    int x2 = x1 + 1;
    int y2 = y1 + 1;

    // Get the pixel values at the four corners
    double q11 = getPixelValue(image, x1, y1);
    double q12 = getPixelValue(image, x1, y2);
    double q21 = getPixelValue(image, x2, y1);
    double q22 = getPixelValue(image, x2, y2);

    // Calculate the interpolation weights
    double x_diff = x - x1;
    double y_diff = y - y1;

    // Perform linear interpolation in the x-direction first
    double r1 = q11 * (1 - x_diff) + q21 * x_diff;
    double r2 = q12 * (1 - x_diff) + q22 * x_diff;

    // Perform linear interpolation in the y-direction
    double p = r1 * (1 - y_diff) + r2 * y_diff;
    
    return p;
}

int main() {
    // Example usage: create a small 2x2 image
    std::vector<std::vector<double>> originalImage = {
        {100, 200},
        {150, 250}
    };

    // Find the interpolated value at a non-integer coordinate (0.5, 0.5)
    double interpolatedValue = bilinearInterpolation(originalImage, 0.5, 0.5);

    std::cout << "Original Image:" << std::endl;
    std::cout << originalImage[0][0] << " " << originalImage[0][1] << std::endl;
    std::cout << originalImage[1][0] << " " << originalImage[1][1] << std::endl;
    std::cout << std::endl;

    std::cout << "Interpolated value at (0.5, 0.5): " << interpolatedValue << std::endl;
    // Expected output: (100 * 0.5 + 200 * 0.5) * 0.5 + (150 * 0.5 + 250 * 0.5) * 0.5 = 175
    // Another way to think about it: the average of all four corners.
    
    return 0;
}