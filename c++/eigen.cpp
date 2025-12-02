#include <iostream>
#include <Eigen/Dense> // Includes Matrix, Array, Vector, and all dense functionality

// Function to print a matrix with a label
void print_matrix(const Eigen::MatrixXd& mat, const std::string& label) {
    std::cout << "\n--- " << label << " ---\n";
    std::cout << mat << "\n";
}

int main() {
    // 1. Matrix Initialization and Basic Properties

    // Declare a 3x3 matrix of doubles (MatrixXd: X for dynamic size, d for double)
    Eigen::MatrixXd A(3, 3); 
    
    // Initialize matrix A with specific values
    A << 1.0, 2.0, 3.0,
         4.0, 5.0, 6.0,
         7.0, 8.0, 9.0;
    
    print_matrix(A, "Matrix A (3x3)");
    
    // Create a 3x1 column vector (VectorXd is an alias for MatrixXd with 1 column)
    Eigen::VectorXd v(3); 
    v << 1.0, 0.5, 2.0;
    
    print_matrix(v, "Vector v (3x1)");

    // Special initialization
    Eigen::MatrixXd I = Eigen::MatrixXd::Identity(3, 3); // 3x3 Identity matrix
    print_matrix(I, "Identity Matrix I");

    // 2. Basic Arithmetic Operations

    // Matrix addition (element-wise)
    Eigen::MatrixXd B = A + I;
    print_matrix(B, "Matrix B (A + I)"); // B is A with 1 added to the diagonal elements

    // Matrix subtraction (element-wise)
    Eigen::MatrixXd C = A - B;
    print_matrix(C, "Matrix C (A - B)"); // C should be the negative of the Identity matrix

    // Scalar multiplication
    Eigen::MatrixXd D = 2.0 * A;
    print_matrix(D, "Matrix D (2 * A)");

    // 3. Core Linear Algebra Operations

    // Matrix Multiplication (A * v = result_vec)
    // The sizes must be compatible (3x3 * 3x1 = 3x1)
    Eigen::VectorXd result_vec = A * v;
    print_matrix(result_vec, "Result of Matrix-Vector Product (A * v)");

    // Matrix-Matrix Multiplication (A * B)
    Eigen::MatrixXd E = A * B;
    print_matrix(E, "Matrix E (A * B)");

    // Transpose (A^T)
    Eigen::MatrixXd A_T = A.transpose();
    print_matrix(A_T, "Transpose of A (A.transpose())");

    // Inverse (A^-1). Note: The example matrix A is singular, so we use B for a proper inverse.
    // The .inverse() method is straightforward but numerically less stable for large systems.
    Eigen::MatrixXd B_inv = B.inverse();
    print_matrix(B_inv, "Inverse of B (B.inverse())");

    // Verification: B * B_inv should be the identity matrix
    Eigen::MatrixXd B_check = B * B_inv;
    print_matrix(B_check, "Check: B * B_inv (Should be Identity)");
    
    // 4. Element-wise Operations (Using Array)

    // For element-wise multiplication, use the array() view and the .* operator
    // This is NOT matrix multiplication.
    Eigen::MatrixXd F = A.array() * B.array();
    print_matrix(F, "Matrix F (Element-wise product A .* B)");

    return 0;
}