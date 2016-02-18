//Imports from C++
#include <iostream>
#include <random>

//Eigen imports
#include <Eigen/Dense>
#include <Eigen/QR>
#include <Eigen/SVD>

using Eigen::MatrixXd;

int get_random_orthonormal_basis(const MatrixXd & A, const int k, const q)
{
    int n = A.rows(); 
    int m = A.cols();
    int max_rank = std::min(n,m);

    if (k > max_rank)
    {
        std::cout << "WARNING: The value of k must not exceed the number of columns or rows of A." << std::endl;
    }

    MatrixXd Omega = get_normal_random_matrix(n, k);
    MatrixXd Y = A * Omega;
    
}

MatrixXd get_normal_random_matrix(const n, const m)
{
    //Random number generation stuff
    std::random_device rd;
    std::mt19937 e2(rd());
    std::normal_distribution<> dist(0,1);

    //Random matrix to be initialized
    MatrixXd A(n, m);

    //Populate entries of random matrix
    for (int i = 0; i < n; ++i)
    {
        for (int j = 0; j < m; ++j)
        {
            A(i, j) = dist(e2);
        }
    }

    return A;
}
