//Imports from C++
#include <iostream>
#include <random>
#include <time.h>

//Eigen imports
#include <Eigen/Dense>
#include <Eigen/QR>
#include <Eigen/SVD>

using namespace Eigen;
using namespace std;

MatrixXd get_normal_random_matrix(const int m, const int n)
{
    //Random number generation stuff
    std::random_device rd;
    std::mt19937 e2(rd());
    std::normal_distribution<> dist(0,1);

    //Random matrix to be initialized
    MatrixXd A(m, n);

    //Populate entries of random matrix
    for (int i = 0; i < m; ++i)
    {
        for (int j = 0; j < n; ++j)
        {
            A(i, j) = dist(e2);
        }
    }

    return A;
}

MatrixXd get_rank_k_matrix(const int m, const int n, const int k)
{
    int max_rank = std::min(m,n);

    if (k > max_rank)
    {
        std::cout << "WARNING: The value of k must not exceed the number of columns or rows." << std::endl;
    }

    MatrixXd A = MatrixXd::Zero(m,n);

    for (int i = 0; i < k; ++i)
    {
        A += MatrixXd::Random(m,1) * MatrixXd::Random(1,n);
    }

    return A;
}

MatrixXd get_random_orthonormal_basis(MatrixXd A, const int k, const int q)
{
    int m = A.rows(); 
    int n = A.cols();
    int max_rank = std::min(m,n);

    if (k > max_rank)
    {
        std::cout << "WARNING: The value of k must not exceed the number of columns or rows of A." << std::endl;
    }

    time_t before, after;

    time(&before);
    MatrixXd Y = A * get_normal_random_matrix(n, k);
    time(&after);
    std::cout << "Acquired random matrix" << difftime(after, before) << std::endl;

    time(&before);
    MatrixXd Q = Eigen::HouseholderQR<MatrixXd>(Y).householderQ();
    time(&after);
    std::cout << "Got first QR" << difftime(after, before) << std::endl;

    time(&before);
    for (int i = 0; i < q; i++)
    {
        Y = A.transpose() * Q;
        Q = Eigen::HouseholderQR<MatrixXd>(Y).householderQ();
        Y = A * Q;
        Q = Eigen::HouseholderQR<MatrixXd>(Y).householderQ();
    }
    time(&after);
    std::cout << "Completed power iteration" << difftime(after, before) << std::endl;

    return Q;
}

std::vector<MatrixXd> get_random_svd(MatrixXd A, const int k, const int q)
{
    int n = A.rows();
    int m = A.cols();
    int max_rank = std::min(n,m);

    if (k > max_rank)
    {
        std::cout << "WARNING: The value of k must not exceed the number of columns or rows of A." << std::endl;
    }

    MatrixXd Q = get_random_orthonormal_basis(A, k, q);

    time_t before, after;

    time(&before);
    Eigen::JacobiSVD<MatrixXd> svd(Q.transpose() * A, ComputeFullU | ComputeFullV);
    time(&after);
    std::cout << "Completed SVD" << difftime(after, before) << std::endl;

    MatrixXd U = Q * svd.matrixU();
    std::vector<MatrixXd> UsV = {U, svd.singularValues(), svd.matrixV()};

    return UsV;
}

int main(){
    MatrixXd m = get_rank_k_matrix(500,500,20);

    time_t before, after;

    time(&before);
    std::vector<MatrixXd> UsV = get_random_svd(m, 20, 1);
    time(&after);

    std::cout << difftime(after, before) << std::endl;

    time(&before);
    Eigen::JacobiSVD<MatrixXd> svd(m, ComputeFullU | ComputeFullV);
    time(&after);

    std::cout << difftime(after, before) << std::endl;

    MatrixXd m_hat = UsV[0] * UsV[1].asDiagonal() * UsV[2].adjoint();

    double cum_error = 0;

    for (int i = 0; i < m_hat.rows(); ++ i) {
        for (int j = 0; j < m_hat.cols(); ++j) {
            //std::cout << m_hat(i,j) << " " << m(i,j) << std::endl;
            cum_error += abs(m_hat(i,j) - m(i,j));
        }
    }

    std::cout << cum_error << std::endl;
}
