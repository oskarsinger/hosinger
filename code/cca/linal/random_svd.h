#ifndef RANDOM_SVD_H_
#define RANDOM_SVD_H_

// Imports from external projects
#include <Eigen/Dense>

namespace linal {
namespace random {

using namespace Eigen;

class RandomSvd
{

 public:
  RandomSvd();
  std::vector<MatrixXd> GetRandomSvd(MatrixXd A);
  std::vector<MatrixXd> GetRandomSvd(MatrixXd A, const int k);
  std::vector<MatrixXd> GetRandomSvd(MatrixXd A, const int k, const int q);
};

} // namespace random
} // namespace linal

#endif // RANDOM_SVD_H_
