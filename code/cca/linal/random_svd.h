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
  // Constructor
  RandomSvd() {}
  ~RandomSvd() {}

  // Methods
  std::vector<MatrixXd> GetRandomSvd(const MatrixXd &A) const;
  std::vector<MatrixXd> GetRandomSvd(const MatrixXd &A, const int k) const;
  std::vector<MatrixXd> GetRandomSvd(const MatrixXd &A, const int k, const int q) const;
};

} // namespace random
} // namespace linal

#endif // RANDOM_SVD_H_
