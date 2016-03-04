#ifndef RANDOM_ORTHONORMAL_BASIS_H_
#define RANDOM_ORTHONORMAL_BASIS_H_

// Imports from external projects
#include <Eigen/Dense>

namespace linal {
namespace random {

using namespace Eigen;

class RandomOrthonormalBasis
{
 public:
  RandomOrthonormalBasis() {}
  ~RandomOrthonormalBasis() {}
  MatrixXd GetEpsilonBasis(MatrixXd A, const double epsilon);
  MatrixXd GetFullRankBasis(MatrixXd A);
  MatrixXd GetRankKBasis(MatrixXd A, const int k);
  MatrixXd GetRankKBasis(MatrixXd A, const int k, const int q);
};

} // namespace random
} // namespace linal

#endif // RANDOM_ORTHONORMAL_BASIS_H_
