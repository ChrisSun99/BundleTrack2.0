#include <iostream>
#include <stdint.h>
#include <typeinfo>
#include <vector>
#include <cmath>
#include "GradientDescent.h"
// #include <Eigen/Geometry>
// #include <qpOASES.hpp>

// Compute the objective function
double objectiveFunction(const Eigen::MatrixXf& A,
                         const Eigen::MatrixXf& B,
                         const Eigen::MatrixXf& N,
                         const Eigen::MatrixXf& M,
                         const Eigen::Matrix3f& R) {
  double loss = 0.0;
  Eigen::MatrixXf term1 = A - R * B;
  // std::cout << "term1 in objectiveFunction min " << term1.minCoeff() << " max " << term1.maxCoeff() << std::endl;
  Eigen::MatrixXf term2 = N - R * M;
  // std::cout << "term2 in objectiveFunction min " << term2.minCoeff() << " max " << term2.maxCoeff() << std::endl;

  // loss = term1.squaredNorm() + lambda * term2.squaredNorm();
  loss = term1.norm() + lambda * term2.norm();
  return loss;
}

// Compute the gradient
Eigen::Matrix3f computeGradient(const Eigen::MatrixXf& A,
                                const Eigen::MatrixXf& B,
                                const Eigen::MatrixXf& N,
                                const Eigen::MatrixXf& M,
                                const Eigen::Matrix3f& R) {
  Eigen::MatrixXf term1 = A - R * B;
  // std::cout << "A min " << A.minCoeff() << " max " << A.maxCoeff() << " mean " << A.mean() << std::endl;
  // std::cout << "B min " << B.minCoeff() << " max " << B.maxCoeff() << " mean " << B.mean() << std::endl;
  // std::cout << "R min " << R.minCoeff() << " max " << R.maxCoeff() << " mean " << R.mean() << std::endl;
  // std::cout << "R * B min " << (R * B).minCoeff() << " max " << (R * B).maxCoeff() << " mean " << (R * B).mean() << std::endl;
  // std::cout << "term1 in computeGradient min " << term1.minCoeff() << " max " << term1.maxCoeff() << std::endl;
  Eigen::MatrixXf term2 = N - R * M;
  // std::cout << "term2 in computeGradient min " << term2.minCoeff() << " max " << term2.maxCoeff() << std::endl;
  Eigen::Matrix3f gradient = -2 * B * term1.transpose() - 2 * lambda * M * term2.transpose();
  // std::cout << "gradient in computeGradient min " << gradient.minCoeff() << " max " << gradient.maxCoeff() << std::endl;

  return gradient;
}

// Gradient descent optimizer
Eigen::Matrix3f optimizeGradientDescent(const Eigen::MatrixXf& A,
                                        const Eigen::MatrixXf& B,
                                        const Eigen::MatrixXf& N,
                                        const Eigen::MatrixXf& M) {
    // initialize the rotation matrix as an identity matrix
    Eigen::Matrix3f R = Eigen::Matrix3f::Identity();
    Eigen::Matrix3f prevGradient = Eigen::Matrix3f::Zero();
    double prevLoss = 0.0;
    int current_lr = learning_rate;

    // Filter out rows with NaN or all-zero values
    std::vector<int> validRows;

    for (int j = 0; j < A.rows(); ++j) {
      // std::cout << "A.row(j) " << A.row(j) << std::endl; 
      bool hasNaN = A.row(j).hasNaN() || B.row(j).hasNaN() || N.row(j).hasNaN() || M.row(j).hasNaN();
      bool allZero = A.row(j).isZero() || B.row(j).isZero() || N.row(j).isZero() || M.row(j).isZero();
      bool invalid1 = (A.row(j).array() < 0.0f).any() || (A.row(j).array() > 1.0f).any() || (B.row(j).array() < 0.0f).any() || (B.row(j).array() > 1.0f).any();
      bool invalid2 = (N.row(j).array() < -M_PI).any() || (N.row(j).array() > M_PI).any() || (M.row(j).array() < -M_PI).any() || (M.row(j).array() > M_PI).any();
      // std::cout << "hasNaN " << hasNaN << " allZero " << allZero << " invalid1 " << invalid1 << " invalid2 " << invalid2 << std::endl; 

      if (!hasNaN && !allZero && !invalid1 && !invalid2) {
        // std::cout << "valid A " << A.row(j) << " invalid1 " << (A.row(j).array() > 1.0f).any() << std::endl; 
        // std::cout << "valid B " << B.row(j) << " invalid1 " << (B.row(j).array() > 1.0f).any() << std::endl; 
        // std::cout << "valid N " << N.row(j) << " invalid2 " << (N.row(j).array() > 1.0f).any() << std::endl; 
        // std::cout << "valid M " << M.row(j) << " invalid2 " << (M.row(j).array() > 1.0f).any() << std::endl; 
        validRows.push_back(j);
      }
    }
    std::cout << "validRows " << validRows.size() << std::endl;
    if (validRows.size() == 0) {
      return R;
    }

    int filteredRowCount = validRows.size();
    Eigen::MatrixXf filteredA(filteredRowCount, A.cols());
    Eigen::MatrixXf filteredB(filteredRowCount, B.cols());
    Eigen::MatrixXf filteredN(filteredRowCount, N.cols());
    Eigen::MatrixXf filteredM(filteredRowCount, M.cols());

    for (int j = 0; j < filteredRowCount; ++j) {
      filteredA.row(j) = A.row(validRows[j]);
      filteredB.row(j) = B.row(validRows[j]);
      filteredN.row(j) = N.row(validRows[j]);
      filteredM.row(j) = M.row(validRows[j]);
    }

    assert(filteredA.array().isNaN().any() == 0);
    assert(filteredB.array().isNaN().any() == 0);
    assert(filteredN.array().isNaN().any() == 0);
    assert(filteredM.array().isNaN().any() == 0);

  for (int i = 0; i < num_iterations; i++) {
    if (i > 0.7 * num_iterations) {
        current_lr *= 0.1;
    }
    // Compute gradient
    Eigen::Matrix3f gradient = computeGradient(filteredA, filteredB, filteredN, filteredM, R);
    // std::cout << "gradient min " << gradient.minCoeff() << " max " << gradient.maxCoeff() << std::endl;

    // Update R with momentum
    Eigen::Matrix3f delta = current_lr * gradient + momentum * prevGradient;
    // std::cout << "delta min " << delta.minCoeff() << " max " << delta.maxCoeff() << std::endl;
    R -= delta;
    prevGradient = delta;

    // Compute current loss
    double currentLoss = objectiveFunction(filteredA, filteredB, filteredN, filteredM, R);
    std::cout << "iter " << i << " currentLoss " << currentLoss << std::endl;

    // Check for convergence
    if (std::abs(currentLoss - prevLoss) < epsilon) {
      break;
    }

    prevLoss = currentLoss;
  }

  // Normalize the matrix by dividing each element by the scaling factor
  float det = R.determinant();
  Eigen::JacobiSVD<Eigen::Matrix3f> svd(R, Eigen::ComputeFullU | Eigen::ComputeFullV);
  Eigen::Vector3f singularValues = svd.singularValues();
  float determinant = singularValues.prod();
  float normalizationFactor = std::cbrt(determinant);
  fprintf(stderr, "Printing the determinant before regularization: %f!!!!!\n", R.determinant());
  fprintf(stderr, "Printing normalizationFactor: %f!!!!!\n", normalizationFactor);
  if (normalizationFactor < 1e-4) {
    return Eigen::Matrix3f::Identity();
  }
  singularValues /= normalizationFactor;
  R = svd.matrixU() * singularValues.asDiagonal() * svd.matrixV().transpose();

  fprintf(stderr, "Printing the determinant after regularization: %f!!!!!\n\n", R.determinant());
  return R;
}

// Eigen::Matrix3f optimizeQuadratic(const Eigen::MatrixXf& A,
//                                   const Eigen::MatrixXf& B,
//                                   const Eigen::MatrixXf& N,
//                                   const Eigen::MatrixXf& M) {
//     // Define the dimensions
//   int numVariables = 9; // 3x3 matrix R has 9 elements
//   int numConstraints = 1; // determinant constraint

//   // Define the quadratic objective matrix
//   Eigen::MatrixXf Q(numVariables, numVariables);
//   Q.setZero();
//   Q.block<3, 3>(0, 0) = A.transpose() * A;
//   Q.block<3, 3>(3, 3) = N.transpose() * N;

//   // Define the linear objective vector
//   Eigen::MatrixXf c(numVariables, 1);
//   c.setZero();

//   // Define the constraint matrix
//   Eigen::MatrixXf Aeq(numConstraints, numVariables);
//   Aeq.setZero();
//   Aeq(0, 0) = 1; // determinant constraint

//   // Define the constraint lower and upper bounds
//   Eigen::MatrixXf beq(numConstraints, 1);
//   beq(0, 0) = 1; // determinant constraint value

//   // Define the lower and upper bounds for variables
//   Eigen::MatrixXf lb(numVariables, 1);
//   lb.setConstant(-std::numeric_limits<float>::infinity()); // no lower bound

//   Eigen::MatrixXf ub(numVariables, 1);
//   ub.setConstant(std::numeric_limits<float>::infinity()); // no upper bound

//   // Solve the quadratic program using qpOASES
//   qpOASES::QProblem problem(numVariables, numConstraints);
//   qpOASES::Options options;
//   options.printLevel = qpOASES::PL_NONE; // Suppress qpOASES output
//   problem.setOptions(options);

//   qpOASES::returnValue status = problem.init(Q.data(), c.data(), Aeq.data(), lb.data(), ub.data(), beq.data(), beq.data());

//   Eigen::MatrixXf solution(numVariables, 1);
//   problem.getPrimalSolution(solution.data(), numVariables);

//   // Extract the solution as a 3x3 matrix
//   Eigen::Matrix3f R;
//   R << solution(0), solution(1), solution(2),
//        solution(3), solution(4), solution(5),
//        solution(6), solution(7), solution(8);
//   return R;
// }