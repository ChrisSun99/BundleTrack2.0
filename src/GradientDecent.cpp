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
    double current_lr = learning_rate;

    assert(A.array().isNaN().any() == 0);
    assert(B.array().isNaN().any() == 0);
    assert(N.array().isNaN().any() == 0);
    assert(M.array().isNaN().any() == 0);

  Eigen::Matrix3f velocity;
  bool lr_scheduler = true;
  for (int i = 0; i < num_iterations; i++) {
    if (i > (0.7*num_iterations) && lr_scheduler) {
        current_lr *= 0.1;
        lr_scheduler = false;
    }

    // Compute gradient
    Eigen::Matrix3f gradient = computeGradient(A, B, N, M, R);
    // std::cout << "gradient min " << gradient.minCoeff() << " max " << gradient.maxCoeff() << std::endl;

    // Update R with momentum
    if (i == 0) {
      velocity = gradient;
    } else {
      velocity = momentum * velocity + gradient;  // It was the previous velocity on RHS from last iter
    }
    // std::cout << "velocity min " << velocity.minCoeff() << " max " << velocity.maxCoeff() << std::endl;
    gradient = velocity;
    R = R - current_lr * gradient;
    // std::cout << "current R " << R << std::endl;

    // Compute current loss
    double currentLoss = objectiveFunction(A, B, N, M, R);
    if (i % 10 == 0) {
      std::cout << "iter " << i << " lr " << current_lr << " currentLoss " << currentLoss << std::endl;
    }

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