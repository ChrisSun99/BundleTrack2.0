#include <random>
#include <iostream>
#include <stdint.h>
#include <typeinfo>
#include <vector>
#include "GradientDescent.h"


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

  loss = term1.squaredNorm() + lambda * term2.squaredNorm();

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

  for (int i = 0; i < num_iterations; i++) {
    // Filter out rows with NaN or all-zero values
    std::vector<int> validRows;

    for (int j = 0; j < A.rows(); ++j) {
      // std::cout << "A.row(j) " << A.row(j) << std::endl; 
      bool hasNaN = A.row(j).hasNaN() || B.row(j).hasNaN() || N.row(j).hasNaN() || M.row(j).hasNaN();
      bool allZero = A.row(j).isZero() || B.row(j).isZero() || N.row(j).isZero() || M.row(j).isZero();
      bool invalid1 = (A.row(j).array() < 0.0f).any() || (A.row(j).array() > 1.0f).any() || (B.row(j).array() < 0.0f).any() || (B.row(j).array() > 1.0f).any();
      bool invalid2 = (N.row(j).array() < -1.0f).any() || (N.row(j).array() > 1.0f).any() || (M.row(j).array() < -1.0f).any() || (M.row(j).array() > 1.0f).any();
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

    // Compute gradient
    Eigen::Matrix3f gradient = computeGradient(filteredA, filteredB, filteredN, filteredM, R);
    // std::cout << "gradient min " << gradient.minCoeff() << " max " << gradient.maxCoeff() << std::endl;

    // Update R with momentum
    Eigen::Matrix3f delta = learning_rate * gradient + momentum * prevGradient;
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
  float scalingFactor = std::pow(det, 1.0f / 3.0f);
  R /= scalingFactor;

  return R;
}