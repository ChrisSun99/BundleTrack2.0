#include <random>
#include <iostream>
#include <stdint.h>
#include <typeinfo>
#include <vector>
#include "GradientDescent.h"


// Generate a random rotation matrix
Eigen::Matrix3f generateRandomRotationMatrix() {
  Eigen::Matrix3f R;
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<float> dis(-1.0, 1.0);

  for (int i = 0; i < 9; ++i) {
    R(i) = dis(gen);
  }

  return R;
}

// Compute the objective function
double objectiveFunction(const Eigen::MatrixXf& A,
                         const Eigen::MatrixXf& B,
                         const Eigen::MatrixXf& N,
                         const Eigen::MatrixXf& M,
                         const Eigen::Matrix3f& R) {
  double loss = 0.0;
  Eigen::MatrixXf term1 = A - R * B;
  Eigen::MatrixXf term2 = N - R * M;

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
  Eigen::MatrixXf term2 = N - R * M;
  Eigen::Matrix3f gradient = -2 * B * term1.transpose() - 2 * lambda * M * term2.transpose();

  return gradient;
}

// Gradient descent optimizer
Eigen::Matrix3f optimizeGradientDescent(const Eigen::MatrixXf& A,
                                        const Eigen::MatrixXf& B,
                                        const Eigen::MatrixXf& N,
                                        const Eigen::MatrixXf& M) {
  Eigen::Matrix3f R = generateRandomRotationMatrix();
  Eigen::Matrix3f prevGradient = Eigen::Matrix3f::Zero();
  double prevLoss = 0.0;

  for (int i = 0; i < num_iterations; i++) {
    // Filter out rows with NaN or all-zero values
    std::vector<int> validRows;

    for (int j = 0; j < A.rows(); ++j) {
      bool hasNaN = A.row(j).hasNaN() || B.row(j).hasNaN() || N.row(j).hasNaN() || M.row(j).hasNaN();
      bool allZero = A.row(j).isZero() && B.row(j).isZero() && N.row(j).isZero() && M.row(j).isZero();

      if (!hasNaN && !allZero) {
        validRows.push_back(j);
      }
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

    // Update R with momentum
    Eigen::Matrix3f delta = learning_rate * gradient + momentum * prevGradient;
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