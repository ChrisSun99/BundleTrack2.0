#ifndef GRADIENTDESCENT_H_
#define GRADIENTDESCENT_H_

#include "FeatureManager.h"

const double lambda = 0;  // Regularization parameter
const double learning_rate = 0.1;  // Learning rate
const double momentum = 0.9;  // Momentum
const double epsilon = 1e-10;  // Stop criterion threshold
const int num_iterations = 100;

Eigen::Matrix3f generateRandomRotationMatrix();
double objectiveFunction(const Eigen::MatrixXf& A,
                         const Eigen::MatrixXf& B,
                         const Eigen::MatrixXf& N,
                         const Eigen::MatrixXf& M,
                         const Eigen::Matrix3f& R);
Eigen::Matrix3f computeGradient(const Eigen::MatrixXf& A,
                                const Eigen::MatrixXf& B,
                                const Eigen::MatrixXf& N,
                                const Eigen::MatrixXf& M,
                                const Eigen::Matrix3f& R);
Eigen::Matrix3f optimizeGradientDescent(const Eigen::MatrixXf& A,
                                        const Eigen::MatrixXf& B,
                                        const Eigen::MatrixXf& N,
                                        const Eigen::MatrixXf& M);
Eigen::Matrix3f optimizeQuadratic(const Eigen::MatrixXf& A,
                                  const Eigen::MatrixXf& B,
                                  const Eigen::MatrixXf& N,
                                  const Eigen::MatrixXf& M);                                       

#endif
