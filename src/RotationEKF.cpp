#include "RotationEKF.h"
#include "Eigen/Dense"
#include <iostream>
#include <cmath>

RotationEKF::RotationEKF(Eigen::Vector3f euler_angles, Eigen::Vector3f normals, Eigen::Matrix3f rotation, Eigen::Vector3f z0)
{
    ekf_.x_ = euler_angles;
    rotation_ = rotation;
    z0_ = z0;
    is_initialized_ = false;
    R_ = Eigen::MatrixXd(2, 2);
    H_jacobian = Eigen::MatrixXd(3, 4);
    
    R_ << 0.0225, 0,
          0, 0.0225;
}

RotationEKF::~RotationEKF() {}

void RotationEKF::processMeasurement(Eigen::Vector3f normals)
{
    ekf_.predict();
    ekf_.H_ = getH();
    ekf_.R_ = R_;
    ekf_.update(normals);
}

Eigen::Matrix3f RotationEKF::getH()
{
    Eigen::Matrix3f jacobian;

    // Compute the individual rotation matrices around each axis
    Eigen::Matrix3f Rz, Ry, Rx;
    float phi = ekf_.x_(0);
    float theta = ekf_.x_(1);
    float psi = ekf_.x_(2);

    jacobian = getF();
    return rotation_ * jacobian * rotation_.transpose() * z0_;
}

Eigen::Matrix3f RotationEKF::getF()
{
    Eigen::Matrix3f jacobian;
    Eigen::Matrix3f Rz, Ry, Rx;
    float phi = ekf_.x_(0);
    float theta = ekf_.x_(1);
    float psi = ekf_.x_(2);

    Rz << cos(psi), -sin(psi), 0,
          sin(psi), cos(psi), 0,
          0, 0, 1;

    Ry << cos(theta), 0, sin(theta),
          0, 1, 0,
          -sin(theta), 0, cos(theta);

    Rx << 1, 0, 0,
          0, cos(phi), -sin(phi),
          0, sin(phi), cos(phi);

    jacobian.row(0) = Rz * Ry * Rx * Eigen::Vector3f::UnitX();
    jacobian.row(1) = Rz * Ry * Rx * Eigen::Vector3f::UnitY();
    jacobian.row(2) = Rz * Ry * Rx * Eigen::Vector3f::UnitZ();
    return jacobian;
}