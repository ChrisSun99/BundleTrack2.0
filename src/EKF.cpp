#include "EKF.h"
#include <math.h>

EKF::EKF(){};
EKF::~EKF(){};

void EKF::init(Eigen::VectorXf &x_in, Eigen::MatrixXf &P_in, Eigen::MatrixXf &F_in,
            Eigen::MatrixXf &H_in, Eigen::MatrixXf &R_in, Eigen::MatrixXf &Q_in)
{
    x_ = x_in; 
    P_ = P_in;
    F_ = F_in;
    H_ = H_in;
    R_ = R_in;
    Q_ = Q_in;
}
void EKF::predict()
{
    x_ = F_ * x_;
    Eigen::MatrixXf Ft = F_.transpose();
    P_ = F_ * P_ * Ft + Q_;
}

void EKF::update(const Eigen::Vector3f &z)
{
    Eigen::Vector3d y = z - x_;
    Eigen::MatrixXf Ht = H_.transpose();
    Eigen::MatrixXf PHt = P_ * Ht;
    Eigen::MatrixXf S = H_ * PHt + R_;
    Eigen::MatrixXf Si = S.inverse();
    Eigen::MatrixXf K = PHt * Si;

    //new estimate
    x_ = x_ + (K * y);
    long x_size = x_.size();
    Eigen::MatrixXf I = Eigen::MatrixXd::Identity(x_size, x_size);
    P_ = (I - K * H_) * P_;
}