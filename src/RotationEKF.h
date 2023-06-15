#ifndef ROTATIONEKF_H_
#define ROTATIONEKF_H_

// #include "Eigen/Dense"
// #include <vector>
// #include <string>
// #include <fstream>
// #include "EKF.h"

// class RotationEKF
// {
//     public:
//     RotationEKF(Eigen::Vector3f euler_angles, Eigen::Vector3f normals, Eigen::Matrix3f rotation, Eigen::Vector3f z0);
//     virtual ~RotationEKF();
//     void processMeasurement(Eigen::Vector3f normals);
//     Eigen::Matrix3f getH();
//     Eigen::Matrix3f getF();
//     EKF ekf_;

//     private:
//     bool is_initialized_;
//     Eigen::Matrix3f rotation_;
//     Eigen::Vector3f z0_; 
//     Eigen::MatrixXd R_;                 // measurement noise
//     Eigen::MatrixXd H_jacobian;         // measurement function for radar
    
// }

#endif