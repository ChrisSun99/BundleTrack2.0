#include "Eigen/Dense";

class EKF
{
    public:
        // State
        Eigen::Vector3f x_;
        // State covariance matrix 
        Eigen::MatrixXf P_;
        // State transition matrix
        Eigen::MatrixXf F_;
        // Process covariance matrix
        Eigen::MatrixXf Q_;
        // Measurement matrix
        Eigen::MatrixXf H_;
        // Measurement covariance matrix
        Eigen::MatrixXf R_;
        // Observation in previous timestamp
        Eigen::Vector3f prev_z_;
        

        EKF();
        virtual ~EKF();
        void init(Eigen::VectorXf &x_in, Eigen::MatrixXf &P_in, Eigen::MatrixXf &F_in,
            Eigen::MatrixXf &H_in, Eigen::MatrixXf &R_in, Eigen::MatrixXf &Q_in);
        void predict();
        void update(const Eigen::Vector3f &z);

    private:
        
};