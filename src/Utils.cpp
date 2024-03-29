/*
Authors: Bowen Wen
Contact: wenbowenxjtu@gmail.com
Created in 2021

Copyright (c) Rutgers University, 2021 All rights reserved.

Bowen Wen and Kostas Bekris. "BundleTrack: 6D Pose Tracking for Novel Objects
 without Instance or Category-Level 3D Models."
 In IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS). 2021.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:
    * Redistributions of source code must retain the above copyright
      notice, this list of conditions and the following disclaimer.
    * Redistributions in binary form must reproduce the above copyright
      notice, this list of conditions and the following disclaimer in the
      documentation and/or other materials provided with the distribution.
    * Neither the name of the Bowen Wen, Kostas Bekris, Rutgers University,
      nor the names of its contributors may be used to
      endorse or promote products derived from this software without
      specific prior written permission.
THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS 'AS IS' AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER BE LIABLE FOR ANY
DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

#include "Utils.h"


namespace Utils
{


float rotationGeodesicDistance(const Eigen::Matrix3f &R1, const Eigen::Matrix3f &R2)
{
  float tmp = ((R1 * R2.transpose()).trace()-1) / 2.0;
  tmp = std::max(std::min(1.0f, tmp), -1.0f);
  return std::acos(tmp);
}


void readDepthImage(cv::Mat &depthImg, std::string path)
{
  cv::Mat depthImgRaw = cv::imread(path, CV_16UC1);
  depthImg = cv::Mat::zeros(depthImgRaw.rows, depthImgRaw.cols, CV_32FC1);
  for (int u = 0; u < depthImgRaw.rows; u++)
    for (int v = 0; v < depthImgRaw.cols; v++)
    {
      unsigned short depthShort = depthImgRaw.at<unsigned short>(u, v);
      float depth = (float)depthShort * 0.001;
      if (depth<0.1)
      {
        depthImg.at<float>(u, v) = 0.0;
      }
      else
      {
        depthImg.at<float>(u, v) = depth;
      }

    }
}

template<class PointT>
void convert3dOrganizedRGB(cv::Mat &objDepth, cv::Mat &colImage, Eigen::Matrix3f &camIntrinsic, boost::shared_ptr<pcl::PointCloud<PointT>> objCloud)
{
  const int imgWidth = objDepth.cols;
  const int imgHeight = objDepth.rows;

  objCloud->height = (uint32_t)imgHeight;
  objCloud->width = (uint32_t)imgWidth;
  objCloud->is_dense = false;
  objCloud->points.resize(objCloud->width * objCloud->height);

  const float bad_point = 0;

  for (int u = 0; u < imgHeight; u++)
    for (int v = 0; v < imgWidth; v++)
    {
      float depth = objDepth.at<float>(u, v);
      cv::Vec3b colour = colImage.at<cv::Vec3b>(u, v);
      if (depth > 0.1 && depth < 2.0)
      {
        (*objCloud)(v, u).x = (float)((v - camIntrinsic(0, 2)) * depth / camIntrinsic(0, 0));
        (*objCloud)(v, u).y = (float)((u - camIntrinsic(1, 2)) * depth / camIntrinsic(1, 1));
        (*objCloud)(v, u).z = depth;
        (*objCloud)(v, u).b = colour[0];
        (*objCloud)(v, u).g = colour[1];
        (*objCloud)(v, u).r = colour[2];
      }
      else
      {
        (*objCloud)(v, u).x = bad_point;
        (*objCloud)(v, u).y = bad_point;
        (*objCloud)(v, u).z = bad_point;
        (*objCloud)(v, u).b = 0;
        (*objCloud)(v, u).g = 0;
        (*objCloud)(v, u).r = 0;
      }
    }
}
template void convert3dOrganizedRGB(cv::Mat &objDepth, cv::Mat &colImage, Eigen::Matrix3f &camIntrinsic, boost::shared_ptr<pcl::PointCloud<pcl::PointXYZRGB>> objCloud);
template void convert3dOrganizedRGB(cv::Mat &objDepth, cv::Mat &colImage, Eigen::Matrix3f &camIntrinsic, boost::shared_ptr<pcl::PointCloud<pcl::PointXYZRGBNormal>> objCloud);


void readDirectory(const std::string& name, std::vector<std::string>& v)
{
  v.clear();
  DIR *dirp = opendir(name.c_str());
  if (dirp==NULL)
  {
    printf("Reading directory failed: %s\n",name.c_str());
  }
  struct dirent *dp;
  while ((dp = readdir(dirp)) != NULL)
  {
    if (std::string(dp->d_name) == "." || std::string(dp->d_name) == "..")
      continue;
    v.push_back(dp->d_name);
  }
  closedir(dirp);
  std::sort(v.begin(),v.end());
}


template<class PointT>
void downsamplePointCloud(boost::shared_ptr<pcl::PointCloud<PointT> > cloud_in, boost::shared_ptr<pcl::PointCloud<PointT> > cloud_out, float vox_size)
{
  pcl::VoxelGrid<PointT> vox;
  vox.setInputCloud(cloud_in);
  vox.setLeafSize(vox_size, vox_size, vox_size);
  vox.filter(*cloud_out);
}
template void downsamplePointCloud<pcl::PointXYZRGBNormal>(boost::shared_ptr<pcl::PointCloud<pcl::PointXYZRGBNormal> > cloud_in, boost::shared_ptr<pcl::PointCloud<pcl::PointXYZRGBNormal> > cloud_out, float vox_size);
template void downsamplePointCloud<pcl::PointXYZRGB>(boost::shared_ptr<pcl::PointCloud<pcl::PointXYZRGB> > cloud_in, boost::shared_ptr<pcl::PointCloud<pcl::PointXYZRGB> > cloud_out, float vox_size);
template void downsamplePointCloud<pcl::PointXYZ>(boost::shared_ptr<pcl::PointCloud<pcl::PointXYZ> > cloud_in, boost::shared_ptr<pcl::PointCloud<pcl::PointXYZ> > cloud_out, float vox_size);
template void downsamplePointCloud<pcl::PointNormal>(boost::shared_ptr<pcl::PointCloud<pcl::PointNormal> > cloud_in, boost::shared_ptr<pcl::PointCloud<pcl::PointNormal> > cloud_out, float vox_size);
template void downsamplePointCloud<pcl::PointSurfel>(boost::shared_ptr<pcl::PointCloud<pcl::PointSurfel> > cloud_in, boost::shared_ptr<pcl::PointCloud<pcl::PointSurfel> > cloud_out, float vox_size);



void parsePoseTxt(std::string filename, Eigen::Matrix4f &out)
{
  parseMatrixTxt<4,4>(filename, out);
}

void normalizeRotationMatrix(Eigen::Matrix3f &R)
{
  for (int col=0;col<3;col++)
  {
    R.col(col).normalize();
  }
}

void normalizeRotationMatrix(Eigen::Matrix4f &pose)
{
  for (int col=0;col<3;col++)
  {
    pose.block(0,col,3,1).normalize();
  }
}


bool isPixelInsideImage(const int H, const int W, float u, float v)
{
  u = std::round(u);
  v = std::round(v);
  if (u<0 || u>=W || v<0 || v>=H) return false;
  return true;
}


void solveRigidTransformBetweenPoints(const Eigen::MatrixXf &points1, const Eigen::MatrixXf &points2, Eigen::Matrix4f &pose)
{
  assert(points1.cols()==3 && points1.rows()>=3 && points2.cols()==3 && points2.rows()>=3);
  pose.setIdentity();

  Eigen::Vector3f mean1 = points1.colwise().mean();
  Eigen::Vector3f mean2 = points2.colwise().mean();

  Eigen::MatrixXf P = points1.rowwise() - mean1.transpose();
  Eigen::MatrixXf Q = points2.rowwise() - mean2.transpose();
  Eigen::MatrixXf S = P.transpose() * Q;
  assert(S.rows()==3 && S.cols()==3);
  Eigen::JacobiSVD<Eigen::MatrixXf> svd(S, Eigen::ComputeThinU | Eigen::ComputeThinV);
  Eigen::Matrix3f R = svd.matrixV() * svd.matrixU().transpose();
  if ( !((R.transpose()*R).isApprox(Eigen::Matrix3f::Identity())) )
  {
    pose.setIdentity();
    return;
  }

  if (R.determinant()<0)
  {
    auto V_new = svd.matrixV();
    V_new.col(2) = (-V_new.col(2)).eval();
    R = V_new * svd.matrixU().transpose();
  }
  pose.block(0,0,3,3) = R;
  pose.block(0,3,3,1) = mean2 - R * mean1;
  if (!isMatrixFinite(pose))
  {
    pose.setIdentity();
    return;
  }

  // assert(points1.cols() == 3 && points1.rows() >= 3 && points2.cols() == 3 && points2.rows() >= 3);
  // pose.setIdentity();

  // Eigen::Vector3f mean1 = points1.colwise().mean();
  // Eigen::Vector3f mean2 = points2.colwise().mean();

  // Eigen::MatrixXf P = points1.rowwise() - mean1.transpose();
  // Eigen::MatrixXf Q = points2.rowwise() - mean2.transpose();
  // Eigen::MatrixXf S = P.transpose() * Q;

  // Eigen::JacobiSVD<Eigen::MatrixXf> svd(S, Eigen::ComputeThinU | Eigen::ComputeThinV);
  // Eigen::Matrix3f R = svd.matrixV() * svd.matrixU().transpose();

  // Eigen::Quaternionf quat(R);  // Convert the rotation matrix to a quaternion

  // pose.block(0, 0, 3, 3) = quat.toRotationMatrix();
  // pose.block(0, 3, 3, 1) = mean2 - quat.toRotationMatrix() * mean1;
}

void drawProjectPoints(PointCloudRGBNormal::Ptr cloud, const Eigen::Matrix3f &K, cv::Mat &out)
{
  Utils::downsamplePointCloud(cloud,cloud,0.01);
  for (const auto &pt:cloud->points)
  {
    int u = std::round(pt.x*K(0,0)/pt.z + K(0,2));
    int v = std::round(pt.y*K(1,1)/pt.z + K(1,2));
    if (u<0 || u>=out.cols || v<0 || v>=out.rows) continue;
    cv::circle(out, {u,v}, 1, {0,255,255}, -1);
  }
}

// Spherical linear interpolation
// https://www.euclideanspace.com/maths/algebra/realNormedAlgebra/quaternions/slerp/index.htm
Eigen::Matrix3f slerp(Eigen::Matrix4f &pose1, Eigen::Matrix4f &pose2, float t)
{
  fprintf(stderr, "REACHED slerp1\n");
  Eigen::Quaternionf qa(pose1.topLeftCorner<3, 3>());
  fprintf(stderr, "REACHED slerp2\n");
  Eigen::Quaternionf qb(pose2.topLeftCorner<3, 3>());
  fprintf(stderr, "REACHED slerp3\n");
	Eigen::Quaternionf qm;
  Eigen::Matrix3f mat;
	// Calculate angle between them.
	double cosHalfTheta = qa.w() * qb.w() + qa.x() * qb.x() + qa.y() * qb.y() + qa.z() * qb.z();
	// if qa=qb or qa=-qb then theta = 0 and we can return qa
  fprintf(stderr, "REACHED slerp4\n");
	if (std::abs(cosHalfTheta) >= 1.0){
		qm.w() = qa.w();
    qm.x() = qa.x();
    qm.y() = qa.y();
    qm.z() = qa.z();
    mat = qm.toRotationMatrix();
		return mat;
	}
  fprintf(stderr, "REACHED slerp4");
	// Calculate temporary values.
	double halfTheta = std::acos(cosHalfTheta);
	double sinHalfTheta = std::sqrt(1.0 - cosHalfTheta*cosHalfTheta);
	// if theta = 180 degrees then result is not fully defined
	// we could rotate around any axis normal to qa or qb
	if (std::fabs(sinHalfTheta) < 0.001){ // fabs is floating point absolute
		qm.w() = (qa.w() * 0.5 + qb.w() * 0.5);
		qm.x() = (qa.x() * 0.5 + qb.x() * 0.5);
		qm.y() = (qa.y() * 0.5 + qb.y() * 0.5);
		qm.z() = (qa.z() * 0.5 + qb.z() * 0.5);
    mat = qm.toRotationMatrix();
		return mat;
	}
	double ratioA = std::sin((1 - t) * halfTheta) / sinHalfTheta;
	double ratioB = std::sin(t * halfTheta) / sinHalfTheta; 
	//calculate Quaternion.
	qm.w() = (qa.w() * ratioA + qb.w() * ratioB);
	qm.x() = (qa.x() * ratioA + qb.x() * ratioB);
	qm.y() = (qa.y() * ratioA + qb.y() * ratioB);
	qm.z() = (qa.z() * ratioA + qb.z() * ratioB);
  mat = qm.toRotationMatrix();
	return mat;
}

// https://gamedev.stackexchange.com/questions/30746/interpolation-between-two-3d-points
Eigen::Matrix4f interpolate(Eigen::Matrix4f &pose1, Eigen::Matrix4f &pose2, Eigen::Matrix4f &pose3, float t)
{
  Eigen::Matrix4f H = Eigen::Matrix4f::Identity();
  Eigen::Matrix3f rot = slerp(pose1, pose2, t);
  fprintf(stderr, "REACHED HERE2");
  H.block<3, 3>(0, 0) = rot;
  Eigen::Vector3f p1 = pose1.block<3, 1>(0, 3);
  Eigen::Vector3f p2 = pose2.block<3, 1>(0, 3);
  Eigen::Vector3f p = pose3.block<3, 1>(0, 3);
  Eigen::Vector3f dist = p2-p1;
  double L = (p-p1).dot(p2-p1) / dist.norm();
  double proportion = L / dist.norm();
  Eigen::Vector3f trans = (1-proportion)*p1 + proportion*p2;
  H.block<3, 1>(0, 3) = trans;
  return H;
}

Eigen::MatrixXf convertToEigenMatrix(const std::vector<std::vector<float>>& input)
{
  size_t rows = input.size();
  size_t cols = input[0].size();
  Eigen::MatrixXf output(rows, cols);
  for (size_t i = 0; i < rows; ++i)
  {
    for (size_t j = 0; j < cols; ++j)
    {
      output(i, j) = input[i][j];
    }
  }
  return output;
}

// Eigen::Vector3f rotationMatrixToEulerAngles(const Eigen::Matrix3f& rotation)
// {
//     Eigen::Vector3f euler_angles;

//     // Extract the rotation matrix elements
//     float r11 = rotation(0, 0);
//     float r12 = rotation(0, 1);
//     float r13 = rotation(0, 2);
//     float r21 = rotation(1, 0);
//     float r22 = rotation(1, 1);
//     float r23 = rotation(1, 2);
//     float r31 = rotation(2, 0);
//     float r32 = rotation(2, 1);
//     float r33 = rotation(2, 2);

//     // Calculate Euler angles
//     euler_angles(0) = atan2(r32, r33); // Roll (around x-axis)
//     euler_angles(1) = asin(-r31);      // Pitch (around y-axis)
//     euler_angles(2) = atan2(r21, r11); // Yaw (around z-axis)

//     return euler_angles;
// }

// Eigen::Matrix3f eulerToRotationMatrix(const Eigen::Vector3f& euler_angles)
// {
//     // Create angle-axis representation from Euler angles
//     Eigen::AngleAxisf rollAngle(euler_angles(0), Eigen::Vector3f::UnitX());
//     Eigen::AngleAxisf pitchAngle(euler_angles(1), Eigen::Vector3f::UnitY());
//     Eigen::AngleAxisf yawAngle(euler_angles(2), Eigen::Vector3f::UnitZ());

//     // Combine the angle-axis rotations
//     Eigen::Quaternionf quaternion = yawAngle * pitchAngle * rollAngle;

//     // Convert quaternion to rotation matrix
//     Eigen::Matrix3f rotationMatrix = quaternion.toRotationMatrix();

//     return rotationMatrix;
// }

// Eigen::Matrix3f estimateRotation(const Eigen::Matrix3Xf& normals1, const Eigen::Matrix3Xf& normals2)
// {
//     // Compute covariance matrices
//     Eigen::Matrix3f covariance1 = normals1 * normals2.transpose();
//     Eigen::Matrix3f covariance2 = normals2 * normals1.transpose();

//     // Compute SVD decomposition
//     Eigen::JacobiSVD<Eigen::Matrix3f> svd1(covariance1, Eigen::ComputeFullU | Eigen::ComputeFullV);
//     Eigen::JacobiSVD<Eigen::Matrix3f> svd2(covariance2, Eigen::ComputeFullU | Eigen::ComputeFullV);

//     // Extract rotation components
//     Eigen::Matrix3f rotation = svd1.matrixV() * svd2.matrixV().transpose();

//     return rotation;
// }

// int main(int argc, char **argv)
// {
//   // Define two transformation matrices
//   Eigen::Matrix4f pose1 = Eigen::Matrix4f::Identity();
//   pose1.block<3, 1>(0, 3) << 1.0, 2.0, 3.0;
//   pose1.block<3, 3>(0, 0) = Eigen::AngleAxisf(M_PI / 4.0, Eigen::Vector3f::UnitY()).toRotationMatrix();

//   Eigen::Matrix4f pose2 = Eigen::Matrix4f::Identity();
//   pose2.block<3, 1>(0, 3) << 4.0, 5.0, 6.0;
//   pose2.block<3, 3>(0, 0) = Eigen::AngleAxisf(M_PI / 2.0, Eigen::Vector3f::UnitX()).toRotationMatrix();

//   Eigen::Matrix4f pose3 = Eigen::Matrix4f::Identity();
//   Eigen::Matrix4f result = interpolate(pose1, pose2, pose3, 0.5);
//   std::cout << result << std::endl;
//   return 0;
// }

} // namespace Utils





