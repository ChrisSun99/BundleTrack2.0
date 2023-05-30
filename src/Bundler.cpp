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

#include <random>
#include <iostream>
#include <stdint.h>
#include "Bundler.h"
#include "LossGPU.h"
#include <typeinfo>
#include <vector>

typedef std::pair<int,int> IndexPair;
using namespace std;
using namespace Eigen;
int BLOCK = 60;
////////////////////////////////////
int num_last_frames_corr = 3;
////////////////////////////////////


Bundler::Bundler(std::shared_ptr<YAML::Node> yml1, DataLoaderBase *data_loader)
{
  _data_loader = data_loader;
  yml = yml1;
  _max_iter = (*yml1)["bundle"]["num_iter_outter"].as<int>();

  _fm = std::make_shared<Lfnet>(yml, this);
}


void Bundler::processNewFrame(std::shared_ptr<Frame> frame)
{
  std::cout<<"\n\n";
  printf("New frame %s\n",frame->_id_str.c_str());
  _newframe = frame;

  std::thread worker;

  const std::string out_dir = (*yml)["debug_dir"].as<std::string>()+"/"+frame->_id_str+"/";
  if ((*yml)["LOG"].as<int>())
  {
    if (!boost::filesystem::exists(out_dir))
    {
      system(std::string("mkdir -p "+out_dir).c_str());
    }
  }

  std::shared_ptr<Frame> last_frame;

  ////////////////////////////////////
  std::vector<std::shared_ptr<Frame>> last_good_frames;
  std::shared_ptr<Frame> next_good_frame;
  bool last_frame_blurry = false;
  ////////////////////////////////////

  if (_frames.size()>0)
  {
    last_frame = _frames.back();

    ////////////////////////////////////
    // check if the last frame is blurry
    int iter = 1;
    int count_found = 0;
    std::shared_ptr<Frame> last_good_frame = last_frame;
    if (last_good_frame->_status != Frame::FAIL) {
      last_good_frames.push_back(last_good_frame);
      count_found++;
    } else {
      last_frame_blurry = true;
    }

    // if needed, keep finding num_last_frames_corr nonblurry frames in _frames
    if (last_frame_blurry) {
      while (count_found < num_last_frames_corr) {
        last_good_frame = *(_frames.rbegin() + iter);

        // if this frame is not blurry, add it to the last_good_frames array
        if (last_good_frame->_status != Frame::FAIL) {
          last_good_frames.push_back(last_good_frame);
          count_found++;
        }
        iter++;
      }
    }

    // assign frame->last_good_frame as the nearest last_good_frame
    frame->last_good_frame = last_good_frames.front();
    frame->_id = frame->last_good_frame->_id + iter;
    frame->_pose_in_model = frame->last_good_frame->_pose_in_model;
    frame->segmentationByMaskFile();
    ////////////////////////////////////
  }
  else
  {
    frame->segmentationByMaskFile();
  }

  // Check if mask is too small
  if (frame->_roi(1)-frame->_roi(0)<10 || frame->_roi(3)-frame->_roi(2)<10)
  {
    frame->_status = Frame::FAIL;
    printf("Frame %s cloud is empty, marked FAIL, roi WxH=%fx%f\n", frame->_id_str.c_str(),frame->_roi(1)-frame->_roi(0),frame->_roi(3)-frame->_roi(2));
    return;
  }


  if (frame->_status==Frame::FAIL)
  {
    _fm->forgetFrame(frame);
    _need_reinit = true;
    return;
  } else {
    ////////////////////////////////////
    // if previous frames has no next_good_frame, either blurry or not, assign them with the current frame
    // they must be consecutive, so stop when we find one that already has next_good_frame
    int iter = 2;
    while (last_frame->next_good_frame) {
      last_frame->next_good_frame = frame;
      last_frame = *(_frames.rbegin() + iter);
      iter++;
    }
    ////////////////////////////////////
  }

  cv::Scalar blurScore;
  fprintf(stderr, "Running blur detection\n");
  blurScore = detectBlur(frame);
  double BLUR_THRES = (*yml)["bundle"]["blur_thres"].as<double>();
  if (blurScore.val[0] < BLUR_THRES)
  {
    frame->_status = Frame::FAIL;
    printf("Frame %s is blurry, marked FAIL\n", frame->_id_str.c_str());
    return;
  }

  if (frame->_status==Frame::FAIL)
  {
    _fm->forgetFrame(frame);
    _need_reinit = true;
    return;
  }

  try
  {
    float rot_deg = 0;
    Eigen::Matrix4f prev_in_init(Eigen::Matrix4f::Identity());

    _fm->detectFeature(frame, rot_deg);
  }
  catch (const std::exception &e)
  {
    printf("frame marked as FAIL since feature detection failed, ERROR\n");
    frame->_status = Frame::FAIL;
    _need_reinit = true;
    _fm->forgetFrame(frame);
    return;
  }

  if (_frames.size()>0)
  {
    ////////////////////////////////////
    // if the last frame is blurry, use num_last_frames_corr of frames to find an average pose
    // by findCorres num_last_frames_corr times, otherwise, findCorres only once
    int num_to_average = 1;
    if (last_frame_blurry) {
      num_to_average = num_last_frames_corr;
    }
    Eigen::Matrix4f mean_offset;

    for (int i = 0; i < num_to_average; i++) {
      // re-assign last_frame as one of the nonblurry frames in the last_good_frames array
      last_frame = last_good_frames[i]

      _fm->findCorres(frame, last_frame);

      if (frame->_status==Frame::FAIL)
      {
        _need_reinit = true;
        _fm->forgetFrame(frame);
        return;
      }

      PointCloudRGBNormal::Ptr cloud = _data_loader->_real_model;
      PointCloudRGBNormal::Ptr tmp(new PointCloudRGBNormal);
      Eigen::Matrix4f model_in_cam = frame->_pose_in_model.inverse();

      mean_offset += _fm->procrustesByCorrespondence(frame, last_frame, _fm->_matches[{frame,last_frame}]);
    }

    // use the average offset calculated from num_to_average previous nonblurry frames
    frame->_pose_in_model = (mean_offset / num_to_average) * frame->_pose_in_model;
    frame->_pose_inited = true;
    ////////////////////////////////////

  }

  if (frame->_status==Frame::FAIL)
  {
    _fm->forgetFrame(frame);
    _need_reinit = true;
    return;
  }

  assert(frame->_pose_in_model!=Eigen::Matrix4f::Identity());

  const int window_size = (*yml)["bundle"]["window_size"].as<int>();
  if (_frames.size()>=window_size+3)
  {
    if (std::find(_keyframes.begin(),_keyframes.end(),_frames.front())==_keyframes.end())
    {
      _fm->forgetFrame(_frames.front());
    }
    _frames.pop_front();
  }

  _frames.push_back(frame);

  if (frame->_id==0)
  {
    checkAndAddKeyframe(frame);
    return;
  }

  if (frame->_id>=1)
  {
    selectKeyFramesForBA();
    optimizeGPU();
  }

  if (frame->_status==Frame::FAIL)
  {
    _fm->forgetFrame(frame);
    _frames.pop_back();
    _need_reinit = true;
    return;
  }

  checkAndAddKeyframe(frame);

}

void Bundler::checkAndAddKeyframe(std::shared_ptr<Frame> frame)
{
  if (frame->_id==0)
  {
    _keyframes.push_back(frame);
    return;
  }
  if (frame->_status!=Frame::OTHER) return;

  const int min_interval = (*yml)["keyframe"]["min_interval"].as<int>();
  const int min_feat_num = (*yml)["keyframe"]["min_feat_num"].as<int>();
  const float min_rot = (*yml)["keyframe"]["min_rot"].as<float>();


  if (frame->_keypts.size()<min_feat_num)
  {
    return;
  }

  for (int i=0;i<_keyframes.size();i++)
  {
    const auto &k_pose = _keyframes[i]->_pose_in_model;
    const auto &cur_pose = frame->_pose_in_model;
    float rot_diff = Utils::rotationGeodesicDistance(cur_pose.block(0,0,3,3), k_pose.block(0,0,3,3));
    float trans_diff = (cur_pose.block(0,3,3,1)-k_pose.block(0,3,3,1)).norm();
    rot_diff = rot_diff*180/M_PI;
    if (rot_diff<min_rot)
    {
      return;
    }
  }


  _keyframes.push_back(frame);
}


void Bundler::selectKeyFramesForBA()
{
  const std::string debug_dir = (*yml)["debug_dir"].as<std::string>();
  std::string keyframe_dir = debug_dir+"/keyframes/";
  if (!boost::filesystem::exists(keyframe_dir))
  {
    system(std::string("mkdir -p "+keyframe_dir).c_str());
  }
  std::set<std::shared_ptr<Frame>> frames = {_newframe};
  const int max_BA_frames = (*yml)["bundle"]["max_BA_frames"].as<int>();
  printf("total keyframes=%d, already chosen _local_frames=%d, want to select %d\n", _keyframes.size(), frames.size(), max_BA_frames);
  if (_keyframes.size()+frames.size()<=max_BA_frames)
  {
    for (const auto &kf:_keyframes)
    {
      frames.insert(kf);
    }
    _local_frames = std::vector<std::shared_ptr<Frame>>(frames.begin(),frames.end());
    fprintf(stderr, "Directly adding new frame into keyframe pool\n");
    cv::Mat color_viz = _newframe->_vis.clone();
    cv::imwrite(debug_dir+"/keyframes/"+_newframe->_id_str+"_directly.jpg",color_viz,{CV_IMWRITE_JPEG_QUALITY, 80});
    return;
  }

  frames.insert(_keyframes[0]);

  const std::string method = (*yml)["bundle"]["subset_selection_method"].as<std::string>();

  if (method=="greedy_rot")
  {
    fprintf(stderr, "Doing greedy selection of keyframes\n");
    cv::Mat color_viz = _newframe->_vis.clone();
    cv::imwrite(debug_dir+"/keyframes/"+_newframe->_id_str+"_greedy.jpg",color_viz,{CV_IMWRITE_JPEG_QUALITY, 80});
    while (frames.size()<max_BA_frames)
    {
      float best_dist = std::numeric_limits<float>::max();
      std::shared_ptr<Frame> best_kf;
      for (int i=0;i<_keyframes.size();i++)
      {
        const auto &kf = _keyframes[i];
        if (frames.find(kf)!=frames.end()) continue;
        float cum_dist = 0;
        for (const auto &f:frames)
        {
          float rot_diff = Utils::rotationGeodesicDistance(kf->_pose_in_model.block(0,0,3,3), f->_pose_in_model.block(0,0,3,3));
          cum_dist += rot_diff;
        }
        if (cum_dist<best_dist)
        {
          best_dist = cum_dist;
          best_kf = kf;
        }
      }
      frames.insert(best_kf);
    }
  }
  else
  {
    std::cout<<"method not exist\n";
    exit(1);
  }

  _local_frames = std::vector<std::shared_ptr<Frame>>(frames.begin(),frames.end());

}




void Bundler::optimizeGPU()
{
  const int num_iter_outter = (*yml)["bundle"]["num_iter_outter"].as<int>();
  const int num_iter_inner = (*yml)["bundle"]["num_iter_inner"].as<int>();
  const int min_fm_edges_newframe = (*yml)["bundle"]["min_fm_edges_newframe"].as<int>();


  std::sort(_local_frames.begin(), _local_frames.end(), FramePtrComparator());
  printf("#_local_frames=%d\n",_local_frames.size());
  for (int i=0;i<_local_frames.size();i++)
  {
    std::cout<<_local_frames[i]->_id_str<<" ";
  }
  std::cout<<std::endl;

  std::vector<EntryJ> global_corres;
  std::vector<int> n_match_per_pair;
  int n_edges_newframe = 0;

  for (int i=0;i<_local_frames.size();i++)
  {
    for (int j=i+1;j<_local_frames.size();j++)
    {
      const auto &frameA = _local_frames[j];
      const auto &frameB = _local_frames[i];
      _fm->findCorres(frameA, frameB);
      _fm->vizCorresBetween(frameA,frameB,"BA");

      const auto &matches = _fm->_matches[{frameA,frameB}];
      for (int k=0;k<matches.size();k++)
      {
        const auto &match = matches[k];
        EntryJ corres;
        corres.imgIdx_j = j;
        corres.imgIdx_i = i;
        corres.pos_j = make_float3(match._ptA_cam.x,match._ptA_cam.y,match._ptA_cam.z);
        corres.pos_i = make_float3(match._ptB_cam.x,match._ptB_cam.y,match._ptB_cam.z);
        global_corres.push_back(corres);
        if (frameA==_newframe || frameB==_newframe)
        {
          n_edges_newframe++;
        }
      }
      n_match_per_pair.push_back(matches.size());
    }
  }

  const int H = _newframe->_H;
  const int W = _newframe->_W;
  const int n_pixels = H*W;

  std::vector<float*> depths_gpu;
  std::vector<uchar4*> colors_gpu;
  std::vector<float4*> normals_gpu;
  std::vector<Eigen::Matrix4f,Eigen::aligned_allocator<Eigen::Matrix4f> > poses;
  for (int i=0;i<_local_frames.size();i++)
  {
    const auto &f = _local_frames[i];
    depths_gpu.push_back(f->_depth_gpu);
    colors_gpu.push_back(f->_color_gpu);
    normals_gpu.push_back(f->_normal_gpu);
    poses.push_back(f->_pose_in_model);
  }

  if (n_edges_newframe<=min_fm_edges_newframe)
  {
    _newframe->_status = Frame::NO_BA;
    return;
  }

  printf("OptimizerGPU begin, global_corres#=%d\n",global_corres.size());
  OptimizerGpu opt(yml);
  opt.optimizeFrames(global_corres, n_match_per_pair, _local_frames.size(), _newframe->_H, _newframe->_W, depths_gpu, colors_gpu, normals_gpu, poses, _newframe->_K);

  for (int i=0;i<_local_frames.size();i++)
  {
    const auto &f = _local_frames[i];
    f->_pose_in_model = poses[i];
  }

}


void Bundler::saveNewframeResult()
{
  const std::string debug_dir = (*yml)["debug_dir"].as<std::string>();
  const std::string out_dir = debug_dir+"/"+_newframe->_id_str+"/";
  const std::string pose_out_dir = debug_dir+"/poses/";
  if (!boost::filesystem::exists(pose_out_dir))
  {
    system(std::string("mkdir -p "+pose_out_dir).c_str());
  }

  Eigen::Matrix4f cur_in_model = _newframe->_pose_in_model;
  Eigen::Matrix4f ob_in_cam = cur_in_model.inverse();

  std::ofstream ff(pose_out_dir+_newframe->_id_str+".txt");
  ff<<std::setprecision(10)<<ob_in_cam<<std::endl;
  ff.close();

  if ((*yml)["LOG"].as<int>()>0)
  {
    cv::Mat color_viz = _newframe->_vis.clone();
    PointCloudRGBNormal::Ptr cur_model(new PointCloudRGBNormal);
    pcl::transformPointCloudWithNormals(*(_data_loader->_real_model),*cur_model,ob_in_cam);
    for (int h=0;h<_newframe->_H;h++)
    {
      for (int w=0;w<_newframe->_W;w++)
      {
        auto &bgr = color_viz.at<cv::Vec3b>(h,w);
        if (_newframe->_fg_mask.at<uchar>(h,w)==0)
        {
          for (int i=0;i<3;i++)
          {
            bgr[i] = (uchar)bgr[i]*0.2;
          }
        }
      }
    }
    Utils::drawProjectPoints(cur_model,_data_loader->_K,color_viz);
    cv::putText(color_viz,_newframe->_id_str,{5,30},cv::FONT_HERSHEY_PLAIN,2,{255,0,0},1,8,false);
    // cv::imshow("color_viz",color_viz);
    // cv::waitKey(1);
    cv::imwrite(debug_dir+"/color_viz/"+_newframe->_id_str+"_color_viz.jpg",color_viz,{CV_IMWRITE_JPEG_QUALITY, 80});
    cv::imwrite(out_dir+"color_viz.jpg",color_viz,{CV_IMWRITE_JPEG_QUALITY, 80});

    const std::string raw_dir = debug_dir+"/color_raw/";
    if (!boost::filesystem::exists(raw_dir))
    {
      system(std::string("mkdir -p "+raw_dir).c_str());
    }
    cv::imwrite(raw_dir+_newframe->_id_str+"_color_raw.png",_newframe->_color);
  }
}

// Blur detection with FFT
// https://docs.opencv.org/4.x/d8/d01/tutorial_discrete_fourier_transform.html
cv::Scalar Bundler::detectBlur(std::shared_ptr<Frame> frame)
{
  int cx = frame->_W / 2;
  int cy = frame->_H / 2;
  cv::Mat fourierTransform;
  cv::Mat colorImage;
  frame->_color.copyTo(colorImage);

  // Convert colorImage to the appropriate type
  cv::cvtColor(colorImage, colorImage, cv::COLOR_BGR2GRAY);
  colorImage.convertTo(colorImage, CV_32FC1);

  cv::dft(colorImage, fourierTransform);
  cv::Mat q0(fourierTransform, cv::Rect(0, 0, cx, cy));       // Top-Left - Create a ROI per quadrant
  cv::Mat q1(fourierTransform, cv::Rect(cx, 0, cx, cy));      // Top-Right
  cv::Mat q2(fourierTransform, cv::Rect(0, cy, cx, cy));      // Bottom-Left
  cv::Mat q3(fourierTransform, cv::Rect(cx, cy, cx, cy));     // Bottom-Right

  cv::Mat tmp;                                            // swap quadrants (Top-Left with Bottom-Right)
  q0.copyTo(tmp);
  q3.copyTo(q0);
  tmp.copyTo(q3);
  q1.copyTo(tmp);                                     // swap quadrant (Top-Right with Bottom-Left)
  q2.copyTo(q1);
  tmp.copyTo(q2);

  fourierTransform(cv::Rect(cx-BLOCK,cy-BLOCK,2*BLOCK,2*BLOCK)).setTo(0);

  //shuffle the quadrants to their original position
  cv::Mat orgFFT;
  fourierTransform.copyTo(orgFFT);
  cv::Mat p0(orgFFT, cv::Rect(0, 0, cx, cy));       // Top-Left - Create a ROI per quadrant
  cv::Mat p1(orgFFT, cv::Rect(cx, 0, cx, cy));      // Top-Right
  cv::Mat p2(orgFFT, cv::Rect(0, cy, cx, cy));      // Bottom-Left
  cv::Mat p3(orgFFT, cv::Rect(cx, cy, cx, cy));     // Bottom-Right

  p0.copyTo(tmp);
  p3.copyTo(p0);
  tmp.copyTo(p3);

  p1.copyTo(tmp);                                     // swap quadrant (Top-Right with Bottom-Left)
  p2.copyTo(p1);
  tmp.copyTo(p2);

  cv::Mat invFFT;
  cv::Mat logFFT;
  double minVal,maxVal;

  cv::dft(orgFFT, invFFT);
  invFFT = cv::abs(invFFT);
  cv::minMaxLoc(invFFT,&minVal,&maxVal,NULL,NULL);

  //check for impossible values
  if(maxVal<=0.0){
      cerr << "No information, complete black image!\n";
      return 1;
  }

  cv::log(invFFT,logFFT);
  logFFT *= 20;

  cv::Scalar result = cv::mean(logFFT);
  std::cout << "Result : "<< result.val[0] << std::endl;
  return result;
}