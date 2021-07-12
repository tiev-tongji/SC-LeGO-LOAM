// Copyright 2013, Ji Zhang, Carnegie Mellon University
// Further contributions copyright (c) 2016, Southwest Research Institute
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// 1. Redistributions of source code must retain the above copyright notice,
//    this list of conditions and the following disclaimer.
// 2. Redistributions in binary form must reproduce the above copyright notice,
//    this list of conditions and the following disclaimer in the documentation
//    and/or other materials provided with the distribution.
// 3. Neither the name of the copyright holder nor the names of its
//    contributors may be used to endorse or promote products derived from this
//    software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.
//
// This is an implementation of the algorithm described in the following papers:
//   J. Zhang and S. Singh. LOAM: Lidar Odometry and Mapping in Real-time.
//     Robotics: Science and Systems Conference (RSS). Berkeley, CA, July 2014.
//   T. Shan and B. Englot. LeGO-LOAM: Lightweight and Ground-Optimized Lidar Odometry and Mapping on Variable Terrain
//      IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS). October 2018.

#include "utility.h"
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <vector>
#include <math.h>
#include <nav_msgs/Odometry.h>
#include <std_msgs/String.h>
#include <geometry_msgs/Quaternion.h>
#include <geometry_msgs/PoseStamped.h>
#include <nav_msgs/Path.h>
#include <tf/tf.h>

using namespace Eigen;

class FeatureAssociation{

private:

	ros::NodeHandle nh;

    ros::Subscriber subLaserCloud;
    ros::Subscriber subLaserCloudInfo;
    ros::Subscriber subOutlierCloud;
    ros::Subscriber subImu;

    ros::Publisher pubCornerPointsSharp;
    ros::Publisher pubCornerPointsLessSharp;
    ros::Publisher pubSurfPointsFlat;
    ros::Publisher pubSurfPointsLessFlat;
    ros::Publisher pubImuOdom;
    ros::Publisher pubImuPath;

    pcl::PointCloud<PointType>::Ptr segmentedCloud;
    pcl::PointCloud<PointType>::Ptr outlierCloud;
    pcl::PointCloud<PointType>::Ptr cloudBeforeAdjust;

    pcl::PointCloud<PointType>::Ptr cornerPointsSharp;
    pcl::PointCloud<PointType>::Ptr cornerPointsLessSharp;
    pcl::PointCloud<PointType>::Ptr surfPointsFlat;
    pcl::PointCloud<PointType>::Ptr surfPointsLessFlat;

    pcl::PointCloud<PointType>::Ptr surfPointsLessFlatScan;
    pcl::PointCloud<PointType>::Ptr surfPointsLessFlatScanDS;

    pcl::VoxelGrid<PointType> downSizeFilter;

    double timeScanCur;
    double timeNewSegmentedCloud;
    double timeNewSegmentedCloudInfo;
    double timeNewOutlierCloud;

    bool newSegmentedCloud;
    bool newSegmentedCloudInfo;
    bool newOutlierCloud;

    cloud_msgs::cloud_info segInfo;
    std_msgs::Header cloudHeader;

    int systemInitCount;
    bool systemInited;

    std::vector<smoothness_t> cloudSmoothness;
    float *cloudCurvature;
    int *cloudNeighborPicked;
    int *cloudLabel;

    int imuPointerFront;
    int imuPointerLast;
    int imuPointerLastIteration;

    float imuRollStart, imuPitchStart, imuYawStart;
    float cosImuRollStart, cosImuPitchStart, cosImuYawStart, sinImuRollStart, sinImuPitchStart, sinImuYawStart;
    float imuRollCur, imuPitchCur, imuYawCur;

    float imuVeloXStart, imuVeloYStart, imuVeloZStart;
    float imuShiftXStart, imuShiftYStart, imuShiftZStart;

    float imuVeloXCur, imuVeloYCur, imuVeloZCur;
    float imuShiftXCur, imuShiftYCur, imuShiftZCur;

    float imuShiftFromStartXCur, imuShiftFromStartYCur, imuShiftFromStartZCur;
    float imuVeloFromStartXCur, imuVeloFromStartYCur, imuVeloFromStartZCur;

    float imuAngularRotationXCur, imuAngularRotationYCur, imuAngularRotationZCur;
    float imuAngularRotationXLast, imuAngularRotationYLast, imuAngularRotationZLast;
    float imuAngularFromStartX, imuAngularFromStartY, imuAngularFromStartZ;

    double imuTime[imuQueLength];
    float imuRoll[imuQueLength];
    float imuPitch[imuQueLength];
    float imuYaw[imuQueLength];

    float imuAccX[imuQueLength];
    float imuAccY[imuQueLength];
    float imuAccZ[imuQueLength];

    float imuVeloX[imuQueLength];
    float imuVeloY[imuQueLength];
    float imuVeloZ[imuQueLength];

    float imuShiftX[imuQueLength];
    float imuShiftY[imuQueLength];
    float imuShiftZ[imuQueLength];

    float imuAngularVeloX[imuQueLength];
    float imuAngularVeloY[imuQueLength];
    float imuAngularVeloZ[imuQueLength];

    float imuAngularRotationX[imuQueLength];
    float imuAngularRotationY[imuQueLength];
    float imuAngularRotationZ[imuQueLength];
    //
//    bool nine_axis;
    float accX_sum;
    float accY_sum;
    float accX_ave;
    float accY_ave;
    float roll_initial;
    float pitch_initial;
    float yaw_initial;

    double roll_back[imuQueLength];
    double pitch_back[imuQueLength];
    double yaw_back[imuQueLength];

    // c = camera , v = velodyne ,p = pandar ,i = imu, vi =virtual imu
    Eigen::Matrix3d R_i2p;// imu to pandar
    Eigen::Vector3d l_i2p;// imu to pandar
    Eigen::Matrix3d R_p2i;// pandar to imu
    Eigen::Vector3d l_p2i;// pandar to imu

    Eigen::Matrix3d R_p2v;// pandar2velodyne
    Eigen::Matrix3d R_v2p;// pandar2velodyne

    Eigen::Matrix3d R_p2c;//velodyne2camera
    Eigen::Matrix3d R_c2p;//camera2pandar

    Eigen::Matrix3d R_imuglobal_initial;
    Eigen::Matrix3d R_veloimuglobal_initial;


    std::vector<Eigen::Vector3d> imuShift;
    std::vector<Eigen::Vector3d> imuVelo;
    std::vector<Eigen::Vector3d> euler_angles;
    std::vector<Eigen::Vector3d> imuAngularRotation_angle;
    std::vector<Eigen::Matrix3d> w_x;
    std::vector<Eigen::Matrix3d> R_c2w;
    std::vector<Eigen::Matrix3d> R_w2c;
    std::vector<Eigen::Matrix3d> R_global;
    std::vector<Eigen::Matrix3d> imuAngularRotation;
    std::vector<Eigen::Matrix3d> realimuAngularRotation;
    std::vector<Eigen::Matrix3d> realimuAngularRotation1;
    std::vector<Eigen::Vector3d> realimuShift;
    std::vector<Eigen::Vector3d> realimuVelo;
//    std::vector<Eigen::Quaterniond> imu_orientation;

    float realimuRoll[imuQueLength];
    float realimuPitch[imuQueLength];
    float realimuYaw[imuQueLength];
    float realimuVeloX[imuQueLength];
    float realimuVeloY[imuQueLength];
    float realimuVeloZ[imuQueLength];

    float realimuShiftX[imuQueLength];
    float realimuShiftY[imuQueLength];
    float realimuShiftZ[imuQueLength];

    float realimuAngularRotationX[imuQueLength];
    float realimuAngularRotationY[imuQueLength];
    float realimuAngularRotationZ[imuQueLength];
    //
    float imuroll[imuQueLength];
    float imupitch[imuQueLength];
    float imuyaw[imuQueLength];
    float realimuVeloX1[imuQueLength];
    float realimuVeloY1[imuQueLength];
    float realimuVeloZ1[imuQueLength];
    float imuAccX1[imuQueLength];
    float imuAccY1[imuQueLength];
    float imuAccZ1[imuQueLength];

    float v_max;
    float time_max;

    float v_max_l;

    float realimuShiftX1[imuQueLength];
    float realimuShiftY1[imuQueLength];
    float realimuShiftZ1[imuQueLength];


    ros::Publisher pubLaserCloudCornerLast;
    ros::Publisher pubLaserCloudSurfLast;
    ros::Publisher pubLaserOdometry;
    ros::Publisher pubOutlierCloudLast;

    int imuCount;
    int laserscannum;
    int skipFrameNum;
    bool systemInitedLM;

    int laserCloudCornerLastNum;
    int laserCloudSurfLastNum;

    int *pointSelCornerInd;
    float *pointSearchCornerInd1;
    float *pointSearchCornerInd2;

    int *pointSelSurfInd;
    float *pointSearchSurfInd1;
    float *pointSearchSurfInd2;
    float *pointSearchSurfInd3;

    float transformCur[6];
    float transformSum[6];

    float imuRollLast, imuPitchLast, imuYawLast;
    float imuShiftFromStartX, imuShiftFromStartY, imuShiftFromStartZ;
    float imuVeloFromStartX, imuVeloFromStartY, imuVeloFromStartZ;

    pcl::PointCloud<PointType>::Ptr laserCloudCornerLast;
    pcl::PointCloud<PointType>::Ptr laserCloudSurfLast;
    pcl::PointCloud<PointType>::Ptr laserCloudOri;
    pcl::PointCloud<PointType>::Ptr coeffSel;

    pcl::KdTreeFLANN<PointType>::Ptr kdtreeCornerLast;
    pcl::KdTreeFLANN<PointType>::Ptr kdtreeSurfLast;

    std::vector<int> pointSearchInd;
    std::vector<float> pointSearchSqDis;

    PointType pointOri, pointSel, tripod1, tripod2, tripod3, pointProj, coeff;

    nav_msgs::Odometry laserOdometry;
    nav_msgs::Odometry imu_odom;
    nav_msgs::Path imu_path;
    geometry_msgs::Quaternion imuQuat1;

    tf::TransformBroadcaster tfBroadcaster;
    tf::StampedTransform laserOdometryTrans;

    bool isDegenerate;
    cv::Mat matP;

    int frameCount;

public:

    FeatureAssociation():
        nh("~")
        {

        subLaserCloud = nh.subscribe<sensor_msgs::PointCloud2>("/segmented_cloud", 1, &FeatureAssociation::laserCloudHandler, this);
        subLaserCloudInfo = nh.subscribe<cloud_msgs::cloud_info>("/segmented_cloud_info", 1, &FeatureAssociation::laserCloudInfoHandler, this);
        subOutlierCloud = nh.subscribe<sensor_msgs::PointCloud2>("/outlier_cloud", 1, &FeatureAssociation::outlierCloudHandler, this);
        subImu = nh.subscribe<sensor_msgs::Imu>(imuTopic, 50, &FeatureAssociation::imuHandler, this);

        pubCornerPointsSharp = nh.advertise<sensor_msgs::PointCloud2>("/laser_cloud_sharp", 1);
        pubCornerPointsLessSharp = nh.advertise<sensor_msgs::PointCloud2>("/laser_cloud_less_sharp", 1);
        pubSurfPointsFlat = nh.advertise<sensor_msgs::PointCloud2>("/laser_cloud_flat", 1);
        pubSurfPointsLessFlat = nh.advertise<sensor_msgs::PointCloud2>("/laser_cloud_less_flat", 1);
        pubImuOdom = nh.advertise<nav_msgs::Odometry>("imu_odom", 50);
        pubImuPath = nh.advertise<nav_msgs::Path>("imu_path", 50);


        pubLaserCloudCornerLast = nh.advertise<sensor_msgs::PointCloud2>("/laser_cloud_corner_last", 2);
        pubLaserCloudSurfLast = nh.advertise<sensor_msgs::PointCloud2>("/laser_cloud_surf_last", 2);
        pubOutlierCloudLast = nh.advertise<sensor_msgs::PointCloud2>("/outlier_cloud_last", 2);
        pubLaserOdometry = nh.advertise<nav_msgs::Odometry> ("/laser_odom_to_init", 5);
        
        initializationValue();
    }

    void initializationValue()
    {
        cloudCurvature = new float[N_SCAN*Horizon_SCAN];
        cloudNeighborPicked = new int[N_SCAN*Horizon_SCAN];
        cloudLabel = new int[N_SCAN*Horizon_SCAN];

        pointSelCornerInd = new int[N_SCAN*Horizon_SCAN];
        pointSearchCornerInd1 = new float[N_SCAN*Horizon_SCAN];
        pointSearchCornerInd2 = new float[N_SCAN*Horizon_SCAN];

        pointSelSurfInd = new int[N_SCAN*Horizon_SCAN];
        pointSearchSurfInd1 = new float[N_SCAN*Horizon_SCAN];
        pointSearchSurfInd2 = new float[N_SCAN*Horizon_SCAN];
        pointSearchSurfInd3 = new float[N_SCAN*Horizon_SCAN];

        imuShift.resize(imuQueLength);
        imuVelo.resize(imuQueLength);
        euler_angles.resize(imuQueLength);
        imuAngularRotation_angle.resize(imuQueLength);
        w_x.resize(imuQueLength);
        R_c2w.resize(imuQueLength);
        R_w2c.resize(imuQueLength);
        R_global.resize(imuQueLength);
//        imu_orientation.resize(imuQueLength);
        imuAngularRotation.resize(imuQueLength);
        realimuShift.resize(imuQueLength);
        realimuVelo.resize(imuQueLength);
        realimuAngularRotation.resize(imuQueLength);
        realimuAngularRotation1.resize(imuQueLength);

        accX_sum = 0;
        accY_sum = 0;
        accX_ave = 0;
        accY_ave = 0;

        cloudSmoothness.resize(N_SCAN*Horizon_SCAN);

        downSizeFilter.setLeafSize(0.2, 0.2, 0.2);

        segmentedCloud.reset(new pcl::PointCloud<PointType>());
        outlierCloud.reset(new pcl::PointCloud<PointType>());
        cloudBeforeAdjust.reset(new pcl::PointCloud<PointType>());

        cornerPointsSharp.reset(new pcl::PointCloud<PointType>());
        cornerPointsLessSharp.reset(new pcl::PointCloud<PointType>());
        surfPointsFlat.reset(new pcl::PointCloud<PointType>());
        surfPointsLessFlat.reset(new pcl::PointCloud<PointType>());

        surfPointsLessFlatScan.reset(new pcl::PointCloud<PointType>());
        surfPointsLessFlatScanDS.reset(new pcl::PointCloud<PointType>());

        timeScanCur = 0;
        timeNewSegmentedCloud = 0;
        timeNewSegmentedCloudInfo = 0;
        timeNewOutlierCloud = 0;

        newSegmentedCloud = false;
        newSegmentedCloudInfo = false;
        newOutlierCloud = false;

        systemInitCount = 0;
        systemInited = false;

        imuCount = -1;
        imuPointerFront = 0;
        imuPointerLast = -1;
        imuPointerLastIteration = 0;

        imuRollStart = 0; imuPitchStart = 0; imuYawStart = 0;
        cosImuRollStart = 0; cosImuPitchStart = 0; cosImuYawStart = 0;
        sinImuRollStart = 0; sinImuPitchStart = 0; sinImuYawStart = 0;
        imuRollCur = 0; imuPitchCur = 0; imuYawCur = 0;

        imuVeloXStart = 0; imuVeloYStart = 0; imuVeloZStart = 0;
        imuShiftXStart = 0; imuShiftYStart = 0; imuShiftZStart = 0;

        imuVeloXCur = 0; imuVeloYCur = 0; imuVeloZCur = 0;
        imuShiftXCur = 0; imuShiftYCur = 0; imuShiftZCur = 0;

        imuShiftFromStartXCur = 0; imuShiftFromStartYCur = 0; imuShiftFromStartZCur = 0;
        imuVeloFromStartXCur = 0; imuVeloFromStartYCur = 0; imuVeloFromStartZCur = 0;

        imuAngularRotationXCur = 0; imuAngularRotationYCur = 0; imuAngularRotationZCur = 0;
        imuAngularRotationXLast = 0; imuAngularRotationYLast = 0; imuAngularRotationZLast = 0;
        imuAngularFromStartX = 0; imuAngularFromStartY = 0; imuAngularFromStartZ = 0;

        // 杭州, pandar 左后上，imu 为右前上，velodyne为前左上，camera为左上前
//        l_i2p << -0.048111, 1.363886, -1.357512;
//        R_i2p << -0.999770, 0.013938, 0.016284,-0.014167, -0.999801, -0.014021,0.016085, -0.014248, 0.999769;//考虑旋转
//        R_i2p << -1,0,0,0,-1,0,0,0,1;//最开始设定方向
//        R_i2p << 0,-1,0,0,0,-1,1,0,0;
//        R_i2p << 1,0,0,0,1,0,0,0,1;
//        R_p2i = R_i2p.transpose();
//        l_p2i = R_p2i*l_i2p;
//
//        R_p2c << 1,0,0,0,0,1,0,-1,0;
//        R_c2p = R_p2c.transpose();
//        R_v2p << 0,1,0,-1,0,0,0,0,1;
//        R_p2v = R_v2p.transpose();

        v_max = 0;
        time_max = 0;
        v_max_l = 0;


        // 松山湖, pandar 安装为左后上，imu 为前左上，velodyne为前左上，camera为左上前，数据为lidar to imu
        l_i2p << -0.080865451,1.637948593,-1.309935873;
//        R_p2i <<  0.036298090009,-0.999333605431,-0.003846260277,0.999326656940,0.036276674109,0.005498694811,-0.005355500980,-0.004043262543,0.999977485065;
//        R_i2p = R_i2p.transpose();
        l_p2i << 1.63475398,0.028594611,1.316095961;
//
        R_p2c << 1,0,0,0,0,1,0,-1,0;
        R_v2p << 0,1,0,-1,0,0,0,0,1;

//        R_imuglobal_initial << 1,0,0,0,1,0,0,0,1;
//        R_veloimuglobal_initial << 1,0,0,0,1,0,0,0,1;

        for (int i = 0; i < imuQueLength; ++i)
        {
            imuTime[i] = 0;
            imuRoll[i] = 0; imuPitch[i] = 0; imuYaw[i] = 0;
            imuAccX[i] = 0; imuAccY[i] = 0; imuAccZ[i] = 0;
            imuVeloX[i] = 0; imuVeloY[i] = 0; imuVeloZ[i] = 0;
            imuShiftX[i] = 0; imuShiftY[i] = 0; imuShiftZ[i] = 0;
            imuAngularVeloX[i] = 0; imuAngularVeloY[i] = 0; imuAngularVeloZ[i] = 0;
            imuAngularRotationX[i] = 0; imuAngularRotationY[i] = 0; imuAngularRotationZ[i] = 0;

            realimuRoll[i] = 0; realimuPitch[i] = 0; realimuYaw[i] = 0;
            realimuVeloX[i] = 0; realimuVeloY[i] = 0; realimuVeloZ[i] = 0;
            realimuShiftX[i] = 0; realimuShiftY[i] = 0; realimuShiftZ[i] = 0;

            realimuVeloX1[i] = 0; realimuVeloY1[i] = 0; realimuVeloZ1[i] = 0;
            realimuShiftX1[i] = 0; realimuShiftY1[i] = 0; realimuShiftZ1[i] = 0;
            imuAccX1[i] = 0; imuAccY1[i] = 0; imuAccZ1[i] = 0;

            realimuAngularRotationX[i] = 0; realimuAngularRotationY[i] = 0; realimuAngularRotationZ[i] = 0;
            imuShift[i]<< 0, 0, 0; imuVelo[i]<< 0, 0, 0; euler_angles[i]<< 0, 0, 0;
            imuAngularRotation_angle[i]<< 0, 0, 0;
            imuAngularRotation[i]<< 0, 0, 0, 0, 0, 0, 0, 0, 0;
            w_x[i]<< 0, 0, 0, 0, 0, 0, 0, 0, 0;
            R_c2w[i] << 0, 0, 0, 0, 0, 0, 0, 0, 0;
            R_w2c[i] << 0, 0, 0, 0, 0, 0, 0, 0, 0;
            R_global[i] << 1, 0, 0, 0, 1, 0, 0, 0, 1;

            roll_back[i] = 0;
            pitch_back[i] = 0;
            yaw_back[i] = 0;
        }


        skipFrameNum = 1;

        for (int i = 0; i < 6; ++i){
            transformCur[i] = 0;
            transformSum[i] = 0;
        }

        systemInitedLM = false;

        imuRollLast = 0; imuPitchLast = 0; imuYawLast = 0;
        imuShiftFromStartX = 0; imuShiftFromStartY = 0; imuShiftFromStartZ = 0;
        imuVeloFromStartX = 0; imuVeloFromStartY = 0; imuVeloFromStartZ = 0;

        laserCloudCornerLast.reset(new pcl::PointCloud<PointType>());
        laserCloudSurfLast.reset(new pcl::PointCloud<PointType>());
        laserCloudOri.reset(new pcl::PointCloud<PointType>());
        coeffSel.reset(new pcl::PointCloud<PointType>());

        kdtreeCornerLast.reset(new pcl::KdTreeFLANN<PointType>());
        kdtreeSurfLast.reset(new pcl::KdTreeFLANN<PointType>());

        laserOdometry.header.frame_id = "/camera_init";
        laserOdometry.child_frame_id = "/laser_odom";

        laserOdometryTrans.frame_id_ = "/camera_init";
        laserOdometryTrans.child_frame_id_ = "/laser_odom";

        imu_odom.header.frame_id = "/camera_init";
        imu_odom.child_frame_id = "/camera";

        imu_path.header.frame_id = "/camera_init";
        imu_path.header.stamp = ros::Time::now();

        
        isDegenerate = false;
        matP = cv::Mat(6, 6, CV_32F, cv::Scalar::all(0));

        frameCount = skipFrameNum;
    }

    void updateImuRollPitchYawStartSinCos(){
        cosImuRollStart = cos(imuRollStart);
        cosImuPitchStart = cos(imuPitchStart);
        cosImuYawStart = cos(imuYawStart);
        sinImuRollStart = sin(imuRollStart);
        sinImuPitchStart = sin(imuPitchStart);
        sinImuYawStart = sin(imuYawStart);
    }


    void ShiftToStartIMU(float pointTime)
    {
        imuShiftFromStartXCur = imuShiftXCur - imuShiftXStart - imuVeloXStart * pointTime;
        imuShiftFromStartYCur = imuShiftYCur - imuShiftYStart - imuVeloYStart * pointTime;
        imuShiftFromStartZCur = imuShiftZCur - imuShiftZStart - imuVeloZStart * pointTime;

        // world to camera start,R_ws
        float x1 = cosImuYawStart * imuShiftFromStartXCur - sinImuYawStart * imuShiftFromStartZCur;
        float y1 = imuShiftFromStartYCur;
        float z1 = sinImuYawStart * imuShiftFromStartXCur + cosImuYawStart * imuShiftFromStartZCur;

        float x2 = x1;
        float y2 = cosImuPitchStart * y1 + sinImuPitchStart * z1;
        float z2 = -sinImuPitchStart * y1 + cosImuPitchStart * z1;

        imuShiftFromStartXCur = cosImuRollStart * x2 + sinImuRollStart * y2;
        imuShiftFromStartYCur = -sinImuRollStart * x2 + cosImuRollStart * y2;
        imuShiftFromStartZCur = z2;
    }

    void VeloToStartIMU()
    {
        imuVeloFromStartXCur = imuVeloXCur - imuVeloXStart;
        imuVeloFromStartYCur = imuVeloYCur - imuVeloYStart;
        imuVeloFromStartZCur = imuVeloZCur - imuVeloZStart;

        float x1 = cosImuYawStart * imuVeloFromStartXCur - sinImuYawStart * imuVeloFromStartZCur;
        float y1 = imuVeloFromStartYCur;
        float z1 = sinImuYawStart * imuVeloFromStartXCur + cosImuYawStart * imuVeloFromStartZCur;

        float x2 = x1;
        float y2 = cosImuPitchStart * y1 + sinImuPitchStart * z1;
        float z2 = -sinImuPitchStart * y1 + cosImuPitchStart * z1;

        imuVeloFromStartXCur = cosImuRollStart * x2 + sinImuRollStart * y2;
        imuVeloFromStartYCur = -sinImuRollStart * x2 + cosImuRollStart * y2;
        imuVeloFromStartZCur = z2;
    }

    void TransformToStartIMU(PointType *p)
    {
        // 先 camera current to world, 再 world to camera start

        float x1 = cos(imuRollCur) * p->x - sin(imuRollCur) * p->y;
        float y1 = sin(imuRollCur) * p->x + cos(imuRollCur) * p->y;
        float z1 = p->z;

        float x2 = x1;
        float y2 = cos(imuPitchCur) * y1 - sin(imuPitchCur) * z1;
        float z2 = sin(imuPitchCur) * y1 + cos(imuPitchCur) * z1;

        float x3 = cos(imuYawCur) * x2 + sin(imuYawCur) * z2;
        float y3 = y2;
        float z3 = -sin(imuYawCur) * x2 + cos(imuYawCur) * z2;

        float x4 = cosImuYawStart * x3 - sinImuYawStart * z3;
        float y4 = y3;
        float z4 = sinImuYawStart * x3 + cosImuYawStart * z3;

        float x5 = x4;
        float y5 = cosImuPitchStart * y4 + sinImuPitchStart * z4;
        float z5 = -sinImuPitchStart * y4 + cosImuPitchStart * z4;

        p->x = cosImuRollStart * x5 + sinImuRollStart * y5 + imuShiftFromStartXCur;
        p->y = -sinImuRollStart * x5 + cosImuRollStart * y5 + imuShiftFromStartYCur;
        p->z = z5 + imuShiftFromStartZCur;
    }

    void AccumulaterealIMUShiftAndRotation()
    {
        float accX = imuAccX1[imuPointerLast];
        float accY = imuAccY1[imuPointerLast];
        float accZ = imuAccZ1[imuPointerLast];

        int imuPointerBack = (imuPointerLast + imuQueLength - 1) % imuQueLength;
        double timeDiff = imuTime[imuPointerLast] - imuTime[imuPointerBack];
//        std::cout << "accX:   " << accX << std::endl;
//        std::cout << "accY:   " << accY << std::endl;
//        std::cout << "accZ:   " << accZ << std::endl;
//        std::cout << "timeDiff:   " << timeDiff << std::endl;

        if (timeDiff < scanPeriod) {

            realimuShiftX1[imuPointerLast] = realimuShiftX1[imuPointerBack] + realimuVeloX1[imuPointerBack] * timeDiff +
                                             accX * timeDiff * timeDiff / 2;
            realimuShiftY1[imuPointerLast] = realimuShiftY1[imuPointerBack] + realimuVeloY1[imuPointerBack] * timeDiff +
                                             accY * timeDiff * timeDiff / 2;
            realimuShiftZ1[imuPointerLast] = realimuShiftZ1[imuPointerBack] + realimuVeloZ1[imuPointerBack] * timeDiff +
                                             accZ * timeDiff * timeDiff / 2;

            realimuVeloX1[imuPointerLast] = realimuVeloX1[imuPointerBack] + accX * timeDiff;
            realimuVeloY1[imuPointerLast] = realimuVeloY1[imuPointerBack] + accY * timeDiff;
            realimuVeloZ1[imuPointerLast] = realimuVeloZ1[imuPointerBack] + accZ * timeDiff;

//            std::cout << "imuPointerBack:  " << imuPointerBack << std::endl;
//            std::cout << "imuPointerLast:  " << imuPointerLast << std::endl;
        }

    }

    void toEulerAngle(const Quaterniond& q, double& roll1, double& pitch1, double& yaw1)
    {
        // roll (x-axis rotation)
        double sinr_cosp = +2.0 * (q.w() * q.x() + q.y() * q.z());
        double cosr_cosp = +1.0 - 2.0 * (q.x() * q.x() + q.y() * q.y());
        roll1 = atan2(sinr_cosp, cosr_cosp);

        // pitch (y-axis rotation)
        double sinp = +2.0 * (q.w() * q.y() - q.z() * q.x());
        if (fabs(sinp) >= 1)
            pitch1 = copysign(M_PI / 2, sinp); // use 90 degrees if out of range
        else
            pitch1 = asin(sinp);

        // yaw (z-axis rotation)
        double siny_cosp = +2.0 * (q.w() * q.z() + q.x() * q.y());
        double cosy_cosp = +1.0 - 2.0 * (q.y() * q.y() + q.z() * q.z());
        yaw1 = atan2(siny_cosp, cosy_cosp);
    }

    void AccumulateIMUShiftAndRotation()
    {
        double roll = realimuRoll[imuPointerLast];
        double pitch = realimuPitch[imuPointerLast];
        double yaw = realimuYaw[imuPointerLast];
        float accX = imuAccX[imuPointerLast];
        float accY = imuAccY[imuPointerLast];
        float accZ = imuAccZ[imuPointerLast];

        // camera的acc转换到world下， R_cw = Ry*Rx*Rz

        //     |cosrz  -sinrz  0|
        //  Rz=|sinrz  cosrz   0|
        //     |0       0      1|
        float x1 = cos(roll) * accX - sin(roll) * accY;
        float y1 = sin(roll) * accX + cos(roll) * accY;
        float z1 = accZ;

        //    |1     0        0|
        // Rx=|0   cosrx -sinrx|
        //    |0   sinrx  cosrx|
        float x2 = x1;
        float y2 = cos(pitch) * y1 - sin(pitch) * z1;
        float z2 = sin(pitch) * y1 + cos(pitch) * z1;

        //    |cosry   0   sinry|
        // Ry=|0       1       0|
        //    |-sinry  0   cosry|
        accX = cos(yaw) * x2 + sin(yaw) * z2;
        accY = y2;
        accZ = -sin(yaw) * x2 + cos(yaw) * z2;

        int imuPointerBack = (imuPointerLast + imuQueLength - 1) % imuQueLength;
        double timeDiff = imuTime[imuPointerLast] - imuTime[imuPointerBack];

//        std::cout << "timeDiff: " << timeDiff << std::endl;
//        std::cout << "imuPointerLast: " << imuPointerLast << std::endl;
//        std::cout << "imuPointerBack: " << imuPointerBack << std::endl;

        if (timeDiff < scanPeriod) {

            realimuShiftX[imuPointerLast] = realimuShiftX[imuPointerBack] + realimuVeloX[imuPointerBack] * timeDiff + accX * timeDiff * timeDiff / 2;
            realimuShiftY[imuPointerLast] = realimuShiftY[imuPointerBack] + realimuVeloY[imuPointerBack] * timeDiff + accY * timeDiff * timeDiff / 2;
            realimuShiftZ[imuPointerLast] = realimuShiftZ[imuPointerBack] + realimuVeloZ[imuPointerBack] * timeDiff + accZ * timeDiff * timeDiff / 2;

            realimuVeloX[imuPointerLast] = realimuVeloX[imuPointerBack] + accX * timeDiff;
            realimuVeloY[imuPointerLast] = realimuVeloY[imuPointerBack] + accY * timeDiff;
            realimuVeloZ[imuPointerLast] = realimuVeloZ[imuPointerBack] + accZ * timeDiff;

            realimuAngularRotationX[imuPointerLast] = realimuAngularRotationX[imuPointerBack] + imuAngularVeloX[imuPointerBack] * timeDiff;
            realimuAngularRotationY[imuPointerLast] = realimuAngularRotationY[imuPointerBack] + imuAngularVeloY[imuPointerBack] * timeDiff;
            realimuAngularRotationZ[imuPointerLast] = realimuAngularRotationZ[imuPointerBack] + imuAngularVeloZ[imuPointerBack] * timeDiff;

        }
        float v_toal = sqrt(realimuVeloX[imuPointerLast]*realimuVeloX[imuPointerLast]
                           +realimuVeloY[imuPointerLast]*realimuVeloY[imuPointerLast]
                           +realimuVeloZ[imuPointerLast]*realimuVeloZ[imuPointerLast]);
        if (v_toal> v_max)
        {
            v_max = v_toal;
            time_max = imuTime[imuPointerLast];
        }

//        std::cout << "v_toal--- " << v_toal << std::endl;
//        std::cout << "v_max--- " << v_max << std::endl;
//        std::cout << fixed<<setprecision(9) << "bag_time--- " << time_max << std::endl;

        // 姿态增量
        float delta_rotationX = imuAngularVeloX[imuPointerBack] * timeDiff;
        float delta_rotationY = imuAngularVeloY[imuPointerBack] * timeDiff;
        float delta_rotationZ = imuAngularVeloZ[imuPointerBack] * timeDiff;

        // 求增量矩阵
        Eigen::AngleAxisd rollAngle(AngleAxisd(delta_rotationX,Vector3d::UnitX()));
        Eigen::AngleAxisd pitchAngle(AngleAxisd(delta_rotationY,Vector3d::UnitY()));
        Eigen::AngleAxisd yawAngle(AngleAxisd(delta_rotationZ,Vector3d::UnitZ()));
//
        Eigen::Matrix3d R_delta,R_global_tmp;// k-1--->k
        R_delta = yawAngle*pitchAngle*rollAngle;

        // 当前时刻的全局姿态=上一时刻的全局姿态矩阵*姿态增量矩阵
        R_global_tmp = R_global[imuPointerBack]*R_delta;// R_k_1初始值为R0
        R_global[imuPointerLast] = R_global_tmp;
    }

    void imuHandler(const sensor_msgs::Imu::ConstPtr& imuIn) {

        // 判断是跑九轴杭州（nine_axis = true）,还是六轴松山湖(nine_axis = false)
        bool nine_axis = true;
        if (nine_axis) {
            imuCount++;
            /*-----------------9 axis for hangzhou_big-----------------*/
            double roll, pitch, yaw;
            tf::Quaternion orientation;
            tf::quaternionMsgToTF(imuIn->orientation, orientation);
            tf::Matrix3x3(orientation).getRPY(roll, pitch, yaw);

//            std::cout << "imu roll:        " <<  yaw << std::endl;
//            std::cout << "imu pitch:       " <<  roll << std::endl;
//            std::cout << "imu yaw:         " <<  pitch << std::endl;


            // 创建一个虚拟IMU,将杭州IMU右前上转变为velodyne imu的前左上,并为虚拟IMU赋值作为其传感器输入
            // creat virtual IMU for camera,virtual IMU is velodyneimu as FrontLeftUP
//            Eigen::Vector3d linear_acceleration_realimu,linear_acceleration_pandar,linear_acceleration_virtualimu;
//            linear_acceleration_realimu << imuIn->linear_acceleration.x, imuIn->linear_acceleration.y,imuIn->linear_acceleration.z;
//            linear_acceleration_pandar = R_i2p*linear_acceleration_realimu;
//            linear_acceleration_virtualimu = R_p2v*linear_acceleration_pandar;
//            float linear_accelerationx = linear_acceleration_virtualimu[0];
//            float linear_accelerationy = linear_acceleration_virtualimu[1];
//            float linear_accelerationz = linear_acceleration_virtualimu[2];
//
//            Eigen::Vector3d angular_velocity_realimu,angular_velocity_pandar,angular_velocity_virtualimu;
//            angular_velocity_realimu << imuIn->linear_acceleration.x, imuIn->linear_acceleration.y,imuIn->linear_acceleration.z;
//            angular_velocity_pandar = R_i2p*angular_velocity_realimu;
//            angular_velocity_virtualimu = R_p2v*angular_velocity_pandar;
//            float angular_velocityx = angular_velocity_virtualimu[0];
//            float angular_velocityy = angular_velocity_virtualimu[1];
//            float angular_velocityz = angular_velocity_virtualimu[2];
//
//            Eigen::Matrix3d  mat_realimu, mat_pandar, mat_virtualimu;
//            Eigen::AngleAxisd rollAngle(AngleAxisd(roll,Vector3d::UnitX()));
//            Eigen::AngleAxisd pitchAngle(AngleAxisd(pitch,Vector3d::UnitY()));
//            Eigen::AngleAxisd yawAngle(AngleAxisd(yaw,Vector3d::UnitZ()));
//            mat_realimu = yawAngle*pitchAngle*rollAngle;
//            mat_pandar =  mat_realimu*R_p2i;
//            mat_virtualimu = mat_pandar*R_v2p;
//
//            Eigen::Quaterniond quaternion_virtualimu;
//            quaternion_virtualimu=mat_virtualimu;
//            toEulerAngle(quaternion_virtualimu,r//            std::cout << "l_i2p:        " <<  l_i2p << std::endl;
//            std::cout << "deltaShift:   " <<  deltaShift << std::endl;
//            std::cout << "deltaVelo:    " <<  deltaVelo << std::endl;oll,pitch,yaw);

            // make the virtual acc as the input, imuIn->linear_acceleration

//            // 若 IMU和pandar 为华为标定结果，那么虚拟IMU和实际的关系： x =y, y= -x
//            R_i2p << -1,0,0,0,-1,0,0,0,1;
//            R_p2i = R_i2p.transpose();
//            float linear_accelerationx = imuIn->linear_acceleration.y;
//            float linear_accelerationy = -imuIn->linear_acceleration.x;
//            float linear_accelerationz = imuIn->linear_acceleration.z;
//
//            float angular_velocityx = imuIn->angular_velocity.y;
//            float angular_velocityy = -imuIn->angular_velocity.x;
//            float angular_velocityz = imuIn->angular_velocity.z;
//
//            double rx = roll;
//            double ry = pitch;
//            double rz = yaw;
//
//            roll = ry;
//            pitch = -rx;
//            yaw = rz;


//            // 若 IMU和pandar 为朱标定结果，那么虚拟IMU和实际的关系： x = -y, y= x
            R_i2p << 1,0,0,0,1,0,0,0,1;
            R_p2i = R_i2p.transpose();
            float linear_accelerationx = -imuIn->linear_acceleration.y;
            float linear_accelerationy = imuIn->linear_acceleration.x;
            float linear_accelerationz = imuIn->linear_acceleration.z;

            float angular_velocityx = -imuIn->angular_velocity.y;
            float angular_velocityy = imuIn->angular_velocity.x;
            float angular_velocityz = imuIn->angular_velocity.z;

            double rx = roll;
            double ry = pitch;
            double rz = yaw;

            roll = -ry;
            pitch = rx;
            yaw = rz;

//             // 若 IMU和pandar 为张晓东判定结果，那么虚拟IMU和实际的关系： x =-y, y= -z, z=x
//            R_i2p << 0,0,-1,0,1,0,1,0,0;
//            R_p2i = R_i2p.transpose();
//
//            float linear_accelerationx = -imuIn->linear_acceleration.y;
//            float linear_accelerationy = -imuIn->linear_acceleration.z;
//            float linear_accelerationz = imuIn->linear_acceleration.x;
//
//            float angular_velocityx = -imuIn->angular_velocity.y;
//            float angular_velocityy = -imuIn->angular_velocity.z;
//            float angular_velocityz = imuIn->angular_velocity.x;
//
//            double rx = roll;
//            double ry = pitch;
//            double rz = yaw;
//
//            roll = -ry;
//            pitch = -rz;
//            yaw = rx;



//            std::cout << "camera roll:        " <<  yaw << std::endl;
//            std::cout << "camera pitch:       " <<  roll << std::endl;
//            std::cout << "camera yaw:         " <<  pitch << std::endl;


//          // 虚拟IMU输入转换后得到camera的加速度
            // virtual IMU to camera,the same as orginal
            float accX = linear_accelerationy;
            float accY = linear_accelerationz;
            float accZ = linear_accelerationx;

            imuPointerLast = (imuPointerLast + 1) % imuQueLength;
            imuTime[imuPointerLast] = imuIn->header.stamp.toSec();

            realimuRoll[imuPointerLast] = roll;
            realimuPitch[imuPointerLast] = pitch;
            realimuYaw[imuPointerLast] = yaw;

            imuAccX[imuPointerLast] = accX;
            imuAccY[imuPointerLast] = accY;
            imuAccZ[imuPointerLast] = accZ;

            imuAngularVeloX[imuPointerLast] = angular_velocityx;
            imuAngularVeloY[imuPointerLast] = angular_velocityy;
            imuAngularVeloZ[imuPointerLast] = angular_velocityz;

//            imuQuat1 = imuIn->orientation;
//            imuAccX1[imuPointerLast] = imuIn->linear_acceleration.x;
//            imuAccY1[imuPointerLast] = imuIn->linear_acceleration.y;
//            imuAccZ1[imuPointerLast] = imuIn->linear_acceleration.z;
//            AccumulaterealIMUShiftAndRotation();

//            publishimuPath1();

            // 加速度，角速度积分得到不含补偿杆臂的camera全局位置，以及虚拟IMU的全局姿态
            AccumulateIMUShiftAndRotation();

            // 创建一个与imu同原点的坐标系为camera 的IMU,用于杆臂补偿
            // real imu to camera imu = R_p2c * R_i2p
            // 根据角速度求解反对称矩阵，为求解杆臂做准备,
            // 虚拟IMU 和camera 的对应关系为xyz-->zxy
            w_x[imuPointerLast] << 0, -imuAngularVeloX[imuPointerLast], imuAngularVeloZ[imuPointerLast],
                    imuAngularVeloX[imuPointerLast], 0, -imuAngularVeloY[imuPointerLast],
                    -imuAngularVeloZ[imuPointerLast], imuAngularVeloY[imuPointerLast], 0;

            // 与IMU原点重合的camera全局姿态
            Eigen::Matrix3d Rx, Rz, Ry;
            Rx << 1, 0, 0, 0, cos(pitch), -sin(pitch), 0, sin(pitch), cos(pitch);
            Rz << cos(roll), -sin(roll), 0, sin(roll), cos(roll), 0, 0, 0, 1;
            Ry << cos(yaw), 0, sin(yaw), 0, 1, 0, -sin(yaw), 0, cos(yaw);
            R_c2w[imuPointerLast] = Ry * Rx * Rz;

            Eigen::Vector3d delta_l_camera;
            delta_l_camera =  R_p2c*R_i2p*R_p2i*(l_i2p);
//            delta_l_camera =  R_p2c*(l_i2p);
//            std::cout << "delta_l_camera:   " <<  delta_l_camera << std::endl;

            realimuShift[imuPointerLast]
                    << realimuShiftX[imuPointerLast], realimuShiftY[imuPointerLast], realimuShiftZ[imuPointerLast];
            realimuVelo[imuPointerLast]
                    << realimuVeloX[imuPointerLast], realimuVeloY[imuPointerLast], realimuVeloZ[imuPointerLast];

            // 杆臂补偿包括：速度补偿，位移补偿
            // 速度补偿 = R_iw *((wx)*l_li)
            // 位置补偿 = R_iw *（l_li)
            bool compensation = true;
            Eigen::Vector3d deltaShift = R_c2w[imuPointerLast] * delta_l_camera;
            Eigen::Vector3d deltaVelo = R_c2w[imuPointerLast] * (w_x[imuPointerLast] * delta_l_camera);
//            std::cout << "l_i2p:        " <<  l_i2p << std::endl;
//            std::cout << "deltaShift:   " <<  deltaShift << std::endl;
//            std::cout << "deltaVelo:    " <<  deltaVelo << std::endl;

            if (compensation) {
                // 速度位移杆臂补偿
                imuShift[imuPointerLast] = realimuShift[imuPointerLast] + deltaShift;
                imuVelo[imuPointerLast] = realimuVelo[imuPointerLast] + deltaVelo;
            } else {
                imuShift[imuPointerLast] = realimuShift[imuPointerLast];
                imuVelo[imuPointerLast] = realimuVelo[imuPointerLast];
            }

            // 输出原本程序对应名称的camera全局变量
            imuShiftX[imuPointerLast] = imuShift[imuPointerLast][0];
            imuShiftY[imuPointerLast] = imuShift[imuPointerLast][1];
            imuShiftZ[imuPointerLast] = imuShift[imuPointerLast][2];

            imuVeloX[imuPointerLast] = imuVelo[imuPointerLast][0];
            imuVeloY[imuPointerLast] = imuVelo[imuPointerLast][1];
            imuVeloZ[imuPointerLast] = imuVelo[imuPointerLast][2];

            imuAngularRotationX[imuPointerLast] = realimuAngularRotationX[imuPointerLast];
            imuAngularRotationY[imuPointerLast] = realimuAngularRotationY[imuPointerLast];
            imuAngularRotationZ[imuPointerLast] = realimuAngularRotationZ[imuPointerLast];

            imuRoll[imuPointerLast] = realimuRoll[imuPointerLast];
            imuPitch[imuPointerLast] = realimuPitch[imuPointerLast];
            imuYaw[imuPointerLast] = realimuYaw[imuPointerLast];

//        /*--------------------------------end--------------------------------*/

        } else {
            /*--------------------------6 axis for songshanhu----------------------*/
            // 根据是否有初始静止求解初始姿态，若有静止则初始姿态根据方程求解，若无则全局姿态初始为0
            // if the vehicle has been static for a while, we can calculate the initial pose
            bool initial_static = false;
            double roll, pitch, yaw;

            // 松山湖, pandar 安装为左后上，imu 为前左上，velodyne为前左上，camera为左上前，数据为lidar to imu
            l_i2p << -0.080865451, 1.637948593, -1.309935873;
            //       R_p2i <<  0.036298090009,-0.999333605431,-0.003846260277,0.999326656940,0.036276674109,0.005498694811,-0.005355500980,-0.004043262543,0.999977485065;
            R_p2i << 0, -1, 0, 1, 0, 0, 0, 0, 1;
            R_i2p = R_p2i.transpose();
            l_p2i << 1.63475398, 0.028594611, 1.316095961;
            R_p2c << 1, 0, 0, 0, 0, 1, 0, -1, 0;
            R_v2p << 0, 1, 0, -1, 0, 0, 0, 0, 1;

            R_veloimuglobal_initial << 1, 0, 0, 0, 1, 0, 0, 0, 1;
            if (imuPointerLast == -1) {
                // 是否为第一帧IMU
                R_global[0] = R_veloimuglobal_initial;
                roll = 0;
                pitch = 0;
                yaw = 0;
            } else {

//                std::cout << "imuPointerLast_afterinitial---" << imuPointerLast << std::endl;
                Eigen::Quaterniond quaternion_global;
                quaternion_global = R_global[imuPointerLast];
                toEulerAngle(quaternion_global, roll, pitch, yaw);
            }

            // 原始IMU加速度去除重力影响，得到各个轴上的数值
            float accX = imuIn->linear_acceleration.x + sin(pitch) * 9.805;
            float accY = imuIn->linear_acceleration.y - sin(roll) * cos(pitch) * 9.805;
            float accZ = imuIn->linear_acceleration.z - cos(roll) * cos(pitch) * 9.805;

            float linear_accelerationx = accX;
            float linear_accelerationy = accY;
            float linear_accelerationz = accZ;

            float angular_velocityx = imuIn->angular_velocity.x;
            float angular_velocityy = imuIn->angular_velocity.y;
            float angular_velocityz = imuIn->angular_velocity.z;

            R_p2i << 0, -1, 0, 1, 0, 0, 0, 0, 1;
            R_i2p = R_p2i.transpose();

            // velodyne imu（前左上）下加速度进行轴变换得到camera （左上前）下的加速度，角速度，姿态不变
            // virtual IMU(FLU) to camera(LUF),
            accX = linear_accelerationy;
            accY = linear_accelerationz;
            accZ = linear_accelerationx;

            imuPointerLast = (imuPointerLast + 1) % imuQueLength;
            imuTime[imuPointerLast] = imuIn->header.stamp.toSec();

            realimuRoll[imuPointerLast] = roll;
            realimuPitch[imuPointerLast] = pitch;
            realimuYaw[imuPointerLast] = yaw;

            imuAccX[imuPointerLast] = accX;
            imuAccY[imuPointerLast] = accY;
            imuAccZ[imuPointerLast] = accZ;

            imuAngularVeloX[imuPointerLast] = angular_velocityx;
            imuAngularVeloY[imuPointerLast] = angular_velocityy;
            imuAngularVeloZ[imuPointerLast] = angular_velocityz;

            // 加速度，角速度积分得到不含补偿杆臂的camera全局位置，以及虚拟IMU的全局姿态
            AccumulateIMUShiftAndRotation();

            // 根据角速度求解反对称矩阵，为求解杆臂做准备
            // 虚拟IMU 和camera 的对应关系为xyz-->zxy
            w_x[imuPointerLast] << 0, -imuAngularVeloZ[imuPointerLast], imuAngularVeloY[imuPointerLast],
                    imuAngularVeloZ[imuPointerLast], 0, -imuAngularVeloX[imuPointerLast],
                    -imuAngularVeloY[imuPointerLast], imuAngularVeloX[imuPointerLast], 0;

            // 与IMU原点重合的camera全局姿态
            Eigen::Matrix3d Rx, Rz, Ry;
            Rx << 1, 0, 0, 0, cos(pitch), -sin(pitch), 0, sin(pitch), cos(pitch);
            Rz << cos(roll), -sin(roll), 0, sin(roll), cos(roll), 0, 0, 0, 1;
            Ry << cos(yaw), 0, sin(yaw), 0, 1, 0, -sin(yaw), 0, cos(yaw);
            R_c2w[imuPointerLast] = Ry * Rx * Rz;

            Eigen::Vector3d delta_l_camera;
            delta_l_camera = R_p2c * R_i2p * R_p2i * (l_i2p);

            realimuShift[imuPointerLast]
                    << realimuShiftX[imuPointerLast], realimuShiftY[imuPointerLast], realimuShiftZ[imuPointerLast];
            realimuVelo[imuPointerLast]
                    << realimuVeloX[imuPointerLast], realimuVeloY[imuPointerLast], realimuVeloZ[imuPointerLast];

            // 杆臂补偿包括：速度补偿，位移补偿
            // 速度补偿 = R_iw *((wx)*l_li)
            // 位置补偿 = R_iw *（l_li)
            bool compensation = true;
            Eigen::Vector3d deltaShift = R_c2w[imuPointerLast] * delta_l_camera;
            Eigen::Vector3d deltaVelo = R_c2w[imuPointerLast] * (w_x[imuPointerLast] * delta_l_camera);

            if (compensation) {
                // 速度位移杆臂补偿
                imuShift[imuPointerLast] = realimuShift[imuPointerLast] + deltaShift;
                imuVelo[imuPointerLast] = realimuVelo[imuPointerLast] + deltaVelo;
            } else {
                imuShift[imuPointerLast] = realimuShift[imuPointerLast];
                imuVelo[imuPointerLast] = realimuVelo[imuPointerLast];
            }

            // 输出原本程序对应名称的camera全局变量
            imuShiftX[imuPointerLast] = imuShift[imuPointerLast][0];
            imuShiftY[imuPointerLast] = imuShift[imuPointerLast][1];
            imuShiftZ[imuPointerLast] = imuShift[imuPointerLast][2];

            imuVeloX[imuPointerLast] = imuVelo[imuPointerLast][0];
            imuVeloY[imuPointerLast] = imuVelo[imuPointerLast][1];
            imuVeloZ[imuPointerLast] = imuVelo[imuPointerLast][2];

            imuAngularRotationX[imuPointerLast] = realimuAngularRotationX[imuPointerLast];
            imuAngularRotationY[imuPointerLast] = realimuAngularRotationY[imuPointerLast];
            imuAngularRotationZ[imuPointerLast] = realimuAngularRotationZ[imuPointerLast];

            imuRoll[imuPointerLast] = realimuRoll[imuPointerLast];
            imuPitch[imuPointerLast] = realimuPitch[imuPointerLast];
            imuYaw[imuPointerLast] = realimuYaw[imuPointerLast];
        }
        // 发布 IMU path
//        publishimuPath();
//
    }

    void laserCloudHandler(const sensor_msgs::PointCloud2ConstPtr& laserCloudMsg){
        laserscannum++;
        cloudHeader = laserCloudMsg->header;

        timeScanCur = cloudHeader.stamp.toSec();
        timeNewSegmentedCloud = timeScanCur;

        segmentedCloud->clear();
        pcl::fromROSMsg(*laserCloudMsg, *segmentedCloud);

        newSegmentedCloud = true;
    }

    void outlierCloudHandler(const sensor_msgs::PointCloud2ConstPtr& msgIn){

        timeNewOutlierCloud = msgIn->header.stamp.toSec();

        outlierCloud->clear();
        pcl::fromROSMsg(*msgIn, *outlierCloud);

        newOutlierCloud = true;
    }

    void laserCloudInfoHandler(const cloud_msgs::cloud_infoConstPtr& msgIn)
    {
        timeNewSegmentedCloudInfo = msgIn->header.stamp.toSec();
        segInfo = *msgIn;
        newSegmentedCloudInfo = true;
    }

    void adjustDistortion()
    {
        // 输出 未去除非匀速畸变的点云
        // 需要坐标变换，由pandar变为velodyne系，但如果在这里转换，程序速度会很慢，影响建图
        // 放慢速度后，由于IMU需要根据rostime打时间戳，运算结果会有影响
        // 如果不做坐标变换，直接在cloudcompare中对齐看，终止程序后，同一个函数保存的两帧点云pcd不完全一致
//        pubCloudBeforeAdjust ();
        pcl::io::savePCDFileASCII(fileDirectory+"cloudBeforeAdjust.pcd", *segmentedCloud);

        bool halfPassed = false;
        int cloudSize = segmentedCloud->points.size();

        PointType point;

        float tempx, tempy, tempz;

//        std::cout << "laserscannum: " << laserscannum  << std::endl;
//        std::cout << fixed<<setprecision(9) <<"laserscantime" << timeScanCur << std::endl;
//        std::cout << fixed<<setprecision(9) << "imuTime[imuPointerLast]: " << imuTime[imuPointerLast] << std::endl;

        for (int i = 0; i < cloudSize; i++) {

            tempx = segmentedCloud->points[i].x;
            tempy = segmentedCloud->points[i].y;
            tempz = segmentedCloud->points[i].z;

            segmentedCloud->points[i].x = -tempy;
            segmentedCloud->points[i].y = tempx;
            segmentedCloud->points[i].z = tempz;

            point.x = segmentedCloud->points[i].y;
            point.y = segmentedCloud->points[i].z;
            point.z = segmentedCloud->points[i].x;


            // *** correct the piont ori  --start *****
            // float ori = atan2(tempy, tempx);
            // float relTime;
            // if (ori< segInfo.startOrientation) {
            //     relTime = (segInfo.startOrientation - ori ) / segInfo.orientationDiff;
            // } else {
            //     relTime = (2 * M_PI + segInfo.startOrientation - ori ) / segInfo.orientationDiff;
            // }
            // std::cout << "lasernumber:   " << i << "    relTime:   " << relTime << std::endl;
            // **** end ******

            float ori = -atan2(point.x, point.z);
            if (!halfPassed) {
                if (ori < segInfo.startOrientation - M_PI / 2)
                    ori += 2 * M_PI;
                else if (ori > segInfo.startOrientation + M_PI * 3 / 2)
                    ori -= 2 * M_PI;

                if (ori - segInfo.startOrientation > M_PI)
                    halfPassed = true;
            } else {
                ori += 2 * M_PI;

                if (ori < segInfo.endOrientation - M_PI * 3 / 2)
                    ori += 2 * M_PI;
                else if (ori > segInfo.endOrientation + M_PI / 2)
                    ori -= 2 * M_PI;
            }

            float relTime = (ori - segInfo.startOrientation) / segInfo.orientationDiff;
            point.intensity = int(segmentedCloud->points[i].intensity) + scanPeriod * relTime;

            if (imuPointerLast >= 0) {
                float pointTime = relTime * scanPeriod;
                imuPointerFront = imuPointerLastIteration;
                while (imuPointerFront != imuPointerLast) {
                    if (timeScanCur + pointTime < imuTime[imuPointerFront]) {
                        break;
                    }
                    imuPointerFront = (imuPointerFront + 1) % imuQueLength;
                }
//                float lasertime = timeScanCur + pointTime;
//                std::cout << "lasertime" << lasertime << std::endl;
//                std::cout << "imuTime[imuPointerFront]: " << imuTime[imuPointerFront] << std::endl;

                if (timeScanCur + pointTime > imuTime[imuPointerFront]) {
                    imuRollCur = imuRoll[imuPointerFront];
                    imuPitchCur = imuPitch[imuPointerFront];
                    imuYawCur = imuYaw[imuPointerFront];

                    imuVeloXCur = imuVeloX[imuPointerFront];
                    imuVeloYCur = imuVeloY[imuPointerFront];
                    imuVeloZCur = imuVeloZ[imuPointerFront];

                    imuShiftXCur = imuShiftX[imuPointerFront];
                    imuShiftYCur = imuShiftY[imuPointerFront];
                    imuShiftZCur = imuShiftZ[imuPointerFront];   
                } else {
                    int imuPointerBack = (imuPointerFront + imuQueLength - 1) % imuQueLength;
                    float ratioFront = (timeScanCur + pointTime - imuTime[imuPointerBack]) 
                                                     / (imuTime[imuPointerFront] - imuTime[imuPointerBack]);
                    float ratioBack = (imuTime[imuPointerFront] - timeScanCur - pointTime) 
                                                    / (imuTime[imuPointerFront] - imuTime[imuPointerBack]);

                    imuRollCur = imuRoll[imuPointerFront] * ratioFront + imuRoll[imuPointerBack] * ratioBack;
                    imuPitchCur = imuPitch[imuPointerFront] * ratioFront + imuPitch[imuPointerBack] * ratioBack;
                    if (imuYaw[imuPointerFront] - imuYaw[imuPointerBack] > M_PI) {
                        imuYawCur = imuYaw[imuPointerFront] * ratioFront + (imuYaw[imuPointerBack] + 2 * M_PI) * ratioBack;
                    } else if (imuYaw[imuPointerFront] - imuYaw[imuPointerBack] < -M_PI) {
                        imuYawCur = imuYaw[imuPointerFront] * ratioFront + (imuYaw[imuPointerBack] - 2 * M_PI) * ratioBack;
                    } else {
                        imuYawCur = imuYaw[imuPointerFront] * ratioFront + imuYaw[imuPointerBack] * ratioBack;
                    }

                    imuVeloXCur = imuVeloX[imuPointerFront] * ratioFront + imuVeloX[imuPointerBack] * ratioBack;
                    imuVeloYCur = imuVeloY[imuPointerFront] * ratioFront + imuVeloY[imuPointerBack] * ratioBack;
                    imuVeloZCur = imuVeloZ[imuPointerFront] * ratioFront + imuVeloZ[imuPointerBack] * ratioBack;

                    imuShiftXCur = imuShiftX[imuPointerFront] * ratioFront + imuShiftX[imuPointerBack] * ratioBack;
                    imuShiftYCur = imuShiftY[imuPointerFront] * ratioFront + imuShiftY[imuPointerBack] * ratioBack;
                    imuShiftZCur = imuShiftZ[imuPointerFront] * ratioFront + imuShiftZ[imuPointerBack] * ratioBack;
                }

                if (i == 0) {
                    imuRollStart = imuRollCur;
                    imuPitchStart = imuPitchCur;
                    imuYawStart = imuYawCur;

                    imuVeloXStart = imuVeloXCur;
                    imuVeloYStart = imuVeloYCur;
                    imuVeloZStart = imuVeloZCur;

                    imuShiftXStart = imuShiftXCur;
                    imuShiftYStart = imuShiftYCur;
                    imuShiftZStart = imuShiftZCur;

                    if (timeScanCur + pointTime > imuTime[imuPointerFront]) {
                        imuAngularRotationXCur = imuAngularRotationX[imuPointerFront];
                        imuAngularRotationYCur = imuAngularRotationY[imuPointerFront];
                        imuAngularRotationZCur = imuAngularRotationZ[imuPointerFront];
                    }else{
                        int imuPointerBack = (imuPointerFront + imuQueLength - 1) % imuQueLength;
                        float ratioFront = (timeScanCur + pointTime - imuTime[imuPointerBack]) 
                                                         / (imuTime[imuPointerFront] - imuTime[imuPointerBack]);
                        float ratioBack = (imuTime[imuPointerFront] - timeScanCur - pointTime) 
                                                        / (imuTime[imuPointerFront] - imuTime[imuPointerBack]);
                        imuAngularRotationXCur = imuAngularRotationX[imuPointerFront] * ratioFront + imuAngularRotationX[imuPointerBack] * ratioBack;
                        imuAngularRotationYCur = imuAngularRotationY[imuPointerFront] * ratioFront + imuAngularRotationY[imuPointerBack] * ratioBack;
                        imuAngularRotationZCur = imuAngularRotationZ[imuPointerFront] * ratioFront + imuAngularRotationZ[imuPointerBack] * ratioBack;
                    }

                    imuAngularFromStartX = imuAngularRotationXCur - imuAngularRotationXLast;
                    imuAngularFromStartY = imuAngularRotationYCur - imuAngularRotationYLast;
                    imuAngularFromStartZ = imuAngularRotationZCur - imuAngularRotationZLast;

                    imuAngularRotationXLast = imuAngularRotationXCur;
                    imuAngularRotationYLast = imuAngularRotationYCur;
                    imuAngularRotationZLast = imuAngularRotationZCur;

                    updateImuRollPitchYawStartSinCos();
                } else {
                    VeloToStartIMU();
                    TransformToStartIMU(&point);
                }
            }
            segmentedCloud->points[i] = point;
        }
        pcl::io::savePCDFileASCII(fileDirectory+"cloudAfterAdjust.pcd", *segmentedCloud);
//        pcl::visualization::CloudViewer viewer();
//        viewer.showCloud(segmentedCloud);

        imuPointerLastIteration = imuPointerLast;
    }

    void calculateSmoothness()
    {
        int cloudSize = segmentedCloud->points.size();
        for (int i = 5; i < cloudSize - 5; i++) {

            float diffRange = segInfo.segmentedCloudRange[i-5] + segInfo.segmentedCloudRange[i-4]
                            + segInfo.segmentedCloudRange[i-3] + segInfo.segmentedCloudRange[i-2]
                            + segInfo.segmentedCloudRange[i-1] - segInfo.segmentedCloudRange[i] * 10
                            + segInfo.segmentedCloudRange[i+1] + segInfo.segmentedCloudRange[i+2]
                            + segInfo.segmentedCloudRange[i+3] + segInfo.segmentedCloudRange[i+4]
                            + segInfo.segmentedCloudRange[i+5];            

            cloudCurvature[i] = diffRange*diffRange;

            cloudNeighborPicked[i] = 0;
            cloudLabel[i] = 0;

            cloudSmoothness[i].value = cloudCurvature[i];
            cloudSmoothness[i].ind = i;
        }
    }

    void markOccludedPoints()
    {
        int cloudSize = segmentedCloud->points.size();

        for (int i = 5; i < cloudSize - 6; ++i){

            float depth1 = segInfo.segmentedCloudRange[i];
            float depth2 = segInfo.segmentedCloudRange[i+1];
            int columnDiff = std::abs(int(segInfo.segmentedCloudColInd[i+1] - segInfo.segmentedCloudColInd[i]));

            if (columnDiff < 10){

                if (depth1 - depth2 > 0.3){
                    cloudNeighborPicked[i - 5] = 1;
                    cloudNeighborPicked[i - 4] = 1;
                    cloudNeighborPicked[i - 3] = 1;
                    cloudNeighborPicked[i - 2] = 1;
                    cloudNeighborPicked[i - 1] = 1;
                    cloudNeighborPicked[i] = 1;
                }else if (depth2 - depth1 > 0.3){
                    cloudNeighborPicked[i + 1] = 1;
                    cloudNeighborPicked[i + 2] = 1;
                    cloudNeighborPicked[i + 3] = 1;
                    cloudNeighborPicked[i + 4] = 1;
                    cloudNeighborPicked[i + 5] = 1;
                    cloudNeighborPicked[i + 6] = 1;
                }
            }

            float diff1 = std::abs(float(segInfo.segmentedCloudRange[i-1] - segInfo.segmentedCloudRange[i]));
            float diff2 = std::abs(float(segInfo.segmentedCloudRange[i+1] - segInfo.segmentedCloudRange[i]));

            if (diff1 > 0.02 * segInfo.segmentedCloudRange[i] && diff2 > 0.02 * segInfo.segmentedCloudRange[i])
                cloudNeighborPicked[i] = 1;
        }
    }

    void extractFeatures()
    {
        cornerPointsSharp->clear();
        cornerPointsLessSharp->clear();
        surfPointsFlat->clear();
        surfPointsLessFlat->clear();

        for (int i = 0; i < N_SCAN; i++) {

            surfPointsLessFlatScan->clear();

            for (int j = 0; j < 6; j++) {

                int sp = (segInfo.startRingIndex[i] * (6 - j)    + segInfo.endRingIndex[i] * j) / 6;
                int ep = (segInfo.startRingIndex[i] * (5 - j)    + segInfo.endRingIndex[i] * (j + 1)) / 6 - 1;

                if (sp >= ep)
                    continue;

                std::sort(cloudSmoothness.begin()+sp, cloudSmoothness.begin()+ep, by_value());

                int largestPickedNum = 0;
                for (int k = ep; k >= sp; k--) {
                    int ind = cloudSmoothness[k].ind;
                    if (cloudNeighborPicked[ind] == 0 &&
                        cloudCurvature[ind] > edgeThreshold &&
                        segInfo.segmentedCloudGroundFlag[ind] == false) {
                    
                        largestPickedNum++;
                        if (largestPickedNum <= 2) {
                            cloudLabel[ind] = 2;
                            cornerPointsSharp->push_back(segmentedCloud->points[ind]);
                            cornerPointsLessSharp->push_back(segmentedCloud->points[ind]);
                        } else if (largestPickedNum <= 20) {
                            cloudLabel[ind] = 1;
                            cornerPointsLessSharp->push_back(segmentedCloud->points[ind]);
                        } else {
                            break;
                        }

                        cloudNeighborPicked[ind] = 1;
                        for (int l = 1; l <= 5; l++) {
                            int columnDiff = std::abs(int(segInfo.segmentedCloudColInd[ind + l] - segInfo.segmentedCloudColInd[ind + l - 1]));
                            if (columnDiff > 10)
                                break;
                            cloudNeighborPicked[ind + l] = 1;
                        }
                        for (int l = -1; l >= -5; l--) {
                            int columnDiff = std::abs(int(segInfo.segmentedCloudColInd[ind + l] - segInfo.segmentedCloudColInd[ind + l + 1]));
                            if (columnDiff > 10)
                                break;
                            cloudNeighborPicked[ind + l] = 1;
                        }
                    }
                }

                int smallestPickedNum = 0;
                for (int k = sp; k <= ep; k++) {
                    int ind = cloudSmoothness[k].ind;
                    if (cloudNeighborPicked[ind] == 0 &&
                        cloudCurvature[ind] < surfThreshold &&
                        segInfo.segmentedCloudGroundFlag[ind] == true) {

                        cloudLabel[ind] = -1;
                        surfPointsFlat->push_back(segmentedCloud->points[ind]);

                        smallestPickedNum++;
                        if (smallestPickedNum >= 4) {
                            break;
                        }

                        cloudNeighborPicked[ind] = 1;
                        for (int l = 1; l <= 5; l++) {

                            int columnDiff = std::abs(int(segInfo.segmentedCloudColInd[ind + l] - segInfo.segmentedCloudColInd[ind + l - 1]));
                            if (columnDiff > 10)
                                break;

                            cloudNeighborPicked[ind + l] = 1;
                        }
                        for (int l = -1; l >= -5; l--) {

                            int columnDiff = std::abs(int(segInfo.segmentedCloudColInd[ind + l] - segInfo.segmentedCloudColInd[ind + l + 1]));
                            if (columnDiff > 10)
                                break;

                            cloudNeighborPicked[ind + l] = 1;
                        }
                    }
                }

                for (int k = sp; k <= ep; k++) {
                    if (cloudLabel[k] <= 0) {
                        surfPointsLessFlatScan->push_back(segmentedCloud->points[k]);
                    }
                }
            }

            surfPointsLessFlatScanDS->clear();
            downSizeFilter.setInputCloud(surfPointsLessFlatScan);
            downSizeFilter.filter(*surfPointsLessFlatScanDS);

            *surfPointsLessFlat += *surfPointsLessFlatScanDS;
        }
    }

    void publishCloud()
    {
        sensor_msgs::PointCloud2 laserCloudOutMsg;

	    if (pubCornerPointsSharp.getNumSubscribers() != 0){
	        pcl::toROSMsg(*cornerPointsSharp, laserCloudOutMsg);
	        laserCloudOutMsg.header.stamp = cloudHeader.stamp;
	        laserCloudOutMsg.header.frame_id = "/camera";
	        pubCornerPointsSharp.publish(laserCloudOutMsg);
	    }

	    if (pubCornerPointsLessSharp.getNumSubscribers() != 0){
	        pcl::toROSMsg(*cornerPointsLessSharp, laserCloudOutMsg);
	        laserCloudOutMsg.header.stamp = cloudHeader.stamp;
	        laserCloudOutMsg.header.frame_id = "/camera";
	        pubCornerPointsLessSharp.publish(laserCloudOutMsg);
	    }

	    if (pubSurfPointsFlat.getNumSubscribers() != 0){
	        pcl::toROSMsg(*surfPointsFlat, laserCloudOutMsg);
	        laserCloudOutMsg.header.stamp = cloudHeader.stamp;
	        laserCloudOutMsg.header.frame_id = "/camera";
	        pubSurfPointsFlat.publish(laserCloudOutMsg);
	    }

	    if (pubSurfPointsLessFlat.getNumSubscribers() != 0){
	        pcl::toROSMsg(*surfPointsLessFlat, laserCloudOutMsg);
	        laserCloudOutMsg.header.stamp = cloudHeader.stamp;
	        laserCloudOutMsg.header.frame_id = "/camera";
	        pubSurfPointsLessFlat.publish(laserCloudOutMsg);
	    }
    }

    void publishimuOdom()
    {
        geometry_msgs::Quaternion imuQuat = tf::createQuaternionMsgFromRollPitchYaw(imuYaw[imuPointerLast],-imuRoll[imuPointerLast],imuPitch[imuPointerLast]);

        imu_odom.header.stamp = ros::Time::now();

        imu_odom.pose.pose.orientation.x = -imuQuat.y;
        imu_odom.pose.pose.orientation.y = -imuQuat.z;
        imu_odom.pose.pose.orientation.z = imuQuat.x;
        imu_odom.pose.pose.orientation.w = imuQuat.w;
        imu_odom.pose.pose.position.x = imuShiftX[imuPointerLast];
        imu_odom.pose.pose.position.y = imuShiftY[imuPointerLast];
        imu_odom.pose.pose.position.z = imuShiftZ[imuPointerLast];
        pubImuOdom.publish(imu_odom);

    }


    void publishimuPath1()
    {
        geometry_msgs::PoseStamped imu_path_stamped;

        imu_path_stamped.pose.orientation.x = imuQuat1.x;
        imu_path_stamped.pose.orientation.y = imuQuat1.y;
        imu_path_stamped.pose.orientation.z = imuQuat1.z;
        imu_path_stamped.pose.orientation.w = imuQuat1.w;

        imu_path_stamped.pose.position.x = realimuShiftX1[imuPointerLast];
        imu_path_stamped.pose.position.y = realimuShiftY1[imuPointerLast];
        imu_path_stamped.pose.position.z = realimuShiftZ1[imuPointerLast];

        imu_path_stamped.header.stamp = ros::Time::now();
        imu_path_stamped.header.frame_id = "/camera_init";
        imu_path.poses.push_back(imu_path_stamped);

        pubImuPath.publish(imu_path);

    }


//    void publishimuPath()
//    {
//        geometry_msgs::Quaternion imuQuat = tf::createQuaternionMsgFromRollPitchYaw(imuYaw[imuPointerLast],-imuRoll[imuPointerLast],imuPitch[imuPointerLast]);
//
//        geometry_msgs::PoseStamped imu_path_stamped;
//
//        imu_path_stamped.pose.orientation.x = -imuQuat.y;
//        imu_path_stamped.pose.orientation.y = -imuQuat.z;
//        imu_path_stamped.pose.orientation.z = imuQuat.x;
//        imu_path_stamped.pose.orientation.w = imuQuat.w;
//        imu_path_stamped.pose.position.x = imuShiftX[imuPointerLast];
//        imu_path_stamped.pose.position.y = imuShiftY[imuPointerLast];
//        imu_path_stamped.pose.position.z = imuShiftZ[imuPointerLast];
//
//
//        imu_path_stamped.header.stamp = ros::Time::now();
//        imu_path_stamped.header.frame_id = "/camera_init";
//        imu_path.poses.push_back(imu_path_stamped);
//
//        pubImuPath.publish(imu_path);
//
//    }

    void pubCloudBeforeAdjust ()
    {
        cloudBeforeAdjust->clear();

        int cloudSize = segmentedCloud->points.size();

        PointType point;

        float tempx, tempy, tempz;

        for (int i = 0; i < cloudSize; i++) {

            tempx = segmentedCloud->points[i].x;
            tempy = segmentedCloud->points[i].y;
            tempz = segmentedCloud->points[i].z;

            segmentedCloud->points[i].x = -tempy;
            segmentedCloud->points[i].y = tempx;
            segmentedCloud->points[i].z = tempz;

            point.x = segmentedCloud->points[i].y;
            point.y = segmentedCloud->points[i].z;
            point.z = segmentedCloud->points[i].x;

            cloudBeforeAdjust->push_back(point);
        }
//            downSizeFilterCorner.setInputCloud(cloudBeforeAjust);
//            downSizeFilterCorner.filter(*cloudBeforeAjustDS);
        pcl::io::savePCDFileASCII(fileDirectory+"cloudBeforeAjust.pcd", *cloudBeforeAdjust);
//        pcl::visualization::CloudViewer viewer();
//        viewer.showCloud(cloudBeforeAjust);

    };









































    // 将current转到start
    void TransformToStart(PointType const * const pi, PointType * const po)
    {
        // 有IMU时候，Pi位于camera的start坐标系，没有IMU时候 pi位于camera的current坐标系下，po位于camera的start坐标系
        // 所以此处存在若有IMU时候多转换一些角度
        float s = 10 * (pi->intensity - int(pi->intensity));
        std::cout << "s = " << s << std::endl;

        float rx = s * transformCur[0];
        float ry = s * transformCur[1];
        float rz = s * transformCur[2];
        float tx = s * transformCur[3];
        float ty = s * transformCur[4];
        float tz = s * transformCur[5];

        // po= s*R*(pi-s*t)
        // Pi 位于current下，去除畸变t，再变换s*R到start坐标系下
        // start frame: 绕y轴转yaw (ry) -- 绕x轴转pitch(rx) -- 绕z轴转roll (rz)--current frame (顺序与IMUroll,pitch,yaw角对应)
        float x1 = cos(rz) * (pi->x - tx) + sin(rz) * (pi->y - ty);
        float y1 = -sin(rz) * (pi->x - tx) + cos(rz) * (pi->y - ty);
        float z1 = (pi->z - tz);

        float x2 = x1;
        float y2 = cos(rx) * y1 + sin(rx) * z1;
        float z2 = -sin(rx) * y1 + cos(rx) * z1;

        po->x = cos(ry) * x2 - sin(ry) * z2;
        po->y = y2;
        po->z = sin(ry) * x2 + cos(ry) * z2;
        po->intensity = pi->intensity;
    }

    // // 先转到start，再从start旋转到end
    void TransformToEnd(PointType const * const pi, PointType * const po)
    {
        // current camera转到start camera，与TransformToStart函数一致
        float s = 10 * (pi->intensity - int(pi->intensity));

        float rx = s * transformCur[0];
        float ry = s * transformCur[1];
        float rz = s * transformCur[2];
        float tx = s * transformCur[3];
        float ty = s * transformCur[4];
        float tz = s * transformCur[5];

//        float vx = transformCur[3]/scanPeriod;
//        float vy = transformCur[4]/scanPeriod;
//        float vz = transformCur[5]/scanPeriod;
//
//        float v_toal_l =sqrt(vx*vx+vy*vy+vz*vz);
//
//        if (v_toal_l> v_max_l)
//        {
//            v_max_l = v_toal_l;
//        }
//        std::cout << "v_toal_l--- " << v_toal_l << std::endl;
//        std::cout << "v_max_l--- " << v_max_l << std::endl;

        float x1 = cos(rz) * (pi->x - tx) + sin(rz) * (pi->y - ty);
        float y1 = -sin(rz) * (pi->x - tx) + cos(rz) * (pi->y - ty);
        float z1 = (pi->z - tz);

        float x2 = x1;
        float y2 = cos(rx) * y1 + sin(rx) * z1;
        float z2 = -sin(rx) * y1 + cos(rx) * z1;

        float x3 = cos(ry) * x2 - sin(ry) * z2;
        float y3 = y2;
        float z3 = sin(ry) * x2 + cos(ry) * z2;

        // start转到end, pe = R_se*ps+t
        // start camera frame: 绕y轴转yaw (ry) -- 绕z轴转pitch(rz) -- 绕x轴转roll (rx)--current camera frame (顺序与IMUroll,pitch,yaw角对应)
        rx = transformCur[0];
        ry = transformCur[1];
        rz = transformCur[2];
        tx = transformCur[3];
        ty = transformCur[4];
        tz = transformCur[5];

        float x4 = cos(ry) * x3 + sin(ry) * z3;
        float y4 = y3;
        float z4 = -sin(ry) * x3 + cos(ry) * z3;

        float x5 = x4;
        float y5 = cos(rx) * y4 - sin(rx) * z4;
        float z5 = sin(rx) * y4 + cos(rx) * z4;

        float x6 = cos(rz) * x5 - sin(rz) * y5 + tx;
        float y6 = sin(rz) * x5 + cos(rz) * y5 + ty;
        float z6 = z5 + tz;

        float x7 = cosImuRollStart * (x6 - imuShiftFromStartX)
                 - sinImuRollStart * (y6 - imuShiftFromStartY);
        float y7 = sinImuRollStart * (x6 - imuShiftFromStartX)
                 + cosImuRollStart * (y6 - imuShiftFromStartY);
        float z7 = z6 - imuShiftFromStartZ;

        float x8 = x7;
        float y8 = cosImuPitchStart * y7 - sinImuPitchStart * z7;
        float z8 = sinImuPitchStart * y7 + cosImuPitchStart * z7;

        float x9 = cosImuYawStart * x8 + sinImuYawStart * z8;
        float y9 = y8;
        float z9 = -sinImuYawStart * x8 + cosImuYawStart * z8;

        // world坐标系转到end camera 坐标系下
        float x10 = cos(imuYawLast) * x9 - sin(imuYawLast) * z9;
        float y10 = y9;
        float z10 = sin(imuYawLast) * x9 + cos(imuYawLast) * z9;

        float x11 = x10;
        float y11 = cos(imuPitchLast) * y10 + sin(imuPitchLast) * z10;
        float z11 = -sin(imuPitchLast) * y10 + cos(imuPitchLast) * z10;

        po->x = cos(imuRollLast) * x11 + sin(imuRollLast) * y11;
        po->y = -sin(imuRollLast) * x11 + cos(imuRollLast) * y11;
        po->z = z11;
        po->intensity = int(pi->intensity);
    }

    // 与accumulateRotatio联合起来更新transformSum的rotation部分的工作
    // 可视为transformToEnd的下部分的逆过程
    void PluginIMURotation(float bcx, float bcy, float bcz, float blx, float bly, float blz, 
                           float alx, float aly, float alz, float &acx, float &acy, float &acz)
    {

        // bcx,bcy,bcz = rx, ry, rz,k+1时刻相对于第一帧的旋转, 来源于AccumulateRotation， transformSum + (-transformCur)
        // blx,bly,blz = imuPitchStart, imuYawStart, imuRollStart, camera start 与 global 姿态关系
        // alx,aly,alz = imuPitchLast, imuYawLast, imuRollLast, camera end 与 global 姿态关系

        float sbcx = sin(bcx);
        float cbcx = cos(bcx);
        float sbcy = sin(bcy);
        float cbcy = cos(bcy);
        float sbcz = sin(bcz);
        float cbcz = cos(bcz);

        float sblx = sin(blx);
        float cblx = cos(blx);
        float sbly = sin(bly);
        float cbly = cos(bly);
        float sblz = sin(blz);
        float cblz = cos(blz);

        float salx = sin(alx);
        float calx = cos(alx);
        float saly = sin(aly);
        float caly = cos(aly);
        float salz = sin(alz);
        float calz = cos(alz);

        float srx = -sbcx*(salx*sblx + calx*caly*cblx*cbly + calx*cblx*saly*sbly) 
                  - cbcx*cbcz*(calx*saly*(cbly*sblz - cblz*sblx*sbly) 
                  - calx*caly*(sbly*sblz + cbly*cblz*sblx) + cblx*cblz*salx) 
                  - cbcx*sbcz*(calx*caly*(cblz*sbly - cbly*sblx*sblz) 
                  - calx*saly*(cbly*cblz + sblx*sbly*sblz) + cblx*salx*sblz);
        acx = -asin(srx);

        float srycrx = (cbcy*sbcz - cbcz*sbcx*sbcy)*(calx*saly*(cbly*sblz - cblz*sblx*sbly) 
                     - calx*caly*(sbly*sblz + cbly*cblz*sblx) + cblx*cblz*salx) 
                     - (cbcy*cbcz + sbcx*sbcy*sbcz)*(calx*caly*(cblz*sbly - cbly*sblx*sblz) 
                     - calx*saly*(cbly*cblz + sblx*sbly*sblz) + cblx*salx*sblz) 
                     + cbcx*sbcy*(salx*sblx + calx*caly*cblx*cbly + calx*cblx*saly*sbly);
        float crycrx = (cbcz*sbcy - cbcy*sbcx*sbcz)*(calx*caly*(cblz*sbly - cbly*sblx*sblz) 
                     - calx*saly*(cbly*cblz + sblx*sbly*sblz) + cblx*salx*sblz) 
                     - (sbcy*sbcz + cbcy*cbcz*sbcx)*(calx*saly*(cbly*sblz - cblz*sblx*sbly) 
                     - calx*caly*(sbly*sblz + cbly*cblz*sblx) + cblx*cblz*salx) 
                     + cbcx*cbcy*(salx*sblx + calx*caly*cblx*cbly + calx*cblx*saly*sbly);
        acy = atan2(srycrx / cos(acx), crycrx / cos(acx));
        
        float srzcrx = sbcx*(cblx*cbly*(calz*saly - caly*salx*salz) 
                     - cblx*sbly*(caly*calz + salx*saly*salz) + calx*salz*sblx) 
                     - cbcx*cbcz*((caly*calz + salx*saly*salz)*(cbly*sblz - cblz*sblx*sbly) 
                     + (calz*saly - caly*salx*salz)*(sbly*sblz + cbly*cblz*sblx) 
                     - calx*cblx*cblz*salz) + cbcx*sbcz*((caly*calz + salx*saly*salz)*(cbly*cblz 
                     + sblx*sbly*sblz) + (calz*saly - caly*salx*salz)*(cblz*sbly - cbly*sblx*sblz) 
                     + calx*cblx*salz*sblz);
        float crzcrx = sbcx*(cblx*sbly*(caly*salz - calz*salx*saly) 
                     - cblx*cbly*(saly*salz + caly*calz*salx) + calx*calz*sblx) 
                     + cbcx*cbcz*((saly*salz + caly*calz*salx)*(sbly*sblz + cbly*cblz*sblx) 
                     + (caly*salz - calz*salx*saly)*(cbly*sblz - cblz*sblx*sbly) 
                     + calx*calz*cblx*cblz) - cbcx*sbcz*((saly*salz + caly*calz*salx)*(cblz*sbly 
                     - cbly*sblx*sblz) + (caly*salz - calz*salx*saly)*(cbly*cblz + sblx*sbly*sblz) 
                     - calx*calz*cblx*sblz);
        acz = atan2(srzcrx / cos(acx), crzcrx / cos(acx));
    }

    void AccumulateRotation(float cx, float cy, float cz, float lx, float ly, float lz, 
                            float &ox, float &oy, float &oz)
    {
        // cx,cy,cz = transformSum[0], transformSum[1], transformSum[2]
        // lx,ly,lz = -transformCur[0], -transformCur[1], -transformCur[2]
        // 0x,oy,oz = rx, ry, rz

        // p_cur_in_global=R_cw*p_cur_in_camera = R_sw*R_cs*p_cur_in_camera,即
        // k+1时刻相对于第一帧的旋转，R_cw =rx, ry, rz = R_sw*R_cs
        // 其中R_sw为初始时刻累积到上一时刻k的位姿，即0-->k，transformSum，R_cs为current到start的位姿，即k-->k+1，transformCur
        float srx = cos(lx)*cos(cx)*sin(ly)*sin(cz) - cos(cx)*cos(cz)*sin(lx) - cos(lx)*cos(ly)*sin(cx);
        ox = -asin(srx);

        float srycrx = sin(lx)*(cos(cy)*sin(cz) - cos(cz)*sin(cx)*sin(cy)) + cos(lx)*sin(ly)*(cos(cy)*cos(cz) 
                     + sin(cx)*sin(cy)*sin(cz)) + cos(lx)*cos(ly)*cos(cx)*sin(cy);
        float crycrx = cos(lx)*cos(ly)*cos(cx)*cos(cy) - cos(lx)*sin(ly)*(cos(cz)*sin(cy) 
                     - cos(cy)*sin(cx)*sin(cz)) - sin(lx)*(sin(cy)*sin(cz) + cos(cy)*cos(cz)*sin(cx));
        oy = atan2(srycrx / cos(ox), crycrx / cos(ox));

        float srzcrx = sin(cx)*(cos(lz)*sin(ly) - cos(ly)*sin(lx)*sin(lz)) + cos(cx)*sin(cz)*(cos(ly)*cos(lz) 
                     + sin(lx)*sin(ly)*sin(lz)) + cos(lx)*cos(cx)*cos(cz)*sin(lz);
        float crzcrx = cos(lx)*cos(lz)*cos(cx)*cos(cz) - cos(cx)*sin(cz)*(cos(ly)*sin(lz) 
                     - cos(lz)*sin(lx)*sin(ly)) - sin(cx)*(sin(ly)*sin(lz) + cos(ly)*cos(lz)*sin(lx));
        oz = atan2(srzcrx / cos(ox), crzcrx / cos(ox));
    }

    double rad2deg(double radians)
    {
        return radians * 180.0 / M_PI;
    }

    double deg2rad(double degrees)
    {
        return degrees * M_PI / 180.0;
    }

    void findCorrespondingCornerFeatures(int iterCount){

        int cornerPointsSharpNum = cornerPointsSharp->points.size();

        for (int i = 0; i < cornerPointsSharpNum; i++) {

            TransformToStart(&cornerPointsSharp->points[i], &pointSel);

            if (iterCount % 5 == 0) {

                kdtreeCornerLast->nearestKSearch(pointSel, 1, pointSearchInd, pointSearchSqDis);
                int closestPointInd = -1, minPointInd2 = -1;
                
                if (pointSearchSqDis[0] < nearestFeatureSearchSqDist) {
                    closestPointInd = pointSearchInd[0];
                    int closestPointScan = int(laserCloudCornerLast->points[closestPointInd].intensity);

                    float pointSqDis, minPointSqDis2 = nearestFeatureSearchSqDist;
                    for (int j = closestPointInd + 1; j < cornerPointsSharpNum; j++) {
                        if (int(laserCloudCornerLast->points[j].intensity) > closestPointScan + 2.5) {
                            break;
                        }

                        pointSqDis = (laserCloudCornerLast->points[j].x - pointSel.x) * 
                                     (laserCloudCornerLast->points[j].x - pointSel.x) + 
                                     (laserCloudCornerLast->points[j].y - pointSel.y) * 
                                     (laserCloudCornerLast->points[j].y - pointSel.y) + 
                                     (laserCloudCornerLast->points[j].z - pointSel.z) * 
                                     (laserCloudCornerLast->points[j].z - pointSel.z);

                        if (int(laserCloudCornerLast->points[j].intensity) > closestPointScan) {
                            if (pointSqDis < minPointSqDis2) {
                                minPointSqDis2 = pointSqDis;
                                minPointInd2 = j;
                            }
                        }
                    }
                    for (int j = closestPointInd - 1; j >= 0; j--) {
                        if (int(laserCloudCornerLast->points[j].intensity) < closestPointScan - 2.5) {
                            break;
                        }

                        pointSqDis = (laserCloudCornerLast->points[j].x - pointSel.x) * 
                                     (laserCloudCornerLast->points[j].x - pointSel.x) + 
                                     (laserCloudCornerLast->points[j].y - pointSel.y) * 
                                     (laserCloudCornerLast->points[j].y - pointSel.y) + 
                                     (laserCloudCornerLast->points[j].z - pointSel.z) * 
                                     (laserCloudCornerLast->points[j].z - pointSel.z);

                        if (int(laserCloudCornerLast->points[j].intensity) < closestPointScan) {
                            if (pointSqDis < minPointSqDis2) {
                                minPointSqDis2 = pointSqDis;
                                minPointInd2 = j;
                            }
                        }
                    }
                }

                pointSearchCornerInd1[i] = closestPointInd;
                pointSearchCornerInd2[i] = minPointInd2;
            }

            if (pointSearchCornerInd2[i] >= 0) {

                tripod1 = laserCloudCornerLast->points[pointSearchCornerInd1[i]];
                tripod2 = laserCloudCornerLast->points[pointSearchCornerInd2[i]];

                float x0 = pointSel.x;
                float y0 = pointSel.y;
                float z0 = pointSel.z;
                float x1 = tripod1.x;
                float y1 = tripod1.y;
                float z1 = tripod1.z;
                float x2 = tripod2.x;
                float y2 = tripod2.y;
                float z2 = tripod2.z;

                float m11 = ((x0 - x1)*(y0 - y2) - (x0 - x2)*(y0 - y1));
                float m22 = ((x0 - x1)*(z0 - z2) - (x0 - x2)*(z0 - z1));
                float m33 = ((y0 - y1)*(z0 - z2) - (y0 - y2)*(z0 - z1));

                float a012 = sqrt(m11 * m11  + m22 * m22 + m33 * m33);

                float l12 = sqrt((x1 - x2)*(x1 - x2) + (y1 - y2)*(y1 - y2) + (z1 - z2)*(z1 - z2));

                float la =  ((y1 - y2)*m11 + (z1 - z2)*m22) / a012 / l12;

                float lb = -((x1 - x2)*m11 - (z1 - z2)*m33) / a012 / l12;

                float lc = -((x1 - x2)*m22 + (y1 - y2)*m33) / a012 / l12;

                float ld2 = a012 / l12;

                float s = 1;
                if (iterCount >= 5) {
                    s = 1 - 1.8 * fabs(ld2);
                }

                if (s > 0.1 && ld2 != 0) {
                    coeff.x = s * la; 
                    coeff.y = s * lb;
                    coeff.z = s * lc;
                    coeff.intensity = s * ld2;
                  
                    laserCloudOri->push_back(cornerPointsSharp->points[i]);
                    coeffSel->push_back(coeff);
                }
            }
        }
    }

    void findCorrespondingSurfFeatures(int iterCount){

        int surfPointsFlatNum = surfPointsFlat->points.size();

        for (int i = 0; i < surfPointsFlatNum; i++) {

            TransformToStart(&surfPointsFlat->points[i], &pointSel);

            if (iterCount % 5 == 0) {

                kdtreeSurfLast->nearestKSearch(pointSel, 1, pointSearchInd, pointSearchSqDis);
                int closestPointInd = -1, minPointInd2 = -1, minPointInd3 = -1;

                if (pointSearchSqDis[0] < nearestFeatureSearchSqDist) {
                    closestPointInd = pointSearchInd[0];
                    int closestPointScan = int(laserCloudSurfLast->points[closestPointInd].intensity);

                    float pointSqDis, minPointSqDis2 = nearestFeatureSearchSqDist, minPointSqDis3 = nearestFeatureSearchSqDist;
                    for (int j = closestPointInd + 1; j < surfPointsFlatNum; j++) {
                        if (int(laserCloudSurfLast->points[j].intensity) > closestPointScan + 2.5) {
                            break;
                        }

                        pointSqDis = (laserCloudSurfLast->points[j].x - pointSel.x) * 
                                     (laserCloudSurfLast->points[j].x - pointSel.x) + 
                                     (laserCloudSurfLast->points[j].y - pointSel.y) * 
                                     (laserCloudSurfLast->points[j].y - pointSel.y) + 
                                     (laserCloudSurfLast->points[j].z - pointSel.z) * 
                                     (laserCloudSurfLast->points[j].z - pointSel.z);

                        if (int(laserCloudSurfLast->points[j].intensity) <= closestPointScan) {
                            if (pointSqDis < minPointSqDis2) {
                              minPointSqDis2 = pointSqDis;
                              minPointInd2 = j;
                            }
                        } else {
                            if (pointSqDis < minPointSqDis3) {
                                minPointSqDis3 = pointSqDis;
                                minPointInd3 = j;
                            }
                        }
                    }
                    for (int j = closestPointInd - 1; j >= 0; j--) {
                        if (int(laserCloudSurfLast->points[j].intensity) < closestPointScan - 2.5) {
                            break;
                        }

                        pointSqDis = (laserCloudSurfLast->points[j].x - pointSel.x) * 
                                     (laserCloudSurfLast->points[j].x - pointSel.x) + 
                                     (laserCloudSurfLast->points[j].y - pointSel.y) * 
                                     (laserCloudSurfLast->points[j].y - pointSel.y) + 
                                     (laserCloudSurfLast->points[j].z - pointSel.z) * 
                                     (laserCloudSurfLast->points[j].z - pointSel.z);

                        if (int(laserCloudSurfLast->points[j].intensity) >= closestPointScan) {
                            if (pointSqDis < minPointSqDis2) {
                                minPointSqDis2 = pointSqDis;
                                minPointInd2 = j;
                            }
                        } else {
                            if (pointSqDis < minPointSqDis3) {
                                minPointSqDis3 = pointSqDis;
                                minPointInd3 = j;
                            }
                        }
                    }
                }

                pointSearchSurfInd1[i] = closestPointInd;
                pointSearchSurfInd2[i] = minPointInd2;
                pointSearchSurfInd3[i] = minPointInd3;
            }

            if (pointSearchSurfInd2[i] >= 0 && pointSearchSurfInd3[i] >= 0) {

                tripod1 = laserCloudSurfLast->points[pointSearchSurfInd1[i]];
                tripod2 = laserCloudSurfLast->points[pointSearchSurfInd2[i]];
                tripod3 = laserCloudSurfLast->points[pointSearchSurfInd3[i]];

                float pa = (tripod2.y - tripod1.y) * (tripod3.z - tripod1.z) 
                         - (tripod3.y - tripod1.y) * (tripod2.z - tripod1.z);
                float pb = (tripod2.z - tripod1.z) * (tripod3.x - tripod1.x) 
                         - (tripod3.z - tripod1.z) * (tripod2.x - tripod1.x);
                float pc = (tripod2.x - tripod1.x) * (tripod3.y - tripod1.y) 
                         - (tripod3.x - tripod1.x) * (tripod2.y - tripod1.y);
                float pd = -(pa * tripod1.x + pb * tripod1.y + pc * tripod1.z);

                float ps = sqrt(pa * pa + pb * pb + pc * pc);

                pa /= ps;
                pb /= ps;
                pc /= ps;
                pd /= ps;

                float pd2 = pa * pointSel.x + pb * pointSel.y + pc * pointSel.z + pd;

                float s = 1;
                if (iterCount >= 5) {
                    s = 1 - 1.8 * fabs(pd2) / sqrt(sqrt(pointSel.x * pointSel.x
                            + pointSel.y * pointSel.y + pointSel.z * pointSel.z));
                }

                if (s > 0.1 && pd2 != 0) {
                    coeff.x = s * pa;
                    coeff.y = s * pb;
                    coeff.z = s * pc;
                    coeff.intensity = s * pd2;

                    laserCloudOri->push_back(surfPointsFlat->points[i]);
                    coeffSel->push_back(coeff);
                }
            }
        }
    }

    bool calculateTransformationSurf(int iterCount){

        int pointSelNum = laserCloudOri->points.size();

        cv::Mat matA(pointSelNum, 3, CV_32F, cv::Scalar::all(0));
        cv::Mat matAt(3, pointSelNum, CV_32F, cv::Scalar::all(0));
        cv::Mat matAtA(3, 3, CV_32F, cv::Scalar::all(0));
        cv::Mat matB(pointSelNum, 1, CV_32F, cv::Scalar::all(0));
        cv::Mat matAtB(3, 1, CV_32F, cv::Scalar::all(0));
        cv::Mat matX(3, 1, CV_32F, cv::Scalar::all(0));

        float srx = sin(transformCur[0]);
        float crx = cos(transformCur[0]);
        float sry = sin(transformCur[1]);
        float cry = cos(transformCur[1]);
        float srz = sin(transformCur[2]);
        float crz = cos(transformCur[2]);
        float tx = transformCur[3];
        float ty = transformCur[4];
        float tz = transformCur[5];

        float a1 = crx*sry*srz; float a2 = crx*crz*sry; float a3 = srx*sry; float a4 = tx*a1 - ty*a2 - tz*a3;
        float a5 = srx*srz; float a6 = crz*srx; float a7 = ty*a6 - tz*crx - tx*a5;
        float a8 = crx*cry*srz; float a9 = crx*cry*crz; float a10 = cry*srx; float a11 = tz*a10 + ty*a9 - tx*a8;

        float b1 = -crz*sry - cry*srx*srz; float b2 = cry*crz*srx - sry*srz;
        float b5 = cry*crz - srx*sry*srz; float b6 = cry*srz + crz*srx*sry;

        float c1 = -b6; float c2 = b5; float c3 = tx*b6 - ty*b5; float c4 = -crx*crz; float c5 = crx*srz; float c6 = ty*c5 + tx*-c4;
        float c7 = b2; float c8 = -b1; float c9 = tx*-b2 - ty*-b1;

        for (int i = 0; i < pointSelNum; i++) {

            pointOri = laserCloudOri->points[i];
            coeff = coeffSel->points[i];

            float arx = (-a1*pointOri.x + a2*pointOri.y + a3*pointOri.z + a4) * coeff.x
                      + (a5*pointOri.x - a6*pointOri.y + crx*pointOri.z + a7) * coeff.y
                      + (a8*pointOri.x - a9*pointOri.y - a10*pointOri.z + a11) * coeff.z;

            float arz = (c1*pointOri.x + c2*pointOri.y + c3) * coeff.x
                      + (c4*pointOri.x - c5*pointOri.y + c6) * coeff.y
                      + (c7*pointOri.x + c8*pointOri.y + c9) * coeff.z;

            float aty = -b6 * coeff.x + c4 * coeff.y + b2 * coeff.z;

            float d2 = coeff.intensity;

            matA.at<float>(i, 0) = arx;
            matA.at<float>(i, 1) = arz;
            matA.at<float>(i, 2) = aty;
            matB.at<float>(i, 0) = -0.05 * d2;
        }

        cv::transpose(matA, matAt);
        matAtA = matAt * matA;
        matAtB = matAt * matB;
        cv::solve(matAtA, matAtB, matX, cv::DECOMP_QR);

        if (iterCount == 0) {
            cv::Mat matE(1, 3, CV_32F, cv::Scalar::all(0));
            cv::Mat matV(3, 3, CV_32F, cv::Scalar::all(0));
            cv::Mat matV2(3, 3, CV_32F, cv::Scalar::all(0));

            cv::eigen(matAtA, matE, matV);
            matV.copyTo(matV2);

            isDegenerate = false;
            float eignThre[3] = {10, 10, 10};
            for (int i = 2; i >= 0; i--) {
                if (matE.at<float>(0, i) < eignThre[i]) {
                    for (int j = 0; j < 3; j++) {
                        matV2.at<float>(i, j) = 0;
                    }
                    isDegenerate = true;
                } else {
                    break;
                }
            }
            matP = matV.inv() * matV2;
        }

        if (isDegenerate) {
            cv::Mat matX2(3, 1, CV_32F, cv::Scalar::all(0));
            matX.copyTo(matX2);
            matX = matP * matX2;
        }

        transformCur[0] += matX.at<float>(0, 0);
        transformCur[2] += matX.at<float>(1, 0);
        transformCur[4] += matX.at<float>(2, 0);

        for(int i=0; i<6; i++){
            if(isnan(transformCur[i]))
                transformCur[i]=0;
        }

        float deltaR = sqrt(
                            pow(rad2deg(matX.at<float>(0, 0)), 2) +
                            pow(rad2deg(matX.at<float>(1, 0)), 2));
        float deltaT = sqrt(
                            pow(matX.at<float>(2, 0) * 100, 2));

        if (deltaR < 0.1 && deltaT < 0.1) {
            return false;
        }
        return true;
    }

    bool calculateTransformationCorner(int iterCount){

        int pointSelNum = laserCloudOri->points.size();

        cv::Mat matA(pointSelNum, 3, CV_32F, cv::Scalar::all(0));
        cv::Mat matAt(3, pointSelNum, CV_32F, cv::Scalar::all(0));
        cv::Mat matAtA(3, 3, CV_32F, cv::Scalar::all(0));
        cv::Mat matB(pointSelNum, 1, CV_32F, cv::Scalar::all(0));
        cv::Mat matAtB(3, 1, CV_32F, cv::Scalar::all(0));
        cv::Mat matX(3, 1, CV_32F, cv::Scalar::all(0));

        float srx = sin(transformCur[0]);
        float crx = cos(transformCur[0]);
        float sry = sin(transformCur[1]);
        float cry = cos(transformCur[1]);
        float srz = sin(transformCur[2]);
        float crz = cos(transformCur[2]);
        float tx = transformCur[3];
        float ty = transformCur[4];
        float tz = transformCur[5];

        float b1 = -crz*sry - cry*srx*srz; float b2 = cry*crz*srx - sry*srz; float b3 = crx*cry; float b4 = tx*-b1 + ty*-b2 + tz*b3;
        float b5 = cry*crz - srx*sry*srz; float b6 = cry*srz + crz*srx*sry; float b7 = crx*sry; float b8 = tz*b7 - ty*b6 - tx*b5;

        float c5 = crx*srz;

        for (int i = 0; i < pointSelNum; i++) {

            pointOri = laserCloudOri->points[i];
            coeff = coeffSel->points[i];

            float ary = (b1*pointOri.x + b2*pointOri.y - b3*pointOri.z + b4) * coeff.x
                      + (b5*pointOri.x + b6*pointOri.y - b7*pointOri.z + b8) * coeff.z;

            float atx = -b5 * coeff.x + c5 * coeff.y + b1 * coeff.z;

            float atz = b7 * coeff.x - srx * coeff.y - b3 * coeff.z;

            float d2 = coeff.intensity;

            matA.at<float>(i, 0) = ary;
            matA.at<float>(i, 1) = atx;
            matA.at<float>(i, 2) = atz;
            matB.at<float>(i, 0) = -0.05 * d2;
        }

        cv::transpose(matA, matAt);
        matAtA = matAt * matA;
        matAtB = matAt * matB;
        cv::solve(matAtA, matAtB, matX, cv::DECOMP_QR);

        if (iterCount == 0) {
            cv::Mat matE(1, 3, CV_32F, cv::Scalar::all(0));
            cv::Mat matV(3, 3, CV_32F, cv::Scalar::all(0));
            cv::Mat matV2(3, 3, CV_32F, cv::Scalar::all(0));

            cv::eigen(matAtA, matE, matV);
            matV.copyTo(matV2);

            isDegenerate = false;
            float eignThre[3] = {10, 10, 10};
            for (int i = 2; i >= 0; i--) {
                if (matE.at<float>(0, i) < eignThre[i]) {
                    for (int j = 0; j < 3; j++) {
                        matV2.at<float>(i, j) = 0;
                    }
                    isDegenerate = true;
                } else {
                    break;
                }
            }
            matP = matV.inv() * matV2;
        }

        if (isDegenerate) {
            cv::Mat matX2(3, 1, CV_32F, cv::Scalar::all(0));
            matX.copyTo(matX2);
            matX = matP * matX2;
        }

        transformCur[1] += matX.at<float>(0, 0);
        transformCur[3] += matX.at<float>(1, 0);
        transformCur[5] += matX.at<float>(2, 0);

        for(int i=0; i<6; i++){
            if(isnan(transformCur[i]))
                transformCur[i]=0;
        }

        float deltaR = sqrt(
                            pow(rad2deg(matX.at<float>(0, 0)), 2));
        float deltaT = sqrt(
                            pow(matX.at<float>(1, 0) * 100, 2) +
                            pow(matX.at<float>(2, 0) * 100, 2));

        if (deltaR < 0.1 && deltaT < 0.1) {
            return false;
        }
        return true;
    }

    bool calculateTransformation(int iterCount){

        int pointSelNum = laserCloudOri->points.size();

        cv::Mat matA(pointSelNum, 6, CV_32F, cv::Scalar::all(0));
        cv::Mat matAt(6, pointSelNum, CV_32F, cv::Scalar::all(0));
        cv::Mat matAtA(6, 6, CV_32F, cv::Scalar::all(0));
        cv::Mat matB(pointSelNum, 1, CV_32F, cv::Scalar::all(0));
        cv::Mat matAtB(6, 1, CV_32F, cv::Scalar::all(0));
        cv::Mat matX(6, 1, CV_32F, cv::Scalar::all(0));

        float srx = sin(transformCur[0]);
        float crx = cos(transformCur[0]);
        float sry = sin(transformCur[1]);
        float cry = cos(transformCur[1]);
        float srz = sin(transformCur[2]);
        float crz = cos(transformCur[2]);
        float tx = transformCur[3];
        float ty = transformCur[4];
        float tz = transformCur[5];

        float a1 = crx*sry*srz; float a2 = crx*crz*sry; float a3 = srx*sry; float a4 = tx*a1 - ty*a2 - tz*a3;
        float a5 = srx*srz; float a6 = crz*srx; float a7 = ty*a6 - tz*crx - tx*a5;
        float a8 = crx*cry*srz; float a9 = crx*cry*crz; float a10 = cry*srx; float a11 = tz*a10 + ty*a9 - tx*a8;

        float b1 = -crz*sry - cry*srx*srz; float b2 = cry*crz*srx - sry*srz; float b3 = crx*cry; float b4 = tx*-b1 + ty*-b2 + tz*b3;
        float b5 = cry*crz - srx*sry*srz; float b6 = cry*srz + crz*srx*sry; float b7 = crx*sry; float b8 = tz*b7 - ty*b6 - tx*b5;

        float c1 = -b6; float c2 = b5; float c3 = tx*b6 - ty*b5; float c4 = -crx*crz; float c5 = crx*srz; float c6 = ty*c5 + tx*-c4;
        float c7 = b2; float c8 = -b1; float c9 = tx*-b2 - ty*-b1;

        for (int i = 0; i < pointSelNum; i++) {

            pointOri = laserCloudOri->points[i];
            coeff = coeffSel->points[i];

            float arx = (-a1*pointOri.x + a2*pointOri.y + a3*pointOri.z + a4) * coeff.x
                      + (a5*pointOri.x - a6*pointOri.y + crx*pointOri.z + a7) * coeff.y
                      + (a8*pointOri.x - a9*pointOri.y - a10*pointOri.z + a11) * coeff.z;

            float ary = (b1*pointOri.x + b2*pointOri.y - b3*pointOri.z + b4) * coeff.x
                      + (b5*pointOri.x + b6*pointOri.y - b7*pointOri.z + b8) * coeff.z;

            float arz = (c1*pointOri.x + c2*pointOri.y + c3) * coeff.x
                      + (c4*pointOri.x - c5*pointOri.y + c6) * coeff.y
                      + (c7*pointOri.x + c8*pointOri.y + c9) * coeff.z;

            float atx = -b5 * coeff.x + c5 * coeff.y + b1 * coeff.z;

            float aty = -b6 * coeff.x + c4 * coeff.y + b2 * coeff.z;

            float atz = b7 * coeff.x - srx * coeff.y - b3 * coeff.z;

            float d2 = coeff.intensity;

            matA.at<float>(i, 0) = arx;
            matA.at<float>(i, 1) = ary;
            matA.at<float>(i, 2) = arz;
            matA.at<float>(i, 3) = atx;
            matA.at<float>(i, 4) = aty;
            matA.at<float>(i, 5) = atz;
            matB.at<float>(i, 0) = -0.05 * d2;
        }

        cv::transpose(matA, matAt);
        matAtA = matAt * matA;
        matAtB = matAt * matB;
        cv::solve(matAtA, matAtB, matX, cv::DECOMP_QR);

        if (iterCount == 0) {
            cv::Mat matE(1, 6, CV_32F, cv::Scalar::all(0));
            cv::Mat matV(6, 6, CV_32F, cv::Scalar::all(0));
            cv::Mat matV2(6, 6, CV_32F, cv::Scalar::all(0));

            cv::eigen(matAtA, matE, matV);
            matV.copyTo(matV2);

            isDegenerate = false;
            float eignThre[6] = {10, 10, 10, 10, 10, 10};
            for (int i = 5; i >= 0; i--) {
                if (matE.at<float>(0, i) < eignThre[i]) {
                    for (int j = 0; j < 6; j++) {
                        matV2.at<float>(i, j) = 0;
                    }
                    isDegenerate = true;
                } else {
                    break;
                }
            }
            matP = matV.inv() * matV2;
        }

        if (isDegenerate) {
            cv::Mat matX2(6, 1, CV_32F, cv::Scalar::all(0));
            matX.copyTo(matX2);
            matX = matP * matX2;
        }

        transformCur[0] += matX.at<float>(0, 0);
        transformCur[1] += matX.at<float>(1, 0);
        transformCur[2] += matX.at<float>(2, 0);
        transformCur[3] += matX.at<float>(3, 0);
        transformCur[4] += matX.at<float>(4, 0);
        transformCur[5] += matX.at<float>(5, 0);

        for(int i=0; i<6; i++){
            if(isnan(transformCur[i]))
                transformCur[i]=0;
        }

        float deltaR = sqrt(
                            pow(rad2deg(matX.at<float>(0, 0)), 2) +
                            pow(rad2deg(matX.at<float>(1, 0)), 2) +
                            pow(rad2deg(matX.at<float>(2, 0)), 2));
        float deltaT = sqrt(
                            pow(matX.at<float>(3, 0) * 100, 2) +
                            pow(matX.at<float>(4, 0) * 100, 2) +
                            pow(matX.at<float>(5, 0) * 100, 2));

        if (deltaR < 0.1 && deltaT < 0.1) {
            return false;
        }
        return true;
    }

    void checkSystemInitialization(){

        pcl::PointCloud<PointType>::Ptr laserCloudTemp = cornerPointsLessSharp;
        cornerPointsLessSharp = laserCloudCornerLast;
        laserCloudCornerLast = laserCloudTemp;

        laserCloudTemp = surfPointsLessFlat;
        surfPointsLessFlat = laserCloudSurfLast;
        laserCloudSurfLast = laserCloudTemp;

        kdtreeCornerLast->setInputCloud(laserCloudCornerLast);
        kdtreeSurfLast->setInputCloud(laserCloudSurfLast);

        laserCloudCornerLastNum = laserCloudCornerLast->points.size();
        laserCloudSurfLastNum = laserCloudSurfLast->points.size();

        sensor_msgs::PointCloud2 laserCloudCornerLast2;
        pcl::toROSMsg(*laserCloudCornerLast, laserCloudCornerLast2);
        laserCloudCornerLast2.header.stamp = cloudHeader.stamp;
        laserCloudCornerLast2.header.frame_id = "/camera";
        pubLaserCloudCornerLast.publish(laserCloudCornerLast2);

        sensor_msgs::PointCloud2 laserCloudSurfLast2;
        pcl::toROSMsg(*laserCloudSurfLast, laserCloudSurfLast2);
        laserCloudSurfLast2.header.stamp = cloudHeader.stamp;
        laserCloudSurfLast2.header.frame_id = "/camera";
        pubLaserCloudSurfLast.publish(laserCloudSurfLast2);

        transformSum[0] += imuPitchStart;
        transformSum[2] += imuRollStart;

        systemInitedLM = true;
    }

    void updateInitialGuess(){

        imuPitchLast = imuPitchCur;
        imuYawLast = imuYawCur;
        imuRollLast = imuRollCur;

        imuShiftFromStartX = imuShiftFromStartXCur;
        imuShiftFromStartY = imuShiftFromStartYCur;
        imuShiftFromStartZ = imuShiftFromStartZCur;

        imuVeloFromStartX = imuVeloFromStartXCur;
        imuVeloFromStartY = imuVeloFromStartYCur;
        imuVeloFromStartZ = imuVeloFromStartZCur;

        if (imuAngularFromStartX != 0 || imuAngularFromStartY != 0 || imuAngularFromStartZ != 0){
            // velodyne imu-->camera :xyz-->yzx
            transformCur[0] = - imuAngularFromStartY;
            transformCur[1] = - imuAngularFromStartZ;
            transformCur[2] = - imuAngularFromStartX;

        }
        
        if (imuVeloFromStartX != 0 || imuVeloFromStartY != 0 || imuVeloFromStartZ != 0){
            transformCur[3] -= imuVeloFromStartX * scanPeriod;
            transformCur[4] -= imuVeloFromStartY * scanPeriod;
            transformCur[5] -= imuVeloFromStartZ * scanPeriod;
        }
    }

    void updateTransformation(){

        if (laserCloudCornerLastNum < 10 || laserCloudSurfLastNum < 100)
            return;

        for (int iterCount1 = 0; iterCount1 < 25; iterCount1++) {
            laserCloudOri->clear();
            coeffSel->clear();

            findCorrespondingSurfFeatures(iterCount1);

            if (laserCloudOri->points.size() < 10)
                continue;
            if (calculateTransformationSurf(iterCount1) == false)
                break;
        }

        for (int iterCount2 = 0; iterCount2 < 25; iterCount2++) {

            laserCloudOri->clear();
            coeffSel->clear();

            findCorrespondingCornerFeatures(iterCount2);

            if (laserCloudOri->points.size() < 10)
                continue;
            if (calculateTransformationCorner(iterCount2) == false)
                break;
        }
    }

    void integrateTransformation(){
        float rx, ry, rz, tx, ty, tz;
        AccumulateRotation(transformSum[0], transformSum[1], transformSum[2], 
                           -transformCur[0], -transformCur[1], -transformCur[2], rx, ry, rz);

        float x1 = cos(rz) * (transformCur[3] - imuShiftFromStartX) 
                 - sin(rz) * (transformCur[4] - imuShiftFromStartY);
        float y1 = sin(rz) * (transformCur[3] - imuShiftFromStartX) 
                 + cos(rz) * (transformCur[4] - imuShiftFromStartY);
        float z1 = transformCur[5] - imuShiftFromStartZ;

        float x2 = x1;
        float y2 = cos(rx) * y1 - sin(rx) * z1;
        float z2 = sin(rx) * y1 + cos(rx) * z1;

        tx = transformSum[3] - (cos(ry) * x2 + sin(ry) * z2);
        ty = transformSum[4] - y2;
        tz = transformSum[5] - (-sin(ry) * x2 + cos(ry) * z2);

        PluginIMURotation(rx, ry, rz, imuPitchStart, imuYawStart, imuRollStart,
                          imuPitchLast, imuYawLast, imuRollLast, rx, ry, rz);

        transformSum[0] = rx;
        transformSum[1] = ry;
        transformSum[2] = rz;
        transformSum[3] = tx;
        transformSum[4] = ty;
        transformSum[5] = tz;
    }

    void publishOdometry(){
        geometry_msgs::Quaternion geoQuat = tf::createQuaternionMsgFromRollPitchYaw(transformSum[2], -transformSum[0], -transformSum[1]);

        // 只改变旋转角不改变三维位姿的坐标变换,按照ZXY（camera to world）的顺序将欧拉角转化为四元数
        laserOdometry.header.stamp = cloudHeader.stamp;
        laserOdometry.pose.pose.orientation.x = -geoQuat.y;
        laserOdometry.pose.pose.orientation.y = -geoQuat.z;
        laserOdometry.pose.pose.orientation.z = geoQuat.x;
        laserOdometry.pose.pose.orientation.w = geoQuat.w;
        laserOdometry.pose.pose.position.x = transformSum[3];
        laserOdometry.pose.pose.position.y = transformSum[4];
        laserOdometry.pose.pose.position.z = transformSum[5];
        pubLaserOdometry.publish(laserOdometry);

        laserOdometryTrans.stamp_ = cloudHeader.stamp;
        laserOdometryTrans.setRotation(tf::Quaternion(-geoQuat.y, -geoQuat.z, geoQuat.x, geoQuat.w));
        laserOdometryTrans.setOrigin(tf::Vector3(transformSum[3], transformSum[4], transformSum[5]));
        tfBroadcaster.sendTransform(laserOdometryTrans);
    }

    void adjustOutlierCloud(){
        PointType point;
        int cloudSize = outlierCloud->points.size();
        float tempx, tempy, tempz;
        for (int i = 0; i < cloudSize; ++i)
        {
            tempx = outlierCloud->points[i].x;
            tempy = outlierCloud->points[i].y;
            tempz = outlierCloud->points[i].z;

            outlierCloud->points[i].x = -tempy;
            outlierCloud->points[i].y = tempx;
            outlierCloud->points[i].z = tempz;

            point.x = outlierCloud->points[i].y;
            point.y = outlierCloud->points[i].z;
            point.z = outlierCloud->points[i].x;
            point.intensity = outlierCloud->points[i].intensity;
            outlierCloud->points[i] = point;
        }
    }

    void publishCloudsLast(){

        updateImuRollPitchYawStartSinCos();

        int cornerPointsLessSharpNum = cornerPointsLessSharp->points.size();
        for (int i = 0; i < cornerPointsLessSharpNum; i++) {
            TransformToEnd(&cornerPointsLessSharp->points[i], &cornerPointsLessSharp->points[i]);
        }


        int surfPointsLessFlatNum = surfPointsLessFlat->points.size();
        for (int i = 0; i < surfPointsLessFlatNum; i++) {
            TransformToEnd(&surfPointsLessFlat->points[i], &surfPointsLessFlat->points[i]);
        }

        pcl::PointCloud<PointType>::Ptr laserCloudTemp = cornerPointsLessSharp;
        cornerPointsLessSharp = laserCloudCornerLast;
        laserCloudCornerLast = laserCloudTemp;

        laserCloudTemp = surfPointsLessFlat;
        surfPointsLessFlat = laserCloudSurfLast;
        laserCloudSurfLast = laserCloudTemp;

        laserCloudCornerLastNum = laserCloudCornerLast->points.size();
        laserCloudSurfLastNum = laserCloudSurfLast->points.size();

        if (laserCloudCornerLastNum > 10 && laserCloudSurfLastNum > 100) {
            kdtreeCornerLast->setInputCloud(laserCloudCornerLast);
            kdtreeSurfLast->setInputCloud(laserCloudSurfLast);
        }

        frameCount++;

        if (frameCount >= skipFrameNum + 1) {

            frameCount = 0;

            adjustOutlierCloud();
            sensor_msgs::PointCloud2 outlierCloudLast2;
            pcl::toROSMsg(*outlierCloud, outlierCloudLast2);
            outlierCloudLast2.header.stamp = cloudHeader.stamp;
            outlierCloudLast2.header.frame_id = "/camera";
            pubOutlierCloudLast.publish(outlierCloudLast2);

            sensor_msgs::PointCloud2 laserCloudCornerLast2;
            pcl::toROSMsg(*laserCloudCornerLast, laserCloudCornerLast2);
            laserCloudCornerLast2.header.stamp = cloudHeader.stamp;
            laserCloudCornerLast2.header.frame_id = "/camera";
            pubLaserCloudCornerLast.publish(laserCloudCornerLast2);
            pcl::io::savePCDFileASCII(fileDirectory+"laserCloudCornerLast2.pcd", *laserCloudCornerLast);

            sensor_msgs::PointCloud2 laserCloudSurfLast2;
            pcl::toROSMsg(*laserCloudSurfLast, laserCloudSurfLast2);
            laserCloudSurfLast2.header.stamp = cloudHeader.stamp;
            laserCloudSurfLast2.header.frame_id = "/camera";
            pubLaserCloudSurfLast.publish(laserCloudSurfLast2);
            pcl::io::savePCDFileASCII(fileDirectory+"laserCloudSurfLast2.pcd", *laserCloudSurfLast);
        }
    }

    int t_num = 0;
    pcl::PointCloud<PointType> cloud;
    void getTransformcloud()
    {
        t_num++;
        int cloudSize =cloud.points.size();

        if(cloudSize!=0){
            pcl::io::savePCDFileASCII(fileDirectory+"laserfullCloudbeore.pcd", cloud);
            pcl::PointCloud<PointType>::Ptr cloudToStart(new pcl::PointCloud<PointType>());
            for(int i=0;i<cloudSize;i++){
                PointType p;
                TransformToStart(&cloud.points[i], &p);
                cloudToStart->points.push_back(p);
            }
            cloudToStart-> width = 1;
            cloudToStart-> height = cloudSize;
            pcl::io::savePCDFileASCII(fileDirectory+"laserfullCloudafter.pcd", *cloudToStart);
        }
    }


    void runFeatureAssociation()
    {

        if (newSegmentedCloud && newSegmentedCloudInfo && newOutlierCloud &&
            std::abs(timeNewSegmentedCloudInfo - timeNewSegmentedCloud) < 0.05 &&
            std::abs(timeNewOutlierCloud - timeNewSegmentedCloud) < 0.05){

            newSegmentedCloud = false;
            newSegmentedCloudInfo = false;
            newOutlierCloud = false;
        }else{
            return;
        }
        /**
        	1. Feature Extraction
        */
        // cloud = *segmentedCloud;
        adjustDistortion();

        calculateSmoothness();

        markOccludedPoints();

        extractFeatures();

        publishCloud(); // cloud for visualization
	
        /**
		2. Feature Association
        */
        if (!systemInitedLM) {
            checkSystemInitialization();
            return;
        }

        updateInitialGuess();

        updateTransformation();

        // getTransformcloud();

        integrateTransformation();

//        if (imuPointerLast >= 0) {
//            publishimuOdom();
//        }

        publishOdometry();

        publishCloudsLast(); // cloud to mapOptimization

        float vx = transformCur[3]/scanPeriod;
        float vy = transformCur[4]/scanPeriod;
        float vz = transformCur[5]/scanPeriod;

        float v_toal_l =sqrt(vx*vx+vy*vy+vz*vz);

        if (v_toal_l> v_max_l)
        {
            v_max_l = v_toal_l;
        }
        std::cout << "scanPeriod--- " << scanPeriod << std::endl;
        std::cout << "v_toal_l--- " << v_toal_l << std::endl;
        std::cout << "v_max_l--- " << v_max_l << std::endl;
    }
};




int main(int argc, char** argv)
{
    ros::init(argc, argv, "lego_loam");

    ROS_INFO("\033[1;32m---->\033[0m Feature Association Started.");

    FeatureAssociation FA;

    ros::Rate rate(200);
    while (ros::ok())
    // while ( 1 )
    {
        ros::spinOnce();

        FA.runFeatureAssociation();

        rate.sleep();
    }
    
    ros::spin();
    return 0;
}
