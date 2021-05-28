#ifndef _UTILITY_LIDAR_ODOMETRY_H_
#define _UTILITY_LIDAR_ODOMETRY_H_


#include <ros/ros.h>

#include <sensor_msgs/Imu.h>
#include <sensor_msgs/PointCloud2.h>
#include <nav_msgs/Odometry.h>

#include "cloud_msgs/cloud_info.h"

#include <opencv/cv.h>

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl_ros/point_cloud.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/range_image/range_image.h>
#include <pcl/filters/filter.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/common/common.h>
#include <pcl/registration/icp.h>

#include <tf/transform_broadcaster.h>
#include <tf/transform_datatypes.h>
 
#include <vector>
#include <cmath>
#include <algorithm>
#include <queue>
#include <deque>
#include <iostream>
#include <fstream>
#include <ctime>
#include <cfloat>
#include <iterator>
#include <sstream>
#include <string>
#include <limits>
#include <iomanip>
#include <array>
#include <thread>
#include <mutex>

#define PI 3.14159265

using namespace std;
// rosbag filter "HK-Data20190316-2 20190331_NJ_LL.bag" "lidaronly_HK-Data20190316-2 20190331_NJ_LL.bag" "topic == '/velodyne_points'"
// rosbag filter "HK-Data20190117.bag" "lidaronly_HK-Data20190117.bag" "topic == '/velodyne_points'"

//typedef pcl::PointXYZI  PointType;
typedef pcl::PointXYZINormal  PointType; //modified to reserve raw intensity to curvature

// extern const string pointCloudTopic = "/velodyne_points";
// extern const string pointCloudTopic = "/kitti_scan";
//  extern const string pointCloudTopic = "/pandar";
extern const string pointCloudTopic = "/pointcloud";
//extern const string imuTopic = "/mti/sensor/imu";
extern const string imuTopic = "/imu";

// Save pcd
extern const string fileDirectory = "/home/tiev_slammer/A_SLAM_Projects/Huawei/data/";

// Using velodyne cloud "ring" channel for image projection (other lidar may have different name for this channel, change "PointXYZIR" below)
extern const bool useCloudRing = true; // if true, ang_res_y and ang_bottom are not used
// extern const bool useCloudRing = false; // if true, ang_res_y and ang_bottom are not used

// Using ICP for refine the matching
extern const bool useICPRefinedMatching = false;

// VLP-16
//extern const int N_SCAN = 16;
//extern const int Horizon_SCAN = 1800;
//extern const float ang_res_x = 0.2;
//extern const float ang_res_y = 2.0;
//extern const float ang_bottom = 15.0+0.1;
//extern const int groundScanInd = 7;

// HDL-32E
// extern const int N_SCAN = 32;
// extern const int Horizon_SCAN = 1800;
// extern const float ang_res_x = 360.0/float(Horizon_SCAN);
// extern const float ang_res_y = 41.33/float(N_SCAN-1);
// extern const float ang_bottom = 30.67;
// extern const int groundScanInd = 20;

//pandar40P
extern const int N_SCAN = 40;
extern const int Horizon_SCAN = 1800;
extern const float ang_res_x = 0.2;
extern const float ang_res_y = 0.33;
extern const float ang_bottom = 25;
extern const int groundScanInd = 40;

//Vel 64
// extern const int N_SCAN = 64;
// extern const int Horizon_SCAN = 1800;
// extern const float ang_res_x = 0.2;
// extern const float ang_res_y = 0.427;
// extern const float ang_bottom = 24.9;
// extern const int groundScanInd = 50;

// VLS-128
// extern const int N_SCAN = 128;
// extern const int Horizon_SCAN = 1800;
// extern const float ang_res_x = 0.2;
// extern const float ang_res_y = 0.3;
// extern const float ang_bottom = 25.0;
// extern const int groundScanInd = 10;

// Ouster users may need to uncomment line 159 in imageProjection.cpp
// Usage of Ouster imu data is not fully supported yet (LeGO-LOAM needs 9-DOF IMU), please just publish point cloud data
// Ouster OS1-16
// extern const int N_SCAN = 16;
// extern const int Horizon_SCAN = 1024;
// extern const float ang_res_x = 360.0/float(Horizon_SCAN);
// extern const float ang_res_y = 33.2/float(N_SCAN-1);
// extern const float ang_bottom = 16.6+0.1;
// extern const int groundScanInd = 7;

// Ouster OS1-64
//extern const int N_SCAN = 64;
//extern const int Horizon_SCAN = 1024;
//extern const float ang_res_x = 360.0/float(Horizon_SCAN);
//extern const float ang_res_y = 33.2/float(N_SCAN-1);
//extern const float ang_bottom = 16.6+0.1;
//extern const int groundScanInd = 15;
extern const bool loadLoop = false; // by cjf
extern const bool loopClosureEnableFlag = true;
extern const double mappingProcessInterval = 0.1;

extern const float scanPeriod = 0.1;
extern const int systemDelay = 0;
extern const int imuQueLength = 200;

extern const float sensorMinimumRange = 1.0;
extern const float sensorMountAngle = 0.0;
extern const float segmentTheta = 60.0/180.0*M_PI; // decrese this value may improve accuracy
extern const int segmentValidPointNum = 5;
extern const int segmentValidLineNum = 3;
extern const float segmentAlphaX = ang_res_x / 180.0 * M_PI;
extern const float segmentAlphaY = ang_res_y / 180.0 * M_PI;


extern const int edgeFeatureNum = 2;
extern const int surfFeatureNum = 4;
extern const int sectionsTotal = 6;
extern const float edgeThreshold = 0.1;
extern const float surfThreshold = 0.1;
extern const float nearestFeatureSearchSqDist = 25;


// Mapping Params
extern const float surroundingKeyframeSearchRadius = 50.0; // key frame that is within n meters from current pose will be considerd for scan-to-map optimization (when loop closure disabled)
extern const int   surroundingKeyframeSearchNum = 50; // submap size (when loop closure enabled)

// history key frames (history submap for loop closure)
extern const float historyKeyframeSearchRadius = 7.0;
//extern const float historyKeyframeSearchRadius = 20.0; // NOT used in Scan Context-based loop detector / default 7.0; key frame that is within n meters from current pose will be considerd for loop closure
extern const int   historyKeyframeSearchNum = 25; // 2n+1 number of history key frames will be fused into a submap for loop closure
//extern const float historyKeyframeFitnessScore = 1.5; // default 0.3; the smaller the better alignment
extern const float historyKeyframeFitnessScore = 0.3; // cjf
extern const float globalMapVisualizationSearchRadius = 1500.0; // key frames with in n meters will be visualized


struct smoothness_t{ 
    float value;
    size_t ind;
};

struct by_value{ 
    bool operator()(smoothness_t const &left, smoothness_t const &right) { 
        return left.value < right.value;
    }
};

/*
    * A point cloud type that has "ring" channel
    */
struct PointXYZIR
{
    PCL_ADD_POINT4D
    PCL_ADD_INTENSITY;
    uint16_t ring;
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
} EIGEN_ALIGN16;

POINT_CLOUD_REGISTER_POINT_STRUCT (PointXYZIR,  
                                   (float, x, x) (float, y, y)
                                   (float, z, z) (float, intensity, intensity)
                                   (uint16_t, ring, ring)
)

/*
    * A point cloud type that has "ring" channel
    */
struct PointXYZIRC
{
    PCL_ADD_POINT4D
    PCL_ADD_INTENSITY;
    float curvature;
    uint16_t ring;
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
} EIGEN_ALIGN16;

POINT_CLOUD_REGISTER_POINT_STRUCT (PointXYZIRC,  
                                   (float, x, x) (float, y, y)
                                   (float, z, z) (float, intensity, intensity)
                                   (float, curvature, curvature)
                                   )
   

/*
    * A point cloud type that has 6D pose info ([x,y,z,roll,pitch,yaw] intensity is time stamp)
    */
struct PointXYZIRPYT
{
    PCL_ADD_POINT4D
    PCL_ADD_INTENSITY;
    float roll;
    float pitch;
    float yaw;
    double time;
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
} EIGEN_ALIGN16;

POINT_CLOUD_REGISTER_POINT_STRUCT (PointXYZIRPYT,
                                   (float, x, x) (float, y, y)
                                   (float, z, z) (float, intensity, intensity)
                                   (float, roll, roll) (float, pitch, pitch) (float, yaw, yaw)
                                   (double, time, time)
)

typedef PointXYZIRPYT  PointTypePose;

//encode intesnsity to curvature john added
PointType EncodeIntensityToCurvature(const PointType& pt_in, PointType& pt_out, short code_size = 10000)
{
    double encoded_intensity = (int)(pt_in.curvature * code_size) + pt_in.intensity;
    pt_out = pt_in;
    pt_out.intensity = encoded_intensity;
}

PointType DecodeIntensityToCurvature(const PointType& pt_in, PointType& pt_out, short code_size = 10000)
{
    double decoded_raw_intensity = pt_in.intensity - (int)(pt_in.intensity / code_size) * code_size;
    double decoded_intensity = pt_in.intensity - decoded_intensity;
    pt_out = pt_in;
    pt_out.curvature = decoded_raw_intensity;
    pt_out.intensity = decoded_intensity;
}

#endif
