#pragma once

#include <ctime>
#include <cassert>
#include <cmath>
#include <utility>
#include <vector>
#include <algorithm> 
#include <cstdlib>
#include <memory>
#include <iostream>

#include <Eigen/Dense>

#include <opencv2/opencv.hpp>
#include <opencv2/core/eigen.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <cv_bridge/cv_bridge.h>

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl_conversions/pcl_conversions.h>

#include "nanoflann.hpp"
#include "KDTreeVectorOfVectorsAdaptor.h"

#include "tictoc.h"
// #include "utility.h"
#include <boost/archive/binary_iarchive.hpp> //二进制序列化
#include <boost/archive/binary_oarchive.hpp> //二进制序列化
#include <boost/serialization/vector.hpp> //序列化STL容器要导入
#include<boost/serialization/map.hpp> //序列化STL容器要导入
#include "eigen_boost_serialization.hpp" // I commented Matrix.h in Eigen/core because of redefinition
using namespace Eigen;
using namespace nanoflann;

using std::cout;
using std::endl;
using std::make_pair;

using std::atan2;
using std::cos;
using std::sin;

// using SCPointType = pcl::PointXYZI; // using xyz only. but a user can exchange the original bin encoding function (i.e., max hegiht) to max intensity (for detail, refer 20 ICRA Intensity Scan Context)
// using PointType = pcl::PointXYZI;

//john added for store raw  intensitive
using SCPointType = pcl::PointXYZINormal; // using xyz only. but a user can exchange the original bin encoding function (i.e., max hegiht) to max intensity (for detail, refer 20 ICRA Intensity Scan Context)
using PointType = pcl::PointXYZINormal;
using KeyMat = std::vector<std::vector<float> >;
using InvKeyTree = KDTreeVectorOfVectorsAdaptor< KeyMat, float >;


// namespace SC2
// {

void coreImportTest ( void );


// sc param-independent helper functions 
float xy2theta( const float & _x, const float & _y );
MatrixXd circshift( MatrixXd &_mat, int _num_shift );
std::vector<float> eig2stdvec( MatrixXd _eigmat );


class SCManager
{
public: 
    SCManager( ) = default; // reserving data space (of std::vector) could be considered. but the descriptor is lightweight so don't care.

    Eigen::MatrixXd makeScancontext( pcl::PointCloud<SCPointType> & _scan_down );
    Eigen::MatrixXd makeRingkeyFromScancontext( Eigen::MatrixXd &_desc );
    Eigen::MatrixXd makeSectorkeyFromScancontext( Eigen::MatrixXd &_desc );

    int fastAlignUsingVkey ( MatrixXd & _vkey1, MatrixXd & _vkey2 ); 
    double distDirectSC ( MatrixXd &_sc1, MatrixXd &_sc2 ); // "d" (eq 5) in the original paper (IROS 18)
    std::pair<double, int> distanceBtnScanContext ( MatrixXd &_sc1, MatrixXd &_sc2 ); // "D" (eq 6) in the original paper (IROS 18)

    // User-side API
    void makeAndSaveScancontextAndKeys( pcl::PointCloud<SCPointType> & _scan_down, PointType &thisPose3D);
    std::pair<int, float> detectLoopClosureID( void ); // int: nearest node index, float: relative yaw
    /** created by cjf*******************************************/
    // std::pair<int, float> detectLoopClosureID(pcl::PointCloud<PointType> & cloudKeyPoses3D); // int: nearest node index, float: relative yaw
    /** created by cjf*******************************************/
public:
    // hyper parameters ()
    const double LIDAR_HEIGHT = 2.0; // lidar height : add this for simply directly using lidar scan in the lidar local coord (not robot base coord) / if you use robot-coord-transformed lidar scans, just set this as 0.
    /** created by cjf*******************************************/
    void SaveDescriptorToFile();
    void LoadDescriptorFromFile();
    const double LOOP_MAX_Z_DIFF = 10.0; // by cjf ensure if the robot is on the same floor or not
    /** created by cjf*******************************************/
    const int    PC_NUM_RING = 20; // 20 in the original paper (IROS 18)
    const int    PC_NUM_SECTOR = 60; // 60 in the original paper (IROS 18)
    const double PC_MAX_RADIUS = 80.0; // 80 meter max in the original paper (IROS 18)
    const double PC_UNIT_SECTORANGLE = 360.0 / double(PC_NUM_SECTOR);
    const double PC_UNIT_RINGGAP = PC_MAX_RADIUS / double(PC_NUM_RING);

    // tree
    const int    NUM_EXCLUDE_RECENT = 50; // simply just keyframe gap, but node position distance-based exclusion is ok. 
    const int    NUM_CANDIDATES_FROM_TREE = 10; // 10 is enough. (refer the IROS 18 paper)

    // loop thres
    const double SEARCH_RATIO = 0.1; // for fast comparison, no Brute-force, but search 10 % is okay. // not was in the original conf paper, but improved ver.
    // const double SC_DIST_THRES = 0.13; // empirically 0.1-0.2 is fine (rare false-alarms) for 20x60 polar context (but for 0.15 <, DCS or ICP fit score check (e.g., in LeGO-LOAM) should be required for robustness)
    const double SC_DIST_THRES = 0.5; // 0.4-0.6 is good choice for using with robust kernel (e.g., Cauchy, DCS) + icp fitness threshold / if not, recommend 0.1-0.15

    // config 
    const int    TREE_MAKING_PERIOD_ = 10; // i.e., remaking tree frequency, to avoid non-mandatory every remaking, to save time cost / in the LeGO-LOAM integration, it is synchronized with the loop detection callback (which is 1Hz) so it means the tree is updated evrey 10 sec. But you can use the smaller value because it is enough fast ~ 5-50ms wrt N.
    int          tree_making_period_conter = 0;

    // data
    /** created by cjf*******************************************/
    class ScanContext
    {
    private:
        friend class boost::serialization::access;
        template<class Archive>
        void serialize(Archive& ar, const unsigned int version)
        {
            ar & polarcontexts_;
            // ar & node_id_container;
            ar & polarcontext_invkeys_;
            ar & polarcontext_vkeys_;
            ar & polarcontext_invkeys_mat_;
            ar & thisPose6D_;        // serialize add pose6D
        }
    public:
        Eigen::MatrixXd polarcontexts_;
        // std::pair<int,int> node_id_container;
        Eigen::MatrixXd polarcontext_invkeys_;
        Eigen::MatrixXd polarcontext_vkeys_;
        std::vector<float> polarcontext_invkeys_mat_;
        float thisPose6D_[6];   // serialize add pose6D
    };

    ScanContext scan_context;
    std::vector<ScanContext> scan_contexts;
    std::vector<std::vector<float>> polarcontext_invkeys_to_search_;
    std::unique_ptr<InvKeyTree> polarcontext_tree_;
    int true_loop_found;
    int false_loop_found;






    // std::vector<double> polarcontexts_timestamp_; // optional.
    // std::vector<Eigen::MatrixXd> polarcontexts_;
    // std::vector<Eigen::MatrixXd> polarcontext_invkeys_;
    // std::vector<Eigen::MatrixXd> polarcontext_vkeys_;

    // KeyMat polarcontext_invkeys_mat_;
    // KeyMat polarcontext_invkeys_to_search_;
    // std::unique_ptr<InvKeyTree> polarcontext_tree_;

}; // SCManager

// } // namespace SC2
