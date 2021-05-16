#include "Scancontext.h"

// namespace SC2
// {

void coreImportTest (void)
{
    cout << "scancontext lib is successfully imported." << endl;
} // coreImportTest


    /**********************************by cjf*********************************/
    void SCManager::SaveDescriptorToFile() {
        std::string sc_desc_path="/home/neousys/huawei-parking-vloc/src/SC-LeGO-LOAM/SC-LeGO-LOAM/LeGO-LOAM/src/serialization_sc.txt";
        std::ofstream fout(sc_desc_path);
        boost::archive::binary_oarchive oa(fout);
        oa & scan_contexts;
        std::cout << "successfully save SC descriptors in sc!" <<"\n";
        fout.close();
    }
    void SCManager::LoadDescriptorFromFile() {
        std::string sc_desc_path="/home/neousys/huawei-parking-vloc/src/SC-LeGO-LOAM/SC-LeGO-LOAM/LeGO-LOAM/src/serialization_sc.txt";
        std::ifstream fin(sc_desc_path);
        boost::archive::binary_iarchive ia(fin);
        ia& scan_contexts;
        std::cout << "successfully load SC descriptors!" <<"\n";
    }
    /**********************************by cjf*********************************/



float rad2deg(float radians)
{
    return radians * 180.0 / M_PI;
}

float deg2rad(float degrees)
{
    return degrees * M_PI / 180.0;
}

// 直角坐标和极坐标转换
float xy2theta( const float & _x, const float & _y )
{
    if ( _x >= 0 & _y >= 0) 
        return (180/M_PI) * atan(_y / _x);

    if ( _x < 0 & _y >= 0) 
        return 180 - ( (180/M_PI) * atan(_y / (-_x)) );

    if ( _x < 0 & _y < 0) 
        return 180 + ( (180/M_PI) * atan(_y / _x) );

    if ( _x >= 0 & _y < 0)
        return 360 - ( (180/M_PI) * atan((-_y) / _x) );
} // xy2theta

// No.5 fastAlignUsingVkey调用,用来得到将列向量平移对应步长后的新矩阵
MatrixXd circshift( MatrixXd &_mat, int _num_shift )
{
    // shift columns to right direction 
    assert(_num_shift >= 0);

    if( _num_shift == 0 )
    {
        MatrixXd shifted_mat( _mat );
        return shifted_mat; // Early return 
    }

    MatrixXd shifted_mat = MatrixXd::Zero( _mat.rows(), _mat.cols() );
    for ( int col_idx = 0; col_idx < _mat.cols(); col_idx++ )
    {
        int new_location = (col_idx + _num_shift) % _mat.cols();
        shifted_mat.col(new_location) = _mat.col(col_idx);
    }

    return shifted_mat;

} // circshift


std::vector<float> eig2stdvec( MatrixXd _eigmat )
{
    std::vector<float> vec( _eigmat.data(), _eigmat.data() + _eigmat.size() );
    return vec;
} // eig2stdvec

/**
 * No.6 distanceBtnScanContext调用
 * @param _sc1
 * @param _sc2
 * @return 每一列余弦距离的求和/列数
 */
double SCManager::distDirectSC ( MatrixXd &_sc1, MatrixXd &_sc2 )
{
    int num_eff_cols = 0; // i.e., to exclude all-nonzero sector
    double sum_sector_similarity = 0;
    for ( int col_idx = 0; col_idx < _sc1.cols(); col_idx++ )
    {
        VectorXd col_sc1 = _sc1.col(col_idx);
        VectorXd col_sc2 = _sc2.col(col_idx);
        
        if( col_sc1.norm() == 0 | col_sc2.norm() == 0 ) // 不计算空列
            continue; // don't count this sector pair. 

        double sector_similarity = col_sc1.dot(col_sc2) / (col_sc1.norm() * col_sc2.norm()); // 余弦距离

        sum_sector_similarity = sum_sector_similarity + sector_similarity;
        num_eff_cols = num_eff_cols + 1;
    }
    
    double sc_sim = sum_sector_similarity / num_eff_cols;
    return 1.0 - sc_sim;

} // distDirectSC

/**
 * No.4 distanceBtnScanContext调用
 * @param _vkey1
 * @param _vkey2
 * @return argmin_vkey_shift:最小模长对应的平移量
 */
int SCManager::fastAlignUsingVkey( MatrixXd & _vkey1, MatrixXd & _vkey2)
{
    int argmin_vkey_shift = 0;
    double min_veky_diff_norm = 10000000;
    // 将_vkey2分别平移0,1,2...步长,求_vkey1与平移后的_vkey2_shifted的差,计算该矩阵的模长,求得最小模长对应的平移量
    for ( int shift_idx = 0; shift_idx < _vkey1.cols(); shift_idx++ )
    {
        MatrixXd vkey2_shifted = circshift(_vkey2, shift_idx);

        MatrixXd vkey_diff = _vkey1 - vkey2_shifted;

        double cur_diff_norm = vkey_diff.norm();
        if( cur_diff_norm < min_veky_diff_norm )
        {
            argmin_vkey_shift = shift_idx;
            min_veky_diff_norm = cur_diff_norm;
        }
    }

    return argmin_vkey_shift;

} // fastAlignUsingVkey

/**
 * No.2 计算两个scan之间的相似度,用余弦距离衡量
 * 1. 先得到两个scan的单行多列描述子vkey_sc1,vkey_sc2
 * 2. 求出两者匹配最好时的平移量,对应的其实就是赵老师之前提出的,比如某个物体在历史scan1中偏外环,而在scan2中位置偏内环,利用该方法可以部分解决其相似性匹配很低的问题
 * 3. 然后确定好平移量后,根据确定的搜索半径确定yaw方向的旋转量,分别计算余弦距离
 * 4. 返回最终的最小距离和旋转方向偏移步长
 * @param _sc1
 * @param _sc2
 * @return pair<min_sc_dist, argmin_shift>:最终的最小距离和旋转方向偏移步长
 */
std::pair<double, int> SCManager::distanceBtnScanContext( MatrixXd &_sc1, MatrixXd &_sc2 )
{
    // 1. fast align using variant key (not in original IROS18) 这个做法和论文中不太一致
    MatrixXd vkey_sc1 = makeSectorkeyFromScancontext( _sc1 );
    MatrixXd vkey_sc2 = makeSectorkeyFromScancontext( _sc2 );
    int argmin_vkey_shift = fastAlignUsingVkey( vkey_sc1, vkey_sc2 );

    const int SEARCH_RADIUS = round( 0.5 * SEARCH_RATIO * _sc1.cols() ); // a half of search range : 0.5*0.1*20 = 1m
    std::vector<int> shift_idx_search_space { argmin_vkey_shift };
    for ( int ii = 1; ii < SEARCH_RADIUS + 1; ii++ )
    {
        shift_idx_search_space.push_back( (argmin_vkey_shift + ii + _sc1.cols()) % _sc1.cols() );
        shift_idx_search_space.push_back( (argmin_vkey_shift - ii + _sc1.cols()) % _sc1.cols() );
    }
    std::sort(shift_idx_search_space.begin(), shift_idx_search_space.end());

    // 2. fast columnwise diff 
    int argmin_shift = 0;
    double min_sc_dist = 10000000;
    for ( int num_shift: shift_idx_search_space )
    {
        MatrixXd sc2_shifted = circshift(_sc2, num_shift);
        double cur_sc_dist = distDirectSC( _sc1, sc2_shifted );
        if( cur_sc_dist < min_sc_dist )
        {
            argmin_shift = num_shift;
            min_sc_dist = cur_sc_dist;
        }
    }

    return make_pair(min_sc_dist, argmin_shift);

} // distanceBtnScanContext

// 构造了scan context矩阵
MatrixXd SCManager::makeScancontext( pcl::PointCloud<SCPointType> & _scan_down )
{
    TicToc t_making_desc;

    int num_pts_scan_down = _scan_down.points.size();

    // main
    const int NO_POINT = -1000;
    MatrixXd desc = NO_POINT * MatrixXd::Ones(PC_NUM_RING, PC_NUM_SECTOR); // 20x60矩阵

    SCPointType pt;
    float azim_angle, azim_range; // wihtin 2d plane
    int ring_idx, sctor_idx;
    for (int pt_idx = 0; pt_idx < num_pts_scan_down; pt_idx++)
    {
        pt.x = _scan_down.points[pt_idx].x; 
        pt.y = _scan_down.points[pt_idx].y;
        pt.z = _scan_down.points[pt_idx].z + LIDAR_HEIGHT; // naive adding is ok (all points should be > 0).

        // xyz to ring, sector
        azim_range = sqrt(pt.x * pt.x + pt.y * pt.y);
        azim_angle = xy2theta(pt.x, pt.y);

        // if range is out of roi, pass
        if( azim_range > PC_MAX_RADIUS )
            continue;

        ring_idx = std::max( std::min( PC_NUM_RING, int(ceil( (azim_range / PC_MAX_RADIUS) * PC_NUM_RING )) ), 1 );
        sctor_idx = std::max( std::min( PC_NUM_SECTOR, int(ceil( (azim_angle / 360.0) * PC_NUM_SECTOR )) ), 1 );

        // taking maximum z 
        if ( desc(ring_idx-1, sctor_idx-1) < pt.z ) // -1 means cpp starts from 0
            desc(ring_idx-1, sctor_idx-1) = pt.z; // update for taking maximum value at that bin
    }

    // reset no points to zero (for cosine dist later)
    for ( int row_idx = 0; row_idx < desc.rows(); row_idx++ )
        for ( int col_idx = 0; col_idx < desc.cols(); col_idx++ )
            if( desc(row_idx, col_idx) == NO_POINT )
                desc(row_idx, col_idx) = 0;

    t_making_desc.toc("PolarContext making");

    return desc;
} // SCManager::makeScancontext

// 环形描述子,输入一个scan的context矩阵,分成20个圆环,每一个圆环是一个row,row里面所有的点的平均值作为其invariant_key
MatrixXd SCManager::makeRingkeyFromScancontext( Eigen::MatrixXd &_desc )
{
    /* 
     * summary: rowwise mean vector
    */
    Eigen::MatrixXd invariant_key(_desc.rows(), 1);
    for ( int row_idx = 0; row_idx < _desc.rows(); row_idx++ )
    {
        Eigen::MatrixXd curr_row = _desc.row(row_idx);
        invariant_key(row_idx, 0) = curr_row.mean();
    }

    return invariant_key;
} // SCManager::makeRingkeyFromScancontext

/**
 * No.3 distanceBtnScanContext调用
 * @param [in] _desc 输入是一个scan的context矩阵
 * @return variant_key:扇区描述子,被分成60个sector,每个扇区是一个col,col里面所有的点的平均值作为variant_key
 */
MatrixXd SCManager::makeSectorkeyFromScancontext( Eigen::MatrixXd &_desc )
{
    /* 
     * summary: columnwise mean vector
     * variant_key因为每个key对应一个扇区sector,受到视角影响,因此是variant_key
    */
    Eigen::MatrixXd variant_key(1, _desc.cols()); // _desc.cols()返回_desc的列数
    for ( int col_idx = 0; col_idx < _desc.cols(); col_idx++ )
    {
        Eigen::MatrixXd curr_col = _desc.col(col_idx);
        variant_key(0, col_idx) = curr_col.mean(); // 把curr_col这一列的元素的平均值赋值给variant_key的第col_idx个元素
    }

    return variant_key;
} // SCManager::makeSectorkeyFromScancontext

/**
 * 输入一帧scan,返回context信息矩阵,RingKey和SectorKey
 * @param _scan_down
 */
void SCManager::makeAndSaveScancontextAndKeys( pcl::PointCloud<SCPointType> & _scan_down,PointType &thisPose3D )
{
        scan_context.polarcontexts_ = makeScancontext(_scan_down);

        scan_context.polarcontext_invkeys_ = makeRingkeyFromScancontext(scan_context.polarcontexts_);

        scan_context.polarcontext_vkeys_ = makeSectorkeyFromScancontext(scan_context.polarcontexts_);

        scan_context.polarcontext_invkeys_mat_ = eig2stdvec(scan_context.polarcontext_invkeys_);
std::cout <<"BBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBB"<<"\n";
        scan_context.thisPose6D_[0] = thisPose3D.x;
        scan_context.thisPose6D_[1] = thisPose3D.y;
        scan_context.thisPose6D_[2] = thisPose3D.z;
std::cout <<"CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC"<<"\n";
        // scan_context.node_id_container = {node_id.trajectory_id,node_id.node_index};

        scan_contexts.emplace_back(scan_context);

    // cout <<polarcontext_vkeys_.size() << endl;

} // SCManager::makeAndSaveScancontextAndKeys

/**
 * No.1 MapOptmization调用
 * @return 历史帧的ID以及yaw角
 */
std::pair<int, float> SCManager::detectLoopClosureID()
// std::pair<int, float> SCManager::detectLoopClosureID(pcl::PointCloud<PointType> & cloudKeyPoses3D)
{
    int loop_id { -1 }; // init with -1, -1 means no loop (== LeGO-LOAM's variable "closestHistoryFrameID")
    // KeyMat类型:二维数组
    auto curr_key = scan_contexts.back().polarcontext_invkeys_mat_; // current observation (query)
    // 所有scan的信息都以矩阵形式存放在polarcontexts_中,取出当前的context
    auto curr_desc = scan_contexts.back().polarcontexts_; // current observation (query)

    /* 
     * step 1: candidates from ringkey tree_
     */
    if( scan_contexts.size() < NUM_EXCLUDE_RECENT + 1) // tree里面不够50个，那还怎么搜索最近的50个,id返回-1,yaw返回0,表示没有loop
    {
        std::pair<int, float> result {loop_id, 0.0};
        return result; // Early return 
    }

    // tree_ reconstruction (not mandatory to make everytime)
    // tree里面够50个,每隔10s进行一次tree重构
    if( tree_making_period_conter % TREE_MAKING_PERIOD_ == 0) // to save computation cost
    {
        TicToc t_tree_construction;

        polarcontext_invkeys_to_search_.clear();
        int num_for_finish = scan_contexts.size();
        for ( auto& iter : scan_contexts){
                polarcontext_invkeys_to_search_.emplace_back(iter.polarcontext_invkeys_mat_);
                num_for_finish--;
                // polarcontext_invkeys_mat_还剩下最近的50个元素没加入搜索,也不需要加入搜索
                if(num_for_finish<=NUM_EXCLUDE_RECENT){cout<<"********************************************************************************************"<<"\n";break;}
        }
        polarcontext_tree_.reset(); 
        polarcontext_tree_ = std::make_unique<InvKeyTree>(PC_NUM_RING /* dim */, polarcontext_invkeys_to_search_, 10 /* max leaf */ );
        // tree_ptr_->index->buildIndex(); // inernally called in the constructor of InvKeyTree (for detail, refer the nanoflann and KDtreeVectorOfVectorsAdaptor)
        t_tree_construction.toc("Tree construction");
    }
    tree_making_period_conter = tree_making_period_conter + 1;
        
    double min_dist = 10000000; // init with somthing large
    int nn_align = 0;
    int nn_idx = 0;

    // knn search 搜10个候选,分别储存其索引和距离
    std::vector<size_t> candidate_indexes( NUM_CANDIDATES_FROM_TREE ); 
    std::vector<float> out_dists_sqr( NUM_CANDIDATES_FROM_TREE );

    TicToc t_tree_search;
    nanoflann::KNNResultSet<float> knnsearch_result( NUM_CANDIDATES_FROM_TREE );
    knnsearch_result.init( &candidate_indexes[0], &out_dists_sqr[0] );
    polarcontext_tree_->index->findNeighbors( knnsearch_result, &curr_key[0] /* query */, nanoflann::SearchParams(10) ); 
    t_tree_search.toc("Tree search");

    /* 
     *  step 2: pairwise distance (find optimal columnwise best-fit using cosine distance)
     */
    TicToc t_calc_dist;   
    for ( int candidate_iter_idx = 0; candidate_iter_idx < NUM_CANDIDATES_FROM_TREE; candidate_iter_idx++ )
    {   // 遍历搜到的每一个context的索引,找到其对应的scan context矩阵信息,赋值给polarcontext_candidate
        MatrixXd polarcontext_candidate = scan_contexts[candidate_indexes[candidate_iter_idx]].polarcontexts_;
        // 然后去和目前的context计算距离,返回距离和旋转方向偏移步长
        std::pair<double, int> sc_dist_result = distanceBtnScanContext( curr_desc, polarcontext_candidate ); 

        double candidate_dist = sc_dist_result.first;
        int candidate_align = sc_dist_result.second;
        // 找到候选scan中距离最小的,得到其距离,旋转方向偏移步长还有索引
        if( candidate_dist < min_dist )
        {
            min_dist = candidate_dist;
            nn_align = candidate_align;

            nn_idx = candidate_indexes[candidate_iter_idx];
        }
    }
    t_calc_dist.toc("Distance calc");



        //loop threshold check 如果距离小于回环设定的阈值,则认为是回环
        if (min_dist < SC_DIST_THRES) {
/**
 * modified by cjf
 * 找到的距离最小的scan,如果满足回环的距离阈值,则判断其索引对应的关键帧的位姿Z值与当前帧Z值进行比较
 * 如果过大说明不在同一层,此时不能认为是回环,否则容易误匹配
 * 如果没有超过,才能认为是最终的真回环
 */
/** created by cjf*******************************************/
            // if (std::abs(cloudKeyPoses3D.at(nn_idx).y - cloudKeyPoses3D.back().y) > LOOP_MAX_Z_DIFF) {
            //     std::cout << "[Not loop] the robot is on the next floor: " << "current_z is: "
            //               << cloudKeyPoses3D.back().y << " and potential_loop_z is: " << cloudKeyPoses3D.at(nn_idx).y
            //               << "\n";
            // }
/** created by cjf*******************************************/
            // else { // 高度差小于设定
                loop_id = nn_idx;

                // std::cout.precision(3);
                cout << "[Loop found] Nearest distance: " << min_dist << " btn " << scan_contexts.size() - 1 << " and "
                     << nn_idx << "." << endl;
                cout << "[Loop found] yaw diff: " << nn_align * PC_UNIT_SECTORANGLE << " deg."
                     << endl; // 极角方向分了60份,所以每份是6°,乘以旋转方向步长即为yaw角偏移量
            // }
        }
            else {
          //     std::cout.precision(3);
          //      cout << "[Not loop] Nearest distance: " << min_dist << " btn " << polarcontexts_.size() - 1 << " and "
          //          << nn_idx << "." << endl;             loop_id = nn_idx;

                // std::cout.precision(3);
                cout << "[Loop found] Nearest distance: " << min_dist << " btn " << scan_contexts.size() - 1 << " and "
                     << nn_idx << "." << endl;
                cout << "[Loop found] yaw diff: " << nn_align * PC_UNIT_SECTORANGLE << " deg."
                     << endl; // 极角方向分了60份,所以每份是6°,乘以旋转方向步长即为yaw角偏移量
          //      cout << "[Not loop] yaw diff: " << nn_align * PC_UNIT_SECTORANGLE << " deg." << endl;
            } // 距离太大不能作为真回环,返回loop_id为-1

    // To do: return also nn_align (i.e., yaw diff)
    float yaw_diff_rad = deg2rad(nn_align * PC_UNIT_SECTORANGLE);
    std::pair<int, float> result {loop_id, yaw_diff_rad};

    return result;

} // SCManager::detectLoopClosureID

// } // namespace SC2
