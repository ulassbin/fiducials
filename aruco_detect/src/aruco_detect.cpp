/*
 * Copyright (c) 2017-20, Ubiquity Robotics Inc., Austin Hendrix
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice,
 *    this list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 *    this list of conditions and the following disclaimer in the documentation
 *    and/or other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 *
 * The views and conclusions contained in the software and documentation are
 * those of the authors and should not be interpreted as representing official
 * policies, either expressed or implied, of the FreeBSD Project.
 *
 */

#include <assert.h>
#include <sys/time.h>
#include <unistd.h>

#include <ros/ros.h>
#include <tf/transform_datatypes.h>
#include <tf2/LinearMath/Transform.h>
#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_broadcaster.h>
#include <tf2_ros/transform_listener.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.h>
#include <visualization_msgs/Marker.h>
#include <image_transport/image_transport.h>
#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/image_encodings.h>
#include <dynamic_reconfigure/server.h>
#include <std_srvs/SetBool.h>
#include <std_msgs/String.h>

#include "fiducial_msgs/Fiducial.h"
#include "fiducial_msgs/FiducialArray.h"
#include "fiducial_msgs/FiducialTransform.h"
#include "fiducial_msgs/FiducialTransformArray.h"
#include "aruco_detect/DetectorParamsConfig.h"

#include <opencv2/highgui.hpp>
#include <opencv2/aruco.hpp>
#include <opencv2/calib3d.hpp>

#include <list>
#include <string>
#include <boost/algorithm/string.hpp>
#include <boost/shared_ptr.hpp>

using namespace std;
using namespace cv;

typedef boost::shared_ptr< fiducial_msgs::FiducialArray const> FiducialArrayConstPtr;
typedef fiducial_msgs::FiducialArray FiducialArray;
typedef vector <vector <Point2d> >  vec2d;
typedef vector <vector <Point2f> >  vec2f;

struct FiducialBundleVector{
    int parent_fid, child_fid;
    tf2::Transform transform;
};

struct FiducialBundleConfig {
    int bundle_id;
    std::vector<int> fiducial_ids;
    std::vector<FiducialBundleVector> control_vectors;
    vector<Point3f> objPoints;
    vector <Point2f> corners;
    int return_index = -1; // The position to return as the center. -1 is mean
    bool valid = false;
    Vec3d rvecs, tvecs;
    double reprojectionError;
    FiducialBundleConfig(std::string config_path)
    {
        // Read config and do stuff
        return;
    }
    FiducialBundleConfig() 
    {
        //Hardcoded initialization for tests
      
    };

    void printInformation()
    {
        ROS_INFO("Printing Bundle information for bundle %d, corners are %d", bundle_id, valid);
        for(int i=0; i<fiducial_ids.size();i++)
        {
            ROS_INFO("Id %d", fiducial_ids[i]);
            for(int k=0; k<4; k++)
            {
                ROS_INFO("ObjectPoints %.2f,%.2f,%.2f", objPoints[i*4+k].x, objPoints[i*4+k].y, objPoints[i*4+k].z);
                if(valid)
                    ROS_INFO("Corners %.2f %.2f", corners[i*4+k].x, corners[i*4+k].y);
            }
        }

    }
};

typedef FiducialBundleConfig bundleConfig;

class FiducialsNode {
  private:
    std::vector<FiducialBundleConfig> fiducial_bundles;
    ros::Publisher vertices_pub;
    ros::Publisher pose_pub;

    ros::Subscriber caminfo_sub;
    ros::Subscriber vertices_sub;
    ros::Subscriber ignore_sub;
    image_transport::ImageTransport it;
    image_transport::Subscriber img_sub;
    tf2_ros::TransformBroadcaster broadcaster;

    ros::ServiceServer service_enable_detections;

    // if set, we publish the images that contain fiducials
    bool publish_images;
    bool enable_detections;

    double fiducial_len;

    bool doPoseEstimation;
    bool haveCamInfo;
    bool publishFiducialTf;
    bool multi_detection = false;
    vector <vector <Point2d> > corners;
    vector <vector <Point2f> > corners_2f;
    vector <int> ids;
    cv_bridge::CvImagePtr cv_ptr;

    cv::Mat cameraMatrix;
    cv::Mat distortionCoeffs;
    int frameNum;
    int bundle_size;
    std::string frameId;
    std::vector<int> ignoreIds;
    std::map<int, double> fiducialLens;
    std::string bundleConfig;
    ros::NodeHandle nh;
    ros::NodeHandle pnh;

    image_transport::Publisher image_pub;

    cv::Ptr<aruco::DetectorParameters> detectorParams;
    cv::Ptr<aruco::Dictionary> dictionary;

    void handleIgnoreString(const std::string& str);

    void estimatePoseSingleMarkers(float markerLength,
                                   const cv::Mat &cameraMatrix,
                                   const cv::Mat &distCoeffs,
                                   vector<Vec3d>& rvecs, vector<Vec3d>& tvecs,
                                   vector<double>& reprojectionError);

    bool estimatePoseMultiMarkers(float markerLength,
                                const cv::Mat &cameraMatrix,
                                const cv::Mat &distCoeffs,
                                vector<Vec3d>& rvecs, vector<Vec3d>& tvecs,
                                vector<double>& reprojectionError);
    void ignoreCallback(const std_msgs::String &msg);
    void imageCallback(const sensor_msgs::ImageConstPtr &msg);
    void poseEstimateCallback(FiducialArray &msg);
    void camInfoCallback(const sensor_msgs::CameraInfo::ConstPtr &msg);
    void configCallback(aruco_detect::DetectorParamsConfig &config, uint32_t level);

    bool enableDetectionsCallback(std_srvs::SetBool::Request &req,
                        std_srvs::SetBool::Response &res);

    dynamic_reconfigure::Server<aruco_detect::DetectorParamsConfig> configServer;
    dynamic_reconfigure::Server<aruco_detect::DetectorParamsConfig>::CallbackType callbackType;

    void checkBundleValidity();
    void drawAndPublishSingle(vector<Vec3d> rvecs, vector<Vec3d> tvecs, vector<double> reprojectionError, fiducial_msgs::FiducialTransformArray& fta);
    void vector2fto2d(const vec2f in_vec, vec2d& out_vec);
    void drawAndPublishMulti(fiducial_msgs::FiducialTransformArray& fta);

  public:
    FiducialsNode();
};


/**
  * @brief Return object points for the system centered in a single marker, given the marker length
  */
static void getSingleMarkerObjectPoints(float markerLength, vector<Point3f>& objPoints) {

    CV_Assert(markerLength > 0);

    // set coordinate system in the middle of the marker, with Z pointing out
    objPoints.clear();
    objPoints.push_back(Vec3f(-markerLength / 2.f, markerLength / 2.f, 0));
    objPoints.push_back(Vec3f( markerLength / 2.f, markerLength / 2.f, 0));
    objPoints.push_back(Vec3f( markerLength / 2.f,-markerLength / 2.f, 0));
    objPoints.push_back(Vec3f(-markerLength / 2.f,-markerLength / 2.f, 0));
}

static void getSingleMarkerObjectPoints(float markerLength, vector<Point3d>& objPoints) {

    CV_Assert(markerLength > 0);

    // set coordinate system in the middle of the marker, with Z pointing out
    objPoints.clear();
    objPoints.push_back(Vec3d(-markerLength / 2.f, markerLength / 2.f, 0));
    objPoints.push_back(Vec3d( markerLength / 2.f, markerLength / 2.f, 0));
    objPoints.push_back(Vec3d( markerLength / 2.f,-markerLength / 2.f, 0));
    objPoints.push_back(Vec3d(-markerLength / 2.f,-markerLength / 2.f, 0));
}

static void getMultiMarkerObjectPoints(float markerLength, vector<Point3f>& objPoints, int num_markers) {

    CV_Assert(markerLength > 0);

    // set coordinate system in the middle of the marker, with Z pointing out
    objPoints.clear();
    if(num_markers == 2 )
    {
        objPoints.push_back(Vec3f(  -markerLength / 2.f,   markerLength / 2.f, 0)); // Base Marker
        objPoints.push_back(Vec3f(   markerLength / 2.f,   markerLength / 2.f, 0));
        objPoints.push_back(Vec3f(   markerLength / 2.f,  -markerLength / 2.f, 0));
        objPoints.push_back(Vec3f(  -markerLength / 2.f,  -markerLength / 2.f, 0));

        objPoints.push_back(Vec3f(   markerLength / 2.f,-3*markerLength / 2.f, 0));
        objPoints.push_back(Vec3f( 3*markerLength / 2.f,-3*markerLength / 2.f, 0));
        objPoints.push_back(Vec3f( 3*markerLength / 2.f,-5*markerLength / 2.f, 0));
        objPoints.push_back(Vec3f(   markerLength / 2.f,-5*markerLength / 2.f, 0));

    }
    else
    {
        objPoints.push_back(Vec3f(  -markerLength / 2.f,   markerLength / 2.f, 0));
        objPoints.push_back(Vec3f(   markerLength / 2.f,   markerLength / 2.f, 0));
        objPoints.push_back(Vec3f(   markerLength / 2.f,  -markerLength / 2.f, 0));
        objPoints.push_back(Vec3f(  -markerLength / 2.f,  -markerLength / 2.f, 0)); 

        objPoints.push_back(Vec3f( 3*markerLength / 2.f,  -markerLength / 2.f, 0));
        objPoints.push_back(Vec3f( 5*markerLength / 2.f,  -markerLength / 2.f, 0));
        objPoints.push_back(Vec3f( 5*markerLength / 2.f,-3*markerLength / 2.f, 0));
        objPoints.push_back(Vec3f( 3*markerLength / 2.f,-3*markerLength / 2.f, 0));

        objPoints.push_back(Vec3f(  -markerLength / 2.f,-3*markerLength / 2.f, 0));
        objPoints.push_back(Vec3f(   markerLength / 2.f,-3*markerLength / 2.f, 0));
        objPoints.push_back(Vec3f(   markerLength / 2.f,-5*markerLength / 2.f, 0));
        objPoints.push_back(Vec3f(  -markerLength / 2.f,-5*markerLength / 2.f, 0)); 

        objPoints.push_back(Vec3f( 3*markerLength / 2.f,-5*markerLength / 2.f, 0));
        objPoints.push_back(Vec3f( 5*markerLength / 2.f,-5*markerLength / 2.f, 0));
        objPoints.push_back(Vec3f( 5*markerLength / 2.f,-7*markerLength / 2.f, 0));
        objPoints.push_back(Vec3f( 3*markerLength / 2.f,-7*markerLength / 2.f, 0));      
    }
}


// Euclidean distance between two points
static double dist(const cv::Point2f &p1, const cv::Point2f &p2)
{
    double x1 = p1.x;
    double y1 = p1.y;
    double x2 = p2.x;
    double y2 = p2.y;

    double dx = x1 - x2;
    double dy = y1 - y2;

    return sqrt(dx*dx + dy*dy);
}

// Compute area in image of a fiducial, using Heron's formula
// to find the area of two triangles
static double calcFiducialArea(const std::vector<cv::Point2f> &pts)
{
    const Point2f &p0 = pts.at(0);
    const Point2f &p1 = pts.at(1);
    const Point2f &p2 = pts.at(2);
    const Point2f &p3 = pts.at(3);

    double a1 = dist(p0, p1);
    double b1 = dist(p0, p3);
    double c1 = dist(p1, p3);

    double a2 = dist(p1, p2);
    double b2 = dist(p2, p3);
    double c2 = c1;

    double s1 = (a1 + b1 + c1) / 2.0;
    double s2 = (a2 + b2 + c2) / 2.0;

    a1 = sqrt(s1*(s1-a1)*(s1-b1)*(s1-c1));
    a2 = sqrt(s2*(s2-a2)*(s2-b2)*(s2-c2));
    return a1+a2;
}

static double calcFiducialArea(const std::vector<cv::Point2d> &pts)
{
    const Point2d &p0 = pts.at(0);
    const Point2d &p1 = pts.at(1);
    const Point2d &p2 = pts.at(2);
    const Point2d &p3 = pts.at(3);

    double a1 = dist(p0, p1);
    double b1 = dist(p0, p3);
    double c1 = dist(p1, p3);

    double a2 = dist(p1, p2);
    double b2 = dist(p2, p3);
    double c2 = c1;

    double s1 = (a1 + b1 + c1) / 2.0;
    double s2 = (a2 + b2 + c2) / 2.0;

    a1 = sqrt(s1*(s1-a1)*(s1-b1)*(s1-c1));
    a2 = sqrt(s2*(s2-a2)*(s2-b2)*(s2-c2));
    return a1+a2;
}

// estimate reprojection error
static double getReprojectionError(const vector<Point3f> &objectPoints,
                            const vector<Point2d> &imagePoints,
                            const Mat &cameraMatrix, const Mat  &distCoeffs,
                            const Vec3d &rvec, const Vec3d &tvec) {

    vector<Point2f> projectedPoints;

    cv::projectPoints(objectPoints, rvec, tvec, cameraMatrix,
                      distCoeffs, projectedPoints);

    // calculate RMS image error
    double totalError = 0.0;
    for (unsigned int i=0; i<objectPoints.size(); i++) {
        double error = dist(imagePoints[i], projectedPoints[i]);
        totalError += error*error;
    }
    double rerror = totalError/(double)objectPoints.size();
    return rerror;
}

static double getReprojectionError(const vector<Point3f> &objectPoints,
                            const vector<Point2f> &imagePoints,
                            const Mat &cameraMatrix, const Mat  &distCoeffs,
                            const Vec3d &rvec, const Vec3d &tvec) {

    vector<Point2f> projectedPoints;

    cv::projectPoints(objectPoints, rvec, tvec, cameraMatrix,
                      distCoeffs, projectedPoints);

    // calculate RMS image error
    double totalError = 0.0;
    for (unsigned int i=0; i<objectPoints.size(); i++) {
        double error = dist(imagePoints[i], projectedPoints[i]);
        totalError += error*error;
    }
    double rerror = totalError/(double)objectPoints.size();
    return rerror;
}

static double getReprojectionError(const vector<Point3d> &objectPoints,
                            const vector<Point2d> &imagePoints,
                            const Mat &cameraMatrix, const Mat  &distCoeffs,
                            const Vec3d &rvec, const Vec3d &tvec) {

    vector<Point2f> projectedPoints;

    cv::projectPoints(objectPoints, rvec, tvec, cameraMatrix,
                      distCoeffs, projectedPoints);

    // calculate RMS image error
    double totalError = 0.0;
    for (unsigned int i=0; i<objectPoints.size(); i++) {
        double error = dist(imagePoints[i], projectedPoints[i]);
        totalError += error*error;
    }
    double rerror = totalError/(double)objectPoints.size();
    return rerror;
}
void FiducialsNode::estimatePoseSingleMarkers(float markerLength,
                                const cv::Mat &cameraMatrix,
                                const cv::Mat &distCoeffs,
                                vector<Vec3d>& rvecs, vector<Vec3d>& tvecs,
                                vector<double>& reprojectionError) {

    CV_Assert(markerLength > 0);

    vector<Point3f> markerObjPoints;
    int nMarkers = (int)corners.size();
    rvecs.reserve(nMarkers);
    tvecs.reserve(nMarkers);
    reprojectionError.reserve(nMarkers);

    // for each marker, calculate its pose
    for (int i = 0; i < nMarkers; i++) {
       double fiducialSize = markerLength;

       std::map<int, double>::iterator it = fiducialLens.find(ids[i]);
       if (it != fiducialLens.end()) {
          fiducialSize = it->second;
       }

       getSingleMarkerObjectPoints(fiducialSize, markerObjPoints);
       cv::solvePnP(markerObjPoints, corners[i], cameraMatrix, distCoeffs,
                    rvecs[i], tvecs[i]);

       reprojectionError[i] =
          getReprojectionError(markerObjPoints, corners[i],
                               cameraMatrix, distCoeffs,
                               rvecs[i], tvecs[i]);
    }
}

bool FiducialsNode::estimatePoseMultiMarkers(float markerLength,
                                const cv::Mat &cameraMatrix,
                                const cv::Mat &distCoeffs,
                                vector<Vec3d>& rvecs, vector<Vec3d>& tvecs,
                                vector<double>& reprojectionError) {

    CV_Assert(markerLength > 0);

    vector<Point3f> markerObjPoints;
    int nMarkers = (int)corners.size();
    rvecs.reserve(1);
    tvecs.reserve(1);
    reprojectionError.reserve(1);
    // vector <vector <Point2f> > corners;
    vector<vector<Point2f>> corner_sets;
    vector <Point2f> corners_combined; // And there are ids
    Point2f temp2f;
    corners_combined.clear();
    Vec3d rvecs_combined, tvecs_combined;
    double reprojectionError_combined;
    int countz = 0; 
    int idmap = 0;
    std::vector<int>::iterator it;
    int marker_start = 0;
    for(int i = 0; i < fiducial_bundles.size(); i++ ) // This should combine based on ids!!! 
    {
        fiducial_bundles[i].corners.clear();
        fiducial_bundles[i].valid = true;
        for(int j = 0; j < fiducial_bundles[i].fiducial_ids.size(); j++) // Check
        {
            it = std::find (ids.begin(), ids.end(), fiducial_bundles[i].fiducial_ids[j]);
            if(it == ids.end())
            {
                ROS_INFO("Couldn't find marker %d required for bundle %d",fiducial_bundles[i].fiducial_ids[j], fiducial_bundles[i].bundle_id);
                for(int k=0; k<ids.size(); k++)
                    ROS_INFO("Existing id %d",ids[k]);
                fiducial_bundles[i].valid = false;
                break;
            }
            else
            {
                marker_start = std::distance(ids.begin(), it);
                fiducial_bundles[i].corners.insert (fiducial_bundles[i].corners.end(), corners[marker_start].begin(), corners[marker_start].end());
            }
        }
        fiducial_bundles[i].printInformation();
    }

    // for all markers, calculate their pose in single iteration
    for (int i = 0; i < fiducial_bundles.size(); i++) {
        if(!fiducial_bundles[i].valid)
        {
            ROS_INFO("Fiducial_bundle %d not valid, skipping",fiducial_bundles[i].bundle_id);
            continue;
        }

       markerObjPoints = fiducial_bundles[i].objPoints;
       // InputArray _opoints = markerObjPoints;
       // InputArray _ipoints = corners_combined;
       // Mat opoints = _opoints.getMat();
       // Mat ipoints = _ipoints.getMat();
       // int npoints = std::max(opoints.checkVector(3, CV_32F), opoints.checkVector(3, CV_64F));
       //ROS_INFO("Assertion is %d , %d", npoints, std::max(ipoints.checkVector(2, CV_32F), ipoints.checkVector(2, CV_64F)));
       // ROS_INFO("Corners %d, ObjectPoints %d, npoints %d", countz, markerObjPoints.size(), npoints);
       cv::solvePnP(markerObjPoints, fiducial_bundles[i].corners, cameraMatrix, distCoeffs, rvecs_combined, tvecs_combined, false);//);

       fiducial_bundles[i].rvecs = rvecs_combined;
       fiducial_bundles[i].tvecs = tvecs_combined;      
        //    return false;
        ROS_INFO_STREAM("Rvec: "<<rvecs_combined<<" tvecs: "<< tvecs_combined);
        reprojectionError_combined = 0;
        fiducial_bundles[i].reprojectionError = reprojectionError_combined;
    }
    ROS_INFO("Out of matching");
    return true;
}


void FiducialsNode::configCallback(aruco_detect::DetectorParamsConfig & config, uint32_t level)
{
    /* Don't load initial config, since it will overwrite the rosparam settings */
    if (level == 0xFFFFFFFF) {
        return;
    }

    detectorParams->adaptiveThreshConstant = config.adaptiveThreshConstant;
    detectorParams->adaptiveThreshWinSizeMin = config.adaptiveThreshWinSizeMin;
    detectorParams->adaptiveThreshWinSizeMax = config.adaptiveThreshWinSizeMax;
    detectorParams->adaptiveThreshWinSizeStep = config.adaptiveThreshWinSizeStep;
    detectorParams->cornerRefinementMaxIterations = config.cornerRefinementMaxIterations;
    detectorParams->cornerRefinementMinAccuracy = config.cornerRefinementMinAccuracy;
    detectorParams->cornerRefinementWinSize = config.cornerRefinementWinSize;
#if CV_MINOR_VERSION==2 and CV_MAJOR_VERSION==3
    detectorParams->doCornerRefinement = config.doCornerRefinement;
#else
    if (config.doCornerRefinement) {
       if (config.cornerRefinementSubpix) {
         detectorParams->cornerRefinementMethod = aruco::CORNER_REFINE_SUBPIX;
       }
       else {
         detectorParams->cornerRefinementMethod = aruco::CORNER_REFINE_CONTOUR;
       }
    }
    else {
       detectorParams->cornerRefinementMethod = aruco::CORNER_REFINE_NONE;
    }
#endif
    detectorParams->errorCorrectionRate = config.errorCorrectionRate;
    detectorParams->minCornerDistanceRate = config.minCornerDistanceRate;
    detectorParams->markerBorderBits = config.markerBorderBits;
    detectorParams->maxErroneousBitsInBorderRate = config.maxErroneousBitsInBorderRate;
    detectorParams->minDistanceToBorder = config.minDistanceToBorder;
    detectorParams->minMarkerDistanceRate = config.minMarkerDistanceRate;
    detectorParams->minMarkerPerimeterRate = config.minMarkerPerimeterRate;
    detectorParams->maxMarkerPerimeterRate = config.maxMarkerPerimeterRate;
    detectorParams->minOtsuStdDev = config.minOtsuStdDev;
    detectorParams->perspectiveRemoveIgnoredMarginPerCell = config.perspectiveRemoveIgnoredMarginPerCell;
    detectorParams->perspectiveRemovePixelPerCell = config.perspectiveRemovePixelPerCell;
    detectorParams->polygonalApproxAccuracyRate = config.polygonalApproxAccuracyRate;
}

void FiducialsNode::ignoreCallback(const std_msgs::String& msg)
{
    ignoreIds.clear();
    pnh.setParam("ignore_fiducials", msg.data);
    handleIgnoreString(msg.data);
}

void FiducialsNode::camInfoCallback(const sensor_msgs::CameraInfo::ConstPtr& msg)
{
    if (haveCamInfo) {
        return;
    }

    if (msg->K != boost::array<double, 9>({0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0})) {
        for (int i=0; i<3; i++) {
            for (int j=0; j<3; j++) {
                cameraMatrix.at<double>(i, j) = msg->K[i*3+j];
            }
        }

        for (int i=0; i<5; i++) {
            distortionCoeffs.at<double>(0,i) = msg->D[i];
        }

        haveCamInfo = true;
        frameId = msg->header.frame_id;
    }
    else {
        ROS_WARN("%s", "CameraInfo message has invalid intrinsics, K matrix all zeros");
    }
}

void FiducialsNode::vector2fto2d(const vec2f in_vec, vec2d& out_vec)
{   
    std::vector <Point2d>  smallvec;
    Point2d p2d;
    out_vec.clear();
    int size1 = in_vec.size();
    int size2 = 0;
    ROS_INFO("Size1 %d", size1);
    out_vec.reserve(size1);
    for(int i = 0; i < in_vec.size(); i++)
    {   
        size2 = in_vec[i].size();
        (out_vec[i]).reserve(size2);
        smallvec.clear();
        //ROS_INFO("Reserving %d for %d from %d vector", i, size2, size1);
        for(int j = 0; j < in_vec[i].size(); j++)
        {
            //ROS_INFO("Accessing %d,%d, sizes %d",i, j, int(out_vec.size()));
            p2d.x = in_vec[i][j].x;
            p2d.y = in_vec[i][j].y;
            smallvec.push_back(p2d);
        }
        out_vec.push_back(smallvec);
    }
}

void FiducialsNode::imageCallback(const sensor_msgs::ImageConstPtr & msg)
{
    if (enable_detections == false) {
        return; //return without doing anything
    }

    ROS_INFO("Got image %d", msg->header.seq);

    fiducial_msgs::FiducialArray fva;
    fva.header.stamp = msg->header.stamp;
    fva.header.frame_id = frameId;
    fva.image_seq = msg->header.seq;

    try {
        cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8);

        aruco::detectMarkers(cv_ptr->image, dictionary, corners_2f, ids, detectorParams);
        vector2fto2d(corners_2f, corners);
        //ROS_INFO("Detected %d markers", (int)ids.size());

        for (size_t i=0; i<ids.size(); i++) {
	    if (std::count(ignoreIds.begin(), ignoreIds.end(), ids[i]) != 0) {
	        ROS_INFO("Ignoring id %d", ids[i]);
	        continue;
	    }
            fiducial_msgs::Fiducial fid;
            fid.fiducial_id = ids[i];

            fid.x0 = corners[i][0].x;
            fid.y0 = corners[i][0].y;
            fid.x1 = corners[i][1].x;
            fid.y1 = corners[i][1].y;
            fid.x2 = corners[i][2].x;
            fid.y2 = corners[i][2].y;
            fid.x3 = corners[i][3].x;
            fid.y3 = corners[i][3].y;
            fva.fiducials.push_back(fid);
        }

        vertices_pub.publish(fva);
        if(doPoseEstimation)
            poseEstimateCallback(fva);
        // if(ids.size() > 0) {
        //     aruco::drawDetectedMarkers(cv_ptr->image, corners, ids);
        // }

        if (publish_images) {
	    image_pub.publish(cv_ptr->toImageMsg());
        }
    }
    catch(cv_bridge::Exception & e) {
        ROS_ERROR("cv_bridge exception: %s", e.what());
    }
    catch(cv::Exception & e) {
        ROS_ERROR("cv exception: %s", e.what());
    }
}

void FiducialsNode::poseEstimateCallback(FiducialArray & msg)
{
    vector <Vec3d>  rvecs, tvecs;

    fiducial_msgs::FiducialTransformArray fta;
    fta.header.stamp = msg.header.stamp;
    fta.header.frame_id = frameId;
    fta.image_seq = msg.header.seq;
    frameNum++;
    int singleEstimation = multi_detection;
    if (doPoseEstimation) {
        try {
            if (!haveCamInfo) {
                if (frameNum > 5) {
                    ROS_ERROR("No camera intrinsics");
                }
                return;
            }

            vector <double>reprojectionError;
            if(!singleEstimation)
            {
                // if(msg.fiducials.size() != bundle_size || ids.size() != bundle_size)
                // {
                //     ROS_WARN("Not enough fids, current %d, %d", int(msg.fiducials.size()), int(ids.size()));
                //     return;
                // }
                if(estimatePoseMultiMarkers((float)fiducial_len,
                              cameraMatrix, distortionCoeffs,
                              rvecs, tvecs,
                              reprojectionError))
                {
                    drawAndPublishMulti(fta);
                }

            }
            else
            {
                ROS_INFO("Doing Single estimation");
                estimatePoseSingleMarkers((float)fiducial_len,
                                      cameraMatrix, distortionCoeffs,
                                      rvecs, tvecs,
                                      reprojectionError);
                drawAndPublishSingle(rvecs, tvecs, reprojectionError, fta);
            }   

        }
        catch(cv_bridge::Exception & e) {
            ROS_ERROR("cv_bridge exception: %s", e.what());
        }
        catch(cv::Exception & e) {
            ROS_ERROR("cv exception: %s", e.what());
        }
    }
    pose_pub.publish(fta);
}


void FiducialsNode::drawAndPublishMulti(fiducial_msgs::FiducialTransformArray& fta)
{

    for (size_t i = 0; i < fiducial_bundles.size(); i++) { // i < (ids.size() - 1) * (singleEstimation) + 1
        if(!fiducial_bundles[i].valid)
            continue;
        aruco::drawAxis(cv_ptr->image, cameraMatrix, distortionCoeffs,
                        fiducial_bundles[i].rvecs, fiducial_bundles[i].tvecs, (float)0.1);

        ROS_INFO("Detected id %d T %.2f %.2f %.2f R %.2f %.2f %.2f", ids[i],
                 fiducial_bundles[i].tvecs[0], fiducial_bundles[i].tvecs[1], fiducial_bundles[i].tvecs[2],
                 fiducial_bundles[i].rvecs[0], fiducial_bundles[i].rvecs[1], fiducial_bundles[i].rvecs[2]);

        if (std::count(ignoreIds.begin(), ignoreIds.end(), ids[i]) != 0) {
            ROS_INFO("Ignoring id %d", ids[i]);
            continue;
        }

        double angle = norm(fiducial_bundles[i].rvecs);
        Vec3d axis = fiducial_bundles[i].rvecs / angle;
        ROS_INFO("angle %f axis %f %f %f",
                 angle, axis[0], axis[1], axis[2]);

        fiducial_msgs::FiducialTransform ft;
        ft.fiducial_id = ids[i];

        ft.transform.translation.x = fiducial_bundles[i].tvecs[0];
        ft.transform.translation.y = fiducial_bundles[i].tvecs[1];
        ft.transform.translation.z = fiducial_bundles[i].tvecs[2];

        tf2::Quaternion q;
        q.setRotation(tf2::Vector3(axis[0], axis[1], axis[2]), angle);

        ft.transform.rotation.w = q.w();
        ft.transform.rotation.x = q.x();
        ft.transform.rotation.y = q.y();
        ft.transform.rotation.z = q.z();

        ft.fiducial_area = 10; //Fix here //calcFiducialArea(corners[i]);
        ft.image_error = fiducial_bundles[i].reprojectionError;

        // Convert image_error (in pixels) to object_error (in meters)
        ft.object_error =
            (fiducial_bundles[i].reprojectionError / dist(fiducial_bundles[i].corners[0], fiducial_bundles[i].corners[2])) *
            (norm(fiducial_bundles[i].tvecs) / fiducial_len);

        fta.transforms.push_back(ft);

        // Publish tf for the fiducial relative to the camera
        if (publishFiducialTf) {
            geometry_msgs::TransformStamped ts;
            ts.transform = ft.transform;
            ts.header.frame_id = frameId;
            ts.header.stamp = fta.header.stamp;
            ts.child_frame_id = "fiducial_" + std::to_string(ft.fiducial_id);
            broadcaster.sendTransform(ts);
        }
    }
}



void FiducialsNode::drawAndPublishSingle(vector<Vec3d> rvecs, vector<Vec3d> tvecs, vector<double> reprojectionError, fiducial_msgs::FiducialTransformArray& fta)
{

    for (size_t i = 0; i < ids.size(); i++) { // i < (ids.size() - 1) * (singleEstimation) + 1
        aruco::drawAxis(cv_ptr->image, cameraMatrix, distortionCoeffs,
                        rvecs[i], tvecs[i], (float)fiducial_len);

        ROS_INFO("Detected id %d T %.2f %.2f %.2f R %.2f %.2f %.2f", ids[i],
                 tvecs[i][0], tvecs[i][1], tvecs[i][2],
                 rvecs[i][0], rvecs[i][1], rvecs[i][2]);

        if (std::count(ignoreIds.begin(), ignoreIds.end(), ids[i]) != 0) {
            ROS_INFO("Ignoring id %d", ids[i]);
            continue;
        }

        double angle = norm(rvecs[i]);
        Vec3d axis = rvecs[i] / angle;
        ROS_INFO("angle %f axis %f %f %f",
                 angle, axis[0], axis[1], axis[2]);

        fiducial_msgs::FiducialTransform ft;
        ft.fiducial_id = ids[i];

        ft.transform.translation.x = tvecs[i][0];
        ft.transform.translation.y = tvecs[i][1];
        ft.transform.translation.z = tvecs[i][2];

        tf2::Quaternion q;
        q.setRotation(tf2::Vector3(axis[0], axis[1], axis[2]), angle);

        ft.transform.rotation.w = q.w();
        ft.transform.rotation.x = q.x();
        ft.transform.rotation.y = q.y();
        ft.transform.rotation.z = q.z();

        ft.fiducial_area = 10; //Fix here //calcFiducialArea(corners[i]);
        ft.image_error = reprojectionError[i];

        // Convert image_error (in pixels) to object_error (in meters)
        ft.object_error =
            (reprojectionError[i] / dist(corners[i][0], corners[i][2])) *
            (norm(tvecs[i]) / fiducial_len);

        fta.transforms.push_back(ft);

        // Publish tf for the fiducial relative to the camera
        if (publishFiducialTf) {
            geometry_msgs::TransformStamped ts;
            ts.transform = ft.transform;
            ts.header.frame_id = frameId;
            ts.header.stamp = fta.header.stamp;
            ts.child_frame_id = "fiducial_" + std::to_string(ft.fiducial_id);
            broadcaster.sendTransform(ts);
        }
    }
}

void FiducialsNode::checkBundleValidity()
{
    // Basic idea is that the markes on the bundles has fixed transforms wtr to each other.
    // If one is deteted false, the transforms should be broken.

    return;
}

void FiducialsNode::handleIgnoreString(const std::string& str)
{
    /*
    ignogre fiducials can take comma separated list of individual
    fiducial ids or ranges, eg "1,4,8,9-12,30-40"
    */
    std::vector<std::string> strs;
    boost::split(strs, str, boost::is_any_of(","));
    for (const string& element : strs) {
        if (element == "") {
           continue;
        }
        std::vector<std::string> range;
        boost::split(range, element, boost::is_any_of("-"));
        if (range.size() == 2) {
           int start = std::stoi(range[0]);
           int end = std::stoi(range[1]);
           ROS_INFO("Ignoring fiducial id range %d to %d", start, end);
           for (int j=start; j<=end; j++) {
               ignoreIds.push_back(j);
           }
        }
        else if (range.size() == 1) {
           int fid = std::stoi(range[0]);
           ROS_INFO("Ignoring fiducial id %d", fid);
           ignoreIds.push_back(fid);
        }
        else {
           ROS_ERROR("Malformed ignore_fiducials: %s", element.c_str());
        }
    }
}

bool FiducialsNode::enableDetectionsCallback(std_srvs::SetBool::Request &req,
                                std_srvs::SetBool::Response &res)
{
    enable_detections = req.data;
    if (enable_detections){
        res.message = "Enabled aruco detections.";
        ROS_INFO("Enabled aruco detections.");
    }
    else {
        res.message = "Disabled aruco detections.";
        ROS_INFO("Disabled aruco detections.");
    }

    res.success = true;
    return true;
}

inline std::vector<bundleConfig> parseBundles(
        XmlRpc::XmlRpcValue &bundles_xml) {
    std::vector<bundleConfig> bundles;
    for (int32_t i = 0; i < bundles_xml.size(); i++) {
        XmlRpc::XmlRpcValue &bundle_xml = bundles_xml[i];
        ROS_INFO("Doing stuff");
        bundleConfig conf;
        conf.bundle_id = i;
        XmlRpc::XmlRpcValue &markers_xml = bundle_xml["markers"];
        ROS_INFO("Casted to markers_xml");
        for (int32_t i = 0; i < markers_xml.size(); i++)
        {
            XmlRpc::XmlRpcValue &marker_xml = markers_xml[i];

            conf.fiducial_ids.push_back(marker_xml["id"]); // tag id
            Vec3f cv_vector;
            auto c = marker_xml["corners"];
            for (int i = 0; i < 4; ++i) 
            {
                conf.objPoints.push_back(Vec3f(double(c[i][0]), double(c[i][1]), double(c[i][2])));
            }
        }
        bundles.push_back(conf);
    }

    return bundles;
}


FiducialsNode::FiducialsNode() : nh(), pnh("~"), it(nh)
{
    frameNum = 0;

    // Camera intrinsics
    cameraMatrix = cv::Mat::zeros(3, 3, CV_64F);

    // distortion coefficients
    distortionCoeffs = cv::Mat::zeros(1, 5, CV_64F);

    haveCamInfo = false;
    enable_detections = true;

    int dicno;

    detectorParams = new aruco::DetectorParameters();
    pnh.param<bool>("multi_detection", multi_detection, false);
    pnh.param<int>("bundle_size", bundle_size, 4);
    pnh.param<bool>("publish_images", publish_images, false);
    pnh.param<double>("fiducial_len", fiducial_len, 0.14);
    pnh.param<int>("dictionary", dicno, 7);
    pnh.param<bool>("do_pose_estimation", doPoseEstimation, true);
    pnh.param<bool>("publish_fiducial_tf", publishFiducialTf, true);
    //pnh.param<string>("bundles", bundleConfig, true);
    XmlRpc::XmlRpcValue param_list;
    if( !pnh.getParam("bundles", param_list) )
        ROS_ERROR("Still failed...");
    else
    {
        ROS_INFO("Got yaml parameter");
        // Printing here
        fiducial_bundles = parseBundles(param_list);
        for(int i = 0; i < fiducial_bundles.size(); i++)
            fiducial_bundles[i].printInformation();
    }

    std::string str;
    std::vector<std::string> strs;

    pnh.param<string>("ignore_fiducials", str, "");
    handleIgnoreString(str);

    /*
    fiducial size can take comma separated list of size: id or size: range,
    e.g. "200.0: 12, 300.0: 200-300"
    */
    pnh.param<string>("fiducial_len_override", str, "");
    boost::split(strs, str, boost::is_any_of(","));
    for (const string& element : strs) {
        if (element == "") {
           continue;
        }
        std::vector<std::string> parts;
        boost::split(parts, element, boost::is_any_of(":"));
        if (parts.size() == 2) {
            double len = std::stod(parts[1]);
            std::vector<std::string> range;
            boost::split(range, element, boost::is_any_of("-"));
            if (range.size() == 2) {
               int start = std::stoi(range[0]);
               int end = std::stoi(range[1]);
               ROS_INFO("Setting fiducial id range %d - %d length to %f",
                        start, end, len);
               for (int j=start; j<=end; j++) {
                   fiducialLens[j] = len;
               }
            }
            else if (range.size() == 1){
               int fid = std::stoi(range[0]);
               ROS_INFO("Setting fiducial id %d length to %f", fid, len);
               fiducialLens[fid] = len;
            }
            else {
               ROS_ERROR("Malformed fiducial_len_override: %s", element.c_str());
            }
        }
        else {
           ROS_ERROR("Malformed fiducial_len_override: %s", element.c_str());
        }
    }

    image_pub = it.advertise("/fiducial_images", 1);

    vertices_pub = nh.advertise<fiducial_msgs::FiducialArray>("fiducial_vertices", 1);

    pose_pub = nh.advertise<fiducial_msgs::FiducialTransformArray>("fiducial_transforms", 1);

    dictionary = aruco::getPredefinedDictionary(dicno);

    img_sub = it.subscribe("camera", 1,
                        &FiducialsNode::imageCallback, this);

    //vertices_sub = nh.subscribe("fiducial_vertices", 1,
    //                &FiducialsNode::poseEstimateCallback, this);
    caminfo_sub = nh.subscribe("camera_info", 1,
                    &FiducialsNode::camInfoCallback, this);

    ignore_sub = nh.subscribe("ignore_fiducials", 1,
                              &FiducialsNode::ignoreCallback, this);

    service_enable_detections = nh.advertiseService("enable_detections",
                        &FiducialsNode::enableDetectionsCallback, this);

    callbackType = boost::bind(&FiducialsNode::configCallback, this, _1, _2);
    configServer.setCallback(callbackType);

    pnh.param<double>("adaptiveThreshConstant", detectorParams->adaptiveThreshConstant, 7);
    pnh.param<int>("adaptiveThreshWinSizeMax", detectorParams->adaptiveThreshWinSizeMax, 53); /* defailt 23 */
    pnh.param<int>("adaptiveThreshWinSizeMin", detectorParams->adaptiveThreshWinSizeMin, 3);
    pnh.param<int>("adaptiveThreshWinSizeStep", detectorParams->adaptiveThreshWinSizeStep, 4); /* default 10 */
    pnh.param<int>("cornerRefinementMaxIterations", detectorParams->cornerRefinementMaxIterations, 30);
    pnh.param<double>("cornerRefinementMinAccuracy", detectorParams->cornerRefinementMinAccuracy, 0.01); /* default 0.1 */
    pnh.param<int>("cornerRefinementWinSize", detectorParams->cornerRefinementWinSize, 5);
#if CV_MINOR_VERSION==2 and CV_MAJOR_VERSION==3
    pnh.param<bool>("doCornerRefinement",detectorParams->doCornerRefinement, true); /* default false */
#else
    bool doCornerRefinement = true;
    pnh.param<bool>("doCornerRefinement", doCornerRefinement, true);
    if (doCornerRefinement) {
       bool cornerRefinementSubPix = true;
       pnh.param<bool>("cornerRefinementSubPix", cornerRefinementSubPix, true);
       if (cornerRefinementSubPix) {
         detectorParams->cornerRefinementMethod = aruco::CORNER_REFINE_SUBPIX;
       }
       else {
         detectorParams->cornerRefinementMethod = aruco::CORNER_REFINE_CONTOUR;
       }
    }
    else {
       detectorParams->cornerRefinementMethod = aruco::CORNER_REFINE_NONE;
    }
#endif
    pnh.param<double>("errorCorrectionRate", detectorParams->errorCorrectionRate , 0.6);
    pnh.param<double>("minCornerDistanceRate", detectorParams->minCornerDistanceRate , 0.05);
    pnh.param<int>("markerBorderBits", detectorParams->markerBorderBits, 1);
    pnh.param<double>("maxErroneousBitsInBorderRate", detectorParams->maxErroneousBitsInBorderRate, 0.04);
    pnh.param<int>("minDistanceToBorder", detectorParams->minDistanceToBorder, 3);
    pnh.param<double>("minMarkerDistanceRate", detectorParams->minMarkerDistanceRate, 0.05);
    pnh.param<double>("minMarkerPerimeterRate", detectorParams->minMarkerPerimeterRate, 0.1); /* default 0.3 */
    pnh.param<double>("maxMarkerPerimeterRate", detectorParams->maxMarkerPerimeterRate, 4.0);
    pnh.param<double>("minOtsuStdDev", detectorParams->minOtsuStdDev, 5.0);
    pnh.param<double>("perspectiveRemoveIgnoredMarginPerCell", detectorParams->perspectiveRemoveIgnoredMarginPerCell, 0.13);
    pnh.param<int>("perspectiveRemovePixelPerCell", detectorParams->perspectiveRemovePixelPerCell, 8);
    pnh.param<double>("polygonalApproxAccuracyRate", detectorParams->polygonalApproxAccuracyRate, 0.01); /* default 0.05 */

    ROS_INFO("Aruco detection ready");
}

int main(int argc, char ** argv) {
    ros::init(argc, argv, "aruco_detect");

    FiducialsNode* fd_node = new FiducialsNode();

    ros::spin();

    return 0;
}
