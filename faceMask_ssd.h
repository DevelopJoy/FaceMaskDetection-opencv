#pragma once

#include <vector>
#include <string>

#include <opencv2\opencv.hpp>
#include <dnn\dnn.hpp>
 
#include "anchor.h"

//using namespace cv;

namespace face_ssd{
	float rectOverlap(cv::Rect2f rect_1, cv::Rect2f rect_2)
	{
		bool bIsOverlapReturn = false;

		float maxleftx = std::max(rect_1.x, rect_2.x);
		float maxlefty = std::max(rect_1.y, rect_2.y);
		float minrightx = std::min(rect_1.x + rect_1.width, rect_2.x + rect_2.width);
		float minrigty = std::min(rect_1.y + rect_1.height, rect_2.y + rect_2.height);
		if (maxleftx >= minrightx || maxlefty >= minrigty)
		{
			bIsOverlapReturn = false;
			return 0;
		}
		else
		{
			float areaa = (minrightx - maxleftx)*(minrigty - maxlefty);
			float  fRatio = areaa / (rect_2.width*rect_2.height);

			return fRatio;
		}

	}

	typedef struct _ObjInfo
	{
		cv::Rect2f _faceLoc;
		float _conf;
		int _faceCla;
	} ObjInfo;

	std::vector<ObjInfo> decode_bbox(cv::Mat& rawBox, cv::Mat& rawCls)
	{
		std::vector<ObjInfo> vR; vR.clear();
		if (rawBox.empty() || rawCls.empty())
			return vR;
		if (rawBox.rows != rawCls.rows)
			return vR;

		float ecsplion = 2.718282;
		float variances[4] = { 0.1, 0.1, 0.2, 0.2 };
		float anchor_centers_x[5972];
		float anchor_centers_y[5972];
		float anchors_w[5972];
		float anchors_h[5972];
		float raw_outputs_rescale[5972][4];
		float predict_center_x[5972];
		float predict_center_y[5972];
		float predict_w[5972];
		float predict_h[5972];
		std::vector<int> vValidateIndexList; vValidateIndexList.clear();
		std::vector<int> vValidateClassList; vValidateClassList.clear();
		std::vector<float> vValidateProbabilityList; vValidateProbabilityList.clear();
		std::vector<cv::Rect2f> vValidateRectList; vValidateRectList.clear();
		for (int i = 0; i < 5972; ++i)
		{
			float cls_0 = rawCls.at<float>(i, 0);
			float cls_1 = rawCls.at<float>(i, 1);
			int maxClass = 0;
			float maxCls = cls_0;
			if (cls_1 > cls_0){
				maxClass = 1;
				maxCls = cls_1;
			}
			if (maxCls > 0.5){
				vValidateProbabilityList.push_back(maxCls);
				vValidateIndexList.push_back(i);
				vValidateClassList.push_back(maxClass);
			}
			else{
				continue;
			}

			anchor_centers_x[i] = (anchor_f[i][0] + anchor_f[i][2]) / 2;
			anchor_centers_y[i] = (anchor_f[i][1] + anchor_f[i][3]) / 2;
			anchors_w[i] = anchor_f[i][2] - anchor_f[i][0];
			anchors_h[i] = anchor_f[i][3] - anchor_f[i][1];
			raw_outputs_rescale[i][0] = rawBox.at<float>(i, 0) * variances[0];
			raw_outputs_rescale[i][1] = rawBox.at<float>(i, 1) * variances[1];
			raw_outputs_rescale[i][2] = rawBox.at<float>(i, 2) * variances[2];
			raw_outputs_rescale[i][3] = rawBox.at<float>(i, 3) * variances[3];

			predict_center_x[i] = raw_outputs_rescale[i][0] * anchors_w[i] + anchor_centers_x[i];
			predict_center_y[i] = raw_outputs_rescale[i][1] * anchors_h[i] + anchor_centers_y[i];
			predict_w[i] = std::pow(ecsplion, raw_outputs_rescale[i][2]) * anchors_w[i];
			predict_h[i] = std::pow(ecsplion, raw_outputs_rescale[i][3]) * anchors_h[i];

			cv::Rect2f tempRt;
			tempRt.x = predict_center_x[i] - predict_w[i] / 2;
			tempRt.y = predict_center_y[i] - predict_h[i] / 2;
			tempRt.width = predict_center_x[i] + predict_w[i] / 2 - tempRt.x;
			tempRt.height = predict_center_y[i] + predict_h[i] / 2 - tempRt.y;
			vValidateRectList.push_back(tempRt);
		}

		//置信度从大到小排序
		std::vector<int> vMaxSortList; vMaxSortList.clear();
		std::vector<float> vTempValidateProbabilityList = vValidateProbabilityList;
		std::sort(vTempValidateProbabilityList.begin(), \
			vTempValidateProbabilityList.end(), \
			[](float a1, float a2){return a1 > a2 ? 1 : 0; });
		for (int i = 0; i < vTempValidateProbabilityList.size(); ++i)
		{
			float iConf = vTempValidateProbabilityList[i];
			int maxInd = i;
			for (int j = 0; j < vValidateProbabilityList.size(); ++j)
			{
				float jconf = vValidateProbabilityList[j];
				if (jconf == iConf){
					maxInd = j;
					break;
				}
			}
			vMaxSortList.push_back(maxInd);
		}
		//合并rect
		for (int i = 0; i < vMaxSortList.size(); ++i){
			int tempInd = vMaxSortList[i];
			cv::Rect2f tempRect = vValidateRectList[tempInd];
			for (int j = i + 1; j < vMaxSortList.size(); ++j){
				int probInd = vMaxSortList[j];
				cv::Rect2f probRect = vValidateRectList[probInd];
				float lapRate = rectOverlap(probRect, tempRect);
				if (lapRate > 0.4){
					vMaxSortList.erase(vMaxSortList.begin() + j);
					--j;
				}
			}
		}
		//输出结果
		vR.resize(vMaxSortList.size());
		for (int i = 0; i < vMaxSortList.size(); ++i)
		{
			int objInd = vMaxSortList[i];
			vR[i]._faceLoc = vValidateRectList[objInd];
			vR[i]._faceCla = vValidateClassList[objInd];
			vR[i]._conf = vValidateProbabilityList[objInd];
		}

		return vR;
	}

	class det_ssd{
	public: 
		det_ssd(){};
		~det_ssd(){};
		bool InitialParam(std::string modelPath = "");

		int SSD_FaceMask(cv::Mat& inMat, std::vector<ObjInfo>& vReList);

		void ReleaseParam();
	private:
		cv::dnn::Net m_mapNet;

	};

	bool det_ssd::InitialParam(std::string modelPath)
	{
		std::string modelTxt = modelPath + "\\face_mask_detection.prototxt";
		std::string modelBin = modelPath + "\\face_mask_detection.caffemodel";

		try{
			m_mapNet = cv::dnn::readNetFromCaffe(modelTxt, modelBin);
		}
		catch (cv::Exception &ee){

			return false;
		}

		return true;
	}
	
	int det_ssd::SSD_FaceMask(cv::Mat& inMat, std::vector<ObjInfo>& vReList)
	{
		vReList.clear();
		if (inMat.empty()){
			return -1;
		}
		cv::Mat img;
		cv::resize(inMat, img, cv::Size(260, 260));

		cv::cvtColor(img, img, CV_BGR2RGB);
		cv::Mat inputBlob = cv::dnn::blobFromImage(img, 1.0 / 255.0, cv::Size(), cv::Scalar(0, 0, 0), false, false);

		cv::Mat pred;
		m_mapNet.setInput(inputBlob, "data");

		//////////////////////////////////////////////////////////////////////////
		std::vector<cv::Mat> outputBlobs;
		std::vector<cv::String> outBlobNames;
		outBlobNames.push_back("loc_branch_concat");
		outBlobNames.push_back("cls_branch_concat");
		m_mapNet.forward(outputBlobs, outBlobNames);
		//loc_0_reshape loc_1_reshape loc_2_reshape loc_3_reshape loc_4_reshape
		/*outBlobNames.push_back("loc_0_reshape");
		outBlobNames.push_back("loc_1_reshape");
		outBlobNames.push_back("loc_2_reshape");
		outBlobNames.push_back("loc_3_reshape");
		outBlobNames.push_back("loc_4_reshape");
		m_mapNet.forward(outputBlobs, outBlobNames);
		cv::Mat out0 = outputBlobs[0].reshape(0, 2);
		cv::Mat out1 = outputBlobs[1].reshape(0, 2);
		cv::Mat out2 = outputBlobs[2].reshape(0, 2);
		cv::Mat out3 = outputBlobs[3].reshape(0, 2);
		cv::Mat out4 = outputBlobs[4].reshape(0, 2);*/
		cv::Mat locBlobs = outputBlobs.front().reshape(0, 4);  //channels  rows
		cv::Mat clsBlobs = outputBlobs.back().reshape(0, 2);  //channels  rows

		int locSize = locBlobs.channels()*locBlobs.rows*locBlobs.cols;
		int clsSize = clsBlobs.channels()*clsBlobs.rows*clsBlobs.cols;
		cv::Mat partLoc(locSize / 4, 4, CV_32F, outputBlobs.front().ptr(0, 0));
		cv::Mat partCls(clsSize / 2, 2, CV_32F, outputBlobs.back().ptr(0, 0));

		std::vector<ObjInfo> vResultList = decode_bbox(partLoc, partCls);
		vReList = vResultList;
		int width = inMat.cols;
		int height = inMat.rows;
		for (int i = 0; i < vResultList.size(); ++i)
		{
			vReList[i]._faceLoc.x = std::max(0, int(vReList[i]._faceLoc.x * width));
			vReList[i]._faceLoc.y = std::max(0, int(vReList[i]._faceLoc.y * height));
			vReList[i]._faceLoc.width = std::max(0, int(vReList[i]._faceLoc.width * width));
			vReList[i]._faceLoc.height = std::max(0, int(vReList[i]._faceLoc.height * height));
			vReList[i]._faceLoc.width = std::min(width - (int)vReList[i]._faceLoc.x, int(vReList[i]._faceLoc.width));
			vReList[i]._faceLoc.height = std::min(height - (int)vReList[i]._faceLoc.y, int(vReList[i]._faceLoc.height));
		}

		return 1;
	}

	void det_ssd::ReleaseParam()
	{

	}
}