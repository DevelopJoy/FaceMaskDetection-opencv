#pragma once

#include <vector>
#include <string>

#include <opencv2\opencv.hpp>
#include <opencv2\dnn\dnn.hpp>
 
#include "anchor.h"
#include "net.h"

#define USING_CV_DNN				0
#define USING_NCNN					1

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
#if USING_CV_DNN
		cv::dnn::Net m_mapNet;
#elif USING_NCNN
		ncnn::Net mobilenet;
#endif
	};

	bool det_ssd::InitialParam(std::string modelPath)
	{
#if USING_CV_DNN	
		std::string modelTxt = modelPath + "\\face_mask_detection.prototxt";
		std::string modelBin = modelPath + "\\face_mask_detection.caffemodel";

		try{
			m_mapNet = cv::dnn::readNetFromCaffe(modelTxt, modelBin);
		}
		catch (cv::Exception &ee){

			return false;
		}
#elif USING_NCNN
		std::string modelTxt = modelPath + "\\faceMask.param";
		std::string modelBin = modelPath + "\\faceMask.bin";
		bool bF = mobilenet.load_param(modelTxt.c_str());
		bF = mobilenet.load_model(modelBin.c_str());
		if (-1 == bF){
			return -1;
		}
#endif
		return true;
	}
	
	int det_ssd::SSD_FaceMask(cv::Mat& inMat, std::vector<ObjInfo>& vReList)
	{
		vReList.clear();
		if (inMat.empty()){
			return -1;
		}

#if USING_CV_DNN	
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
		cv::Mat concatLoc(locSize / 4, 4, CV_32F, outputBlobs.front().ptr(0, 0));
		cv::Mat concatCls(clsSize / 2, 2, CV_32F, outputBlobs.back().ptr(0, 0));
#elif USING_NCNN
		const int target_size = 260;
		int img_w = inMat.cols;
		int img_h = inMat.rows;
		cv::cvtColor(inMat, inMat, CV_BGR2RGB);
		ncnn::Mat in = ncnn::Mat::from_pixels_resize(\
			inMat.data, ncnn::Mat::PIXEL_BGR, \
			inMat.cols, inMat.rows, \
			target_size, target_size);

		const float mean_vals[3] = { 0, 0, 0 };
		const float norm_vals[3] = { 1.0 / 255, 1.0 / 255, 1.0 / 255 };
		in.substract_mean_normalize(mean_vals, norm_vals);

		ncnn::Extractor ex = mobilenet.create_extractor();
		//ex.set_num_threads(4);

		ex.input("data", in);

		//loc
		ncnn::Mat loc_0;
		ncnn::Mat loc_1;
		ncnn::Mat loc_2;
		ncnn::Mat loc_3;
		ncnn::Mat loc_4;
		ex.extract("loc_0_reshape", loc_0);
		float* ptrLoc0 = (float*)loc_0.data;
		ex.extract("loc_1_reshape", loc_1);
		float* ptrLoc1 = (float*)loc_1.data;
		ex.extract("loc_2_reshape", loc_2);
		float* ptrLoc2 = (float*)loc_2.data;
		ex.extract("loc_3_reshape", loc_3);
		float* ptrLoc3 = (float*)loc_3.data;
		ex.extract("loc_4_reshape", loc_4);
		float* ptrLoc4 = (float*)loc_4.data;

		//concat
		int locSize_0 = loc_0.w * loc_0.h * loc_0.c;
		int locSize_1 = loc_1.w * loc_1.h * loc_1.c;
		int locSize_2 = loc_2.w * loc_2.h * loc_2.c;
		int locSize_3 = loc_3.w * loc_3.h * loc_3.c;
		int locSize_4 = loc_4.w * loc_4.h * loc_4.c;
		cv::Mat partLoc(1, locSize_0 + locSize_1 + locSize_2 + locSize_3 + locSize_4, CV_32F);
		cv::Mat loc0(1, locSize_0, CV_32F, (float*)ptrLoc0);
		cv::Mat loc1(1, locSize_1, CV_32F, (float*)ptrLoc1);
		cv::Mat loc2(1, locSize_2, CV_32F, (float*)ptrLoc2);
		cv::Mat loc3(1, locSize_3, CV_32F, (float*)ptrLoc3);
		cv::Mat loc4(1, locSize_4, CV_32F, (float*)ptrLoc4);
		loc0.copyTo(partLoc.colRange(0, locSize_0));
		loc1.copyTo(partLoc.colRange(locSize_0, locSize_0 + locSize_1));
		loc2.copyTo(partLoc.colRange(locSize_0 + locSize_1, locSize_0 + locSize_1 + locSize_2));
		loc3.copyTo(partLoc.colRange(locSize_0 + locSize_1 + locSize_2, locSize_0 + locSize_1 + locSize_2 + locSize_3));
		loc4.copyTo(partLoc.colRange(locSize_0 + locSize_1 + locSize_2 + locSize_3, locSize_0 + locSize_1 + locSize_2 + locSize_3 + locSize_4));

		//cls
		ncnn::Mat cls_0;
		ncnn::Mat cls_1;
		ncnn::Mat cls_2;
		ncnn::Mat cls_3;
		ncnn::Mat cls_4;
		ex.extract("cls_0_reshape_cls_0_activation", cls_0);
		float* ptrCls0 = (float*)cls_0.data;
		ex.extract("cls_1_reshape_cls_1_activation", cls_1);
		float* ptrCls1 = (float*)cls_1.data;
		ex.extract("cls_2_reshape_cls_2_activation", cls_2);
		float* ptrCls2 = (float*)cls_2.data;
		ex.extract("cls_3_reshape_cls_3_activation", cls_3);
		float* ptrCls3 = (float*)cls_3.data;
		ex.extract("cls_4_reshape_cls_4_activation", cls_4);
		float* ptrCls4 = (float*)cls_4.data;
		int clsSize_0 = cls_0.w * cls_0.h * cls_0.c;
		int clsSize_1 = cls_1.w * cls_1.h * cls_1.c;
		int clsSize_2 = cls_2.w * cls_2.h * cls_2.c;
		int clsSize_3 = cls_3.w * cls_3.h * cls_3.c;
		int clsSize_4 = cls_4.w * cls_4.h * cls_4.c;
		cv::Mat partCls(1, clsSize_0 + clsSize_1 + clsSize_2 + clsSize_3 + clsSize_4, CV_32F);
		cv::Mat cls0(1, clsSize_0, CV_32F, (float*)ptrCls0);
		cv::Mat cls1(1, clsSize_1, CV_32F, (float*)ptrCls1);
		cv::Mat cls2(1, clsSize_2, CV_32F, (float*)ptrCls2);
		cv::Mat cls3(1, clsSize_3, CV_32F, (float*)ptrCls3);
		cv::Mat cls4(1, clsSize_4, CV_32F, (float*)ptrCls4);
		cls0.copyTo(partCls.colRange(0, clsSize_0));
		cls1.copyTo(partCls.colRange(clsSize_0, clsSize_0 + clsSize_1));
		cls2.copyTo(partCls.colRange(clsSize_0 + clsSize_1, clsSize_0 + clsSize_1 + clsSize_2));
		cls3.copyTo(partCls.colRange(clsSize_0 + clsSize_1 + clsSize_2, clsSize_0 + clsSize_1 + clsSize_2 + clsSize_3));
		cls4.copyTo(partCls.colRange(clsSize_0 + clsSize_1 + clsSize_2 + clsSize_3, clsSize_0 + clsSize_1 + clsSize_2 + clsSize_3 + clsSize_4));

		int locSize = partLoc.channels()*partLoc.rows*partLoc.cols;
		int clsSize = partCls.channels()*partCls.rows*partCls.cols;
		cv::Mat concatLoc(locSize / 4, 4, CV_32F, partLoc.ptr(0, 0));
		cv::Mat concatCls(clsSize / 2, 2, CV_32F, partCls.ptr(0, 0));
#endif
		std::vector<ObjInfo> vResultList = decode_bbox(concatLoc, concatCls);
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
