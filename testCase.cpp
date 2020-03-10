

#include <stdlib.h>
#include <stdio.h>
#include <iostream>  
#include <vector>
#include <string>
#include <opencv2/opencv.hpp>  

#include "anchor.h"
#include "faceMask_ssd.h"

using namespace std;
using namespace cv;

void SSD_FaceMask()
{
	face_ssd::det_ssd *pD = new face_ssd::det_ssd();
	pD->InitialParam("..\\models");

	
	//ÈËÏñMASK
	cv::VideoCapture cap;
	bool is_op = cap.open(0);
	int iCount = 0;
	while (is_op)
	{
		cv::Mat srcMat2;
		cap >> srcMat2;
		if (srcMat2.empty()){
			continue;
		}

		std::vector<face_ssd::ObjInfo> vReList;
		pD->SSD_FaceMask(srcMat2, vReList);
		for (int i = 0; i < vReList.size(); ++i)
		{
			if (vReList[i]._faceCla == 0){
				rectangle(srcMat2, cv::Rect(vReList[i]._faceLoc), \
					cv::Scalar(0, 255, 0), 2);
			}
			else{
				rectangle(srcMat2, cv::Rect(vReList[i]._faceLoc), \
					cv::Scalar(255, 0, 0), 2);
			}

		}
		cv::namedWindow("re", 0);
		cv::imshow("re", srcMat2);
		cvWaitKey(10);
	}
}

int main()
{
	SSD_FaceMask();

	system("Pause");

	return 0;
}