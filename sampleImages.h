//
//  sampleImages.h
//  ConvolutionNN
//
//  Created by apple on 5/22/14.
//  Copyright (c) 2014 SYSU. All rights reserved.
//

#ifndef __ConvolutionNN__sampleImages__
#define __ConvolutionNN__sampleImages__

#include <iostream>
#include <vector>

const int patchCount = 10000;    //patch 的数目

const int imageX = 64;
const int imageY = 48;

using namespace std;
vector<vector<double> >* sampleImages(const vector<vector <double> > *trainX,const int patchSizeX,const int patchSizeY){
    vector<vector<double> >* sampledPatches = new vector<vector<double> >(patchCount);
    unsigned long  trainXCount = trainX->size();  //原始数据的大小
    srand(300);
    
    vector<double> patch;
    for (int i = 0; i < patchCount; i++) {
        
        int sampleIndex = rand() % trainXCount;
        int row = rand() % (imageX - patchSizeX);
        int col = rand() % (imageY - patchSizeY);
        for (int r = row ;r < patchSizeX + row ;r++) {
            for (int c = col; c < patchSizeY + col; c++) {
                
                patch.push_back( (*trainX)[sampleIndex][r * imageY + c] );
            }
            
        }
        (*sampledPatches)[i]  = patch;
        patch.clear();
        
    }
    return sampledPatches;
    
}


#endif /* defined(__ConvolutionNN__sampleImages__) */
