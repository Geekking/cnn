//
//  Convolution.h
//  ConvolutionNN
//
//  Created by apple on 5/25/14.
//  Copyright (c) 2014 SYSU. All rights reserved.
//

#ifndef __ConvolutionNN__Convolution__
#define __ConvolutionNN__Convolution__

#include <iostream>
#include <fstream>
#include "FileReader.h"
#include <vector>

using namespace std;

/* read parameters from disk include
 * theta,
 * b1,
 * means,
 * stds
 */
vector<vector<double> > *w1;
vector<double> *b1;
vector<double> *menns;
vector<double> *stds;

int hiddenNodes;

void getParas(string filepPath){
    fstream *fstream;
}

/* forward a patch in the origin photo to middle layer
 * @para patch
 * @return middlelayer activation value
 */
vector<double>* forward(){
    vector<double> *middleLayerOutput;
    return middleLayerOutput;
}

/* save the pooled outputvalue to disk file
 * @para pooled outpuvalues
 */
void savetoDisk(const vector<vector<vector<double> > > *pooledOutput,string filePath,bool isTrain,const FileReader<int>* fileReader){
    
}

/* pool the midlayerOutput
 * @sample rate is the pool size ,example,2 means every 2*2 value get the max value
 */
void pool(vector<vector<vector<double> > > *originOutput,const int sampleRate){
    
}

/* preprocess the origin data
 */
void preprocess(){
    
}

/* convolution layer computation function
 * @para fileReader 
 * @para patchSizeX
 * @para patchSizeY
 */
void convol(const FileReader<int>* fileReader,const int patchSizeX,const int patchSizeY,const int middleNodes){
    /*for each patch in the trainX forward to get middle layer activation
     *then pool
     *save to file
     */
    
    /*for each patch in the testX forward to get middle layer activation
     *then pool
     *save to file
     */
    
}




#endif /* defined(__ConvolutionNN__Convolution__) */
