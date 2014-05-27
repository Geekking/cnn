//
//  main.cpp
//  ConvolutionNN
//
//  Created by apple on 5/22/14.
//  Copyright (c) 2014 SYSU. All rights reserved.
//

#include <iostream>
#include "FileReader.h"
#include "sampleImages.h"
#include "sparseEncoder.h"
#include "extract.h"
#include "Convolution.h"
//#include "Softmax.h"

const int patchX = 9;
const int patchY = 9;
const int conv_middle_nodes = 30;
const int soft_middle_nodes = 18;
const int numberofClasses = 10;

int main(int argc, const char * argv[])
{

    
    
    FileReader<double> tmp("/Users/apple/Documents/develop/ufldl/cnn/cifar-10-train.csv","/Users/apple/Documents/develop/ufldl/cnn/cifar-10-test.csv");
    tmp.getTrainData();
    // sample patches in the train file
    vector<vector<double> > *samplePatches =sampleImages( tmp.getTrainX(),patchX,patchY);
    
    // train theta in sparse encoder network
    //trainTheta(samplePatches,conv_middle_nodes);
    
    // calculate the convalution output and save to file
    convol(&tmp, patchX, patchY, conv_middle_nodes);
    /*
    // using the data from convolution to softmax netowork
    softMax(numberofClasses, soft_middle_nodes);
    */
    return 0;
}

