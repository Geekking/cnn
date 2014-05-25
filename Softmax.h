//
//  Softmax.h
//  ConvolutionNN
//
//  Created by apple on 5/25/14.
//  Copyright (c) 2014 SYSU. All rights reserved.
//

#ifndef __ConvolutionNN__Softmax__
#define __ConvolutionNN__Softmax__

#include <iostream>
#include "FileReader.h"
using namespace std;

FileReader<double> *fileReader;

int hiddenNodes = 0;
string trainFilePath,testFilePath;
vector<vector<double> > *w1;
vector<vector<double> > *w2;
vector<double> *b1;
vector<double> *b2;

vector<vector<double> > *gradW1;
vector<vector<double> > *gradW2;
vector<double> *gradB1;
vector<double> *gradB2;

vector<vector<double> > *midOutput;
vector<vector<double> > *output;

/*initialize the theta and gradW1,and allocate the middle and output layer
 */
void initialize(const double episilon){
    
}

/* get maxvalue index
 * @para prababilityoutputs
 * @return the maxvalut index in the vector
 */
int getIndex(const vector<double> output){
    int index = 0;
    return index;
}

/* forward from the input to hiden layer ,then to output layer
 *
 */
void forward(){
    
}

/*
 * backward to calculate the grad for w1,w2,b1,b2
 */
double backward(){
    double cost = 0;
    return cost;
}

/*
 * update w1,w2 according to grad for w1,w2,b1,b2
 */
void update(const double alpha,const double lumda){
    
}

/*
 * calculate numeric gradient
 */
void numericGradient(const double episilon){
    
}

/*
 *check gradient
 * remember to save the grad and theta before check
 */
void checkGradient(){
    
}

/* save the predict output to file
 *
 */
void saveToDisk(vector<int> *predict,string filePath){
    
}

/* test the network and save to output file
 *
 */
void test(){
    
}

/*
 * learn the softmax network
 * @para classes # of classes
 * @middleNodes
 */
void softMax(const int classes,const int hiddenNodes){
    
    test();
}

#endif /* defined(__ConvolutionNN__Softmax__) */
