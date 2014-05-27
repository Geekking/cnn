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
#include <cmath>
#include <sstream>
#include <string>

using namespace std;

typedef vector<vector<double> > Patch;
typedef vector<vector<double> > Image;

vector<vector<double> > *w1 = new vector<vector<double> >();
vector<double> *b1 = new vector<double>();
vector<double> *means = new vector<double>();
vector<double> *stds = new vector<double>();

int conv_hiddenNodes;
int inputNodes;

const int sizeX = 64;
const int sizeY = 48;
const double e = 2.71828182846;
const int samRateX  = 2;
const int samRateY = 2;
bool trainFirstFlag = true;
bool testFirstFlag = true;

namespace CNN {
    
    /*split a string to float vector
     */
    vector<double> split(string& str,const char* c)
    {
        char *cstr, *p;
        vector<double> res;
        cstr = new char[str.size()+1];
        strcpy(cstr,str.c_str());
        p = strtok(cstr,c);
        while(p!=NULL)
        {
            res.push_back((double)atof(p) ); // change to float
            p = strtok(NULL,c);
        }
        
        return res;
    }
    
    /*sigmoid function
     *@para e
     @return 1/(1+(e-z))
     */
    double sigmoid(double z){
        return 1.0/(1.0 + pow(e,-z));
    }

}

/* pool the midlayerOutput
 * @sampleRateX,sampeRateY rate is the pool size ,example,2 means every 2*2 value get the max value
 */
vector<Patch>* pool(vector<vector<vector<double> > > *originOutput,const int sampleRateX,const int sampleRateY){
    vector<Patch> *pooledData = new vector<Patch>(originOutput->size());
    for (int i = 0; i < originOutput->size(); i++) {
        Patch *layerOutput = new Patch();
        for (int j = 0; j < (*originOutput)[i].size();j += sampleRateX ) {
            vector<double> *oneRow = new vector<double>();
            for (int k = 0; k < (*originOutput)[i][j].size(); k += sampleRateY ) {
                double max = 0.0;
                for (int m = 0; m < sampleRateX; m++) {
                    for (int n = 0; n < sampleRateY; n++) {
                        
                        if (max < (*originOutput)[i][j + m][k + n]) {
                            max = (*originOutput)[i][j + m][k + n];
                        }
                    }
                }
                oneRow->push_back(max);
            }
            layerOutput->push_back((*oneRow));
        }
        (*pooledData)[i] = (*layerOutput);
    }
    originOutput->clear();
    originOutput = new vector<Patch>(*pooledData);
    return originOutput;
}

/* read parameters from disk include
 * theta,
 * b1,
 * means,
 * stds
 */
void getParas(string filepPath){
    ifstream ifs;
    
    try {
        ifs.open((filepPath).c_str());
        
        string str;
        while (!ifs.eof()) {
            //get w1
            for (int i = 0; i < conv_hiddenNodes; i++ ) {
                getline(ifs, str);
                w1->push_back(CNN::split(str,","));
            }
            cout<<w1->size()<<endl;
            //get b1
            getline(ifs, str);
            *b1 = CNN::split(str,",");
            cout<<b1->size()<<endl;
            //get means
            getline(ifs, str);
            *means = CNN::split(str,",");
            cout<<means->size()<<endl;
            
            //get stds
            getline(ifs, str);
            *stds = CNN::split(str, ",");
            cout<<stds->size()<<endl;
            break;
        }

        
        ifs.close();
    } catch (ifstream::failure e) {
        cout<<"Exception opening/reading/closing file\n";
    }

    
}

/* vetorize a patch to a vector
 * @para patch
 * @return a vector
 */
vector<double>* vetorized(Patch *patch){
    vector<double> *res = new vector<double>();
    for (int i = 0; i < patch->size(); i++) {
        for (int j = 0; j < (*patch)[i].size(); j++) {
            res->push_back((*patch)[i][j]);
        }
    }
    return res;
}


/* preprocess the origin data
 */
void preprocess(vector<double> *vec_pat){
    for (int i = 0; i < vec_pat->size() ; i++) {
        (*vec_pat)[i] -= (*means)[i];
        (*vec_pat)[i] /= (*stds)[i];
        
    }
}

/*forwar a patch to middle layer
 * @para patch
 @return all the middle activation value for this output
 */
vector<double> *forward2Middle(Patch *patch){
    vector<double> *res = new vector<double>();
    vector<double> *patch_vec = vetorized(patch);
    preprocess(patch_vec);
    
    for (int i = 0; i < conv_hiddenNodes ; i ++) {
        double h = 0.0;
        for (int j = 0; j < inputNodes; j++) {
            h += (*patch_vec)[j] * (*w1)[i][j];
        }
        h += (*b1)[i];
        h = CNN::sigmoid(h);
        
        res->push_back(h);
    }
    
    return res;
}

/* forward patches in the origin photo to middle layer
 * @para patches
 * @return middlelayer activation value
 */
vector<Patch>* forward(vector<vector<Patch> > *patches){
    int patchesSizeX = (int)patches->size();
    int patchesSizeY = (int)(*patches)[0].size();
    vector<Patch> *middleLayerOutput  = new vector<Patch>(conv_hiddenNodes,vector<vector<double> >(patchesSizeX,vector<double>(patchesSizeY)));
    for (int i = 0; i < patchesSizeX; i++) {
        for (int j = 0; j < patchesSizeY; j++) {
            vector<double> *res =  forward2Middle(&(*patches)[i][j]);
            for (int k = 0; k < res->size(); k++) {
                (*middleLayerOutput)[k][i][j] = (*res)[k];
            }
        }
    }
    return middleLayerOutput;
}

/* vectorize pooled output
 *
 */
string stringlized(const vector<vector<vector<double> > > *pooledOutput){
    stringstream ss;
    
    for (int i = 0; i < pooledOutput->size(); i++) {
        for (int j = 0; j < (*pooledOutput)[i].size(); j++) {
            for (int k = 0; k < (*pooledOutput)[i][j].size(); k++) {
                ss<<","<< (*pooledOutput)[i][j][k];
            }
        }
    }
    return ss.str();
}
string int2str(const int value){
    string str;
    
    return str;
}
/* save the pooled outputvalue to disk file
 * @para pooled outpuvalues
 */
void savetoDisk(const vector<vector<vector<double> > > *pooledOutput,string filePath,bool isTrain,int sampleId = 0,int ylabel = -1){
    ofstream fs;
    try {
        
        if (isTrain) {
            if (!trainFirstFlag) {
                fs.open(filePath.c_str(),ios::app|ios::out);
            }else{
                fs.open(filePath.c_str(),ios::ate|ios::out);
                trainFirstFlag = false;
                fs<<"Id,label,value"<<endl;
                
            }
            stringstream ss;
            
            ss << sampleId<<","<<ylabel;
            string str = ss.str();
            
            str += stringlized(pooledOutput);
            fs<<str<<endl;
        
        }else{
            if (!testFirstFlag) {
                fs.open(filePath.c_str(),ios::app|ios::out);
            }else{
                fs.open(filePath.c_str(),ios::ate|ios::out);
                fs<<"Id,value"<<endl;
                testFirstFlag = false;
            }
            stringstream ss;
            ss << sampleId;
            string str = ss.str();
            str += stringlized(pooledOutput);
            fs<<str<<endl;
            
        }
        fs.close();
        
    } catch (fstream::failure err) {
        cout<<"Cannot open output file!"<<endl;
    }
}



/*convert a vector 2 two dimention vector
 */
vector<vector<double> >* transfer2Twodimention(const vector<double> *rowVec ,const int first_dimension){
    vector<vector<double> > *res = new vector<vector<double> >(first_dimension);
    int cols = (int)rowVec->size() / first_dimension;
    
    for (int i = 0 ; i < first_dimension; i++) {
        for (int j = 0; j < cols; j++) {
            (*res)[i].push_back( (*rowVec)[i * cols + j]);
        }
    }
    
    return res;
    
}

/* get patch in a images
 * @para image
 * @para patchX,patchY
 * @return a 2dimention vector contains all the patch
 */
vector<vector<Patch> >* getPatches(Image *image,int patchX,int patchY){
    int patchesX = ((int)image->size() - patchX +1)  ;
    int patchesY = ((int)(*image)[0].size() - patchY  + 1);
    
    vector< vector<Patch> > *patches = new vector<vector<Patch> >(patchesX,vector<Patch>(patchesY));
    
    for (int i = 0; i< patchesX; i++) {
        Patch apatch = Patch(patchX);
        for (int j = 0; j < patchesY; j++) {
            for (int k = 0; k < patchX; k++) {
                for (int m = 0; m < patchY; m++) {
                    apatch[k].push_back( (*image)[i + k][ j + m]);
                }
            }
            (*patches)[i][j] = apatch;
        }
    }
    return patches;
    
}

/* convolution layer computation function
 * @para fileReader 
 * @para patchSizeX
 * @para patchSizeY
 */
void convol( FileReader<double>* fileReader,const int patchSizeX,const int patchSizeY,const int middleNodes){
    conv_hiddenNodes = middleNodes;
    inputNodes = patchSizeX * patchSizeY;
    string filePath = "sparsePara.csv";
    string trainPath = "train.csv";
    string testPath = "test.csv";
    
    getParas(filePath);
    /*for each patch in the trainX forward to get middle layer activation
     *then pool
     *save to file
     */
    const vector<vector<double> > *trainX = (fileReader->getTrainX());
    const vector<vector<double> > *trainY = fileReader -> getTrainY();
    const vector<vector<double> > *testID = fileReader->getTestID();
    
    for (int i = 0; i < trainX->size(); i++) {
        Image *image = transfer2Twodimention(&(*trainX)[i], sizeX);
        
        vector<vector<Patch> > *patches;
        patches = getPatches(image, patchSizeX, patchSizeY);
        
        vector<Patch > *middleOutput = forward(patches);
        
        middleOutput = pool(middleOutput, samRateX, samRateY);     // now middleOutput saves the pooled output
        int y = int((*trainY)[i][0]);
        savetoDisk(middleOutput, trainPath, true,0,y);
        delete patches;
        delete middleOutput;
        
        cout<<i<<endl;
     }
    
    
    /*for each patch in the testX forward to get middle layer activation
     *then pool
     *save to file
     */
    /*
    const vector<vector<double> > *testX = fileReader->getTestX();
    
    for (int i = 0; i < testX->size(); i++) {
        Image *image = transfer2Twodimention(&(*testX)[i], sizeX);
        
        vector<vector<Patch> > *patches;
        patches = getPatches(image, patchSizeX, patchSizeY);
        
        vector<Patch > *middleOutput = forward(patches);
        
        middleOutput = pool(middleOutput, samRateX, samRateY);     // now middleOutput saves the pooled output
        int sID = int((*testID)[i][0]);
        savetoDisk(middleOutput, testPath, false,sID);
        delete patches;
        delete middleOutput;
        
        cout<<i<<endl;
    }
    
}




#endif /* defined(__ConvolutionNN__Convolution__) */
