//
//  sparseEncoder.h
//  ConvolutionNN
//
//  Created by apple on 5/22/14.
//  Copyright (c) 2014 SYSU. All rights reserved.
//

#ifndef __ConvolutionNN__sparseEncoder__
#define __ConvolutionNN__sparseEncoder__

#include <iostream>
#include <vector>
#include <math.h>
#include <stdio.h>
#include <fstream>
using namespace std;
const int testGradCount = 10;
const int startCount = 0;
int counter = 0;
typedef struct COST_GRAD{
    double cost;
    vector<double> *grad;
} COST_GRAD;

const double rho = 0.95;
const double beta = 5;
 int hiddenNodes = 30;
const double EPISILON = 0.0001;
vector<vector<double> > midOutput;

vector<vector<double> > output;
const bool debug = false;

void initilizeTheta(vector<double> *theta,const int intputNodes){
    srand(1000);
    //initialize w1
    if(debug){
        cout<<"Initializing"<<endl;
    }
    for (int i =0 ; i < hiddenNodes * intputNodes; i ++ ) {
        theta->push_back( (rand()% 100 - 50)*EPISILON);
    }
    //initialize b1
    for (int i =0 ; i < hiddenNodes ; i ++ ) {
        theta->push_back( (rand()% 100 - 50)*EPISILON);
    }
    //initialize w2
    for (int i =0 ; i <  intputNodes * hiddenNodes ; i ++ ) {
        theta->push_back( (rand()% 100 - 50)*EPISILON);
    }
    //initialzed b2
    
    for (int i =0 ; i <  intputNodes ; i ++ ) {
        theta->push_back( (rand()% 100 - 50)*EPISILON);
    }
    
    if(debug){
        cout<<"Initialized"<<endl;
    }
    
    
}

void extractArg(const vector<double>* theta,
                vector<vector<double> > *w1,
                vector<vector<double> > *w2,
                vector<double>  *b1,
                vector<double>  *b2,const int inputNodes){
    int i = 0;
    int offset = 0;
    w1->clear();
    w2->clear();
    b1->clear();
    b2->clear();
    
    if(debug){
        cout<<"Extracting"<<endl;
    }
    vector<double> thea1 = vector<double> ();
    for (i = 0; i < hiddenNodes * inputNodes; i++) {
        
        thea1.push_back((*theta)[i+offset]);
        if ((i+1) % inputNodes == 0) {
            w1->push_back(thea1);
            thea1.clear();
        }
    }
    offset += hiddenNodes * inputNodes;
    for (i =0 ; i < hiddenNodes; i++) {
        b1->push_back((*theta)[i+offset]);
    }
    
    offset += hiddenNodes;
    vector<double> thea2 = vector<double>();
    for (i = 0; i < hiddenNodes * inputNodes; i++) {
        thea2.push_back((*theta)[i+offset]);
        
        if ((i+1) % hiddenNodes == 0) {
            w2->push_back(thea2);
            thea2.clear();
        }
    }
    
    offset += hiddenNodes *inputNodes;
    for (i = 0; i < inputNodes; i++) {
        b2->push_back((*theta)[i+offset]);
    }
    
    if(debug){
        cout<<"Extracted"<<endl;
    }
}
double arrayMultiply(const vector<int> a,const vector<double> b){
    double res = 0;
    if (a.size() != b.size()) {
        cout<<"Error"<<endl;
    }
    for (int i = 0; i < a.size(); i++) {
        res += double(a[i]) *b[i];
    }
    return res;
}
double arrayMultiply(const vector<double> a,const vector<double> b){
    double res = 0;
    if (a.size() != b.size()) {
        cout<<"Error"<<endl;
    }
    for (int i = 0; i < a.size(); i++) {
        res += a[i] *b[i];
    }
    return res;
}

double sigmoid(double z){
    return 1.0/(1.0+pow(2.71828182846,-z));
}

double forward(const vector<vector<double> > *sampleData,const vector<double> *theta ,const int inputNodes){
    
    vector<vector<double> > w1 = vector<vector<double> >();
    vector<vector<double> > w2 = vector<vector<double> >();
    vector<double> b1 = vector<double>();
    vector<double> b2 = vector<double>();
    
    double res = 0;
    extractArg(theta, &w1, &w2, &b1, &b2, inputNodes);
    
    midOutput = vector<vector<double> >(sampleData->size(),vector<double>(hiddenNodes,0.0) );
    output = vector<vector<double> >(sampleData->size(),vector<double>(inputNodes, 0.0));
    for (int i = 0;  i < sampleData ->size(); i ++) {
        vector<double> X = (*sampleData)[i];
        
        //forward to middle layer
        
        for (int j = 0; j < hiddenNodes; j ++) {
            double h = arrayMultiply(X,w1[j]) + b1[j];
            h = sigmoid(h);
            
            midOutput[i][j] = h;
          
         }
        
        //forward to third layer
        for (int j = 0; j < inputNodes; j ++) {
            double h = arrayMultiply(midOutput[i],w2[j]) + b2[j];
            
            //cout<<w2[63][500]<<endl; //BUG
            output[i][j] = h;
            //sigmoid(h);
            
        }
        
    }
    return res;
}
void formTheta( vector<double> *theta,
               const vector<vector<double> > w1,
               const vector<vector<double> > w2,
               const vector<double> b1,
               const vector<double> b2,
               const int inputSize){
    int offset = 0;
    
    if(debug){
        cout<<"forming"<<endl;
    }
    
    for(int i  = 0;i < hiddenNodes;i++){
        for (int j = 0; j < inputSize; j++) {
            (*theta)[i*inputSize + j + offset] = w1[i][j];
        }
    }
    offset += hiddenNodes * inputSize;
    
    for (int i =0 ; i < hiddenNodes; i++) {
        (*theta)[offset + i] = b1[i];
    }
    offset += hiddenNodes;
    vector<double> thea2;
    
    for (int i = 0; i < inputSize; i++) {
        for (int j = 0; j < hiddenNodes; j++) {
            (*theta)[i*hiddenNodes + j + offset] = w2[i][j];
        }
    }
    
    offset += hiddenNodes * inputSize;
    
    for (int i = 0; i < inputSize; i++) {
        (*theta)[i + offset] = b2[i];
    }
    if (debug) {
        cout<<"formed"<<endl;
    }
    
    
}

double KL(double rho,double r_h){
    double res = 0;
    res = rho*log((double)rho/r_h) +(1-rho)*log((double)(1-rho)/(1-r_h));
    return res;
}

COST_GRAD Stoch_backward(const vector<vector<int> > *sampleData,const vector<double> *theta,const double numda,const int sampleIndex){
    vector<double>  delta3;
    vector<double>  delta2;
    
    
    vector<double> rhohead = vector<double>(hiddenNodes,0);
    
    int inputSize = (int)(*sampleData)[0].size();
    int sampleSize = (int)(*sampleData).size();
    
    vector<vector<double> > w1;
    vector<vector<double> > w2;
    vector<double> b1;
    vector<double> b2;
    extractArg(theta, &w1, &w2, &b1, &b2, inputSize);
    
    
    
    vector<vector<double> > gradW2 = vector<vector<double> >(inputSize ,vector<double> (hiddenNodes,0.0));
    vector<vector<double> > gradW1 = vector<vector<double> >(hiddenNodes,vector<double> (inputSize,0.0));
    vector<double> gradB1  = vector<double>(hiddenNodes,0.0);
    vector<double> gradB2 = vector<double>(inputSize,0.0);
    
    double cost  = 0.0;
    
    for (int i =0; i < sampleData->size(); i++) {
        for (int j = 0; j < hiddenNodes; j++) {
            rhohead[j] += midOutput[i][j];
        }
    }
    
    for (int j = 0; j < hiddenNodes; j++) {
        rhohead[j] /= (double)sampleSize;
    }
    
    
    for(int i = 0;i < sampleData->size(); i++){
        delta3.clear();
        delta2.clear();
        
        for (int j  = 0; j < inputSize; j++) {
            double tmp = (output[i][j] - (*sampleData)[i][j]);
            cost +=  tmp * tmp;
            
            
            delta3.push_back(tmp);
        }
        
        for (int j = 0; j < hiddenNodes; j++) {
            double tmp = 0.0;
            
            for (int k = 0; k < inputSize; k++) {
                tmp += delta3[k] * w2[k][j];
            }
            
            double sparsePanalty = (-rho/rhohead[j] + (1.0 - rho)/(1.0 - rhohead[j]));
            tmp += beta * sparsePanalty;
            
            //乘上simoid 函数的倒数
            
            tmp *= (midOutput[i][j]*(1.0 - midOutput[i][j])); // hpi
            
            delta2.push_back(tmp);
        }
        
        for (int j = 0; j < inputSize;j++) {
            for (int k = 0; k < hiddenNodes; k++) {
                gradW2[j][k] += delta3[j] * midOutput[i][k] ;
            }
            
            gradB2[j] += delta3[j];
        }
        for (int j = 0; j < hiddenNodes; j++) {
            for (int k = 0; k < inputSize; k++) {
                gradW1[j][k] += delta2[j]*(*sampleData)[i][k];
            }
            gradB1[j] += delta2[j];
        }
    }
    
    cost /= (double)(2.0*sampleSize);
    
    //normalization
    double weight_decay = 0.0;
    
    for (int j = 0; j < inputSize;j++) {
        for (int k = 0; k < hiddenNodes; k++) {
            
            gradW2[j][k] /= (double)sampleSize;
            gradW2[j][k] +=  numda * w2[j][k];
            
            weight_decay += (w2[j][k]*w2[j][k]);
            
        }
        gradB2[j] /= (double)sampleSize;
    }
    for (int j = 0; j < hiddenNodes; j++) {
        for (int k = 0; k < inputSize; k++) {
            gradW1[j][k] /= (double)sampleSize;
            gradW1[j][k] +=  numda *w1[j][k];
            
            weight_decay += (w1[j][k])*(w1[j][k]);
        }
        gradB1[j] /= (double)sampleSize;
    }
    
    cost += weight_decay*numda/2.0;
    
    double sparsity = 0.0;
    
    for (int j = 0; j < hiddenNodes; j++) {
        sparsity += KL(rho, rhohead[j]);
    }
    
    cost += beta* sparsity;
    
    COST_GRAD res;
    res.cost = cost;
    
    vector<double> *grad = new vector<double>(*theta);
    
    formTheta(grad, gradW1, gradW2, gradB1, gradB2, inputSize);
    res.grad = grad;
    
    return res;

}

COST_GRAD backward(const vector<vector<double> > *sampleData,const vector<double> *theta,const double numda){
    
    
    vector<double> rhohead = vector<double>(hiddenNodes,0.0);
    
    int inputSize = (int)(*sampleData)[0].size();
    int sampleSize = (int)(*sampleData).size();
    
    vector<vector<double> > w1 = vector<vector<double> >();
    vector<vector<double> > w2 = vector<vector<double> >();
    vector<double> b1 = vector<double> ();
    vector<double> b2 = vector<double> ();
    extractArg(theta, &w1, &w2, &b1, &b2, inputSize);
    
    
    
    vector<vector<double> > gradW2 = vector<vector<double> >(inputSize ,vector<double> (hiddenNodes,0.0));
    vector<vector<double> > gradW1 = vector<vector<double> >(hiddenNodes,vector<double> (inputSize,0.0));
    vector<double> gradB1  = vector<double>(hiddenNodes,0.0);
    vector<double> gradB2 = vector<double>(inputSize,0.0);
    
    double cost  = 0.0;
    
    for (int i =0; i < sampleData->size(); i++) {
        for (int j = 0; j < hiddenNodes; j++) {
            rhohead[j] += midOutput[i][j];
        }
    }
    
    for (int j = 0; j < hiddenNodes; j++) {
        rhohead[j] /= (double)sampleSize;
        
    }
    
    
    for(int i = 0;i < sampleData->size(); i++){
        vector<double>  delta3 = vector<double> ();
        vector<double>  delta2 = vector<double> ();
        
        for (int j  = 0; j < inputSize; j++) {
            double tmp = (output[i][j] - (*sampleData)[i][j]);
            
            cost +=  tmp * tmp;
            //tmp = tmp*output[i][j]*(1.0-output[i][j]);
            delta3.push_back(tmp);
        }
        
        for (int j = 0; j < hiddenNodes; j++) {
            double tmp = 0.0;
            
            for (int k = 0; k < inputSize; k++) {
                tmp += delta3[k] * w2[k][j];
                
            }
            
            double sparsePanalty = (-rho/rhohead[j] + (1.0 - rho)/(1.0 - rhohead[j]));
            
            //tmp += beta * sparsePanalty;
            
            //乘上simoid 函数的倒数
            
            tmp *= (midOutput[i][j]*(1.0 - midOutput[i][j])); // hpi
            
            delta2.push_back(tmp);
            
        }
        
        for (int j = 0; j < inputSize;j++) {
            for (int k = 0; k < hiddenNodes; k++) {
                gradW2[j][k] += delta3[j] * midOutput[i][k] ;
            }
            
            gradB2[j] += delta3[j];
        }
        
        for (int j = 0; j < hiddenNodes; j++) {
            for (int k = 0; k < inputSize; k++) {
                
                gradW1[j][k] += delta2[j]*(*sampleData)[i][k];
                
            }
            gradB1[j] += delta2[j];
        }
    }
    
    cost /= (double)(2.0*sampleSize);
    
    //normalization
    double weight_decay = 0.0;
    
    for (int j = 0; j < inputSize;j++) {
        for (int k = 0; k < hiddenNodes; k++) {
            
            gradW2[j][k] /= (double)sampleSize;
            gradW2[j][k] +=  numda * w2[j][k];
            
            weight_decay += (w2[j][k]*w2[j][k]);
            
        }
        gradB2[j] /= (double)sampleSize;
    }
    for (int j = 0; j < hiddenNodes; j++) {
        for (int k = 0; k < inputSize; k++) {
            
            gradW1[j][k] /= (double)sampleSize;
            
            gradW1[j][k] +=  numda *w1[j][k];
            
            weight_decay += (w1[j][k])*(w1[j][k]);
        }
        gradB1[j] /= (double)sampleSize;
    }
    
    cost += weight_decay*numda/2.0;
    
    double sparsity = 0.0;
    
    for (int j = 0; j < hiddenNodes; j++) {
        sparsity += KL(rho, rhohead[j]);
    }
    
    //cost += beta* sparsity;
    
    COST_GRAD res;
    
    res.cost = cost;
    
    vector<double> *grad = new vector<double>(*theta);
    
    formTheta(grad, gradW1, gradW2, gradB1, gradB2, inputSize);
    res.grad = grad;
    
    return res;
    
}
double testNumbericGradient(double x1,double x2,bool showValue){
    double res = x1*x1 + 3*x1*x2;
    double grad1 = 2*x1 + 3*x2;
    if (showValue) {
        cout<<grad1<<endl;
    }
    return res;
}

vector<double>* numbericGradient(const vector<vector<double> > *sampleData ,const vector<double > *theta,const int intputNodes,const double numda ){
    
    vector<double> *nubericGrad = new vector<double>(startCount);
    
    for (int i = startCount; i-startCount < testGradCount; i++) {
        
        vector<double> *tmpTheta = new vector<double>((*theta));
        (*tmpTheta)[i] += EPISILON;
        
        forward(sampleData, tmpTheta,intputNodes );
        
        COST_GRAD cost_grad1 = backward(sampleData, tmpTheta,numda);
        tmpTheta = new vector<double>((*theta));
        (*tmpTheta)[i] -= EPISILON;
        forward(sampleData, tmpTheta,intputNodes );
        
        COST_GRAD cost_grad2  = backward(sampleData, tmpTheta,numda);
        
        double valueGrad = (cost_grad1.cost - cost_grad2.cost)/(2*EPISILON);
        
        nubericGrad->push_back(valueGrad);
    }
    return nubericGrad;
}
void checkGradient(const vector<vector<double> > *sampleData ,const vector<double > *theta,const int inputNodes,vector<double> *grad,const double numda){
    
    vector<double> *numbericGrad = numbericGradient(sampleData, theta, inputNodes,numda);
    
    for (int i = startCount ; i-startCount < testGradCount; i++) {
        cout<<(*numbericGrad)[i]<<" vs "<<(*grad)[i]<<endl;
    }
    cout<<"ena"<<endl;
    
}

void updateTheta(vector<double> *theta,const int inputNodes, const vector<double> *grad,const double alpha,const  double numda ){
    vector<vector<double> > w1 = vector<vector<double> >();
    vector<vector<double> > w2 = vector<vector<double> >();
    vector<double> b1 = vector<double>();
    vector<double> b2 = vector<double>();
    
    vector<vector<double> > gradW2 = vector<vector<double> >(); //= vector<vector<double> >(inputNodes ,vector<double> (hiddenNodes,0));
    vector<vector<double> > gradW1 = vector<vector<double> >(); //= vector<vector<double> >(hiddenNodes,vector<double> (inputNodes,0));
    vector<double> gradB1 = vector<double>(); // = vector<double>(hiddenNodes,0);
    vector<double> gradB2 = vector<double>(); //= vector<double>(inputNodes,0);
    
    
    extractArg(theta, &w1, &w2, &b1, &b2, inputNodes);
//    for (int i = 0; i < grad->size(); i++) {
//        cout<<(*grad)[i]<<endl;
//    }
    extractArg(grad, &gradW1, &gradW2, &gradB1, &gradB2, inputNodes);
    
    if (debug) {
        cout<<"updating w2"<<endl;
    }
    for (int j = 0; j < inputNodes;j++) {
        for (int k = 0; k < hiddenNodes; k++) {
            //double tmp = w2[j][k];
            w2[j][k] -=  alpha * (gradW2[j][k] );
           // cout<<w2[j][k]<<" vs "<< tmp<<" + "<<gradW2[j][k]<<endl;
        }
    }
    
    if (debug) {
        cout<<"updating w1"<<endl;
    }
    for (int j = 0; j < hiddenNodes; j++) {
        for (int k = 0; k < inputNodes; k++) {
            
            w1[j][k] -= alpha *(gradW1[j][k] );
            
        }
        b1[j] -= alpha*gradB1[j];
    }
    if (debug) {
        cout<<"update over "<<endl;
    }
    
    formTheta(theta, w1, w2,b1, b2,inputNodes);
    
}
void preProcess(vector<vector<double> > *sampleData,vector<double> &means,vector<double> &stds){
    int inputNodes = (int)(*sampleData)[0].size();
    int sampleSize= (int)sampleData->size();
    means = vector<double> (inputNodes,0.0);
    stds = vector<double> (inputNodes,0.0);


    for (int i =0; i < sampleSize; i++) {
        for (int j = 0; j < inputNodes; j++) {
            
            means[j] += (*sampleData)[i][j];
            
        }
    }
    for (int i = 0; i < inputNodes; i++) {
        
        means[i] /= sampleSize;
    }
    
    for (int i =0; i < sampleSize; i++) {
        for (int j = 0; j < inputNodes; j++) {
            stds[j] += ((*sampleData)[i][j] - means[j])*((*sampleData)[i][j] - means[j]);
        }
    }
    
    for (int i = 0; i < inputNodes; i++) {
        stds[i] /= sampleSize;
        stds[i] = sqrt(stds[i]);
    }
    for (int i =0; i < sampleSize; i++) {
        for (int j = 0; j < inputNodes; j++) {
            (*sampleData)[i][j]  = (double)((*sampleData)[i][j]-means[j])/(stds[j]);
        }
    }
    
    
}
void getOrigin(const vector<vector<double> > *orginData,vector<double> means , vector<double> stds){
    float cost = 0.0;
    for (int i = 0; i < output.size(); i++) {
        for (int j = 0; j < output[0].size(); j++) {
            
            output[i][j]  = output[i][j] * stds[j] + means[j];
            //output[i][j] *= 255.0;
            cout<<output[i][j]<<" vs "<<(*orginData)[i][j]<<endl;
            
            cost += ((*orginData)[i][j] - output[i][j])*((*orginData)[i][j] - output[i][j]);
        }
    }
    cost /= output.size();
    cout<<cost<<endl;
    
}
void saveToDisk(const vector<double>  *theta,const int inputNodes, const vector<double> means, const vector<double> stds){
    ofstream fileStream;
    string outputFilePath = "sparsePara.csv";
    
    try {
        fileStream.open((outputFilePath).c_str());
        string str;
        for (int i =0; i < hiddenNodes; i++) {
            for (int j = 0; j < inputNodes; j++) {
                if(j ==  0){
                    fileStream<<(*theta)[i*inputNodes + j];
                }else{
                    fileStream<<","<<(*theta)[i*inputNodes + j];
                }
            }
            
            fileStream<<endl;
        }
        int offset = hiddenNodes*inputNodes;
        for (int j = 0; j < hiddenNodes; j++) {
            if(j ==  0){
                fileStream<<(*theta)[offset + j];
            }else{
                fileStream<<","<<(*theta)[offset + j];
            }
        }
        fileStream<<endl;
        for (int i = 0; i < means.size(); i++) {
            if(i ==  0){
                fileStream<<means[i];
            }else{
                fileStream<<","<<means[i];
            }
        }
        fileStream<<endl;
        for (int i = 0; i < stds.size(); i++) {
            if(i ==  0){
                fileStream<<stds[i];
            }else{
                fileStream<<","<<stds[i];
            }
        }
        fileStream.close();
    } catch (ifstream::failure e) {
        cerr<<"Exception opening/reading/closing file\n";
    }
}
void trainTheta( const vector<vector<double> > *sampleData,const int hidNodes){
    
    hiddenNodes = hidNodes;
    vector<vector<double> > *tmpSample = new vector<vector<double> >(*sampleData);
    int  inputNodes = (int)(*tmpSample)[0].size();
    vector<double> means;
    vector<double> stds;
    preProcess(tmpSample, means, stds);
    
    vector<double> theta  = vector<double>();
    vector<double> tmpTheta;
    bool CHECK_GRAD = false;
    
    initilizeTheta(&theta,inputNodes);
    double alpha = 0.1;
    double numda = 0.01;
    
    COST_GRAD res;
    

    for (int i = 0; i < 30; i++) {
        counter = i;
        if ((i+1) % 30 == 0) {
            alpha *= 0.9;
        }
        tmpTheta = theta;
        
        forward(tmpSample, &theta, inputNodes);
        
        res =  backward(tmpSample,&theta,numda);
        cout<<res.cost<<endl;
        if (CHECK_GRAD ) {
            checkGradient(tmpSample, &theta,inputNodes,res.grad,numda);
        }
        
        updateTheta(&theta, inputNodes, res.grad, alpha, numda);
        
    }
    getOrigin(sampleData,means,stds);
    saveToDisk( &theta, inputNodes, means, stds);
    
    
    
}
#endif /* defined(__ConvolutionNN__sparseEncoder__) */
