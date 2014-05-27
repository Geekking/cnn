//
//  FileReader.h
//  ConvolutionNN
//
//  Created by apple on 5/22/14.
//  Copyright (c) 2014 SYSU. All rights reserved.
//

#ifndef __ConvolutionNN__FileReader__
#define __ConvolutionNN__FileReader__

#include <iostream>
#include <vector>
#include <string>
using namespace std;

const int DEFAULT_SIZE = 50000;
const int DEFAULT_DIM = 1000;

template <class T>
class FileReader {
private:
     string trainPath;
     string testPath;
    vector<vector<T> > *testData;
    vector<vector<T> > *trainData;
    int testCount;
    int trainCount;
    
public:
    FileReader(const string trPath, const string tePath):testPath(tePath),trainPath(trPath),testCount(0),trainData(0){};
    ~FileReader(){
        delete testData;
        delete trainData;
    }
    const vector< vector<T> >* getTrainData();
    const vector< vector<T> >* getTestData() ;
    const vector< vector<T> >* getTrainX();
    const vector< vector<T> >* getTrainY();
    const vector< vector<T> >* getTestX();
    const vector< vector<T> >* getTestID() ;
    
};
#include <fstream>

template <class T>
vector<T> split(string& str,const char* c)
{
    char *cstr, *p;
    vector<T> res;
    cstr = new char[str.size()+1];
    strcpy(cstr,str.c_str());
    p = strtok(cstr,c);
    while(p!=NULL)
    {
        res.push_back((T)atof(p) ); // change to float
        p = strtok(NULL,c);
    }
    
    return res;
}

template <class T>
const vector< vector<T> >* FileReader<T>::getTrainData() {
    if ( this->trainData == NULL) {
        this -> trainData = new vector< vector<T> >(DEFAULT_SIZE,vector<T>());
        ifstream file;
        try {
            file.open((this->trainPath).c_str());
            string str;
            int count = 0;
            getline(file, str);
            while (!file.eof()) {
                getline(file, str);
                if (str.size() ==0) {
                    break;
                }
                vector<T> res =  split<T>(str,",");
                (*trainData)[count++] = res;
           }
            trainCount = count;
            cout<<"Train Size:"<<count<<endl;
            cout<<"Train Dim +2 :"<<(*trainData)[0].size()<<endl;
        } catch (ifstream::failure e) {
            cerr<<"Exception opening/reading/closing file\n";
        }
        
    }
    return this->trainData;
}

template <class T>
const vector< vector<T> >* FileReader<T>::getTestData() {
    if (this->testData == NULL) {
        this->testData = new vector<vector<T> >(DEFAULT_SIZE,vector<T>());
        ifstream file;
        try {
            file.open((this->testPath).c_str());
            string str;
            int count = 0;
            getline(file, str);
            while (!file.eof()) {
                getline(file, str);
                if (str.size() ==0) {
                    break;
                }
                vector<T> res =  split<T>(str,",");
                (*testData)[count++] = res;
            }
            testCount = count;
            cout<<"Test Size:"<<count<<endl;
            cout<<"Test Dim +1 :"<<(*testData)[0].size()<<endl;
        } catch (ifstream::failure e) {
            cerr<<"Exception opening/reading/closing file\n";
        }
        

    }
    
    return this->testData;
}

template <class T>
const vector< vector<T> >* FileReader<T>::getTrainX() {
    if (this->trainData == NULL) {
        this->getTrainData();
    }
    cout<<"TrainX"<<endl;
    vector<vector<T> > *trainX = new vector<vector<T> >(trainCount);
    for (int i = 0; i < trainCount; i++) {
        vector<T> oneRow;
        for (int j =2; j < (*trainData)[i].size(); j++) {
            oneRow.push_back((T) ((*trainData)[i][j]) / 255.0);
        }
        (*trainX)[i] = oneRow;
    }
    return trainX;
}

template <class T>
const vector< vector<T> >* FileReader<T>::getTrainY() {
    if (this->trainData == NULL) {
        this->getTrainData();
    }
    vector<vector<T> > *trainY = new vector<vector<T> >(trainCount);
    cout<<"TrainY"<<endl;
    for (int i=0; i < trainCount; i++) {
        vector<T> oneRow;
        oneRow.push_back((*trainData)[i][1]);
        (*trainY)[i] = oneRow;
     }
    return trainY;
    
}
template <class T>
const vector< vector<T> >* FileReader<T>::getTestX() {
    if (this->testData == NULL) {
        this->getTestData();
    }
    cout<<"Test X"<<endl;
    vector<vector<T> > *testX = new vector<vector<T> >(testCount);
    for (int i =0; i < testCount; i++) {
        vector<T> oneRow;
        for(int j = 1;j < (*testData)[i].size();j++){
            oneRow.push_back((T)((*testData)[i][j])/255.0);
         }
        (*testX)[i] = oneRow;
    }
    return testX;
}

template <class T>
const vector< vector<T> >* FileReader<T>::getTestID() {
    if (this->testData == NULL) {
        this->getTestData();
    }
    cout<<"testID"<<endl;
    vector<vector<T> > *testID = new vector<vector<T> >(testCount);
    for (int i = 0; i < testCount; i++) {
        vector<T> oneRow;
        oneRow.push_back((*testData)[i][0]);
        (*testID)[i] = oneRow;
    }
    return testID;
}
#endif /* defined(__ConvolutionNN__FileReader__) */
