//
//  extract.h
//  ConvolutionNN
//
//  Created by apple on 5/22/14.
//  Copyright (c) 2014 SYSU. All rights reserved.
//

#ifndef ConvolutionNN_extract_h
#define ConvolutionNN_extract_h
#include <vector>
#include <fstream>

void extract(string filePath){
    
    ifstream file;
    ofstream oFile;
    string smallPath = "/Users/apple/Documents/develop/ufldl/cnn/cifar-10-train_m.csv";
    try {
        file.open(filePath.c_str());
        oFile.open(smallPath.c_str());
        string str;
        int count = 0;
        while (!file.eof()) {
            getline(file, str);
            if(count > 100){
                break;
            }
            count ++;
            oFile << str<<endl;
        }
        
    } catch (ifstream::failure e) {
            cerr<<"Exception opening/reading/closing file\n";
    }
    

}
#endif
