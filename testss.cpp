#include <string>
#include <sstream>
#include <iostream>

using namespace std;

int main(){
	stringstream ss;
	ss << ",";
	cout<<ss.str()<<endl;
	ss.flush();
	ss<<"2";
	cout<<ss.str()<<endl;
	return 0;
};
