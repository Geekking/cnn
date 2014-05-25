#include <string>
#include <vector>
#include <iostream>
using namespace std;
template <class T>
class CLA{
private:
	T a;
public:
	CLA(T b):a(b){};
	void show(){
		cout<<a<<endl;
	}
};
int main(){
	vector<vector<int> > a =  vector<vector<int> >(10,vector<int>(10));
	a[0][0] = 10;
	CLA<int> b(9);
	b.show();
	srand(10);
	float mm  = rand();
	cout<<mm<<endl<<rand()<<endl;
	cout<<a[0][0]<<endl;
	string str = "abcd";
}
