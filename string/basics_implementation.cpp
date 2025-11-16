#include<bits/stdc++.h>
using namespace std;

int main(){
    string s;
    s = "hello";
    s+= " world";
    cout<< s<<endl;
    s.size();
    // cout<<s.substr(0,5);
    for (int i=0;i<s.size();++i){
        cout<< s[i];

    }
    //string to int
    string q ="1234";
    for (int i=0;i<q.size();++i){
        int a = q[i];
        cout<< stoi("a");

    }
}