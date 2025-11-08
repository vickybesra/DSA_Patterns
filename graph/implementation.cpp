#include<bits/stdc++.h>
using namespace std;


void buildGraph(int V, int E, vector<vector<int>>&adj){
    int u, v;
    for (int i = 0; i< E; ++i){

        //input vertices
        cin>> u>> v;
        adj[u].push_back(v);
        
    }
}
void printGraph(int V, vector<vector<int>>&adj){
    cout<<"adjency list representation"<<endl;
    for (int i=0; i<V; ++i){
        cout<<"vertex"<< i <<"is connected to ";
        for (int neighbour : adj[i]){
            cout<< neighbour << endl;
        }
        cout<< endl;
    }
}


int main(){
    int V, E;
    cout<<"input the Vertices:V"<< endl;
    cin>>V;
    cout<<"input the Edges:E"<< endl;
    cin>>E;

    vector<vector<int>> adj(V);

    buildGraph(V, E, adj);
    printGraph(V, adj);
}