#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>
#include <pthread.h>

#include <fstream>
#include <sstream>
#include <iostream>
#include <string>
#include <string.h>
#include <map>
#include <vector>
#include <algorithm>

#include "plsa.h"

using namespace std;

string FilterWord(string s_word)
{
  const char* p = s_word.c_str();
  char ch_ret[256];
  char* ret = ch_ret;
  while((*p)!=0)
  {
    if((65<=*p&&*p<=90)||(97<=*p&&*p<=122))
    {
      *ret = *p;
      ret++;
    }
    p++; 
  }
  *ret = 0;
  string s_ret(ch_ret);
  return s_ret;
}


int WriteTrainData(string infilename,string outfilename)
{
  std::ifstream infile;
  char str[1024*1024];
  char substr1[256];
  char substr2[256];
  map<string,double> word_map;
  infile.open(infilename.c_str());
  if(!infile)
  {
    std::cout<<"open file error"<<std::endl;
  }
  infile.getline(str,sizeof(str));
  int doc_num = 0;
  while(infile.getline(str,sizeof(str)))
  {
    doc_num++;
    std::stringstream ss(str);
    ss>>substr1>>substr2;
    char word[256];
    while(ss>>word)
    {
      string s_word(word);
      s_word = FilterWord(s_word);
      if(s_word.size()==0) continue;
      map<string,double>::iterator itr = word_map.find(s_word);
      if(itr==word_map.end())
      {
	word_map.insert(pair<string,double>(s_word,1));
      }
      else
      {
        itr->second++;
      }
    }
  }
  infile.close();

  std::ofstream outfile;
  outfile.open(outfilename.c_str());
  for(map<string,double>::iterator itr=word_map.begin();itr!=word_map.end();itr++)
  {
    if(itr->second>(doc_num*0.005)&&itr->second<(doc_num*0.7))
    {
      outfile<<itr->first<<" "<<itr->second<<endl;
    }
  }
  outfile.close();
  return 0;
}

map<string,int> ReadTrainDataByHash(string filename)
{
  map <string,int> ret;
  std::ifstream infile;
  infile.open(filename.c_str());
  char str[256];
  char sub_str[256];
  char dft_str[256];
  int index = 0;
  while(infile.getline(str,sizeof(str)))
  {
    string tmpstr(str);
    std::stringstream ss(tmpstr);
    ss>>sub_str;
    ss>>dft_str;
    ret.insert(pair<string,int>(sub_str,index++));
  }
  return ret;
}

int GetTrainData(double** data,double* label,int doc_num,int dimension,string filename,map<string,int> word_map)
{
  int tmpnum = 0;
  char str[1024*1024];
  char word[256];
  char s_id[256];
  char s_label[256];
  ifstream infile;
  infile.open(filename.c_str());
  infile.getline(str,sizeof(str));
  while(tmpnum<doc_num)
  {
    infile.getline(str,sizeof(str));
    std::stringstream ss(str);
    ss>>s_id;
    ss>>s_label;
    label[tmpnum] = atoi(s_label);
    while(ss>>word)
    {
      string s_word(word);
      s_word = FilterWord(s_word);
      map<string,int>::iterator itr = word_map.find(s_word);
      if(itr!=word_map.end()&&itr->second<dimension)
      {
        data[tmpnum][itr->second] += 1;
      }
    }
    tmpnum++;
  }
  infile.close();
  return 0;
}

int main(int argc,char* argv[])
{
  if(argc<2)
  {
    cout<<"no enough arguments"<<endl;
  }
  int shownum = atoi(argv[1]);
  WriteTrainData("labeledTrainData.tsv","plsa_word.txt");
  map<string,int> word_map = ReadTrainDataByHash("plsa_word.txt");
  int word_size = word_map.size();
  int topic_num = 10;
  int doc_num = 25000;
  double** train_data = new double*[doc_num];
  double* label_data = new double[doc_num];
  for(int i=0;i<doc_num;i++)
  {
    train_data[i] = new double[word_size];
    bzero((void*)train_data[i],word_size*sizeof(double));
  }
  GetTrainData(train_data,label_data,doc_num,word_size,"labeledTrainData.tsv",word_map);
  PLSAOBJ model;
  model.TrainModel(train_data,1000,word_size,topic_num,0.1,100);
  model.SaveModel("plsa_test.model");
  //model.LoadModel("plsa_test.model");
  float** pw_t = model.GetPW_T();
  for(int i=0;i<topic_num;i++)
  {
    map<float,string> sorted_map;
    int j = 0;
    for(map<string,int>::iterator itr=word_map.begin();itr!=word_map.end();itr++)
    {
      sorted_map.insert(pair<float,string>(pw_t[i][j],itr->first));
      j++;
    }
    cout<<"topic "<<i<<endl;
    j = 0;
    for(map<float,string>::reverse_iterator itr=sorted_map.rbegin();(itr!=sorted_map.rend())&&(j<shownum);itr++)
    {
      cout<<itr->second<<" ";
      j++;
    }
    cout<<endl;
  }
  return 0;
}
