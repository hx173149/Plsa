#include <math.h>
#include <iostream>
#include <fstream>

#include <stdlib.h>
#include <stdio.h>
#include <sys/time.h>

#include "plsa.h"

using namespace std;

PLSAOBJ::PLSAOBJ()
{
  doc_num = MAX_DOC_NUM_;
  word_num = MAX_WORD_NUM_;
  topic_num = MAX_TOPIC_NUM_;

  pt_d = new float* [doc_num];
  for(int i=0;i<doc_num;i++)
  {
    double sum = 0;
    pt_d[i] = new float[topic_num];
    for(int j=0;j<topic_num;j++)
    {
      pt_d[i][j] = (float)(rand()%99+1)/100;
      sum += pow(pt_d[i][j],2);
    }
    sum = sqrt(sum);
    for(int j=0;j<topic_num;j++)
    {
      pt_d[i][j] = pt_d[i][j]/sum;
    }
  }

  pw_t = new float* [topic_num];
  for(int i=0;i<topic_num;i++)
  {
    pw_t[i] = new float[word_num];
    double sum = 0;
    for(int j=0;j<word_num;j++)
    {
      pw_t[i][j] = (float)(rand()%99+1)/100;//(float)(rand()%((int)(restP*100))+1)/100;
      sum += pow(pw_t[i][j],2);
    }
    sum = sqrt(sum);
    for(int j=0;j<word_num;j++)
    {
      pw_t[i][j] = pw_t[i][j]/sum;//(float)(rand()%((int)(restP*100))+1)/100;
    }
  }
  
}

PLSAOBJ::~PLSAOBJ()
{
  doc_num = MAX_DOC_NUM_;
  word_num = MAX_WORD_NUM_;
  topic_num = MAX_TOPIC_NUM_;
  for(int i=0;i<doc_num;i++)
  {
    delete[] pt_d[i];
  }
  delete[] pt_d;
  for(int i=0;i<topic_num;i++)
  {
    delete[] pw_t[i];
  }
  delete[] pw_t;
  
}

void PLSAOBJ::EStep(double** train_data)
{
  //calculate p(tk|di)
  cout<<"e-step"<<endl;
  for(int i=0;i<doc_num;i++)
    for(int j=0;j<word_num;j++)
    {
      double down = 0;
      for(int kk=0;kk<topic_num;kk++)
      {
        down += pw_t[kk][j]*pt_d[i][kk];
      }
      if(train_data[i][j]!=0)
      {
        int tmpindex = i*word_num+j;
        map<int,float*>::iterator itr;
        itr = pt_wd.find(tmpindex);
        float* tmpP = itr->second;
        for(int k=0;k<topic_num;k++)
        {
          double top = pw_t[k][j]*pt_d[i][k];
          tmpP[k] = top/down;
        }
     }
   }  
}

void PLSAOBJ::MStep(double** train_data)
{
  //calc p(w|t)
  cout<<"m-step"<<endl;
  for(int i=0;i<topic_num;i++)
  {
    double down = 0;
    for(int k=0;k<doc_num;k++)
    {
      for(int kk=0;kk<word_num;kk++)
      {
        if(train_data[k][kk]!=0)
        {
          int tmpindex = k*word_num+kk;
          map<int,float*>::iterator itr = pt_wd.find(tmpindex);
          float* tmpP = itr->second;
          down += train_data[k][kk]*tmpP[i]; 
         }
      }
    }
    for(int j=0;j<word_num;j++)
    {
      double top = 0;
      for(int k=0;k<doc_num;k++)
      {
        if(train_data[k][j]!=0)
        {
          int tmpindex = k*word_num+j;
          map<int,float*>::iterator itr = pt_wd.find(tmpindex);
          float* tmpP = itr->second;
          top += train_data[k][j]*tmpP[i]; 
        }
      }
      if(top>0&&down>0)
        pw_t[i][j] = top/down;
    }
  }

  //calc p(t|d)
  for(int i=0;i<doc_num;i++)
  {
    double down = 0;
    for(int k=0;k<word_num;k++)
    {
      down += train_data[i][k];
    }
    for(int j=0;j<topic_num;j++)
    {
      double top = 0;
      for(int k=0;k<word_num;k++)
      {
        if(train_data[i][k]!=0)
        {
          int tmpindex = i*word_num + k;
          map<int,float*>::iterator itr = pt_wd.find(tmpindex);
          float* tmpP = itr->second;
          top += train_data[i][k]*tmpP[j];
        }
      }	
      if(top>0&&down>0)
        pt_d[i][j] = top/down;
    }
  }
}

double PLSAOBJ::LogLikehood(double** train_data)
{
  double likehood = 0;
  for(int i=0;i<doc_num;i++)
    for(int j=0;j<word_num;j++)
    {
      if(train_data[i][j]!=0)
      {
        double tmp1 = 0;
        for(int k=0;k<topic_num;k++)
        {
          tmp1 += pt_d[i][k]*pw_t[k][j];
        }
        if(tmp1!=0)
        {
          likehood += train_data[i][j]*log(tmp1);
        }
      }
    }
  return likehood;
}

int PLSAOBJ::TrainModel(double** train_data,int d_num,int w_num,int t_num,double eps,int max_iter)
{
  doc_num = d_num;
  word_num = w_num;
  topic_num = t_num;

  double tmpsum = 0;
  for(int i=0;i<doc_num;i++)
    for(int j=0;j<word_num;j++)
    {
      if(train_data[i][j]!=0)
      {
        int tmpindex = i*word_num+j;
	float* tmpP = new float[topic_num];
	///*
	for(int k=0;k<topic_num;k++)
	{
          tmpP[k] = (float)(rand()%99+1)/100;//(double)(rand()%((int)(100*restP))+1)/100;
          tmpsum += pow(tmpP[k],2);
	}
	//*/
	pt_wd.insert(pair<int,float*>(tmpindex,tmpP));
      }      
    }
  tmpsum = sqrt(tmpsum);

  for(map<int,float*>::iterator itr=pt_wd.begin();itr!=pt_wd.end();itr++)
  {
    for(int k=0;k<topic_num;k++)
      itr->second[k] = (itr->second[k])/tmpsum;
  } 
 
  double likehood = 99;
  double last_likehood = 0;
  struct timeval start_t,end_t;
  int cur_iter = 0;
  while(abs(likehood-last_likehood)>eps&&cur_iter<max_iter)
  //while(cur_iter<max_iter)
  {
    cur_iter++;
    gettimeofday(&start_t,NULL);
    last_likehood = likehood;
    //E step
    EStep(train_data); 

    //M step
    MStep(train_data); 

    //calculate likelyhood
    likehood = LogLikehood(train_data);
     
    gettimeofday(&end_t,NULL);
    double timeuse = end_t.tv_sec-start_t.tv_sec;
    cout<<"likehood: "<<likehood<<endl;
    cout<<"time consume: "<<timeuse<<endl;
  }
  for(map<int,float*>::iterator itr=pt_wd.begin();itr!=pt_wd.end();itr++)
  {
    delete[] itr->second;
  }
  return 0;
}


int PLSAOBJ::Inference(double* src,double* dst,int& len,int iter_max)
{
  len = topic_num;
  float** tmpP_t_w = new float*[word_num];
  for(int i=0;i<word_num;i++)
  {
    tmpP_t_w[i] = new float[topic_num];
  }
  //rand initial 
  double sum = 0;
  for(int i=0;i<topic_num;i++)
  {
    dst[i] = (float)(rand()%99+1)/100;
    sum += pow(dst[i],2);
  }
  sum = sqrt(sum);
  for(int i=0;i<topic_num;i++)
  {
    dst[i] = dst[i]/sum;
  }
  int iter = 0;
  while(iter<iter_max)
  {
    iter++;
    //E step:
    for(int i=0;i<word_num;i++)
    {
      double down = 0;	
      for(int j=0;j<topic_num;j++)
      {
        down += dst[j]*pw_t[j][i];
      }
      for(int j=0;j<topic_num;j++)
      {
        double top = dst[j]*pw_t[j][i];
        tmpP_t_w[i][j] = top/down;
      }
    }
    //M step
    double down1 = 0;
    for(int i=0;i<word_num;i++)
      for(int j=0;j<topic_num;j++)
      {
	if(src[i]!=0)
          down1 += src[i]*tmpP_t_w[i][j];	
      }
    for(int i=0;i<topic_num;i++)
    {
      double top = 0;
      for(int j=0;j<word_num;j++)
      {
        if(src[j]!=0)
	{
          top += src[j]*tmpP_t_w[j][i];
	}
      } 
      dst[i] = top/down1;
    }
  }
  for(int i=0;i<word_num;i++)
  {
    delete[] tmpP_t_w[i];
  }
  delete[] tmpP_t_w;

  return 0;
}

int PLSAOBJ::SaveModel(const string& filepath)
{
  ofstream outfile(filepath.c_str());
  outfile<<doc_num<<endl;
  outfile<<topic_num<<endl;
  outfile<<word_num<<endl;
  for(int i=0;i<doc_num;i++)
    for(int j=0;j<topic_num;j++)
    {
      outfile<<pt_d[i][j]<<endl;
    }
  for(int i=0;i<topic_num;i++)
    for(int j=0;j<word_num;j++)
    {
      outfile<<pw_t[i][j]<<endl;
    }
  outfile.close();
  return 0;
}


int PLSAOBJ::LoadModel(const string& filepath)
{
  ifstream infile(filepath.c_str());
  if(!infile)
    return -1;
  char str[256];
  infile.getline(str,sizeof(str));
  doc_num = atoi(str);
  infile.getline(str,sizeof(str));
  topic_num = atoi(str);
  infile.getline(str,sizeof(str));
  word_num = atoi(str);
  for(int i=0;i<doc_num;i++)
    for(int j=0;j<topic_num;j++)
    {
      infile.getline(str,sizeof(str));
      pt_d[i][j] = atof(str); 
    }
  for(int i=0;i<topic_num;i++)
    for(int j=0;j<word_num;j++)
    {
      infile.getline(str,sizeof(str));
      pw_t[i][j] = atof(str); 
    }
  return 0; 
}

int PLSAOBJ::GetSize(int& d_num,int& t_num,int& w_num)
{
  d_num = doc_num;
  t_num = topic_num;
  w_num = word_num;
  return 0;
}

float** PLSAOBJ::GetPT_D()
{
  return pt_d;
}

