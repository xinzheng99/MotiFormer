import random
from sklearn.metrics import f1_score,recall_score
import torch
import numpy as np
import pickle
import scipy.sparse as sp
import torch

from itertools import combinations


import os
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
import networkx as nx

os.environ["KMP_DUPLICATE_LIB_OK"] = "True"

np.seterr(divide='ignore',invalid='ignore')

class PositionEncoder_J(torch.nn.Module):
    def __init__(self, motif_type='1234', device='cuda', path='../Data/programweb/', k=16,neighbors='all'):

        super(PositionEncoder_J, self).__init__()

        with open(f'{path}madjs1-7.pickle', 'rb') as file:
                madjs= pickle.load(file)

        self.neighbors=neighbors
        self.adj_matrices = []
        
        for i in list(motif_type):
            self.adj_matrices.append(madjs[int(i)-1])
        
        self.adj_matrices=np.array(self.adj_matrices)
        self.adjm1=madjs[0]
        
        self.k = k
        self.path=path
    def get_LPE(self, k):
        """
        :return:使用特征值/向量组成的拉普拉斯位置嵌入
        """
        result = []
        adjs = self.adj_matrices  
        # print(adjs[0, :].shape)
        for i in range(len(self.adj_matrices)):
            print(i)
            adj_matrix = adjs[i, :, :]
            adj_matrix[adj_matrix > 0] = 1
            # #  adj_matrix[adj_matrix <= 0] = 0  # not necessary
            #
            R = np.sum(adj_matrix, axis=1)  #
            R_sqrt = 1 / (np.sqrt(R) + (1e-6))
            D_sqrt = np.diag(R_sqrt)
            I = np.eye(adjs[i].shape[0])
            laplacian_matrix = I - np.matmul(np.matmul(D_sqrt, adj_matrix), D_sqrt)
            eigVals, eigVecs = np.linalg.eigh(laplacian_matrix)
            """
            R = np.sum(adj_matrix, axis=1)
            R_half = np.power(R, -0.5)  # 负0.5次方
            degreeMatrix = np.diag(R)
            degreeMatrix_half = np.diag(R_half)
            laplacian_matrix = degreeMatrix - adj_matrix
            sym_laplacian_matrix = degreeMatrix_half @ laplacian_matrix @ degreeMatrix_half
            # 正则化以后有nan，可能是有除0 ：np.power(R, -0.5)
            # 可以把为0的变成1
            eigVals, eigVecs = np.linalg.eigh(laplacian_matrix)


            """

            
            eigVecs = eigVecs[0:k, :].T

            

            result.append(eigVecs)
        return np.array(result)

    def get_LLMPE(self, ):
        if self.neighbors=='all':
            with open(f'{self.path}llm_emb_result_noid.pkl','rb') as f:
                print('all neighbors')
                llm_emb= pickle.load(f)
                llm_emb=torch.tensor(llm_emb)
        else:
            with open(f'{self.path}llm_emb_result_noid_100neighbors.pkl','rb') as f:
                print('100 neighbors')
                llm_emb= pickle.load(f)
                llm_emb=torch.tensor(llm_emb)
            
        return llm_emb

    def get_PE(self, devices,k):
        result = self.get_LPE(k)

        PE = torch.from_numpy(result).float().to(devices)
        PE = PE.permute(1, 0, 2)
 
        LLM_PE=self.get_LLMPE().to(devices)
        

        return PE,LLM_PE, torch.tensor(self.adj_matrices).to(devices),self.adjm1.to(devices)

class PositionEncoder_movie_J(torch.nn.Module):
    def __init__(self, motif_type='1234', device='cuda', path='../Data/movie/', k=16,neighbors='all'):

        super(PositionEncoder_movie_J, self).__init__()

        with open(f'{path}movie_m123456_madjs_V2.pickle', 'rb') as file:
                madjs= pickle.load(file)
                

        self.neighbors=neighbors
        self.adj_matrices = []
        
        for i in list(motif_type):
            self.adj_matrices.append(madjs[int(i)-1].todense())
        
        self.adj_matrices=np.array(self.adj_matrices)
        self.adjm1=madjs[0].todense()
        #print(self.adj_matrices.shape)
        self.k = k
        self.path=path
    def get_LPE(self, k):
        """
        :return:使用特征值/向量组成的拉普拉斯位置嵌入
        """
        result = []
        adjs = self.adj_matrices  
        # print(adjs[0, :].shape)
        for i in range(len(self.adj_matrices)):
            print(i)
            adj_matrix = adjs[i, :, :]
            adj_matrix[adj_matrix > 0] = 1
            # #  adj_matrix[adj_matrix <= 0] = 0  # not necessary
            #
            R = np.sum(adj_matrix, axis=1)  #
            R_sqrt = 1 / (np.sqrt(R) + (1e-6))
            D_sqrt = np.diag(R_sqrt)
            I = np.eye(adjs[i].shape[0])
            laplacian_matrix = I - np.matmul(np.matmul(D_sqrt, adj_matrix), D_sqrt)
            eigVals, eigVecs = np.linalg.eigh(laplacian_matrix)
            """
            R = np.sum(adj_matrix, axis=1)
            R_half = np.power(R, -0.5)  # 负0.5次方
            degreeMatrix = np.diag(R)
            degreeMatrix_half = np.diag(R_half)
            laplacian_matrix = degreeMatrix - adj_matrix
            sym_laplacian_matrix = degreeMatrix_half @ laplacian_matrix @ degreeMatrix_half
            # 正则化以后有nan，可能是有除0 ：np.power(R, -0.5)
            # 可以把为0的变成1
            eigVals, eigVecs = np.linalg.eigh(laplacian_matrix)


            """

            
            eigVecs = eigVecs[0:k, :].T


            result.append(eigVecs)
        return np.array(result)

    def get_LLMPE(self, ):
        if self.neighbors=='all':
            with open(f'{self.path}llm_emb_result_noid.pkl','rb') as f:
                print('all neighbors')
                llm_emb= pickle.load(f)
                llm_emb=torch.tensor(llm_emb)
        else:
            with open(f'{self.path}llm_emb_result_noid_100neighbors.pkl','rb') as f:
                print('100 neighbors')
                llm_emb= pickle.load(f)
                llm_emb=torch.tensor(llm_emb)
            
        return llm_emb

    def get_PE(self, devices,k):
        result = self.get_LPE(k)

        PE = torch.from_numpy(result).float().to(devices)
        PE = PE.permute(1, 0, 2)

        LLM_PE=self.get_LLMPE().to(devices)
        

        return PE,LLM_PE, torch.tensor(self.adj_matrices).to(devices),torch.tensor(self.adjm1).to(devices)


