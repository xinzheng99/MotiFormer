import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import os

class GATLayer(nn.Module):
    """GAT层"""
    def __init__(self,input_feature,output_feature,dropout,alpha=0,concat=True):
        super(GATLayer,self).__init__()
        self.input_feature = input_feature
        self.output_feature = output_feature
        self.alpha = alpha
        self.dropout = dropout
        self.concat = concat
        self.a = nn.Parameter(torch.empty(size=(2*output_feature,1)))
        self.w = nn.Parameter(torch.empty(size=(input_feature,output_feature)))
        self.leakyrelu = nn.LeakyReLU()#self.alpha
        self.reset_parameters()
    
    def reset_parameters(self):
        nn.init.xavier_uniform_(self.w.data,gain=1.414)
        nn.init.xavier_uniform_(self.a.data,gain=1.414)
    
    def forward(self,h,adj):
        Wh = torch.mm(h,self.w)
        e = self._prepare_attentional_mechanism_input(Wh)
        zero_vec = -9e15*torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec) # adj>0的位置使用e对应位置的值替换，其余都为-9e15，这样设定经过Softmax后每个节点对应的行非邻居都会变为0。
        attention = F.softmax(attention, dim=1) # 每行做Softmax，相当于每个节点做softmax
        attention = F.dropout(attention, self.dropout, training=self.training)
        h_prime = torch.mm(attention, Wh) # 得到下一层的输入
        
        if self.concat:
            return F.selu(h_prime) #激活
        else:
            return h_prime
        
    def _prepare_attentional_mechanism_input(self,Wh):
        
        Wh1 = torch.matmul(Wh,self.a[:self.output_feature,:]) # N*out_size @ out_size*1 = N*1
        
        Wh2 = torch.matmul(Wh,self.a[self.output_feature:,:]) # N*1
        
        e = Wh1+Wh2.T # Wh1的每个原始与Wh2的所有元素相加，生成N*N的矩阵
        return self.leakyrelu(e)
    
class GAT(nn.Module):
    """GAT模型"""
    def __init__(self,input_size,hidden_size,output_size,dropout,nheads,alpha=0,concat=True):
        super(GAT,self).__init__()
        self.dropout= dropout
        self.concat=concat
        self.attention = [GATLayer(input_size, hidden_size, dropout=dropout, alpha=alpha,concat=concat) for _ in range(nheads)]
        for i,attention in enumerate(self.attention):
            self.add_module('attention_{}'.format(i),attention)
        if concat:
            self.out_att = GATLayer(hidden_size*nheads, output_size, dropout=dropout, alpha=alpha,concat=False)
        else:
            self.out_att = GATLayer(hidden_size, output_size, dropout=dropout, alpha=alpha,concat=False)
    def forward(self,x,adj):
        x = F.dropout(x,self.dropout,training=self.training)
        if self.concat:
            x = torch.cat([att(x,adj) for att in self.attention],dim=1)
        else:
            x = torch.cat([att(x,adj).unsqueeze(0) for att in self.attention],dim=0)
            x=torch.mean(x,dim=0)
        x = F.dropout(x,self.dropout,training=self.training)
        x = F.selu(self.out_att(x,adj))
        
        return x

class MGATLayer(nn.Module):

    def __init__(self, in_features, out_features, dropout, alpha, concat=True, max_pool=False, motif_len=3, N=6861,device='cpu'):
        """
        分别权重
        :param in_features: 输入的特征维数
        :param out_features: 输出特征维数
        :param dropout:
        :param alpha:
        :param concat:
        :param max_pool:
        :param motif_len: 使用几个motif
        :param N: 所有节点个数
        """
        super(MGATLayer, self).__init__()

        self.dropout = dropout
        self.in_features = in_features  # features.shape[1]
        self.out_features = out_features  # 默认 8
        self.alpha = alpha
        self.concat = concat
        self.max_pool = max_pool
        self.motif_len = motif_len
        self.device=device
        self.motif_att=[]
        self.ln=torch.nn.BatchNorm1d(out_features)
        
        # nodel-level attention
        self.W2 = nn.Parameter(torch.empty(size=( in_features, out_features)))
        nn.init.xavier_uniform_(self.W2.data, gain=1.414)
        self.a2 = nn.Parameter(torch.empty(size=( 2 * out_features, 1)))
        nn.init.xavier_uniform_(self.a2.data, gain=1.414)
        self.W1 = nn.Parameter(torch.empty(size=( in_features, out_features)))
        nn.init.xavier_uniform_(self.W1.data, gain=1.414)
        self.a1 = nn.Parameter(torch.empty(size=( 2 * out_features, 1)))
        nn.init.xavier_uniform_(self.a1.data, gain=1.414)
        self.W3 = nn.Parameter(torch.empty(size=( in_features, out_features)))
        nn.init.xavier_uniform_(self.W3.data, gain=1.414)
        self.a3 = nn.Parameter(torch.empty(size=( 2 * out_features, 1)))
        nn.init.xavier_uniform_(self.a3.data, gain=1.414)



        self.elu = F.elu

    def forward(self, motif_adjs,features):
        """
        :param h: 上一层集合了所有motif以及所有邻居节点的特征
        :param motif_adjs:motif邻接矩阵

        """
        # 进行project，获得h
        Whs1 = torch.mm(features, self.W1)
        # (N,in_features) * (in_features,out_features) = (N,out_features)
        # 注意力分数计算
        
        e1 = self._prepare_attentional_mechanism_input1(Whs1)  # (N,N)
        
        self.zero_vec = (-9e15) * torch.ones_like(e1).to(self.device)
        self.one_zero_vec =torch.ones(1).to(self.device)* (-9e15)

        attention1 = torch.where(motif_adjs > 0, e1, self.zero_vec)  # (num_motifs,N,N) 是motif邻居的保留e
        attention1 = F.softmax(attention1, dim=1)
        attention1 = F.dropout(attention1, self.dropout, training=self.training)
        h_m1 = torch.matmul(attention1, Whs1)

        Whs2 = torch.mm(features, self.W2)
        
        # (N,in_features) * (in_features,out_features) = (N,out_features)
        # 注意力分数计算
        e2 = self._prepare_attentional_mechanism_input2(Whs2)  # (N,N)
        


        attention2 = torch.where(motif_adjs > 0, e2, self.zero_vec)  # (num_motifs,N,N) 是motif邻居的保留e
        attention2 = F.softmax(attention2, dim=1)
        attention2 = F.dropout(attention2, self.dropout, training=self.training)

        # 集合了本身和所有带注意力分数邻居信息
        h_m2 = torch.matmul(attention2, Whs2)

        Whs3 = torch.mm(features, self.W3)
        
        # (N,in_features) * (in_features,out_features) = (N,out_features)
        # 注意力分数计算
        e3 = self._prepare_attentional_mechanism_input2(Whs3)  # (N,N)
        


        attention3 = torch.where(motif_adjs > 0, e3, self.zero_vec)  # (num_motifs,N,N) 是motif邻居的保留e
        attention3 = F.softmax(attention3, dim=1)
        attention3 = F.dropout(attention3, self.dropout, training=self.training)

        # 集合了本身和所有带注意力分数邻居信息
        h_m3 = torch.matmul(attention3, Whs3)


        h_m = h_m1+h_m2+h_m3  # (num_motif,N,out_features) =(num_motifs,N,N)*(num_motif,N,out_features)
        #h_m=self.ln(h_m)
        # motif-level attention

        
        """ 测试：
        e = torch.Tensor([[0.2, 0.8]])

        f = torch.Tensor([
                    [[1, 2], [1, 2]],
                  [[3, 2], [3, 2]]
                ])

        result=e.matmul(f.permute(1,0,2))  # 应该为 2.6  2.0

        """

        return self.elu(h_m)

    def _prepare_attentional_mechanism_input1(self, Whs):
        """

        :param Whs:
        :return: 根据Wh计算所有N与N的注意力分数，
        """
       
        
        # 对于每个motif分别计算
        
        Wh1 = torch.matmul(Whs, self.a1[ :self.out_features, :])
        Wh2 = torch.matmul(Whs, self.a1[ self.out_features:, :])

        e = Wh1 + Wh2.t()
        
        


        
        #output = self.elu(e)
        output = torch.nn.functional.softmax(e,dim=1)
        return output
    def _prepare_attentional_mechanism_input2(self, Whs):
        """

        :param Whs:
        :return: 根据Wh计算所有N与N的注意力分数，
        """
       
        
        # 对于每个motif分别计算
        
        Wh1 = torch.matmul(Whs, self.a2[ :self.out_features, :])
        Wh2 = torch.matmul(Whs, self.a2[ self.out_features:, :])

        e = Wh1 + Wh2.t()
        
        


        
        #output = self.elu(e)
        output = torch.nn.functional.softmax(e,dim=1)
        return output
    def _prepare_attentional_mechanism_input3(self, Whs):
        """

        :param Whs:
        :return: 根据Wh计算所有N与N的注意力分数，
        """
       
        
        # 对于每个motif分别计算
        
        Wh1 = torch.matmul(Whs, self.a3[ :self.out_features, :])
        Wh2 = torch.matmul(Whs, self.a3[ self.out_features:, :])

        e = Wh1 + Wh2.t()
        
        


        
        #output = self.elu(e)
        output = torch.nn.functional.softmax(e,dim=1)
        return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'

class MGATLayer_gat(nn.Module):

    def __init__(self, in_features, out_features, dropout, alpha, concat=True, max_pool=False, motif_len=3, N=6861,device='cpu'):
        """
        分别权重
        :param in_features: 输入的特征维数
        :param out_features: 输出特征维数
        :param dropout:
        :param alpha:
        :param concat:
        :param max_pool:
        :param motif_len: 使用几个motif
        :param N: 所有节点个数
        """
        super(MGATLayer_gat, self).__init__()

        self.dropout = dropout
        self.in_features = in_features  # features.shape[1]
        self.out_features = out_features  # 默认 8
        self.alpha = alpha
        self.concat = concat
        self.max_pool = max_pool
        self.motif_len = motif_len
        self.device=device
        self.motif_att=[]
        self.ln=torch.nn.BatchNorm1d(out_features)
        
        # nodel-level attention

        self.W1 = nn.Parameter(torch.empty(size=( in_features, out_features)))
        nn.init.xavier_uniform_(self.W1.data, gain=1.414)
        self.a1 = nn.Parameter(torch.empty(size=( 2 * out_features, 1)))
        nn.init.xavier_uniform_(self.a1.data, gain=1.414)



        self.elu = F.elu

    def forward(self, motif_adjs,features):
        """
        :param h: 上一层集合了所有motif以及所有邻居节点的特征
        :param motif_adjs:motif邻接矩阵

        """
        # 进行project，获得h
        Whs1 = torch.mm(features, self.W1)
        # (N,in_features) * (in_features,out_features) = (N,out_features)
        # 注意力分数计算
        
        e1 = self._prepare_attentional_mechanism_input1(Whs1)  # (N,N)
        
        self.zero_vec = (-9e15) * torch.ones_like(e1).to(self.device)
        self.one_zero_vec =torch.ones(1).to(self.device)* (-9e15)

        attention1 = torch.where(motif_adjs > 0, e1, self.zero_vec)  # (num_motifs,N,N) 是motif邻居的保留e
        attention1 = F.softmax(attention1, dim=1)
        attention1 = F.dropout(attention1, self.dropout, training=self.training)
        h_m1 = torch.matmul(attention1, Whs1)

        

        h_m = h_m1 
        #h_m=self.ln(h_m)
        # motif-level attention

        
        """ 测试：
        e = torch.Tensor([[0.2, 0.8]])

        f = torch.Tensor([
                    [[1, 2], [1, 2]],
                  [[3, 2], [3, 2]]
                ])

        result=e.matmul(f.permute(1,0,2))  # 应该为 2.6  2.0

        """

        return self.elu(h_m)

    def _prepare_attentional_mechanism_input1(self, Whs):
        """

        :param Whs:
        :return: 根据Wh计算所有N与N的注意力分数，
        """
       
        
        # 对于每个motif分别计算
        
        Wh1 = torch.matmul(Whs, self.a1[ :self.out_features, :])
        Wh2 = torch.matmul(Whs, self.a1[ self.out_features:, :])

        e = Wh1 + Wh2.t()
        
        


        
        #output = self.elu(e)
        output = torch.nn.functional.softmax(e,dim=1)
        return output
    def _prepare_attentional_mechanism_input2(self, Whs):
        """

        :param Whs:
        :return: 根据Wh计算所有N与N的注意力分数，
        """
       
        
        # 对于每个motif分别计算
        
        Wh1 = torch.matmul(Whs, self.a2[ :self.out_features, :])
        Wh2 = torch.matmul(Whs, self.a2[ self.out_features:, :])

        e = Wh1 + Wh2.t()
        
        


        
        #output = self.elu(e)
        output = torch.nn.functional.softmax(e,dim=1)
        return output
    def _prepare_attentional_mechanism_input3(self, Whs):
        """

        :param Whs:
        :return: 根据Wh计算所有N与N的注意力分数，
        """
       
        
        # 对于每个motif分别计算
        
        Wh1 = torch.matmul(Whs, self.a3[ :self.out_features, :])
        Wh2 = torch.matmul(Whs, self.a3[ self.out_features:, :])

        e = Wh1 + Wh2.t()
        
        


        
        #output = self.elu(e)
        output = torch.nn.functional.softmax(e,dim=1)
        return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'



class MLPForRec(nn.Module):
    def weight_init(m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)
            nn.init.constant_(m.bias, 0)

    def __init__(self, in_features_num, hideen_num, out_features_num):
        super(MLPForRec, self).__init__()
        # for mashup
        self.fc1_m = nn.Linear(in_features_num, hideen_num)
        self.fc2_m = nn.Linear(hideen_num, int(hideen_num / 2))
        self.fc3_m = nn.Linear(int(hideen_num / 2), out_features_num)
        # for api 
        self.fc1_a = nn.Linear(in_features_num, hideen_num)
        self.fc2_a = nn.Linear(hideen_num, int(hideen_num / 2))
        self.fc3_a = nn.Linear(int(hideen_num / 2), out_features_num)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)

    def forward(self, mashup, api):
        assert torch.is_tensor(mashup) and torch.is_tensor(api), "ERROR: The input of the embedding layer is not a Tensor."

        mashup = nn.functional.elu(self.fc1_m(mashup))
        mashup = nn.functional.elu(self.fc2_m(mashup))
        mashup = nn.functional.elu(self.fc3_m(mashup))

        api = nn.functional.elu(self.fc1_a(api))
        api = nn.functional.elu(self.fc2_a(api))
        api = nn.functional.elu(self.fc3_a(api))

        result = torch.mm(mashup, api.T)
        result = torch.where(result == 0, (1e-9), result.to(torch.float64))
        result = nn.functional.softmax(result, dim=1)

        return result

class MotiFormerAttention(nn.Module):

    def __init__(
        self,
        embed_dim,
        out_dim,
        num_heads,
        kdim=None, # dimensions are consistent (Default)
        vdim=None, # dimensions are consistent (Default)
        dropout_rate=0.0,
        has_outproj=True, # project for output （optional）
        
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.kdim = kdim if kdim is not None else embed_dim
        self.vdim = vdim if kdim is not None else embed_dim

        self.num_heads = num_heads
        self.has_outproj = has_outproj
        self.act_fun = self.get_act_fun()
        # q, k, v projection
        self.k_proj = nn.Linear(self.kdim, embed_dim)
        self.v_proj = nn.Linear(self.vdim, embed_dim)
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        # out projection
        self.out_proj = nn.Linear(embed_dim, out_dim)
        # dropout rate
        self.dropout_rate = dropout_rate


        assert (self.embed_dim % self.num_heads == 0), "embed_dim must be divisible by num_heads"

    def get_index(self, motif_adjs):
        """
        motif_adjs: the adjacency matrix of the motif graph
        cauculate the sum of all motif neighbors for each node
        """
        adjs_sum = torch.einsum("mnk->nk",motif_adjs)
        # cauculate the sum of all motif neighbors for each node
        adjs_col_sum=torch.sum(adjs_sum,dim=1)  
        index = (adjs_col_sum.reshape(1, -1, 1)-torch.min(adjs_col_sum))/(torch.max(adjs_col_sum)-torch.min(adjs_col_sum))
        return nn.Parameter(index, requires_grad=False)

    def get_act_fun(self):
            return F.selu

    def forward(
        self,
        query,
        motif_adjs,
        key = None,
        value = None,
        eps = 1e-6,
    ):
        # default key and value are the same as query
        if key == None:
            key = query
        if value == None:
            value = query
        
        num_heads = self.num_heads
        tgt_len, bsz, embed_dim = query.size()
        src_len = key.size(0)
        head_dim = embed_dim // num_heads

        q = self.q_proj(query)
        k = self.k_proj(key)
        v = self.v_proj(value)

        q = self.act_fun(q)+2
        k = self.act_fun(k)+2


        q = q.contiguous().view(-1, bsz * num_heads, head_dim).transpose(0, 1)
        k = k.contiguous().view(-1, bsz * num_heads, head_dim).transpose(0, 1)
        v = v.contiguous().view(-1, bsz * num_heads, head_dim).transpose(0, 1)

        weight_index = self.get_index(motif_adjs).to(q)

        q_=q
        k_=k * torch.sqrt(1-torch.pow((1-weight_index[:, :src_len, ]),2))  
        kv_ = torch.einsum('nld,nlm->ndm', k_, v)
        z_ = 1 / torch.clamp_min(torch.einsum('nld,nd->nl', q_, torch.sum(k_, axis=1)), eps)
        attn_output = torch.einsum('nld,ndm,nl->nlm', q_, kv_, z_)
        attn_output = attn_output.transpose(0, 1).contiguous().view(tgt_len, bsz, -1)
        if self.has_outproj:
            attn_output = self.out_proj(attn_output)
        return attn_output

class PositionEncoder(torch.nn.Module):
    def __init__(self, madjs,k):

        super(PositionEncoder, self).__init__()
        self.adj_matrices = madjs
        self.adj_len = len(madjs)
        self.k=k


    def get_LPE(self, k):
        result = []
        adjs = self.adj_matrices.numpy()
        # print(adjs[0, :].shape)
        for i in range(self.adj_len):
            # print(i)
            adj_matrix = adjs[i, :, :]
            adj_matrix[adj_matrix > 0] = 1

            R = np.sum(adj_matrix, axis=1)  #
            R_sqrt = 1 / (np.sqrt(R) + (1e-6))
            D_sqrt = np.diag(R_sqrt)
            I = np.eye(adjs[i].shape[0])
            laplacian_matrix = I - np.matmul(np.matmul(D_sqrt, adj_matrix), D_sqrt)

            eigVals, eigVecs = np.linalg.eigh(laplacian_matrix)
            eigVals = eigVals[0:k]
            eigVecs = eigVecs[0:k, :]

            node_PE_list = []
            for j in range(eigVecs.shape[1]):
                node_vecs = eigVecs[:, j:j + 1].squeeze(1)
                node_PE = node_vecs
                node_PE_list.append(node_PE)
            result.append(node_PE_list)
        return np.array(result)


    def get_PE(self, devices):
        result = self.get_LPE(self.k)
        PE = torch.from_numpy(result).float().to(devices)
        PE = PE.permute(1, 0, 2)
        PE = PE.reshape(self.adj_matrices.shape[1],
                        self.adj_matrices.shape[0] * self.k).to(devices)
        return PE, self.adj_matrices.to(devices)


# run demo
if __name__=='__main__':
    num_nodes = 40
    num_motifs = 3
    k=16

    adj_mat=torch.randint(0,10,(num_motifs, num_nodes, num_nodes))
    adj_mat=adj_mat+torch.transpose(adj_mat, 1, 2,)
    GAT=GATLayer(num_motifs*k, num_motifs*k, 0, motif_len=num_motifs).train()
    PE=PositionEncoder(adj_mat,k)
    mlp=MLPForRec(num_motifs*k, 2*num_motifs*k, num_motifs*k)
    motiformer=MotiFormerAttention(embed_dim=num_motifs*k,out_dim=num_motifs*k,num_heads=4)
    # local PE
    node_f,motif_adj=PE.get_PE("cpu")
    # final PE
    node_f=GAT.forward(adj_mat[0],node_f)
    # MotiFormer
    node_f=node_f.unsqueeze(0).permute(1,0,2)
    node_f=motiformer.forward(node_f,motif_adj)
    node_f=node_f.permute(1,0,2).squeeze()
    # MLP
    m=node_f[:20,:]
    a=node_f[20:,:]
    rec_result=mlp.forward(m,a)
    print(rec_result.shape)
    