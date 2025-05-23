import torch
import torch.nn as nn
import torch.nn.functional as F

import math
import pdb
import numpy as np
import typing as ty
from sklearn.preprocessing import OneHotEncoder
# from transformers.utils import torch_only_method

'''
# Description:
- Using normalized std for pivots, with option to scale it by number of pivots.
- We allow the weights to become -ve, so no piece wise convexity.
- This is for getting important kernel points /pivot points from dataseet, so selecting constant std for all and multiplying by weights and
    training them to get important points by weights.
- Making this for linear + linear score
- All vectors have +ve and -ve kernel points. So along every direction we get scores from +ve kernel points and -ve kernel points.

- Does score computed by +ve kernel should show +ve score ?? 
    This makes sence because +ve kernels gives distance of test point from +ve points.
    Whereas -ve kernel gives the distance of test point from -ve points.

# Notation:
n: number of data points, like batch size
J: n_vectors i.e. total number of vectors 
K/P: n_pivot i.e. total number of pivots mean/centers along every vector
d: dimension of data 

'''

NORMALIZATION_PROJ = ty.Literal['max', 'max-detached', 'max/n_kernel']
KERNEL_TYPE = ty.Literal['lin', 'rbf']
SCORE_TYPE = ty.Literal['lin', 'rbf']
    
class TV7(nn.Module):
    def __init__(
        self, 
        categories: ty.Optional[ty.List[int]],
        d_embedding: int,
        d_in: int,
        n_vectors: int = 5,
        n_kernel: int = 10,  # Unused, just taking it as input
        use_kernel: KERNEL_TYPE = 'rbf',
        use_score: SCORE_TYPE = 'lin',
        reg_type: ty.Literal['lin'] = 'lin',
        reg_wt: float = 0.1,
        train_std = True,
        gamma: float = 1.0,
        d: float = 2,
        s: float = 1,
        c: float = 0,
        train_gamma: bool = False,
        use_sparse: bool = False,
        alpha: float = 0.5,
        kernel_std: float = 1,
        use_kernel_weight: bool =True,
        use_svm_score: bool = False,
        normalize_proj_diff: NORMALIZATION_PROJ = "max-detached",
        detached_vec_norm: bool = True,
        detached_pivots: bool = False,
        combine_type: ty.Optional[ty.Literal['embd', 'default', 'ohe']] = 'embd',
    ) -> None:
        '''
        # Parameters
        '''
        super().__init__()

        # Encoding categorical values, as done for all other models
        self.categories = categories
        self.n_categories = 0 if categories is None else len(categories)
        if categories is not None:
            if combine_type == 'default':
                d_in += len(categories)

            if combine_type == 'ohe':
                extra_dim = sum(categories)  # Just using one hot encoding
                if extra_dim > 200:
                    print('Very high number of categorical features, switching from OHE to embedding vector')
                    combine_type = 'embd'
                else:
                    d_in += extra_dim

            if combine_type == 'embd':
                d_in += len(categories) * d_embedding
                category_offsets = torch.tensor([0] + categories[:-1]).cumsum(0)
                self.register_buffer('category_offsets', category_offsets)
                self.category_embeddings = nn.Embedding(sum(categories), d_embedding)
                nn.init.kaiming_uniform_(self.category_embeddings.weight, a=math.sqrt(5))
                # print(f'{self.category_embeddings.weight.shape}')

        self.n_vectors = n_vectors
        self.data_dim = d_in
        self.sqrt_data_dim = math.sqrt(d_in)
        self.kernel_std = kernel_std
        self.use_weight = use_kernel_weight
        # self.same_kernel = True if len(kernel_points[0]) == 2 else False
        self.use_svm_score = use_svm_score
        self.train_gamma = train_gamma
        self.train_pivot_std = train_std
        self.normalize_proj_diff = normalize_proj_diff
        self.detached_vec_norm = detached_vec_norm
        self.detached_pivots = detached_pivots
        self.reg_type = reg_type
        self.reg_wt = reg_wt
        self.d = d
        self.s = s
        self.c = c
        self.use_kernel = use_kernel
        self.use_score = use_score
        self.use_sparse = use_sparse
        self.alpha = alpha
        self.combine_type = combine_type
        # self.max_proj_diff = (None, None)

        self.vec_batch_size = self.n_vectors
        self.optimize_infer = False # If turned on you can't further train the model
        self.use_regularizer = True # use regularizer by default

        if gamma == 'scale':
            raise Exception(f"Don't use for {gamma = } TV")
            self.gamma = torch.tensor(1/(data_dim * kernel_points.var()))
        elif gamma == 'auto':
            self.gamma = torch.tensor(1/d_in)
        else:
            self.gamma = torch.tensor(gamma)
        if train_gamma:
            self.gamma = nn.Parameter(torch.tensor(self.gamma))
        # print(f'{self.gamma=}')

        vectors = torch.randn(n_vectors, d_in) # [J,d]
        vectors_norm = vectors.norm(dim=1)
        if (vectors_norm == 0).any():
            print(f"{'ERROR vector initialized to all zeros':}")
        vectors = vectors/vectors_norm.unsqueeze(dim = 1)
        self.vectors = nn.Parameter(vectors)
        return

    @classmethod
    def get_kernel_points(
        cls,
        X,
        y,
        n_kernel,
        J,
        np_r = None,
        same_kernel_points = True
    ):
        kernel_points = []
        all_p_points = []
        all_n_points = []

        p_idx = y == 1
        n_idx = y == 0
        all_p_points = X[p_idx]
        all_n_points = X[n_idx]

        n_pos_kernel = min(n_kernel//2, all_p_points.shape[0])
        n_neg_kernel = min(n_kernel - n_kernel//2, all_n_points.shape[0])
        if all_p_points.shape[0] < n_kernel//2 or all_n_points.shape[0] < n_kernel//2:
            print(f'Reducing kernel points, as very less +ve and or -ve points found')
            print(f'Update Kernel points: {n_pos_kernel} + {n_neg_kernel}')
        if same_kernel_points:
            if np_r is None:
                p_idx = np.random.choice(all_p_points.shape[0],n_pos_kernel, replace=False)
                n_idx = np.random.choice(all_n_points.shape[0],n_neg_kernel, replace=False)
            else:
                p_idx = np_r.choice(all_p_points.shape[0], n_pos_kernel, replace=False)
                n_idx = np_r.choice(all_n_points.shape[0], n_neg_kernel, replace=False)

            # print(f'Using same kernel, printing the dimension')
            p_kernel_points, n_kernel_points = torch.tensor(all_p_points[p_idx]), torch.tensor(all_n_points[n_idx])
            # print(f'{p_kernel_points.shape=}, {n_kernel_points.shape=}')
            return (p_kernel_points,n_kernel_points)

        ## Code for different kernel along different directions
        if np_r is None:
            p_idx = np.zeros(shape= (J, n_kernel//2), dtype=int)
            n_idx = np.zeros(shape= (J, n_kernel//2), dtype=int)
            for j in range(J):
                p_idx[j] = np.random.choice(all_p_points.shape[0], n_kernel//2, replace=False) # choose +ve points for first half
                n_idx[j] = np.random.choice(all_n_points.shape[0], n_kernel//2, replace=False)  # choose -ve points for second half 
        else:
            p_idx = np.zeros(shape= (J, n_kernel//2), dtype=int)
            n_idx = np.zeros(shape= (J, n_kernel//2), dtype=int)
            for j in range(J):
                p_idx[j] = np_r.choice(all_p_points.shape[0], n_kernel//2, replace=False) # choose +ve points for first half
                n_idx[j] = np_r.choice(all_n_points.shape[0], n_kernel//2, replace=False)  # choose -ve points for second half 

        if type(all_p_points) == torch.Tensor:
            p_kernel_points, n_kernel_points = all_p_points[[p_idx]], all_n_points[[n_idx]]
        else:
            p_kernel_points, n_kernel_points = torch.tensor(all_p_points[p_idx]), torch.tensor(all_n_points[n_idx])

        return p_kernel_points, n_kernel_points

    def combine_cat_num(self, x_cat, x_num, phase: ty.Optional[ty.Literal['init']]= 'init'):
        X = []
        combine_type = self.combine_type
        if combine_type == 'embd':
            if x_num is not None:
                X.append(x_num)
            if x_cat is not None:
                X.append(
                    self.category_embeddings(x_cat + self.category_offsets[None]).view(
                        x_cat.size(0), -1
                    )
                )
            return torch.cat(X, dim = -1)
        elif combine_type == 'default':
            if x_cat is not None:
                X.append(x_cat.float())
            if x_num is not None:
                X.append(x_num)
            return torch.cat(X, dim = -1)
        elif combine_type == 'ohe':
            if x_cat is None:
                return torch.cat([x_num], dim =-1)
            x_cat_new = []
            if phase == 'init':
                # self.encoder = OneHotEncoder(
                #     handle_unknown='ignore', sparse=False, dtype='float32'
                # )
                for col_id in range(x_cat.shape[1]):
                    x_cat_new.append(F.one_hot(x_cat[:,col_id], num_classes=self.categories[col_id]))
                # x_cat = F.one_hot(x_cat)
                # x_cat = self.encoder.fit_transform(x_cat)
            else:
                # x_cat = F.one_hot(x_cat)
                for col_id in range(x_cat.shape[1]):
                    x_cat_new.append(F.one_hot(x_cat[:,col_id], num_classes=self.categories[col_id]))
                # x_cat = self.encoder.transform(x_cat.cpu())
            x_cat_new = torch.cat(x_cat_new, dim = -1)
            if x_num is not None:
                if torch.is_tensor(x_num):
                    X.append(x_num)
                else:
                    X.append(torch.tensor(x_num))
            X.append(x_cat_new.float())
            return torch.cat(X, dim = -1)

    def init_phase2(self, dataset, n_kernel):
        
        '''
        Setting
        - pivot points
        - pivot std
        - weight for (vector, pivot)
        '''

        x_cat = None if dataset.X_cat is None else torch.tensor(dataset.X_cat['train'])
        x_num = None if dataset.X_num is None else torch.tensor(dataset.X_num['train'])
        X = self.combine_cat_num(x_cat=x_cat, x_num=x_num, phase='init')

        kernel_points = self.get_kernel_points(
            X = X.detach().numpy(),
            y = dataset.y['train'],
            n_kernel = n_kernel,
            J = self.n_vectors,
            np_r = None,
            same_kernel_points = True
        )

        # Setting rest properties that depend on kernel_points size
        self.n_pos_pivot = kernel_points[0].shape[0]
        self.n_neg_pivot = kernel_points[1].shape[0]
        self.n_pivot = self.n_pos_pivot + self.n_neg_pivot

        # setting pivot std
        if self.train_pivot_std:
            self.pivot_std = nn.Parameter(torch.randn(self.n_vectors, self.n_pivot,1)) # [J,K,1]
        else:
            # pivot_std_val = kernel_std # * torch.ones(n_vectors,self.n_pivot,1) # [J,K,1]
            self.register_buffer('pivot_std', torch.tensor(self.kernel_std)) 

        # weights for (pivot, vector) pair
        if self.use_weight:
            self.score_weight = nn.Parameter(torch.ones(self.n_vectors, self.n_pivot, 1)) # [J,K,1]
        else:
            score_weight = torch.ones(self.n_vectors, self.n_pivot,1)
            self.register_buffer('score_weight', score_weight)

        # setting pivot points
        self.register_buffer('p_pivot_pts', kernel_points[0]) # [J,K//2,d] or [Kp,d]
        self.register_buffer('n_pivot_pts', kernel_points[1]) # [J,K//2,d] or [Kn,d]

        # Uncomment this for global normalization of projected difference
        proj_points = self.kernel(X.detach())
        proj_p_pivot = self.kernel(self.p_pivot_pts)
        proj_n_pivot = self.kernel(self.n_pivot_pts)

        res_p = proj_points.unsqueeze(dim = 1) - proj_p_pivot.unsqueeze(dim = 2)  # [J,1,n] - [J,K//2,1] => [J,K//2,n]
        res_n = proj_points.unsqueeze(dim = 1) - proj_n_pivot.unsqueeze(dim = 2)  # [J,1,n] - [J,K//2,1] => [J,K//2,n]
        temp_n = torch.max(torch.abs(res_n), dim=2).values.detach()
        temp_p = torch.max(torch.abs(res_p), dim=2).values.detach()
        self.register_buffer('max_proj_diff_p', temp_p)
        self.register_buffer('max_proj_diff_n', temp_n)
        # self.max_proj_diff = (self.max_proj_diff_p, self.max_proj_diff_n)
        # self.max_proj_diff = (temp_p, temp_n)

    def get_size(self):
        '''
        returns  the model's size in Bytes and number of variables (parameters + buffers)
        '''
        param_size = 0
        param_count = 0
        for param in self.parameters():
            param_size += param.nelement() * param.element_size()
            param_count += param.nelement()
        
        buffer_size = 0
        buffer_count =0
        for buffer in self.buffers():
            buffer_size += buffer.nelement() * buffer.element_size()
            buffer_count += buffer.nelement()

        return param_size + buffer_size, param_count + buffer_count

    def optimize_inference(self):
        '''
        Optimize inference, by precomputing projections
        This take more space as new self. objects are needed.
        It saves the projection of pivots (instead of pivots in data's dimension)
        ---
        We can create a separate TV instance whith only required params for testing to lessen the space
        '''
        self.proj_p_pivot = self.kernel(self.p_pivot_pts)
        self.proj_n_pivot = self.kernel(self.n_pivot_pts)

        # del self.p_pivot_pts
        # del self.n_pivot_pts

        self.optimize_infer = True
        
    def kernel(self, points):
        '''
        ## Parameters-
        poitns: datapoints or kernel_points
            shape: [n, d] or [P, d] (for same pivot points) or [J, K//2, d] for diff pivot points
        ## Output-
        res: kernel applied to every point/kernel and vectors
            shape: [J,n] or [J, P]
        '''

        vectors = self.vectors # [J,d]
        if self.detached_vec_norm:
            vectors = vectors / vectors.norm(dim = 1, keepdim=True).detach()

        if self.use_kernel == 'lin': # k(v_j, x_i) = <v_j,x_i> (linear projection)
            if len(points.shape) == 3: # for 3D kernel_points (i.e, different kernel along diff vector)
                res = torch.matmul(points, vectors.unsqueeze(dim = -1)).squeeze(dim = -1)  # [J,K,d] @ [J,d,1] => [J,K]
            else:
                # res = torch.matmul(vectors, points.mT)  # [J,d] @ [n,d].T => [J,n] or [J,d] @ [P,d].T => [J,P]
                res = torch.matmul(vectors, points.transpose(-2,-1))  # [J,d] @ [n,d].T => [J,n] or [J,d] @ [P,d].T => [J,P]
            res = res /( vectors.norm(dim=1, keepdim=True))  # Because of normalized vectors, it's equivalent to divide by 1
        elif self.use_kernel == 'rbf': # k(v_j, x_i) = exp(- \gamma ||v_j - x_i||^2)
            if len(points.shape) == 3:  # for 3D kernel_points (i.e, different kernel along diff vector)
                res = ((vectors/vectors.norm(dim=1,keepdim=True)).unsqueeze(dim=1) - points).norm(dim = 2)  # [J, 1, d] - [J, K, d] -> norm => [J,K]
            else:
                res = ((vectors/vectors.norm(dim=1, keepdim=True)).unsqueeze(dim=1) - points.unsqueeze(dim = 0)).norm(dim = 2)  # [J,1,d] - [1,n,d] => [J,n] or [J,P]
            res = torch.exp( -torch.abs(self.gamma)* res * res)

            # TODO: numerical stable kernel implementation
            # self.take_kernel_later = True
            # res =  -torch.abs(self.gamma)* res * res
            
        elif self.use_kernel == 'poly':
            if len(points.shape) == 3:  # for 3D kernel_points
                res = torch.matmul(points, vectors.unsqueeze(dim = -1)).squeeze()/vectors.norm(dim=1, keepdim=True)  # [J,K,d] @ [J,d,1] => [J,K]
            else:
                # res = (vectors @ points.mT)/vectors.norm(dim=1, keepdim=True)  # [J,d] @ [n,d] => [J,n]
                res = (vectors @ points.transpose(-2,-1))/vectors.norm(dim=1, keepdim=True)  # [J,d] @ [n,d] => [J,n]
            res = (self.s * res/math.sqrt(self.data_dim) + self.c).pow(self.d)  # [J,n] => [J,n]
        else:
            error_msg = f'{"ERROR":!^50},\n Unknown {self.use_kernel=}'
            raise Exception(error_msg)

        return res
     
    def get_score(self, proj_point, proj_pivot, pivot_std, pos_pivot: bool, score_weight = None):
        '''
        # Parameters
        proj_point  : projected(or kernelized) datapoints
            shape: [J,n]
        proj_pivot : projected (or kernelized) pivots
            shape: [J,P] or [J,K//2] (what about different pivot points?)
        pivot_std : std of pivots (either scaler or vector)
            shape: scaler or [J,P] or [J,K//2]

        # Output
        res      : Score of every point 
            shape: [n]
        '''
        # print('-'*20)
        # print(f'{proj_point.shape=}\n {proj_pivot.shape=}\n {pivot_std.shape=}')

        # Setting normalization factor
        # self.max_proj_diff = (self.max_proj_diff_p, self.max_proj_diff_n)
        # max_proj_diff_idx = 0 if pos_pivot else 1
        norm_factor = None
        if self.normalize_proj_diff:
            norm_factor = self.max_proj_diff_p if pos_pivot else self.max_proj_diff_n

        # First calculate the difference between projection of pivots and projection of data points 
        if self.detached_pivots:
            res = proj_point.unsqueeze(dim = 1) - proj_pivot.unsqueeze(dim = 2).detach()  # [J,1,n] - [J,K//2,1] => [J,K//2,n]
        else:
            res = proj_point.unsqueeze(dim = 1) - proj_pivot.unsqueeze(dim = 2)  # [J,1,n] - [J,K//2,1] => [J,K//2,n]

        # if self.take_kernel_later:
        #     # self.max_proj_diff is just -gamma || v_i - x_max||^2. That is same for +ve and neg
        #     a = torch.exp(- proj_point + norm_factor) # [J, n] - [J, K//2, 1]
        #     b = torch.exp(- proj_point.unsqueeze(dim = 1) + proj_pivot.unsqueeze(dim = 2))
        #     c = torch.exp(-proj_pivot - norm_factor) - 1
        #     res = 1/(a - b) - 1/c
        if self.normalize_proj_diff == 'max':
            if norm_factor is None:
                norm_factor = torch.max(torch.abs(res), dim=2).values

            # if self.max_proj_diff[max_proj_diff_idx] is None:
            #     norm_factor = torch.max(torch.abs(res), dim=2).values
            # else:
            #     norm_factor = self.max_proj_diff[max_proj_diff_idx]
            res = res/norm_factor.unsqueeze(dim = 2)  # [J,K,n] / [J,K,1] => [J,K,n]
        elif self.normalize_proj_diff == 'max-detached':
            if norm_factor is None:
                norm_factor = torch.max(torch.abs(res), dim=2).values.detach()
            # if self.max_proj_diff[max_proj_diff_idx] is None:
            #     norm_factor = torch.max(torch.abs(res), dim=2).values.detach()
            # else:
            #     norm_factor = self.max_proj_diff[max_proj_diff_idx]
            res = res/norm_factor.unsqueeze(dim = 2)  # [J,K,n] / [J,K,1] => [J,K,n]
        elif self.normalize_proj_diff == 'max/n_kernel':
            if norm_factor is None:
                norm_factor = torch.max(torch.abs(res), dim=2).values
            # if self.max_proj_diff[max_proj_diff_idx] is None:
            #     norm_factor = torch.max(torch.abs(res), dim=2).values
            # else:
            #     norm_factor = self.max_proj_diff[max_proj_diff_idx]
            norm_factor = norm_factor/self.n_pivot
            res = res/norm_factor.unsqueeze(dim = 2)  # [J,K,n] / [J,K,1] => [J,K,n]

        if self.use_score == 'inv':
            res = pivot_std* pivot_std/(res * res + pivot_std*pivot_std)
        elif self.use_score == 'rbf':
            # res = torch.exp(- res * res /(2 * pivot_std * pivot_std*self.sqrt_data_dim * self.sqrt_data_dim))
            res = torch.exp(- res * res /(2 * pivot_std * pivot_std))
        elif self.use_score == 'lin':
            res = F.relu(1 - torch.abs(res/pivot_std))  # max{ 1- | (x - x_0)/m |, 0}
        else:
            error_msg = f'{"ERROR":!^50},\n Unknown {self.use_score=}'
            raise Exception(error_msg)

        # print(f'{res.shape=}')
        if self.use_weight:
            if self.use_svm_score:
                # res =  torch.abs(score_weight)*res # [J,K,n] * [J,K,1] => [J,K,n]
                res =  score_weight*res #  [J,K,1] * [J,K,n] => [J,K,n]
            else:
                res =  1/(1+torch.abs(score_weight))*res # [J,K,1] * [J,K,n] => [J,K,n]

        res = res.mean(dim=1)  # [J,K,n] => [J,n]
        res = res.mean(dim=0)  # [J,n] => [n]
        # p_res = res[:self.n_vectors//2].mean(dim = 0) # [J/2, n] => [n]
        # n_res = res[self.n_vectors//2:].mean(dim = 0) # [J/2, n] => [n]
        # return p_res, n_res
        return res
    
    def regularizer(self, use_low_gpu = True):
        '''
        Taking pair wise dot product of all vectors and using it as regularizer
        it's only used if data_dim < n_vectors and data_dim < 1000
        '''
        if self.n_vectors <= 1:
            return 0
        if not self.use_regularizer: # not using regularizer at all
            return 0
        # only use if data_dim is smaller than number of vectors and data_dim < 1000
        if self.data_dim > 1000 or self.data_dim > self.n_vectors:
            return 0
        if use_low_gpu:
            dot_prod_mat = 0
            for j in range(self.vectors.shape[0]):
                dot_prod_mat += (F.cosine_similarity(
                            self.vectors[j].unsqueeze(dim = 0),
                            self.vectors, dim=-1)**2).sum()
        else:
            try:  # Faster method for smaller number of vectors
                dot_prod_mat = F.cosine_similarity(
                    self.vectors.unsqueeze(dim = 0),
                    self.vectors.unsqueeze(dim =1),
                    dim= -1)
                dot_prod_mat = dot_prod_mat * dot_prod_mat
                dot_prod_mat = dot_prod_mat.sum()
                ## [1,J,d], [J,1,d] => [J,J], gives normalized dot product
            except RuntimeError: # again fall back to lower gpu implementation
                dot_prod_mat = 0
                for j in range(self.vectors.shape[0]):
                    dot_prod_mat += (F.cosine_similarity(
                                self.vectors[j].unsqueeze(dim = 0),
                                self.vectors, dim=-1)**2).sum()

        res = (dot_prod_mat - self.n_vectors)/(self.n_vectors*(self.n_vectors-1))
        return res
    
    def score_regularizer(self, pred, label, method=None):
        '''
        Not in use
        This is for maximizing the score value i.e. 
        1. For score in 0,1 we want it to be away from 0.5.
            - We want most points to have this property `(score - 0.5)**2` to be large
            - Note that this should be along all vectors.
            - !!! Just using `(score - 0.5) wont work, you need to consider label for direction as well
        2. for score in -1,1 we want the scores to be away from 0
            - We want most points to have `score**2` to be large
        '''
        reg = - (pred - 0.5)*2 * (label - 0.5)*2
        return reg.mean()
        # return torch.square(pred - 0.5).sum()
    
    def weight_regularizer(self):
        """
        To give bigger weight values to very few kernel points
        """
        
        reg = self.score_weight
        raise Exception('Function not implemented')


    def forward(self, x_num, x_cat):
        '''
        Parameters
        x_num: [n,d_num] numerical features
        x_cat: [n,d_cat] categorical features
        
        Output
        score: [n] score of each point
        '''
        X = self.combine_cat_num(x_cat = x_cat, x_num = x_num, phase=None)
        X = X.view(X.shape[0], -1)
        proj_point = self.kernel(X)
        # print(f'{self.p_kernel_points.shape=}')
        if self.optimize_infer:
            proj_p_pivot = self.proj_p_pivot
            proj_n_pivot = self.proj_n_pivot
        else:
            proj_p_pivot = self.kernel(self.p_pivot_pts)
            proj_n_pivot = self.kernel(self.n_pivot_pts)
        if self.train_pivot_std:
            p_score = self.get_score(
                proj_point,
                proj_p_pivot,
                self.pivot_std[:, :self.n_pos_pivot, :],
                pos_pivot=True,
                score_weight=self.score_weight[:,:self.n_pos_pivot,:]
            )
            n_score = self.get_score(
                proj_point,
                proj_n_pivot,
                self.pivot_std[:, self.n_pos_pivot:, :],
                pos_pivot=False,
                score_weight=self.score_weight[:,self.n_pos_pivot:,:]
            )
        else:
            p_score, n_score = 0, 0
            if self.vec_batch_size == -1 or self.vec_batch_size is None:
                self.vec_batch_size = self.n_vectors
            self.n_vec_batches = self.n_vectors//self.vec_batch_size
            batch_start=0
            batch_end = self.vec_batch_size
            for batch_id in range(self.n_vec_batches):
                p_score += self.get_score(
                    proj_point[batch_start:batch_end],
                    proj_p_pivot[batch_start: batch_end],
                    self.pivot_std,
                    pos_pivot=True,
                    score_weight=self.score_weight[batch_start:batch_end,:self.n_pos_pivot,:],
                )
                n_score += self.get_score(
                    proj_point[batch_start: batch_end],
                    proj_n_pivot[batch_start: batch_end],
                    self.pivot_std,
                    pos_pivot=False,
                    score_weight=self.score_weight[batch_start:batch_end,self.n_pos_pivot:,:],
                )
                batch_start += self.vec_batch_size
                batch_end += self.vec_batch_size
            p_score, n_score = p_score/self.n_vec_batches, n_score/self.n_vec_batches  # taking avg actoss batch
        # score = (1 - self.alpha) * p_score - self.alpha * n_score + self.alpha # Adding + self.alpha to make score in 0,1 (instead of - self.alpha  , 1 - self.alpha)
        if self.use_svm_score:
            score = p_score - n_score
        else:
            score = (1 + p_score - n_score)/2
        return score

