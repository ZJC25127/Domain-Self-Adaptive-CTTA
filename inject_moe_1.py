import logging
import math
from typing import List

import torch
import torch.nn.functional as F
from adapter import Adapter
from torch import nn



class SparseDispatcher(object):
    """Helper for implementing a mixture of experts.
    The purpose of this class is to create input minibatches for the
    experts and to combine the results of the experts to form a unified
    output tensor.
    There are two functions:
    dispatch - take an input Tensor and create input Tensors for each expert.
    combine - take output Tensors from each expert and form a combined output
      Tensor.  Outputs from different experts for the same batch element are
      summed together, weighted by the provided "gates".
    The class is initialized with a "gates" Tensor, which specifies which
    batch elements go to which experts, and the weights to use when combining
    the outputs.  Batch element b is sent to expert e iff gates[b, e] != 0.
    The inputs and outputs are all two-dimensional [batch, depth].
    Caller is responsible for collapsing additional dimensions prior to
    calling this class and reshaping the output to the original shape.
    See common_layers.reshape_like().
    Example use:
    gates: a float32 `Tensor` with shape `[batch_size, num_experts]`
    inputs: a float32 `Tensor` with shape `[batch_size, input_size]`
    experts: a list of length `num_experts` containing sub-networks.
    dispatcher = SparseDispatcher(num_experts, gates)
    expert_inputs = dispatcher.dispatch(inputs)
    expert_outputs = [experts[i](expert_inputs[i]) for i in range(num_experts)]
    outputs = dispatcher.combine(expert_outputs)
    The preceding code sets the output for a particular example b to:
    output[b] = Sum_i(gates[b, i] * experts[i](inputs[b]))
    This class takes advantage of sparsity in the gate matrix by including in the
    `Tensor`s for expert i only the batch elements for which `gates[b, i] > 0`.
    """

    def __init__(self, num_experts, gates):
        """Create a SparseDispatcher."""

        self._gates = gates       #（B*token, num_experts）
        self._num_experts = num_experts  

        sorted_experts, index_sorted_experts = torch.nonzero(gates).sort(0) 
        #torch.nonzero(gates) (B*token*top_k, 2)    每个索引为 (batch_token_idx, expert_idx)，表示第batch_token_idx个样本分配给第expert_idx个专家
        #(B*token*top_k, 2)   sorted_experts,index_sorted_experts  每列排序(0-B*token,0-num_experts),每列原来的索引
        
        #保留专家索引  范围0-4
        _, self._expert_index = sorted_experts.split(1, dim=1)  #(B*token*top_k,1)
        
        #取出按专家排序后的batch索引  专家排序0-4时对应的batch索引范围0-B*token  比如 0 1 0 1  前两个表示选择了专家1，后两个表示选择了专家2
        self._batch_index = torch.nonzero(gates)[index_sorted_experts[:, 1], 0] #(B*token*top_k,)
        #得到每个专家的样本数量
        self._part_sizes = (gates > 0).sum(0).tolist()  #list[num_experts]  每个专家的样本数量
        # 权重按照专家索引排序
        gates_exp = gates[self._batch_index.flatten()]  # (B*token*top_k, num_experts) 会有重复
        # 取出每个专家的权重
        self._nonzero_gates = torch.gather(gates_exp, 1, self._expert_index)  # (B*token*top_k, 1)  每个专家的权重
        #把所有专家排序，选择专家0的bach在前面，专家4的在后面

    def dispatch(self, inp):
        """Create one input Tensor for each expert.
        The `Tensor` for a expert `i` contains the slices of `inp` corresponding
        to the batch elements `b` where `gates[b, i] > 0`.
        Args:
          inp: a `Tensor` of shape "[batch_size, <extra_input_dims>]`
        Returns:
          a list of `num_experts` `Tensor`s with shapes
            `[expert_batch_size_i, <extra_input_dims>]`.
        """

        # assigns samples to experts whose gate is nonzero

        inp_exp = inp[self._batch_index].squeeze(1)  #(B*token*top_k,1,dim)
        return torch.split(inp_exp, self._part_sizes, dim=0)

    def combine(self, expert_out, multiply_by_gates=True):
        """Sum together the expert output, weighted by the gates.
        The slice corresponding to a particular batch element `b` is computed
        as the sum over all experts `i` of the expert output, weighted by the
        corresponding gate values.  If `multiply_by_gates` is set to False, the
        gate values are ignored.
        Args:
          expert_out: a list of `num_experts` `Tensor`s, each with shape
            `[expert_batch_size_i, <extra_output_dims>]`.
          multiply_by_gates: a boolean
        Returns:
          a `Tensor` with shape `[batch_size, <extra_output_dims>]`.
        """
        # apply exp to expert outputs, so we are not longer in log space

        stitched = torch.cat(expert_out, 0)  # (B*token*top_k, dim)  
        if multiply_by_gates:
            stitched = stitched.mul(self._nonzero_gates)  # 加权

        zeros = torch.zeros(self._gates.size(0), expert_out[-1].size(1), device=stitched.device)
        #(B*token, dim)  
       
        # 0 1 0 1  zeros[self._batch_index[i]] += stitched[i].float()  self._batch_index (B*token*top_k,)
        combined = zeros.index_add(0, self._batch_index, stitched.float())
        # add eps to all zero values in order to avoid nans when going back to log space
        # back to log space
        return combined

    def expert_to_gates(self):
        """Gate values corresponding to the examples in the per-expert `Tensor`s.
        Returns:
          a list of `num_experts` one-dimensional `Tensor`s with type `tf.float32`
              and shapes `[expert_batch_size_i]`
        """
        # split nonzero gates for each expert
        return torch.split(self._nonzero_gates, self._part_sizes, dim=0)


class Mlp(nn.Module):
    """ MLP as used in Vision Transformer, MLP-Mixer and related networks
    """
    def __init__(self, in_features, hidden_features, out_features, bias,act_layer=nn.GELU, drop=0.):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features,bias)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features,bias)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class ParameterSlot(nn.Module):
    """A single parameter wrapped in a module for DataParallel replication."""

    def __init__(self, shape):
        super().__init__()
        self.param = nn.Parameter(torch.zeros(*shape), requires_grad=True)


class ParameterBank(nn.ModuleList):
    """Indexed parameters that replicate correctly under DataParallel."""

    def __init__(self, count, shape):
        super().__init__([ParameterSlot(shape) for _ in range(count)])

    def __getitem__(self, index):
        return super().__getitem__(index).param


class MOEInjectedLinear(nn.Module):
    def __init__(self, in_features,hidden_features, out_features, bias, rank=4, router_num = 4, experts_num=1,top_k=1,adapter_scalar=0.1, shared_ratio=0.9, adapter_dropout=0.1, noisy_gating=True, domain_slots=14):
        super().__init__()

        self.register_buffer("mean", torch.tensor([0.0]))
        self.register_buffer("std", torch.tensor([1.0]))

        self.mlp_moe = Mlp(in_features, hidden_features, out_features,bias)

        
        self.step = router_num
        self.top_k = top_k
        self.ffn_num = rank
        self.experts_num = experts_num
        self.softmax = nn.Softmax(1)
        self.softplus = nn.Softplus()
        self.noisy_gating = noisy_gating
        self.global_taskid = 1
        self.train_flag = 0
        self.shared_ratio = shared_ratio
        self.domain_slots = domain_slots
        # self.loss = None


        
        self.moe_mlp_list = nn.ModuleList()
        self.moe_router_list = ParameterBank(
            self.step * self.domain_slots, (in_features, self.experts_num)
        )
        self.moe_noise_list = ParameterBank(
            self.step * self.domain_slots, (in_features, self.experts_num)
        )
        for i in range(self.experts_num*self.domain_slots):  #专家lora
        # for i in range(self.experts_num):  #专家lora
            moe_adaptmlp = Adapter(d_model=in_features, dropout=adapter_dropout, bottleneck=self.ffn_num,
                                    init_option='lora',
                                    adapter_scalar=adapter_scalar,
                                    adapter_layernorm_option='none',
                                    )
            self.moe_mlp_list.append(moe_adaptmlp)

        self.vis_list=[[],[],[]]
        


    def noisy_top_k_gating(self, x, train, w_gate, w_noise, noise_epsilon=1e-2):
        """Noisy top-k gating.
          See paper: https://arxiv.org/abs/1701.06538.
          Args:
            x: input Tensor with shape [batch_size, input_size]
            train: a boolean - we only add noise at training time.
            noise_epsilon: a float
          Returns:
            gates: a Tensor with shape [batch_size, num_experts]
            load: a Tensor with shape [num_experts]
        """
        # x (B,token, dim)
        clean_logits = x @ w_gate.to(x)    #(B*token, num_experts) (40,576,4)
        if self.noisy_gating :
            raw_noise_stddev = x @ w_noise.to(x)
            noise_stddev = ((self.softplus(raw_noise_stddev) + noise_epsilon))
            noisy_logits = clean_logits + (torch.randn_like(clean_logits) * noise_stddev)
            logits = noisy_logits
        else:
            logits = clean_logits
        # logits (B* token, num_experts)
        # calculate topk + 1 that will be needed for the noisy gates
        top_logits, top_indices = logits.topk(min(self.top_k + 1, self.experts_num), dim=1)
        top_k_logits = top_logits[:,:self.top_k]        #(B* token, top_k)  存值
        top_k_indices = top_indices[:,:self.top_k]      #(B* token, top_k)  存索引
        top_k_gates = self.softmax(top_k_logits)        # (B* token, top_k)  归一化
        zeros = torch.zeros_like(logits)
        gates = zeros.scatter(1, top_k_indices, top_k_gates)  #将top_k_gates的值放到top_k_indices对应索引中

        return gates
    
    def token_level_contrastive_loss(self,x, temperature=0.07):
        """
        x: [batch_size, 1 + num_tokens, dim], 其中第一个token是class_token
        temperature: 温度系数
        """
        batch_size, all_token, dim = x.size()

        class_token = x[:, 0, :]    # [B, dim]
        tokens = x[:, 1:, :]        # [B, T-1, dim]
        T_minus_1 = tokens.size(1)

        # 计算cosine similarity (B, T-1)
        sim = F.cosine_similarity(tokens, class_token.unsqueeze(1), dim=-1)

        # 取 top 50%为正样本，bottom 50%为负样本
        k = max(1, int(0.2 * T_minus_1))  

        sorted_indices = torch.argsort(sim, dim=1)  # 升序排列

        pos_indices = sorted_indices[:, -k:]    # 正样本
        neg_indices = sorted_indices[:, :k]     # 负样本

        # 收集正负样本的embedding
        tokens_pos = torch.gather(tokens, 1, pos_indices.unsqueeze(-1).expand(-1, -1, dim))  # [B, k, dim]
        tokens_neg = torch.gather(tokens, 1, neg_indices.unsqueeze(-1).expand(-1, -1, dim))  # [B, k, dim]

        # 计算 class_token与tokens_pos的相似度 [B, k]
        pos_sim = torch.sum(class_token.unsqueeze(1) * tokens_pos, dim=-1) / temperature
        neg_sim = torch.sum(class_token.unsqueeze(1) * tokens_neg, dim=-1) / temperature

        # 拼接正负样本相似度 [B, 2k]
        logits = torch.cat([pos_sim, neg_sim], dim=1)  # [B, 2k]

        # 创建标签 (正样本在前，label为0到k-1)
        labels = torch.arange(k, device=x.device).unsqueeze(0).repeat(batch_size, 1)  # [B, k]

        # 由于每个anchor对应多个正负样本，使用cross_entropy，调整为(B*k, 1+负样本数)
        logits = torch.cat([
            pos_sim.unsqueeze(-1),  # 正样本 [B,k,1]
            neg_sim.unsqueeze(1).expand(-1, k, -1)  # 负样本 [B,k,k]
        ], dim=-1)  # [B, k, 1+k]

        logits = logits.reshape(-1, k + 1)  # [B*k, 1+k]
        labels = torch.zeros(batch_size * k, dtype=torch.long, device=x.device)  # 正样本总在第0个位置

        losses = F.cross_entropy(logits, labels,reduction='none')
        losses = losses.view(batch_size, k).mean(1)


        return losses


    ### 前背景分离
    # def forward(self, x: torch.Tensor):
    #     # x: [batch_size, class_token + ptach_token , dim]
    #     # import ipdb; ipdb.set_trace()
    #     batch_size, all_token, dim = x.size()
    #     global_taskid = self.global_taskid
    #     if global_taskid==0:
    #         import ipdb;ipdb.set_trace()
    #     # self.loss = self.token_level_contrastive_loss(x)

    #     # 分离 class token 和其他 token
    #     class_token = x[:, 0, :]       # [B, dim]
    #     tokens = x[:, 1:, :]           # [B, T-1, dim]

        
    #     T_minus_1 = tokens.size(1)     # 576 T-1
    #     sim = torch.sum(tokens * class_token.unsqueeze(1), dim=2)  # [B, T-1]
       
    #     k_top = max(1, int(1.0 * T_minus_1))             # k
    #     k_bottom = max(1, int(1.0* T_minus_1))              # k
        
    #     sorted_indices = torch.argsort(sim, dim=1)   # 升序排列：最小在前，最大在后 (B,T-1)
        
    #     top_indices = sorted_indices[:, -k_top:]         # 相似度最高的20%  (B,k)  # 前景token
    #     bottom_indices = sorted_indices[:, :k_bottom]       # 相似度最低的20%  (B,k)  # 背景token

    #     tokens_top = torch.gather(tokens, 1, top_indices.unsqueeze(-1).expand(-1, -1, dim))      # [B, k, dim]
    #     tokens_bottom = torch.gather(tokens, 1, bottom_indices.unsqueeze(-1).expand(-1, -1, dim))  # [B, k, dim]




    #     x_re = tokens_top.reshape(-1, dim)
    #     gates0 = self.noisy_top_k_gating(x_re, False, self.moe_router_list[0],
    #                                             self.moe_noise_list[0])
    #     dispatcher0 = SparseDispatcher(self.experts_num, gates0)
    #     expert_inputs0 = dispatcher0.dispatch(x_re)  
    #     expert_outputs0 = [self.moe_mlp_list[0:2][i](expert_inputs0[i].to(x), add_residual=False)
    #                         for i in range(self.experts_num)]
    #     y0 = dispatcher0.combine(expert_outputs0) 
    #     y0 = y0.reshape(batch_size, k_top, dim)
        
    #     y0_out = torch.zeros_like(tokens)  # [B, T-1, dim]
    #     for b in range(batch_size):
    #         y0_out[b].index_copy_(0, top_indices[b], y0[b])
        
        
    #     x_re = tokens_bottom.reshape(-1, dim)
    #     if self.train_flag == 0:
    #         # #输入加随机噪声
    #         # noise = torch.randn_like(x_re)  # 与 x_re 形状相同的 N(0,1) 张量
    #         # alpha = 1.0                     # 控制噪声强度的系数
    #         # x_re = x_re + alpha * noise
    #         gates = self.noisy_top_k_gating(x_re, False, self.moe_router_list[global_taskid],
    #                                                 self.moe_noise_list[global_taskid])
    #         # gates (B* token, num_experts)  每行top_k个值和为1，其他都是0
    #         dispatcher = SparseDispatcher(self.experts_num, gates)
    #         expert_inputs = dispatcher.dispatch(x_re)  
    #         # (batch*token, dim)  每个专家的输入  [(x1,dim), (x2,dim), (x3,dim), (x4,dim)] x1+x2+x3+x4 = batch*token*top_k
    #         expert_outputs = [self.moe_mlp_list[global_taskid*2:(global_taskid+1)*2][i](expert_inputs[i].to(x), add_residual=False)
    #                             for i in range(self.experts_num)]
    #         y = dispatcher.combine(expert_outputs) #(B*token, dim)  #将每个专家的输出加权求和
    #         y = y.reshape(batch_size, k_bottom, dim)  #(B, all_token-1, dim)
        
    #         y_out = torch.zeros_like(tokens)  # [B, T-1, dim]
    #         for b in range(batch_size):
    #             y_out[b].index_copy_(0, bottom_indices[b], y[b])
    #     if self.train_flag == 1:
    #         for j in range(8):
    #             taskid = j+1
    #             # #输入加随机噪声
    #             # noise = torch.randn_like(x_re)  # 与 x_re 形状相同的 N(0,1) 张量
    #             # alpha = 1.0                     # 控制噪声强度的系数
    #             # x_re = x_re + alpha * noise
    #             gates = self.noisy_top_k_gating(x_re, False, self.moe_router_list[taskid],
    #                                                     self.moe_noise_list[taskid])
    #             # gates (B* token, num_experts)  每行top_k个值和为1，其他都是0
    #             dispatcher = SparseDispatcher(self.experts_num, gates)
    #             expert_inputs = dispatcher.dispatch(x_re)  
    #             # (batch*token, dim)  每个专家的输入  [(x1,dim), (x2,dim), (x3,dim), (x4,dim)] x1+x2+x3+x4 = batch*token*top_k
    #             expert_outputs = [self.moe_mlp_list[taskid*2:(taskid+1)*2][i](expert_inputs[i].to(x), add_residual=False)
    #                                 for i in range(self.experts_num)]
    #             y = dispatcher.combine(expert_outputs) #(B*token, dim)  #将每个专家的输出加权求和
    #             y = y.reshape(batch_size, k_bottom, dim)  #(B, all_token-1, dim)
            
    #             y_out0 = torch.zeros_like(tokens)  # [B, T-1, dim]
    #             for b in range(batch_size):
    #                 y_out0[b].index_copy_(0, bottom_indices[b], y[b])
    #             self.vis_list[1].append(y_out0)
    #             if j == global_taskid:
    #                 y_out = y_out0
    #         import ipdb;ipdb.set_trace()
                


    #     x = self.mlp_moe(x)      #(B, all_token, dim)  #原来的线性层的输出
    #     # x[:, 1:, :] = x[:, 1:, :] + 0.9*y_out + 0.1*y0_out  # (B, all_token, dim)
    #     x[:, 1:, :] = x[:, 1:, :] + y0_out  # (B, all_token, dim)
    #     # x[:, 1:, :] = x[:, 1:, :] + y_out   # (B, all_token, dim)
    #     # x[:, 1:, :] = x[:, 1:, :] + 0.5*y_out + 0.5*y0_out  # (B, all_token, dim)
    #     if self.train_flag == 1:
    #         self.vis_list[0].append(y0_out)
    #         self.vis_list[2].append(x)
    #         # self.vis_list[2].append(y_out)


    #     return x
    
    def forward(self, x: torch.Tensor):      #前景背景不分离
        # x: [batch_size, class_token + ptach_token , dim]
        # import ipdb; ipdb.set_trace()
        batch_size, all_token, dim = x.size()
        x_re = x[:,1:,:].reshape(-1, dim)  #x(batch, token, dim) ---> x_re(batch*token, dim)
        
        global_taskid = self.global_taskid
        if global_taskid==0:
            import ipdb;ipdb.set_trace()
        
        
        ### 共享
        gates0 = self.noisy_top_k_gating(x_re, False, self.moe_router_list[0],
                                                    self.moe_noise_list[0])
        dispatcher0 = SparseDispatcher(self.experts_num, gates0)
        expert_inputs0 = dispatcher0.dispatch(x_re)  
        expert_outputs0 = [self.moe_mlp_list[0:2][i](expert_inputs0[i].to(x), add_residual=False)
                                for i in range(self.experts_num)]
        y0 = dispatcher0.combine(expert_outputs0) 
        y0 = y0.reshape(batch_size, all_token-1, dim)
        
        if self.train_flag == 0:
            gates = self.noisy_top_k_gating(x_re, False, self.moe_router_list[global_taskid],
                                                    self.moe_noise_list[global_taskid])
            # gates (B* token, num_experts)  每行top_k个值和为1，其他都是0
            dispatcher = SparseDispatcher(self.experts_num, gates)
            expert_inputs = dispatcher.dispatch(x_re)  
            # (batch*token, dim)  每个专家的输入  [(x1,dim), (x2,dim), (x3,dim), (x4,dim)] x1+x2+x3+x4 = batch*token*top_k
            
            expert_outputs = [self.moe_mlp_list[global_taskid*2:(global_taskid+1)*2][i](expert_inputs[i].to(x), add_residual=False)
                                for i in range(self.experts_num)]
            y = dispatcher.combine(expert_outputs) #(B*token, dim)  #将每个专家的输出加权求和
            y = y.reshape(batch_size, all_token-1, dim)  #(B, all_token-1, dim)
        
        if self.train_flag == 1:
            for j in range(8):
                taskid = j+1
                gates = self.noisy_top_k_gating(x_re, False, self.moe_router_list[taskid],
                                                        self.moe_noise_list[taskid])
                # gates (B* token, num_experts)  每行top_k个值和为1，其他都是0
                dispatcher = SparseDispatcher(self.experts_num, gates)
                expert_inputs = dispatcher.dispatch(x_re)  
                # (batch*token, dim)  每个专家的输入  [(x1,dim), (x2,dim), (x3,dim), (x4,dim)] x1+x2+x3+x4 = batch*token*top_k
                # import ipdb;ipdb.set_trace()
                expert_outputs = [self.moe_mlp_list[taskid*2:(taskid+1)*2][i](expert_inputs[i].to(x), add_residual=False)
                                    for i in range(self.experts_num)]
                y_1 = dispatcher.combine(expert_outputs) #(B*token, dim)  #将每个专家的输出加权求和
                y_1 = y_1.reshape(batch_size, all_token-1, dim)  #(B, all_token-1, dim)
                self.vis_list[1].append(y_1)
                if taskid == global_taskid:
                    y = y_1
            
        x = self.mlp_moe(x)      #(B, all_token, dim)  #原来的线性层的输出
        specific_ratio = 1.0 - self.shared_ratio
        x[:, 1:, :] = x[:, 1:, :] + specific_ratio*y + self.shared_ratio*y0  # (B, all_token, dim)
        # x[:, 1:, :] = x[:, 1:, :] + y  # (B, all_token, dim)
        # x[:, 1:, :] = x[:, 1:, :] + y0  # (B, all_token, dim)
        if self.train_flag == 1:
            self.vis_list[0].append(y0)
            self.vis_list[2].append(x)

        return x


def get_parent_module(model, module_name: str):
    """
    根据 'a.b.c' 的模块名称路径，返回父模块以及子模块在父模块中的属性名。
    """
    names = module_name.split('.')
    parent = model
    for n in names[:-1]:
        parent = getattr(parent, n)
    return parent, names[-1]

def inject_trainable_moe_1(
    model: nn.Module,
    target_replace_module: List[str] = ["Mlp"],
    rank: int = 32,
    router_num: int = 4,
    experts_num: int = 1,
    top_k: int = 1,
    adapter_scalar: float = 0.1,
    shared_ratio: float = 0.9,
    adapter_dropout: float = 0.1,
    noisy_gating: bool = True,
    domain_slots: int = 14,
    moe_layer_mode: str = "all",
):
    # model = model, target_replace_module = ["CrossAttention", "Attention"], 
    #         r = cfg.TEST.vida_rank1   1           r2 = cfg.TEST.vida_rank2   128
    """
    inject vida into model, and returns vida parameter groups.
    """

    if moe_layer_mode not in {"all", "odd", "even", "last-half"}:
        raise ValueError(
            f"Unsupported moe_layer_mode={moe_layer_mode!r}; "
            "expected all, odd, even, or last-half"
        )

    require_grad_params = []
    names = []
    injected_layers = 0
    blocks = model.module.blocks if hasattr(model, "module") else model.blocks
    half_start = len(blocks) // 2

    def selected_layer(module_name):
        parts = module_name.split(".")
        try:
            blocks_pos = parts.index("blocks")
            block_idx = int(parts[blocks_pos + 1])
        except (ValueError, IndexError):
            raise ValueError(f"Cannot determine Transformer block from {module_name!r}")

        if moe_layer_mode == "all":
            return True
        if moe_layer_mode == "odd":
            # Human layer numbers 1, 3, 5, ... correspond to 0-based even indices.
            return block_idx % 2 == 0
        if moe_layer_mode == "even":
            # Human layer numbers 2, 4, 6, ... correspond to 0-based odd indices.
            return block_idx % 2 == 1
        return block_idx >= half_start

    for name,_module in model.named_modules():
        # if "mlp" in name:
        #     parts = name.split('.')  #'module.blocks.0.mlp'
        #     block_idx = int(parts[2])
        #     if block_idx < 6:
        #         continue
        if _module.__class__.__name__ in target_replace_module:
            if not selected_layer(name):
                continue
            # 类名  "MlP"
            # print(name,_module)
            for _child_name, _child_module in _module.named_modules():
                
                if _child_name == "fc1":
                    # print(name) 实例名
                    # print(_child_module.__class__.__name__) 类名 
                    weight1 = _child_module.weight
                    bias1 = _child_module.bias
                    in_features = _child_module.in_features
                    hidden_features = _child_module.out_features
                if _child_name == "fc2":
                    weight2 = _child_module.weight
                    bias2 = _child_module.bias
                    out_features = _child_module.out_features


                    #初始化一个线性层，两个lora 1 128
                    #如果是Mlp层，定义一个新的MOE层
            _tmp = MOEInjectedLinear(
                        in_features,
                        hidden_features,
                        out_features,
                        bias1 is not None,
                        rank,
                        router_num,
                        experts_num,
                        top_k,
                        adapter_scalar,
                        shared_ratio,
                        adapter_dropout,
                        noisy_gating,
                        domain_slots,
                    )

            #把原来线性层的权重重新赋值给新的线性层
            _tmp.mlp_moe.fc1.weight = weight1
            if bias1 is not None:
                _tmp.mlp_moe.fc1.bias = bias1
            _tmp.mlp_moe.fc2.weight = weight2
            if bias2 is not None:
                _tmp.mlp_moe.fc2.bias = bias2

            # switch the module
            parent, attr_name = get_parent_module(model, name)
            # print(attr_name)
            setattr(parent, attr_name, _tmp)
            injected_layers += 1
        

            # require_grad_params.extend(
            #     list(_module[name].vida_up.parameters())
            # )
            # require_grad_params.extend(
            #     list(_module[name].vida_down.parameters())
            # )
            # require_grad_params.extend(
            #     list(_module[name].vida_up2.parameters())
            # )
            # _module[name].vida_up.weight.requires_grad = True
            # _module[name].vida_down.weight.requires_grad = True


            # _module[name].vida_up2.weight.requires_grad = True
            # _module[name].vida_down2.weight.requires_grad = True                    
            # names.append(name)

    logging.getLogger(__name__).info(
        "MoE layer mode=%s injected MLP layers=%d/%d",
        moe_layer_mode,
        injected_layers,
        len(blocks),
    )
    return require_grad_params, names
