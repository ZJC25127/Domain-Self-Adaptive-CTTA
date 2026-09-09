from copy import deepcopy

import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.jit
import numpy as np
from inject_moe_1 import inject_trainable_moe_1
import logging
from inject_moe_1 import MOEInjectedLinear
from adapter import Adapter
logger = logging.getLogger(__name__)
import math
from collections import OrderedDict


def update_ema_variables(ema_model, model, alpha_teacher, alpha_moe):#, iteration):
    # for ema_param, param in zip(ema_model.parameters(), model.parameters()):
    #     ema_param.data[:] = alpha_teacher * ema_param[:].data[:] + (1 - alpha_teacher) * param[:].data[:]
    # return ema_model
    for ema_param, (name, param) in zip(ema_model.parameters(), model.named_parameters()):
        #ema_param.data.mul_(alpha).add_(1 - alpha, param.data)
        if "moe_" in name:
            # print(alpha_moe)
            ema_param.data[:] = alpha_moe * ema_param[:].data[:] + (1 - alpha_moe) * param[:].data[:]
        else:
            ema_param.data[:] = alpha_teacher * ema_param[:].data[:] + (1 - alpha_teacher) * param[:].data[:]
    return ema_model

class MOE(nn.Module):
    """MOE adapts a model by entropy minimization during testing.

    Once tented, a model adapts itself by updating on every forward.
    """
    def __init__(self, model, optimizer, steps=1, episodic=False, ema=0.99, ema_moe = 0.99,
                 rst_m=0.1, legacy_variance=False, class_entropy_weight=0.0,
                 grad_clip_norm=0.0, restore_prob=0.0, sam_rho=0.0,
                 redundancy_margin=0.0,
                 eata_entropy_weighting=False,
                 eata_weight_scale=1.0,
                 anchor_reg_weight=0.0,
                 entropy_ratio=0.6, high_thresh=2.5,
                 low_thresh=1.5, domain_feature_size=72, fix_ema_teacher=False,
                 deterministic_prediction=False, post_update_prediction=False,
                 new_domain_init="zero",
                 sample_entropy_mode="full", sample_entropy_thresh=None,
                 dynamic_entropy_thresh=False, dynamic_entropy_quantile=0.6,
                 dynamic_entropy_warmup=5, dynamic_entropy_cold_start=0.9):
        super().__init__()
        self.model = model
        self.optimizer = optimizer
        self.steps = steps
        assert steps > 0, "MOE requires >= 1 step(s) to forward and update"
        self.episodic = episodic
        
        self.model_state, self.optimizer_state, self.model_ema, self.model_anchor = \
            copy_model_and_optimizer(self.model, self.optimizer)
        self.alpha_teacher = ema     # 0.999
        self.alpha_moe = ema_moe   # 0.999
        self.rst = rst_m  
        self.legacy_variance = legacy_variance
        self.class_entropy_weight = class_entropy_weight
        self.grad_clip_norm = grad_clip_norm
        self.restore_prob = restore_prob
        self.sam_rho = sam_rho
        self.redundancy_margin = redundancy_margin
        self.eata_entropy_weighting = eata_entropy_weighting
        self.eata_weight_scale = eata_weight_scale
        self.anchor_reg_weight = anchor_reg_weight
        self.current_model_probs = None
        self.entropy_ratio = entropy_ratio
        self.high_thresh = high_thresh
        self.low_thresh = low_thresh
        self.domain_feature_size = domain_feature_size
        self.fix_ema_teacher = fix_ema_teacher
        self.deterministic_prediction = deterministic_prediction
        self.post_update_prediction = post_update_prediction
        self.new_domain_init = new_domain_init
        self.sample_entropy_mode = sample_entropy_mode
        self.sample_entropy_thresh = sample_entropy_thresh
        self.dynamic_entropy_thresh = dynamic_entropy_thresh
        self.dynamic_entropy_quantile = dynamic_entropy_quantile
        self.dynamic_entropy_warmup = dynamic_entropy_warmup
        self.dynamic_entropy_cold_start = dynamic_entropy_cold_start
        self.pre_entropys = 0.0
        
        self.domain_class=0
        self.batch=0
        self.task_id=0
        self.class_centers = []
        self.domain_threshold = 0.01  #0.15  
        self.update=0
        from collections import deque
        self.distance_queue = deque(maxlen=3)  # 存储最近3次min_distance
        self.feature_queue = deque(maxlen=3)
        self.entropy_history = {}
        self.shared_weight = {}
        # self.entropy_buffer = DynamicEntropyBuffer(buffer_size=30,entropy_threshold=math.log(10) * 0.1)
  
        ## 域检测模型
        for name, module in self.model_ema.named_modules():
            if hasattr(module, 'global_taskid'):
                module.global_taskid = 0
        

    def forward(self, x, only_test=False, class_indices=None):
        if self.episodic:
            self.reset()

        for _ in range(self.steps):
            outputs,a = self.forward_and_adapt(
                x, self.model, self.optimizer, only_test=only_test,
                class_indices=class_indices
            )

        return outputs,a

    def reset(self):
        if self.model_state is None or self.optimizer_state is None:
            raise Exception("cannot reset without saved model/optimizer state")
        # Injected MoE layers keep a per-forward visualization/loss tensor.
        # After an adaptation step it can still reference an autograd graph;
        # clear that transient state before rebuilding the reset snapshots.
        for module in self.model.modules():
            if hasattr(module, 'loss'):
                module.loss = None
            if hasattr(module, 'attention_out'):
                module.attention_out = None
            if hasattr(module, 'vis_list'):
                module.vis_list = [[], [], []]
        load_model_and_optimizer(self.model, self.optimizer,
                                 self.model_state, self.optimizer_state)
        # Use this line to also restore the teacher model                         
        self.model_state, self.optimizer_state, self.model_ema, self.model_anchor = \
            copy_model_and_optimizer(self.model, self.optimizer)
   
    def set_task_id(self,task_id):  #如果包含这个属性或方法更改
        for target in (self.model, self.model_ema):
            for name, module in target.named_modules():
                if hasattr(module, 'global_taskid'):
                    module.global_taskid = task_id

    def _initialize_new_domain_parameters(self, new_task_id, nearest_task_id):
        """Initialize a new task slot from the nearest or shared slot.

        Slot 0 is the shared branch and task ids 1..N are domain-specific
        branches.  Both the live model and EMA model are initialized so the
        option remains consistent when the EMA teacher is enabled.
        """
        if self.new_domain_init == "zero":
            return 0

        source_task_id = nearest_task_id if self.new_domain_init == "nearest" else 0
        copied_layers = 0
        for target_model in (self.model, self.model_ema):
            for module in target_model.modules():
                if not isinstance(module, MOEInjectedLinear):
                    continue
                if new_task_id >= len(module.moe_router_list):
                    logger.warning(
                        "new domain task id %d exceeds available slots %d; skip initialization",
                        new_task_id, len(module.moe_router_list))
                    continue
                with torch.no_grad():
                    module.moe_router_list[new_task_id].copy_(
                        module.moe_router_list[source_task_id]
                    )
                    module.moe_noise_list[new_task_id].copy_(
                        module.moe_noise_list[source_task_id]
                    )
                for expert_id in range(module.experts_num):
                    target_expert = new_task_id * module.experts_num + expert_id
                    source_expert = source_task_id * module.experts_num + expert_id
                    module.moe_mlp_list[target_expert].load_state_dict(
                        module.moe_mlp_list[source_expert].state_dict()
                    )
                copied_layers += 1

        logger.info(
            "initialized new domain task=%d from %s task=%d across %d model slots",
            new_task_id,
            self.new_domain_init,
            source_task_id,
            copied_layers,
        )
        return copied_layers

    def _predict_without_adapter_dropout(self, x):
        """Return a deterministic prediction while preserving train mode."""
        states = [(module, module.training)
                  for module in self.model.modules()
                  if isinstance(module, Adapter)]
        for module, _ in states:
            module.eval()
        try:
            with torch.no_grad():
                return self.model(x)
        finally:
            for module, training in states:
                module.train(training)

    def _source_anchor_regularization(self, model):
        """Mean squared drift of trainable MoE parameters from source weights."""
        if self.anchor_reg_weight <= 0.0:
            return torch.zeros((), device=next(model.parameters()).device)
        anchor_params = dict(self.model_anchor.named_parameters())
        total = torch.zeros((), device=next(model.parameters()).device)
        count = 0
        for name, param in model.named_parameters():
            if not param.requires_grad or "moe_" not in name:
                continue
            total = total + (param - anchor_params[name]).pow(2).sum()
            count += param.numel()
        return total / max(count, 1)
    # ## 均值距离
    # @torch.enable_grad()  # ensure grads in possible no grad context for testing
    # def forward_and_adapt(self, x, model, optimizer,only_test=False):
    #     self.batch +=1
    #     if self.batch==101:
    #         self.domain_class+=1
    #         self.batch=1

    #     if x.dim() == 5:
    #         x1= x[:, 0, :, :, :]
    #         x2= x[:, 1:, :, :, :].reshape(-1, 3, 224, 224)
    #     else:
    #         x1= x

    #     ############ 任务判断 ###########
    #     # image_gray= x1.mean(dim=1) 
    #     weights = torch.tensor([0.2989, 0.5870, 0.1140], device=x1.device).view(1, 3, 1, 1)
    #     image_gray = (x1 * weights).sum(dim=1)
        
    #     f_transform = torch.fft.fft2(image_gray)                                 # 结果 shape: (B, 224, 224)，每个是复数
    #     f_transform_shifted = torch.fft.fftshift(f_transform, dim=(-2, -1))    # 频率居中
    #     magnitude_spectrum = torch.abs(f_transform_shifted)          # 强度谱，也是 (B, 224, 224)
    #     # psd = magnitude_spectrum ** 2 
    #     # magnitude_spectrum_pooled = magnitude_spectrum[:, :112, 112:]  
    #     k = 72
    #     h2, w2 = 224//2, 224//2
    #     magnitude_spectrum_pooled = magnitude_spectrum[:, h2-k//2:h2+k//2, w2-k//2:w2+k//2]  

    #     flattened_feature = magnitude_spectrum_pooled.reshape(magnitude_spectrum_pooled.size(0), -1)#（B, 112*112）
    #     flattened_feature = flattened_feature.mean(0, keepdim=True)  
    #     flattened_feature = flattened_feature.cpu().numpy()   
    #     batch_feature_normalized = flattened_feature / (np.linalg.norm(flattened_feature, axis=1,ord=1, keepdims=True) + 1e-8)
            
    #     if not only_test:
    #         result_tmp=[]
    #         if len(self.class_centers) == 0:
    #             self.update = 20
    #             self.class_centers.append((batch_feature_normalized, 1))  # 保存特征和当前中心的样本数
    #             logger.info("add new class center")
    #             self.task_id=1              #表示扩展moe的个数
    #             min_index = 0    #第一次就指定选哪个
    #         else:
    #             distances = []
    #             for center, _ in self.class_centers:
    #                 # 确保 center 已归一化（如未归一化则手动归一化）
    #                 center_normalized = center / (np.linalg.norm(center, axis=1, ord=1, keepdims=True) + 1e-8)
    #                 l2_distance = np.linalg.norm(batch_feature_normalized - center_normalized, ord=2)
    #                 # import ipdb; ipdb.set_trace()
    #                 distances.append(l2_distance)

    #             min_index = np.argmin(distances) #选择出最近的域中心
    #             logger.info(f"batch: {self.batch} class: {self.domain_class} threshold: {self.domain_threshold:.4f}  min_index: {min_index+1}/{len(self.class_centers)}, min_distance: {distances[min_index]:.7f}")
                
    #             min_distance = distances[min_index]
    #             self.distance_queue.append(min_distance)
    #             self.feature_queue.append(batch_feature_normalized)
                
    #             ########## 如何更新或新建中心
    #             # 判断连续三个batch的最小距离  大于阈值1  --- 小于阈值0
    #             for i in range(len(self.distance_queue)):
    #                 if self.distance_queue[i]>= self.domain_threshold:
    #                     result_tmp.append(1)
    #                 else:
    #                     result_tmp.append(0)
    #             # if sum(result_tmp)<len(result_tmp):
    #             #     valid_features = [self.feature_queue[i] for i in range(len(result_tmp)) if result_tmp[i] == 0]
    #             #     sum_feature = np.sum(valid_features, axis=0) 
    #             #     old_center, old_num = self.class_centers[min_index]
    #             #     new_center = (old_center * old_num + sum_feature) / (old_num+ len(valid_features))
    #             #     self.class_centers[min_index] = (new_center, old_num + len(valid_features))
    #             if sum(result_tmp)<len(result_tmp):
    #                 if result_tmp[-1]==0: # 0 1 2
    #                     valid_features = [self.feature_queue[-1]]
    #                     sum_feature = np.sum(valid_features, axis=0)
    #                     old_center, old_num = self.class_centers[min_index]
    #                     new_center = (old_center * old_num + sum_feature) / (old_num+ len(valid_features))
    #                     new_center = new_center / (np.linalg.norm(new_center, ord=1) + 1e-8)  # 归一化
    #                     self.class_centers[min_index] = (new_center, old_num + len(valid_features))
    #             else:
    #                 if self.update == 0 :
    #                     averaged_feature = np.sum(self.feature_queue, axis=0) / len(self.distance_queue)
    #                     self.class_centers.append((averaged_feature, len(self.distance_queue)))
    #                     logger.info("add new class center")
    #                     self.update=20
    #                     self.task_id = self.task_id + 1
            
    #         if self.update>0:
    #             self.update -= 1


    #         self.set_task_id(min_index+1) 
    #         logger.info(f"set task id {min_index+1}")
            


    #         ######   原样本 #########
    #         a = 0
    #         outputs = self.model(x1)
    #         standard_ema = outputs
    #         entropys0 = (softmax_entropy(outputs, standard_ema))

    #         e_margin = math.log(100) * 0.3
            
    #         # intersection_ids = torch.where(entropys0 < e_margin)
        

    #         filter_ids_1 = torch.where(entropys0 < e_margin)[0]
    #         filter_ids_2 = torch.argsort(entropys0, descending=False)[:int(entropys0.size()[0] * 0.6)]
    #         intersection_ids = torch.tensor(
    #                                 list(set(filter_ids_1.tolist()) & set(filter_ids_2.tolist())),
    #                                 dtype=torch.long,   # 强制为 long
    #                                 device=entropys0.device
    #                             )

    #         entropys = entropys0[intersection_ids]
    #         a = entropys.shape[0]
    #         ########################


    #         # ##########  共享loss ##########
    #         # delta_loss = 0.0
    #         # device = next(model.parameters()).device
    #         # if self.shared_weight!={}:
    #         #     delta_loss = compute_shared_delta_loss(model, self.shared_weight,device)
    #         # self.shared_weight = cache_shared_params(model)
    #         # print(f"delta_loss: {delta_loss:.5f}")
    #         # # import ipdb; ipdb.set_trace()

    #         # #######################
            

    #         ########## 增强样本 ############
    #         entropys_arg = None
    #         if x.dim()==5:
    #             outputs_arg  = self.model(x2)
    #             entropys_arg0 = (softmax_entropy(outputs_arg, outputs_arg))
    #             # e_margin = math.log(10) * 0.20
    #             # filter_ids_arg = torch.where(entropys_arg0 < e_margin)
    #             filter_ids_arg = torch.argsort(entropys_arg0, descending=False)[:int(entropys_arg0.size()[0] * 0.2)]
    #             entropys_arg = entropys_arg0[filter_ids_arg]
    #         ########################



    #         # if result_tmp[-1] == 0 or self.update>0 :  #必须是在这个域里才更新 或者在连续阶段
    #         if result_tmp and result_tmp[-1] == 0 or self.update > 0:
    #             print("update loss")
    #             loss0=None
    #             for name, module in self.model.named_modules():
    #                 if hasattr(module, 'loss') and module.loss is not None:
    #                     if loss0 is None:
    #                         loss0 = module.loss
    #                     else:
    #                         loss0 += module.loss

    #             if loss0 is not None: 
    #                 loss0 = loss0[intersection_ids]       
    #                 loss = entropys.mean(0)+0.001*loss0.mean(0)
    #             else:
    #                 if entropys_arg is None:
    #                     loss = entropys.mean(0)
    #                 else:
    #                     loss = entropys.mean(0) + 1.0*entropys_arg.mean(0) #+ 10.0*delta_loss   

    #             loss.backward()
    #             optimizer.step()
    #             optimizer.zero_grad()
        
    #     else:
    #         distances = []
    #         for center, _ in self.class_centers:
    #             # 确保 center 已归一化（如未归一化则手动归一化）
    #             center_normalized = center / (np.linalg.norm(center, axis=1, ord=1, keepdims=True) + 1e-8)
    #             l2_distance = np.linalg.norm(batch_feature_normalized - center_normalized, ord=2)
    #             # import ipdb; ipdb.set_trace()
    #             distances.append(l2_distance)

    #         min_index = np.argmin(distances)
    #         self.set_task_id(min_index+1) 
    #         logger.info(f"set task id {min_index+1}")
    #         a = 0
    #         outputs = self.model(x1)
    #         standard_ema = outputs
        
    #     return standard_ema,a


    @torch.enable_grad()  # ensure grads in possible no grad context for testing
    def forward_and_adapt(self, x, model, optimizer, only_test=False,
                          class_indices=None):
        self.batch += 1
        if self.batch == 101:
            self.domain_class += 1
            self.batch = 1

        x1 = x

        centers_before = len(self.class_centers)

        weights = torch.tensor([0.2989, 0.5870, 0.1140], device=x1.device).view(1, 3, 1, 1)
        image_gray = (x1 * weights).sum(dim=1)

        f_transform = torch.fft.fft2(image_gray)
        f_transform_shifted = torch.fft.fftshift(f_transform, dim=(-2, -1))
        magnitude_spectrum = torch.abs(f_transform_shifted)

        k = self.domain_feature_size
        h2, w2 = 224 // 2, 224 // 2
        magnitude_spectrum_pooled = magnitude_spectrum[:, h2-k//2:h2+k//2, w2-k//2:w2+k//2]

        flattened_feature = magnitude_spectrum_pooled.reshape(magnitude_spectrum_pooled.size(0), -1)
        flattened_feature = flattened_feature.mean(0, keepdim=True)
        flattened_feature = flattened_feature.cpu().numpy()
        batch_feature_normalized = flattened_feature / (np.linalg.norm(flattened_feature, axis=1, ord=1, keepdims=True) + 1e-8)
        z = batch_feature_normalized[0]

        # Mahalanobis distance with diagonal covariance
        def mahalanobis_diag(z, mu, sigma_diag, eps=1e-4):
            sigma_diag_reg = (1 - eps) * sigma_diag + eps
            diff = z - mu
            return np.sum((diff ** 2) / sigma_diag_reg)

        if not only_test:
            # 双阈值判断
            low_thresh = self.low_thresh
            high_thresh = self.high_thresh
            if len(self.class_centers) == 0:
                mu = z
                sigma_diag = np.ones_like(z) * 1e-2
                c = 1
                self.class_centers.append((mu, sigma_diag, c))
                self.update = 20
                min_index = 0
                self.task_id = 1
                min_dist = 0.0
                logger.info("add new class center (first)")
            else:
                distances = [mahalanobis_diag(z, mu, sigma_diag) for (mu, sigma_diag, c) in self.class_centers]
                min_dist = min(distances)
                min_index = np.argmin(distances)

                logger.info(f"batch: {self.batch} class: {self.domain_class} min_index: {min_index+1}/{len(self.class_centers)}, min_distance: {min_dist:.6f}")

                self.distance_queue.append(min_dist)
                self.feature_queue.append(z)



                if min_dist < low_thresh:
                    mu_old, sigma_diag_old, c_old = self.class_centers[min_index]
                    w = np.exp(-0.5 * min_dist)
                    c_new = c_old + w
                    delta = z - mu_old
                    mu_new = mu_old + (w / c_new) * delta
                    if self.legacy_variance:
                        sigma_diag_new = (c_old * sigma_diag_old + w * (delta ** 2)) / c_new
                    else:
                        m2_old = c_old * sigma_diag_old
                        m2_new = m2_old + w * delta * (z - mu_new)
                        sigma_diag_new = m2_new / c_new
                    self.class_centers[min_index] = (mu_new, sigma_diag_new, c_new)
                    logger.info(f"✅ updated domain {min_index+1} with weight={w:.4f}")

                elif min_dist > high_thresh:
                    if self.update == 0:
                        z_avg = np.sum(self.feature_queue, axis=0) / len(self.feature_queue)
                        mu = z_avg
                        sigma_diag = np.ones_like(z) * 1e-2
                        c = float(len(self.feature_queue))
                        self.class_centers.append((mu, sigma_diag, c))
                        logger.info("🚨 add new class center (new domain)")
                        new_task_id = len(self.class_centers)
                        self._initialize_new_domain_parameters(
                            new_task_id=new_task_id,
                            nearest_task_id=int(min_index) + 1,
                        )
                        self.task_id += 1
                        self.update = 20
                else:
                    logger.info("⚠️ uncertain region: no update, no new center")

                if self.update > 0:
                    self.update -= 1

            self.set_task_id(min_index + 1)
            logger.info(f"set task id {min_index + 1}")
            # self.set_task_id(self.domain_class)
            # logger.info(f"set task id {min_index + 1}")

            # 原样本
            a = 0
            outputs = self.model(x1)
            prediction_outputs = outputs
            if self.deterministic_prediction:
                prediction_outputs = self._predict_without_adapter_dropout(x1)
            teacher_outputs = outputs
            if self.fix_ema_teacher:
                self.model_ema.eval()
                with torch.no_grad():
                    teacher_outputs = self.model_ema(x1)
            adapt_outputs = (outputs if class_indices is None
                             else outputs[:, class_indices])
            adapt_teacher_outputs = (teacher_outputs if class_indices is None
                                     else teacher_outputs[:, class_indices])
            entropy_fn = (per_sample_topk_entropy
                          if self.sample_entropy_mode == "per_sample_top_half"
                          else softmax_entropy)
            entropys0 = entropy_fn(adapt_outputs, adapt_teacher_outputs)
            batch_ent = batch_class_entropy(adapt_outputs)

            entropy_domain_index = (
                len(self.class_centers) - 1
                if len(self.class_centers) > centers_before
                else int(min_index)
            )
            entropy_history_size = 0
            dynamic_entropy_ready = False
            if self.dynamic_entropy_thresh:
                from collections import deque

                history = self.entropy_history.setdefault(
                    entropy_domain_index,
                    deque(maxlen=self.dynamic_entropy_warmup * entropys0.size(0) * 4),
                )
                entropy_history_size = len(history)
                if entropy_history_size >= self.dynamic_entropy_warmup * entropys0.size(0):
                    history_tensor = torch.tensor(list(history), dtype=entropys0.dtype,
                                                   device=entropys0.device)
                    e_margin = torch.quantile(
                        history_tensor, self.dynamic_entropy_quantile
                    ).item()
                    dynamic_entropy_ready = True
                else:
                    e_margin = self.dynamic_entropy_cold_start
            else:
                e_margin = (self.sample_entropy_thresh
                            if self.sample_entropy_thresh is not None
                            else math.log(100) * 0.3)
            filter_ids_1 = torch.where(entropys0 < e_margin)[0]
            filter_ids_2 = torch.argsort(entropys0, descending=False)[:int(entropys0.size(0) * self.entropy_ratio)]
            intersection_ids = torch.tensor(list(set(filter_ids_1.tolist()) & set(filter_ids_2.tolist())),
                                            dtype=torch.long, device=entropys0.device)
            reliable_ids = intersection_ids
            if self.redundancy_margin > 0.0 and reliable_ids.numel() > 0:
                current_probs = outputs.softmax(dim=1).detach()
                if self.current_model_probs is not None:
                    similarities = F.cosine_similarity(
                        self.current_model_probs.unsqueeze(0),
                        current_probs[reliable_ids], dim=1
                    )
                    intersection_ids = reliable_ids[
                        similarities.abs() < self.redundancy_margin
                    ]
                if intersection_ids.numel() > 0:
                    new_probs = current_probs[intersection_ids].mean(dim=0)
                    self.current_model_probs = (
                        new_probs if self.current_model_probs is None else
                        0.9 * self.current_model_probs + 0.1 * new_probs
                    )
                logger.info(
                    "redundancy filter: reliable=%d kept=%d margin=%.4f",
                    reliable_ids.numel(), intersection_ids.numel(),
                    self.redundancy_margin,
                )
            entropys = entropys0[intersection_ids]
            a = entropys.shape[0]

            if self.dynamic_entropy_thresh:
                self.entropy_history[entropy_domain_index].extend(
                    entropys0.detach().cpu().tolist()
                )

            # 损失优化
            if (min_dist < low_thresh or self.update > 0) and entropys.numel() > 0:
                loss0 = None
                for name, module in self.model.named_modules():
                    if hasattr(module, 'loss') and module.loss is not None:
                        loss0 = module.loss if loss0 is None else loss0 + module.loss

                if loss0 is not None:
                    loss = entropys.mean(0) + 0.001 * loss0[intersection_ids].mean(0)
                else:
                    if self.eata_entropy_weighting:
                        confidence_weight = torch.exp(
                            (self.eata_weight_scale *
                             (e_margin - entropys.detach())).clamp(max=5.0)
                        )
                        sample_loss = (entropys * confidence_weight).mean(0)
                    else:
                        sample_loss = entropys.mean(0)
                    loss = sample_loss

                # Minimize the sample entropy while maximizing entropy of the
                # batch-level marginal class distribution.
                loss = loss - self.class_entropy_weight * batch_ent
                loss = loss + self.anchor_reg_weight * self._source_anchor_regularization(model)

                loss.backward()
                sam_perturbations = []
                if self.sam_rho > 0.0 and loss0 is None:
                    grad_params = [p for p in model.parameters()
                                   if p.requires_grad and p.grad is not None]
                    grad_norm = torch.norm(torch.stack([
                        p.grad.detach().norm(2) for p in grad_params
                    ])) if grad_params else torch.tensor(0.0, device=x1.device)
                    scale = self.sam_rho / (grad_norm + 1e-12)
                    with torch.no_grad():
                        for param in grad_params:
                            perturbation = param.grad * scale
                            param.add_(perturbation)
                            sam_perturbations.append((param, perturbation))
                    optimizer.zero_grad()

                    # Re-evaluate the same selected samples at the perturbed
                    # weights, then take one optimizer step from this gradient.
                    outputs_sam = model(x1)
                    teacher_outputs_sam = outputs_sam
                    if self.fix_ema_teacher:
                        teacher_outputs_sam = teacher_outputs
                    adapt_outputs_sam = (
                        outputs_sam if class_indices is None
                        else outputs_sam[:, class_indices]
                    )
                    adapt_teacher_outputs_sam = (
                        teacher_outputs_sam if class_indices is None
                        else teacher_outputs_sam[:, class_indices]
                    )
                    entropys_sam = entropy_fn(
                        adapt_outputs_sam, adapt_teacher_outputs_sam
                    )
                    sam_entropys = entropys_sam[intersection_ids]
                    if self.eata_entropy_weighting:
                        sam_weights = torch.exp(
                            (self.eata_weight_scale *
                             (e_margin - sam_entropys.detach())).clamp(max=5.0)
                        )
                        sam_loss = (sam_entropys * sam_weights).mean(0)
                    else:
                        sam_loss = sam_entropys.mean(0)
                    sam_loss = sam_loss - self.class_entropy_weight * batch_class_entropy(adapt_outputs_sam)
                    sam_loss = sam_loss + self.anchor_reg_weight * self._source_anchor_regularization(model)
                    sam_loss.backward()
                    logger.info("SAM gradient: first_norm=%.6f radius=%.6f",
                                float(grad_norm), self.sam_rho)

                    with torch.no_grad():
                        for param, perturbation in sam_perturbations:
                            param.sub_(perturbation)
                if self.grad_clip_norm > 0.0:
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        [p for p in model.parameters() if p.requires_grad],
                        self.grad_clip_norm,
                    )
                    logger.info("gradient clip: total_norm=%.6f cap=%.6f",
                                float(grad_norm), self.grad_clip_norm)
                optimizer.step()
                if self.restore_prob > 0.0:
                    anchor_params = dict(self.model_anchor.named_parameters())
                    restored = 0
                    eligible = 0
                    with torch.no_grad():
                        for name, param in model.named_parameters():
                            if not param.requires_grad or "moe_" not in name:
                                continue
                            anchor = anchor_params[name].to(param.device)
                            mask = torch.rand_like(param) < self.restore_prob
                            param.data.copy_(torch.where(mask, anchor, param))
                            restored += int(mask.sum().item())
                            eligible += mask.numel()
                    logger.info("anchor restore: restored=%d/%d probability=%.4f",
                                restored, eligible, self.restore_prob)
                optimizer.zero_grad()
            if self.fix_ema_teacher:
                update_ema_variables(self.model_ema, self.model,
                                     self.alpha_teacher, self.alpha_moe)

            if self.post_update_prediction:
                standard_ema = self._predict_without_adapter_dropout(x1)

            logger.info(
                "adapt diagnostics: selected=%d/%d, "
                "entropy_mean=%.4f, entropy_min=%.4f, entropy_max=%.4f, "
                "entropy_thresh=%.4f, entropy_domain=%d, entropy_history=%d, "
                "dynamic_ready=%s, centers=%d, cooldown=%d, optimized=%s",
                a,
                entropys0.size(0),
                entropys0.mean().item(),
                entropys0.min().item(),
                entropys0.max().item(),
                e_margin,
                entropy_domain_index,
                entropy_history_size,
                dynamic_entropy_ready,
                len(self.class_centers),
                self.update,
                min_dist < low_thresh or self.update > 0,
            )
            logger.info("class entropy: %.6f, weighted contribution: %.6f",
                        batch_ent.item(),
                        (self.class_entropy_weight * batch_ent).item())

            standard_ema = (self._predict_without_adapter_dropout(x1)
                            if self.fix_ema_teacher
                            else prediction_outputs)

        else:
            for name, module in self.model.named_modules():
                if hasattr(module, 'train_flag'):
                    # Inference should use the selected domain and must not
                    # retain visualization tensors from every domain.
                    module.train_flag = 0
            distances = [mahalanobis_diag(z, mu, sigma_diag) for (mu, sigma_diag, c) in self.class_centers]
            min_index = np.argmin(distances)
            # min_index = 0
            self.set_task_id(min_index + 1)
            logger.info(f"set task id {min_index + 1}")
            a = 0
            outputs = self.model(x1)
            standard_ema = outputs

        return standard_ema, a







    # @torch.enable_grad()  # ensure grads in possible no grad context for testing
    # def forward_and_adapt(self, x, model, optimizer, only_test=False):
        

        
    #     self.batch +=1
    #     if self.batch==101:
    #         self.domain_class+=1
    #         self.batch=1

    #     # upper
    #     # self.set_task_id(self.domain_class+1) 
    #     # logger.info(f"set task id {self.domain_class+1}")

    #     # # random
    #     rand_id = random.randint(1, 8)      # 随机生成 1‑8（含 1 和 8）
    #     self.set_task_id(rand_id) 
    #     logger.info(f"set task id {rand_id}")
    
        
    #     if x.dim()==5:
    #         x1= x[:, 0, :, :, :]
    #         x2= x[:, 1:, :, :, :].reshape(-1, 3, 224, 224)
    #     else:
    #         x1= x
    #     ######   原样本 #########
    #     a = 0
    #     outputs = self.model(x1)
    #     standard_ema = outputs
    #     entropys0 = softmax_entropy(outputs, standard_ema)

    #     e_margin = math.log(100) * 0.3
    #     filter_ids_1 = torch.where(entropys0 < e_margin)[0]
    #     filter_ids_2 = torch.argsort(entropys0, descending=False)[:int(entropys0.size(0) * 0.6)]
    #     intersection_ids = torch.tensor(list(set(filter_ids_1.tolist()) & set(filter_ids_2.tolist())),
    #                                     dtype=torch.long, device=entropys0.device)
    #     entropys = entropys0[intersection_ids]
    #     a = entropys.shape[0]
    #     ########################

        

    #     ########## 增强样本 ############
    #     entropys_arg = None
    #     if x.dim()==5:
    #         outputs_arg  = self.model(x2)
    #         entropys_arg0 = (softmax_entropy(outputs_arg, outputs_arg))
    #         filter_ids_arg = torch.argsort(entropys_arg0, descending=False)[:int(entropys_arg0.size()[0] * 0.2)]
    #         entropys_arg = entropys_arg0[filter_ids_arg]
    #     ########################

    #     loss0=None
    #     for name, module in self.model.named_modules():
    #         if hasattr(module, 'loss') and module.loss is not None:
    #             if loss0 is None:
    #                 loss0 = module.loss
    #             else:
    #                 loss0 += module.loss

    #     if loss0 is not None: 
    #         loss0 = loss0[filter_ids_1]       
    #         loss = entropys.mean(0)+0.001*loss0.mean(0)
    #     else:
    #         if entropys_arg is None:
    #             loss = entropys.mean(0)
    #         else:
    #             loss = entropys.mean(0) + 1.0*entropys_arg.mean(0)   
    #             # x= entropys.mean(0)
    #             # y= entropys_arg.mean(0)
                
    #             # par = x.item()/(y.item()+1e-8) * 0.5
    #             # loss= x + par*y
    
    #             # print(entropys.mean(0).item(), entropys_arg.mean(0).item()) 


    #         # loss = 0.5*entropys.mean(0) + 0.5*entropys_arg.mean(0) + 0.1*loss_arg.mean(0) #2
    #         # loss = entropys.mean(0) + 0.1*loss_arg.mean(0) #2
    #         # loss = loss_arg.mean(0)


    #     loss.backward()
    #     optimizer.step()
    #     optimizer.zero_grad()
    #     return standard_ema,a

@torch.jit.script
def softmax_entropy(x, x_ema):# -> torch.Tensor:
    """Entropy of softmax distribution from logits."""
    return -(x_ema.softmax(1) * x.log_softmax(1)).sum(1)


def per_sample_topk_entropy(x, x_ema, top_ratio=0.5, eps=1e-8):
    """Cross-entropy restricted to each sample's top half of classes."""
    teacher_probs = x_ema.softmax(dim=1)
    student_probs = x.softmax(dim=1)
    top_k = max(1, int(teacher_probs.size(1) * top_ratio))
    top_indices = torch.topk(
        teacher_probs, top_k, dim=1, largest=True, sorted=False
    ).indices
    teacher_top = torch.gather(teacher_probs, 1, top_indices)
    student_top = torch.gather(student_probs, 1, top_indices)
    teacher_top = teacher_top / teacher_top.sum(dim=1, keepdim=True).clamp_min(eps)
    student_top = student_top / student_top.sum(dim=1, keepdim=True).clamp_min(eps)
    return -(teacher_top * student_top.clamp_min(eps).log()).sum(dim=1)


def batch_class_entropy(logits, eps=1e-8):
    """Entropy of the batch-mean class probability distribution."""
    mean_probs = logits.softmax(dim=1).mean(dim=0)
    return -(mean_probs * mean_probs.clamp_min(eps).log()).sum()


def collect_params(model,freeze=False):
    """Collect all trainable parameters.

    Walk the model's modules and collect all parameters.
    Return the parameters and their names.

    Note: other choices of parameterization are possible!
    """
    # 假设你已有一个 PyTorch 模型对象 model

    # num_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    num_trainable_params = sum(p.numel() for p in model.parameters())

    logger.info(f"Number of trainable parameters (initial): {num_trainable_params}")

    moe_params_list = []
    model_params_lst = []
    for name, param in model.named_parameters():
        if 'moe_' in name:
            moe_params_list.append(param)
            # logger.info(f"moe trainable parameters{name}")
        else:
            if freeze:
                param.requires_grad = False  # freeze non-vida parameters
            else:
                model_params_lst.append(param)  # collect only if not freezing
    
    # Report final number of trainable parameters    
    num_trainable_params_after = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Number of trainable parameters (after freezing): {num_trainable_params_after}")

    return model_params_lst, moe_params_list


def copy_model_and_optimizer(model, optimizer):
    """Copy the model and optimizer states for resetting after adaptation."""
    model_state = deepcopy(model.state_dict())
    model_anchor = deepcopy(model)
    optimizer_state = deepcopy(optimizer.state_dict())
    ema_model = deepcopy(model)
    for param in ema_model.parameters():
        param.detach_()
    return model_state, optimizer_state, ema_model, model_anchor


def load_model_and_optimizer(model, optimizer, model_state, optimizer_state):
    """Restore the model and optimizer states from copies."""
    model.load_state_dict(model_state, strict=True)
    optimizer.load_state_dict(optimizer_state)


def configure_model(model, cfg, shared_ratio=0.9, adapter_dropout=0.1,
                    noisy_gating=True, domain_slots=14, moe_layer_mode="all"):
    """Inject the project's MoE adapters and load the source checkpoint."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.cpu()
   
    if cfg.TEST.moe_new:
        vida_params, vida_names = inject_trainable_moe_1(model = model, target_replace_module = ["Mlp"], rank = cfg.TEST.moe_rank, router_num = cfg.TEST.moe_router_num,experts_num=cfg.TEST.moe_exp_num,top_k=cfg.TEST.moe_top_k,adapter_scalar=cfg.TEST.adapter_scalar, shared_ratio=shared_ratio, adapter_dropout=adapter_dropout, noisy_gating=noisy_gating, domain_slots=domain_slots, moe_layer_mode=moe_layer_mode)
        
        if cfg.TEST.ckpt!=None:
            checkpoint = torch.load(cfg.TEST.ckpt,map_location='cuda:0')
            state_dict = checkpoint['model'] if 'model' in checkpoint else checkpoint
            model.load_state_dict(state_dict, strict=False)
        
        new_state_dict = model.state_dict()
        updated_state_dict = OrderedDict()
                
        # 'module.blocks.11.mlp.moe_mlp_list.26.up_proj.bias' 1-32
        # 'module.blocks.11.mlp.moe_router_list.6' 1-16
        # 'module.blocks.11.mlp.moe_noise_list.1'  1-16
        for key, param in new_state_dict.items():
            if 'moe_mlp_list' in key:
                # key示例: module.blocks.{block_idx}.mlp.moe_mlp_list.{adapter_idx}.down_proj.weight
                parts = key.split('.')

                # block_idx = int(parts[2])
                # if block_idx < 6:
                #     continue

                # block_idx = int(parts[2])
                # if block_idx % 2 == 0:  # 偶数块
                #     continue

                adapter_idx = int(parts[5])
                old_adapter_idx = adapter_idx  % 2  #偶数对0 奇数对1
                old_key = '.'.join(parts[:5] + [str(old_adapter_idx)] + parts[6:])
                # import ipdb;ipdb.set_trace()
                if old_key in state_dict:
                    updated_state_dict[key] = state_dict[old_key]
                else:
                    print(f"Warning: {old_key} not found in checkpoint, using original param.")
                    updated_state_dict[key] = param

            elif 'moe_router_list' in key or 'moe_noise_list' in key:
                # key示例：module.blocks.{block_idx}.mlp.moe_router_list.{router_idx}
                parts = key.split('.')
                router_idx = int(parts[5])

                # block_idx = int(parts[2])
                # if block_idx < 6:
                #     continue

                # block_idx = int(parts[2])
                # if block_idx % 2 == 0:  # 偶数块
                #     continue

                old_router_idx = router_idx % 1  # 这里旧模型只有1个，因此始终为0
                old_key = '.'.join(parts[:5] + [str(old_router_idx)] + parts[6:])

                # ParameterBank stores each tensor as ``.param`` while older
                # checkpoints store the tensor directly under the index.
                if old_key not in state_dict and len(parts) > 6:
                    old_key = '.'.join(parts[:5] + [str(old_router_idx)])

                if old_key in state_dict:
                    updated_state_dict[key] = state_dict[old_key]
                else:
                    print(f"Warning: {old_key} not found in checkpoint, using original param.")
                    updated_state_dict[key] = param

        model.load_state_dict(updated_state_dict, strict=False)
        # model = torch.nn.DataParallel(model) 

        logger.info("model has update param")
    else:
        raise ValueError("The clean project requires TEST.moe_new=True")


    model.to(device)
    model.train()
    return model
