import torch
import torch.nn as nn


class BlockLLM:
    def __init__(self, model, sparsity, patience, learning_rate, logger):
        self.model = model
        self.sparsity = sparsity  # 稀疏率 s
        self.patience = patience  # 参数选择的耐心参数 m
        self.learning_rate = learning_rate  # 学习率 η
        self.logger = logger
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.moment1 = {name: torch.zeros_like(param) for name, param in model.named_parameters()}
        self.moment2 = {name: torch.zeros_like(param) for name, param in model.named_parameters()}
        self.loss_history = []  # 存储损失值的历史 H
        self.beta1 = 0.9  # Adam的 β1 参数
        self.beta2 = 0.999  # Adam的 β2 参数
        self.epsilon = 1e-8  # Adam的 ε 参数

    def compute_gradient(self, data):
        # 前向传播与损失计算
        output = self.model(data['input'])
        loss = nn.CrossEntropyLoss()(output, data['label'])
        loss.backward()  # 反向传播
        gradients = {name: param.grad for name, param in self.model.named_parameters()}
        return loss.item(), gradients

    def select_parameters(self, gradients):
        # 参数选择逻辑
        layer_norms = {name: torch.norm(grad) for name, grad in gradients.items()}
        sorted_layers = sorted(layer_norms.items(), key=lambda item: item[1], reverse=True)

        total_params = sum(param.numel() for param in self.model.parameters())
        num_selected_params = int(total_params * (1 - self.sparsity))

        selected_layers = []
        current_count = 0

        for name, _ in sorted_layers:
            selected_layers.append(name)
            current_count += self.model.state_dict()[name].numel()
            if current_count >= num_selected_params:
                break

        # 生成二进制mask
        masks = {name: torch.zeros_like(grad) for name in selected_layers}
        for name in selected_layers:
            layer_grad = gradients[name]
            threshold = torch.quantile(layer_grad.abs(), 1 - self.sparsity)
            masks[name] = (layer_grad.abs() >= threshold).float()

        return masks, selected_layers

    def update_parameters(self, masks, selected_layers, gradients):
        # 使用选定的mask更新参数
        for name, param in self.model.named_parameters():
            if name in selected_layers:
                grad = gradients[name]
                mask = masks[name]

                # Adam 更新
                self.moment1[name] = self.beta1 * self.moment1[name] + (1 - self.beta1) * grad
                self.moment2[name] = self.beta2 * self.moment2[name] + (1 - self.beta2) * grad.pow(2)

                m1_unbiased = self.moment1[name] / (1 - self.beta1)
                m2_unbiased = self.moment2[name] / (1 - self.beta2)

                param_update = mask * (m1_unbiased / (m2_unbiased.sqrt() + self.epsilon))
                param.data -= self.learning_rate * param_update

    def train_step(self, data):
        loss, gradients = self.compute_gradient(data)
        self.loss_history.append(loss)

        if len(self.loss_history) >= self.patience and loss >= sum(self.loss_history[-self.patience:]) / self.patience:
            masks, selected_layers = self.select_parameters(gradients)
            self.loss_history = []  # 重置损失历史
        else:
            masks, selected_layers = {}, []

        self.update_parameters(masks, selected_layers, gradients)
        return loss
