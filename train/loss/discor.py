import torch

def lip_motion_decorrelation_loss(A, B):
    batch_size, D = A.size()
    
    # 计算特征之间的相关性矩阵
    correlation_matrix = torch.matmul(A.t(), B)  # A的转置乘以B
    
    # 计算相关性的平方并求和
    squared_correlation = torch.square(correlation_matrix).sum()
    
    # 计算损失
    loss = squared_correlation / D
    
    return loss
