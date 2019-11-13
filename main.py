from trainer import Trainer

if __name__ == '__main__':
    # trainer = Trainer("models/net.pth")
    # trainer = Trainer("models/net_0.8.pth")
    # trainer = Trainer("models/net_SGD_without_normal.pth")
    # trainer = Trainer("models/net_Adam_with_normal.pth")
    # trainer = Trainer("models/net_SGD_with_normal.pth")
    # trainer = Trainer("models/net_Adam_with_normal_new_net.pth")# 残差块2层
    # trainer = Trainer("models/net_Adam_with_normal_old_net.pth")  # 残差块3层
    # trainer = Trainer("models/net_Adam_tiny_GroupNorm_net.pth")  # 使用GroupNorm代替BatchNorm,使用yolov3-tiny代替yolov3
    # trainer = Trainer("models/net_Adam_add_net.pth")  # 使用add代替cat
    # trainer = Trainer("models/net_Adam_not_garbage.pth")  # 使用adam训练原样本
    # trainer = Trainer("models/net_Adam_garbage_new_stack_cls.pth")  # 使用adam以及tiny网络训练垃圾分类样本
    # trainer = Trainer("models/net_Adam_garbage_old_stack_cls.pth")
    trainer = Trainer("models/net_Adam_garbage_with_normal.pth")
    trainer.train()
