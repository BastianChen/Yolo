from trainer import Trainer

if __name__ == '__main__':
    trainer = Trainer("models/net.pth")
    trainer = Trainer("models/net_0.8.pth")
    trainer.train()