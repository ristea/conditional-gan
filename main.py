import torch
from torch.autograd import Variable
import numpy as np
from networks import Generator
from torchvision.utils import save_image
import train as T


def view_all():
    gen = Generator(T.H, T.W)
    checkpoint = torch.load('./latest_checkpoint.pkl', map_location='cpu')
    gen.load_state_dict(checkpoint['G_weights'])
    gen.eval()

    noise = Variable(torch.FloatTensor(np.random.normal(0, 1, (10 ** 2, 100))).to('cpu'))

    # Labels
    y_ = torch.LongTensor(np.array([num for num in range(10)])).view(10, 1).expand(-1, 10).contiguous()
    y_fixed = torch.zeros(10 ** 2, 10)
    y_fixed = Variable(y_fixed.scatter_(1, y_.view(10 ** 2, 1), 1).to('cpu'))

    gen_imgs = gen(noise, y_fixed).view(-1, 1, 28, 28)

    save_image(gen_imgs.data, './test.png', nrow=10, normalize=True)


def view_digit(digit=5):
    gen = Generator(T.H, T.W)
    checkpoint = torch.load('./latest_checkpoint.pkl', map_location='cpu')
    gen.load_state_dict(checkpoint['G_weights'])
    gen.eval()

    noise = Variable(torch.FloatTensor(np.random.normal(0, 1, (10 ** 2, 100))).to('cpu'))

    # Labels
    y_ = torch.LongTensor(np.array([digit for num in range(10)])).view(10, 1).expand(-1, 10).contiguous()
    y_fixed = torch.zeros(10 ** 2, 10)
    y_fixed = Variable(y_fixed.scatter_(1, y_.view(10 ** 2, 1), 1).to('cpu'))

    gen_imgs = gen(noise, y_fixed).view(-1, 1, 28, 28)

    save_image(gen_imgs.data, './test.png', nrow=10, normalize=True)


if __name__ == '__main__':
    view_digit()
