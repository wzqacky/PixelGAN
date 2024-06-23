import argparse


def train_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_epochs', default=50, type=int)
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--gp', default=10, type=int, help='gradient penalty')
    parser.add_argument('--disc_iter', default=5, type=int, help='discriminator iterations')
    parser.add_argument('--save_model', default=True, type=bool)
    return parser 


def test_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='generator25000.pt', type=str, help='generator model to be used')
    return parser
    