import time
import torch
import torch.backends.cudnn as cudnn
import torch.optim
from torch import nn
from models.SSRNET import SSRNET
from models.SingleCNN import SpatCNN, SpecCNN
from models.TFNet import TFNet, ResTFNet
from models.SSFCNN import SSFCNN, ConSSFCNN
from models.MSDCNN import MSDCNN
from models.SSINR import SSINR
from utils import *
from data_loader import build_datasets
from validate import validate
from train import train
import pdb
import args_parser
from torch.optim import lr_scheduler
from skimage.io.tests.test_mpl_imshow import plt
from torch.nn import functional as F


args = args_parser.args_parser()
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

print (args)


def main():
    # Custom dataloader
    train_list, test_list = build_datasets(args.root, 
                                           args.dataset, 
                                           args.image_size, 
                                           args.n_select_bands, 
                                           args.scale_ratio)
    if args.dataset == 'PaviaU':
      args.n_bands = 103
    elif args.dataset == 'Pavia':
      args.n_bands = 102
    elif args.dataset == 'Botswana':
      args.n_bands = 145
    elif args.dataset == 'KSC':
      args.n_bands = 176
    elif args.dataset == 'Urban':
      args.n_bands = 162
    elif args.dataset == 'IndianP':
      args.n_bands = 200
    elif args.dataset == 'Washington':
      args.n_bands = 191
    elif args.dataset == 'MUUFL_HSI':
      args.n_bands = 64
    elif args.dataset == 'salinas_corrected':
      args.n_bands = 204
    elif args.dataset == 'Houston_HSI':
      args.n_bands = 144
    # Build the models
    if args.arch == 'SSFCNN':
      model = SSFCNN(args.scale_ratio, 
                     args.n_select_bands, 
                     args.n_bands).cuda()
    elif args.arch == 'ConSSFCNN':
      model = ConSSFCNN(args.scale_ratio, 
                        args.n_select_bands, 
                        args.n_bands).cuda()
    elif args.arch == 'TFNet':
      model = TFNet(args.scale_ratio, 
                    args.n_select_bands, 
                    args.n_bands).cuda()
    elif args.arch == 'ResTFNet':
      model = ResTFNet(args.scale_ratio, 
                       args.n_select_bands, 
                       args.n_bands).cuda()
    elif args.arch == 'MSDCNN':
      model = MSDCNN(args.scale_ratio, 
                     args.n_select_bands, 
                     args.n_bands).cuda()
    elif args.arch == 'SSINR':
      model = SSINR(args.arch,
                     args.scale_ratio,
                     args.n_select_bands,
                     args.n_bands,
                     args.image_size).cuda()
    elif args.arch == 'SSRNET' or args.arch == 'SpatRNET' or args.arch == 'SpecRNET':
      model = SSRNET(args.arch,
                     args.scale_ratio,
                     args.n_select_bands, 
                     args.n_bands).cuda()
    elif args.arch == 'SpatCNN':
      model = SpatCNN(args.scale_ratio, 
                     args.n_select_bands, 
                     args.n_bands).cuda()
    elif args.arch == 'SpecCNN':
      model = SpecCNN(args.scale_ratio, 
                     args.n_select_bands, 
                     args.n_bands).cuda()

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    ####################学习率衰减#################
    scheduler = lr_scheduler.StepLR(optimizer, step_size=10000, gamma=0.9)
    #######################################
    # Load the trained model parameters
    model_path = args.model_path.replace('dataset', args.dataset) \
                                .replace('arch', args.arch) 
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path), strict=False)
        print ('Load the chekpoint of {}'.format(model_path))
        recent_psnr = validate(test_list, 
                                args.arch,
                                model,
                                0,
                                args.n_epochs)
        print ('psnr: ', recent_psnr)

    # Loss and Optimizer
    criterion = nn.MSELoss().cuda()


    best_psnr = 0

    best_psnr = validate(test_list,
                          args.arch, 
                          model,
                          0,
                          args.n_epochs)
    print ('psnr: ', best_psnr)

    # Epochs
    print ('Start Training: ')
    best_epoch=0
    lr_list = []
    for epoch in range(args.n_epochs):
        # One epoch's traininginceptionv3
        print ('Train_Epoch_{}: '.format(epoch))
        train(train_list, 
              args.image_size,
              args.scale_ratio,
              args.n_bands, 
              args.arch,
              model, 
              optimizer, 
              criterion, 
              epoch, 
              args.n_epochs)

        # One epoch's validation
        print ('Val_Epoch_{}: '.format(epoch))
        recent_psnr = validate(test_list, 
                                args.arch,
                                model,
                                epoch,
                                args.n_epochs)
        print ('psnr: ', recent_psnr)
        ########################
        scheduler.step()
        lr_list.append(optimizer.state_dict()['param_groups'][0]['lr'])
        ########################
        # # save model
        is_best = recent_psnr > best_psnr
        best_psnr = max(recent_psnr, best_psnr)
        print('best_psnr=======================================>  ', best_psnr)
        print('best_epoch======================================>  ', best_epoch)
        if is_best:
            best_epoch = epoch
            if best_psnr > 44:
                torch.save(model.state_dict(), model_path)
                print ('Saved!')
                print ('')

    print ('best_psnr: ', best_psnr)
    plt.plot(range(args.n_epochs), lr_list, color='r')
    plt.show()
if __name__ == '__main__':
    main()
