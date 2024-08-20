import argparse
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from scipy.signal import hilbert
from torch.utils.data import Dataset, DataLoader
import matplotlib as mpl
import matplotlib.pyplot as plt

from model import Autoencoder
from training_function import traces, CustomDataset, split_indices, plot_metrics, psnr_loss, get_peak_amplitude, calculate_psnr_with_peak, peak_to_peak_ratio, psnr
from train import train_validate
from test import test 
from hilbert import peak_amplitude, peak_time
# /home/923714256/0422_simulation/sim_Xiaodushan_20221025_220000_RUN0_CD_ZHAireS_0000
# /home/923714256/0422_simulation/sim_Xiaodushan_20221025_220000_RUN0_CD_ZHAireS_0000NJ

def main(args):
    current_directory = os.getcwd()
    save_folder = args.save_folder_name

    if not os.path.exists(save_folder):
        os.makedirs(current_directory + '/' + save_folder) 

    print("Empty Folder is created")   
    mpl.rcParams['figure.max_open_warning'] = 50

    ##### data preparation
    time, frequency, noised_trace_x, clean_trace_x, noised_trace_y, clean_trace_y, noised_trace_z, clean_trace_z = traces(
        args.directory, args.NJ_directory, nb_events=args.nb_events, mpe=args.mpe, 
        min_zenith=args.min_zenith, max_zenith=args.max_zenith, band_filter=args.band_filter
    )

    print(f'shape of time:{np.shape(time)}')
    print(f'shape of frequency{np.shape(frequency)}')
    print(f'shape of noised_trace_x:{np.shape(noised_trace_x)}')
    print(f'shape of noised_trace_y:{np.shape(noised_trace_y)}')
    print(f'shape of noised_trace_z:{np.shape(noised_trace_z)}')        
    print(f'shape of clean_trace_x:{np.shape(clean_trace_x)}')
    print(f'shape of clean_trace_y:{np.shape(clean_trace_y)}')
    print(f'shape of clean_trace_z:{np.shape(clean_trace_z)}')

    noised_signals = (noised_trace_x, noised_trace_y, noised_trace_z)
    clean_signals = (clean_trace_x, clean_trace_y, clean_trace_z)
    total_samples = len(noised_trace_x)
    train_indices, valid_indices, test_indices = split_indices(total_samples)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f'Device set to : {device}')
    train_dataset = CustomDataset(noised_signals, clean_signals, indices=train_indices)
    valid_dataset = CustomDataset(noised_signals, clean_signals, indices=valid_indices)
    test_dataset = CustomDataset(noised_signals, clean_signals,  indices=test_indices)

    train_loader = DataLoader(train_dataset, batch_size=4, num_workers=4, shuffle=True, pin_memory=True)
    valid_loader = DataLoader(valid_dataset, batch_size=4, num_workers=4, shuffle=True, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=1, num_workers=4, shuffle=False, pin_memory=True)

    model = Autoencoder(kernel_size=3).to(device)
    base_lr = 0.0001
    optimizer = optim.Adam(model.parameters(), lr=base_lr, weight_decay=0.0005)
        # Set the criterion based on the argument
    if args.criterion == 'mse':
        criterion = nn.MSELoss()
    elif args.criterion == 'psnr':
        criterion = psnr_loss 

    train_validate(train_loader=train_loader, valid_loader=valid_loader, model=model, optimizer=optimizer,
                   criterion=args.criterion, num_epochs=args.epochs, device=device, save_folder=save_folder)

    torch.save(model.state_dict(), current_directory + '/' + save_folder + 'model.pth')

    if args.test_mode:
        time, frequency, noised_trace_x, clean_trace_x, noised_trace_y, clean_trace_y, noised_trace_z, clean_trace_z = traces(
            args.directory, args.NJ_directory, nb_events=args.nb_events, mpe=args.mpe, 
            min_zenith=args.min_zenith, max_zenith=args.max_zenith, voltage=False, ADC=True, 
            efield = False, random_mode=True
        )
        print(f'shape of noised_time:{np.shape(time)}')
        print(f'shape of noised_trace_x:{np.shape(noised_trace_x)}')
        print(f'shape of noised_trace_y:{np.shape(noised_trace_y)}')
        print(f'shape of noised_trace_z:{np.shape(noised_trace_z)}')        
        print(f'shape of clean_trace_x:{np.shape(clean_trace_x)}')
        print(f'shape of clean_trace_y:{np.shape(clean_trace_y)}')
        print(f'shape of clean_trace_z:{np.shape(clean_trace_z)}')

        test_dataset = CustomDataset(noised_signals, clean_signals, indices=test_indices)    
        noised_signals = (noised_trace_x, noised_trace_y, noised_trace_z)
        clean_signals = (clean_trace_x, clean_trace_y, clean_trace_z)
        total_samples = len(noised_trace_x)
        train_indices, valid_indices, test_indices = split_indices(total_samples)
        test_dataset = CustomDataset(noised_signals, clean_signals, indices=test_indices)
        test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, pin_memory=True)

        test(time=time, frequency=frequency, testloader=test_loader, model=model, device="cpu", min_snr=args.min_snr, save_folder=save_folder)
        peak_amplitude(dataloader=test_loader, model=model, device="cpu", min_snr=args.min_snr, save_folder=save_folder)
        peak_time(dataloader=test_loader, model=model, device="cpu", min_snr=args.min_snr, save_folder=save_folder)
        print(f'All process are complete.')
    else:
        test(time=time, frequency=frequency, testloader=test_loader, model=model, device="cpu", min_snr=args.min_snr, save_folder=save_folder, voltage=True)
        peak_amplitude(dataloader=test_loader, model=model, device="cpu", min_snr=args.min_snr, save_folder=save_folder)
        peak_time(dataloader=test_loader, model=model, device="cpu", min_snr=args.min_snr, save_folder=save_folder)
        print(f'All process are complete.')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process some traces.')
    parser.add_argument('--directory', type=str, required=True, help='Path to the main directory')

    parser.add_argument('--NJ_directory', type=str, required=True, help='Path to the NJ directory')

    parser.add_argument('--nb_events', type=int, default=1000, help='Number of events')

    parser.add_argument('--criterion', default='mse', choices=['mse', 'psnr'], help='Loss function: either MSE or PSNR')

    parser.add_argument('--mpe', type=float, default=1e9, help='MPE value')

    parser.add_argument('--epochs', type = float, default = 100, help ='nums of epochs')

    parser.add_argument('--min_zenith', type=float, default=80, help='Minimum zenith angle')

    parser.add_argument('--max_zenith', type=float, default=88, help='Maximum zenith angle')

    parser.add_argument('--band_filter', type=bool, default=True, help='Whether to apply a band filter')

    parser.add_argument('--save_folder_name', type=str, required=True, help='Folder to save results')

    parser.add_argument('--test_mode',default=False, help='Whether to run in test mode')
    
    parser.add_argument('--min_snr', type = float, default = 3, help ='minimum of snr for the data display')

    args = parser.parse_args()
    main(args)