import torch
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import hilbert 
import os

def peak_amplitude(dataloader, model, device = "cpu", min_snr = 1, save_folder =''):
    peak_amplitudes_noisy = { 'X Channel': [], 'Y Channel': [], 'Z Channel': [] }
    peak_amplitudes_clean = { 'X Channel': [], 'Y Channel': [], 'Z Channel': [] }
    peak_amplitudes_denoised = { 'X Channel': [], 'Y Channel': [], 'Z Channel': [] }

    snr_values = {'X Channel': [], 'Y Channel': [], 'Z Channel': []}
    device = torch.device(device)
    model = model.to(device)
    model.eval()

    with torch.no_grad():
        for noisy_data, clean_data in dataloader:
            noisy_data, clean_data = noisy_data.to(device), clean_data.to(device)
            denoised_output = model(noisy_data)

            sample_idx = 0  # Index of the sample to plot
            channel_names = ['X Channel', 'Y Channel', 'Z Channel']
            
            for channel_idx in range(3): 
                # Convert tensors to numpy for processing
                clean_np = clean_data[sample_idx, channel_idx].cpu().numpy()
                noisy_np = noisy_data[sample_idx, channel_idx].cpu().numpy()
                denoised_np = denoised_output[sample_idx, channel_idx].cpu().numpy()

                if np.std(noisy_np) != 0:
                    snr = np.max(clean_np) / np.std(noisy_np)
                else:
                    snr = float('inf')


                if snr > min_snr:
    
                    # Calculate Hilbert envelopes and find peaks
                    envelope_clean = np.abs(hilbert(clean_np))
                    envelope_noisy = np.abs(hilbert(noisy_np))
                    envelope_denoised = np.abs(hilbert(denoised_np))

                    peak_amplitude_noisy = np.max(envelope_noisy)
                    peak_amplitude_clean = np.max(envelope_clean)
                    peak_amplitude_denoised = np.max(envelope_denoised)

                    peak_amplitudes_clean[channel_names[channel_idx]].append(peak_amplitude_clean)
                    peak_amplitudes_noisy[channel_names[channel_idx]].append(peak_amplitude_noisy)
                    peak_amplitudes_denoised[channel_names[channel_idx]].append(peak_amplitude_denoised)
                    
                    snr_values[channel_names[channel_idx]].append(snr)

    plt.figure(figsize=(15, 18)) 

    for i, channel in enumerate(channel_names):
        if len(peak_amplitudes_clean[channel]) > 0:
        # Plotting peak Amplitude comparison with noisy
            plt.subplot(3, 2, 2*i+1)
            scatter = plt.scatter(peak_amplitudes_clean[channel], peak_amplitudes_noisy[channel], 
                                c=snr_values[channel], cmap='viridis', alpha=0.6, label='Noisy vs Clean')
            plt.plot([min(peak_amplitudes_clean[channel]), max(peak_amplitudes_clean[channel])], 
                    [min(peak_amplitudes_clean[channel]), max(peak_amplitudes_clean[channel])], 
                    'blue', linestyle='--', linewidth=2, label='x=y')
            # plt.xlim(1660,1680)
            # plt.ylim(1660,1680)
            plt.colorbar(scatter, label='SNR')
            plt.xlabel('Peak Amplitude of Clean Data(µV)')
            plt.ylabel('Peak Amplitude of Noisy Data(µV)')
            plt.title(f'Noisy vs Clean - {channel}')
            plt.xscale('log')
            plt.yscale('log')
            plt.legend()
            plt.grid(True)

            # Plotting peak Amplitude comparison with denoised
            plt.subplot(3, 2, 2*i+2)
            scatter = plt.scatter(peak_amplitudes_clean[channel], peak_amplitudes_denoised[channel], 
                                c=snr_values[channel], cmap='viridis', alpha=0.6, label='Denoised vs Clean')
            plt.plot([min(peak_amplitudes_clean[channel]), max(peak_amplitudes_clean[channel])], 
                    [min(peak_amplitudes_clean[channel]), max(peak_amplitudes_clean[channel])], 
                    'blue', linestyle='--', linewidth=2, label='x=y')
            # plt.xlim(1660,1680)
            # plt.ylim(1660,1680)
            plt.colorbar(scatter, label='SNR')
            plt.xlabel('Peak Amplitude of Clean Data (µV)')
            plt.ylabel('Peak Amplitude of Denoised Data (µV)')
            plt.title(f'Denoised vs Clean - {channel}')
            plt.xscale('log')
            plt.yscale('log')
            plt.legend()
            plt.grid(True)

    plt.tight_layout()
    plt.savefig(os.path.join(save_folder, f'Peak_Amplitude.png'))
    plt.close()
    print('peak-amplitude graph had been saved')


def peak_time(dataloader, model, device = "cpu", min_snr = 1, save_folder = ''):
    peak_times_noisy = { 'X Channel': [], 'Y Channel': [], 'Z Channel': [] }
    peak_times_clean = { 'X Channel': [], 'Y Channel': [], 'Z Channel': [] }
    peak_times_denoised = { 'X Channel': [], 'Y Channel': [], 'Z Channel': [] }

    snr_values = {'X Channel': [], 'Y Channel': [], 'Z Channel': []}
    device = torch.device(device)
    model = model.to(device)
    model.eval()
    with torch.no_grad():
        for noisy_data, clean_data in dataloader:
            noisy_data, clean_data = noisy_data.to(device), clean_data.to(device)
            denoised_output = model(noisy_data)

            sample_idx = 0  # Index of the sample to plot
            channel_names = ['X Channel', 'Y Channel', 'Z Channel']

            for channel_idx in range(3): 
                # Convert tensors to numpy for processing
                clean_np = clean_data[sample_idx, channel_idx].cpu().numpy()
                noisy_np = noisy_data[sample_idx, channel_idx].cpu().numpy()
                denoised_np = denoised_output[sample_idx, channel_idx].cpu().numpy()
                timing = np.array([i for i in range(clean_np.size)])
                snr = np.max(clean_np) / np.std(noisy_np)
                if snr > min_snr:
                    # Calculate Hilbert envelopes and find peaks
                    envelope_clean = np.abs(hilbert(clean_np))
                    envelope_noisy = np.abs(hilbert(noisy_np))
                    envelope_denoised = np.abs(hilbert(denoised_np))

                    peak_time_noisy = timing[np.argmax(envelope_noisy)]
                    peak_time_clean = timing[np.argmax(envelope_clean)]
                    peak_time_denoised = timing[np.argmax(envelope_denoised)]

                    # Store peak times for later plotting
                    peak_times_clean[channel_names[channel_idx]].append(peak_time_clean)
                    peak_times_noisy[channel_names[channel_idx]].append(peak_time_noisy)
                    peak_times_denoised[channel_names[channel_idx]].append(peak_time_denoised)

                    snr_values[channel_names[channel_idx]].append(snr)

    plt.figure(figsize=(15, 18))  
    for i, channel in enumerate(channel_names):
        # Plotting peak times comparison with noisy
        plt.subplot(3, 2, 2*i+1)
        scatter = plt.scatter(peak_times_clean[channel], peak_times_noisy[channel], 
                            c=snr_values[channel], cmap='viridis', alpha=0.6, label='Noisy vs Clean')
        plt.plot([min(peak_times_clean[channel]), max(peak_times_clean[channel])], 
                [min(peak_times_clean[channel]), max(peak_times_clean[channel])], 
                '--', label='x = y')  # 'b--' specifies a blue dashed line
        #plt.xlim(1650,1680)
        #plt.ylim(1650,1680)
        plt.colorbar(scatter, label='SNR')
        plt.xlabel('Peak Times of Clean Data (ns)')
        plt.ylabel('Peak Times of Noisy Data (ns)')
        plt.title(f'Noisy vs Clean - {channel}')
        # plt.xscale('log')
        # plt.yscale('log')
        plt.legend()
        plt.grid(True)
        
        # Plotting peak times comparison with denoised
        plt.subplot(3, 2, 2*i+2)
        scatter = plt.scatter(peak_times_clean[channel], peak_times_denoised[channel], 
                            c=snr_values[channel], cmap='viridis', alpha=0.6, label='Denoised vs Clean')
        plt.plot([min(peak_times_clean[channel]), max(peak_times_clean[channel])], 
                [min(peak_times_clean[channel]), max(peak_times_clean[channel])], 
                '--', label='x = y')  # 'b--' specifies a blue dashed line
        #plt.xlim(1650,1680)
        #plt.ylim(1650,1680)
        plt.colorbar(scatter, label='SNR')
        plt.xlabel('Peak Times of Clean Data (ns)')
        plt.ylabel('Peak Times of Denoised Data (ns)')
        plt.title(f'Denoised vs Clean - {channel}')
        # plt.xscale('log')
        # plt.yscale('log')
        plt.legend()
        plt.grid(True)

    plt.tight_layout()
    plt.savefig(os.path.join(save_folder, f'Peak_Time.png'))
    plt.close()
    print('peak-time graph had been saved')