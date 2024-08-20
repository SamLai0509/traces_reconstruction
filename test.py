from training_function import psnr
import sys
import matplotlib.pyplot as plt
import torch
import numpy as np 
import os 

def test(testloader, time, frequency, model, num_images = 200, device="cpu", min_snr =1, save_folder ='', fft_mode = False,
         voltage = True, efield = False, ADC = False):
    with torch.no_grad():  
        device = torch.device(device)
        model = model.to(device)
        model.eval() 
        count = 0  # To count the number of images saved
        for noisy_data, clean_data in testloader:
            if count >= num_images: 
                break
            
            noisy_data, clean_data = noisy_data.to(device), clean_data.to(device)
            denoised_output = model(noisy_data)
            sample_idx = 0  # Index of the sample to plot
            channel_names = ['X Channel', 'Y Channel', 'Z Channel']
            
            for channel_idx in range(3): 
                clean_np = clean_data[sample_idx, channel_idx].cpu().numpy()
                noisy_np = noisy_data[sample_idx, channel_idx].cpu().numpy()
                denoised_np = denoised_output[sample_idx, channel_idx].cpu().numpy()
                snr = np.max(clean_np) / np.std(noisy_np)

                if 3 > snr > min_snr:
                    plt.figure(figsize=(25, 16))  # Set figure size for each channel
                    
                    # Metrics calculations
                    mse_value = np.mean((clean_np - denoised_np) ** 2)
                    psnr_value = psnr(clean_np, denoised_np, np.max(clean_np))

                    # Plot the clean signal
                    plt.subplot(2, 1, 1)


                    if fft_mode:
                        plt.title(f'FFT of the Signal- {channel_names[channel_idx]}, SNR:{snr:.2f}',fontsize = 18)
                        plt.yscale('log')
                        plt.plot(frequency[0][:4096], clean_np, label=f'Pure - {channel_names[channel_idx]}', color='blue')
                        plt.plot(frequency[0][:4096], denoised_np, label=f'Denoised - MSE: {mse_value:.2f}, PSNR: {psnr_value:.2f}', linestyle='--', color='orange')
                        plt.xlabel('Frequency (MHz)',fontsize = 18)
                        plt.ylabel('Amplitude (mV/MHz)',fontsize = 18)
                        plt.xlim(0, 300)  # Only plot up to Nyquist frequency, converted to MHz
                        plt.legend(fontsize=18)
                        plt.xticks(fontsize=18)
                        plt.yticks(fontsize=18)
                    elif voltage:
                        plt.plot(time[0], clean_np, label=f'Pure - {channel_names[channel_idx]}', color='blue')
                        plt.plot(time[0], denoised_np, label=f'Denoised - MSE: {mse_value:.2f}, PSNR: {psnr_value:.2f}', linestyle='--', color='orange')
                        plt.xlabel('Time (ns)',fontsize = 18)
                        plt.ylabel('Voltage (µV)',fontsize = 18)
                        plt.title(f'Denoised vs Pure Signal - {channel_names[channel_idx]}', fontsize=18)
                        plt.legend(fontsize=18)
                        plt.xticks(fontsize=18)
                        plt.yticks(fontsize=18)
                    elif ADC:
                        plt.plot(time[0], clean_np, label=f'Pure - {channel_names[channel_idx]}', color='blue')
                        plt.plot(time[0], denoised_np, label=f'Denoised - MSE: {mse_value:.2f}, PSNR: {psnr_value:.2f}', linestyle='--', color='orange')
                        plt.xlabel('Time (ns)',fontsize = 18)
                        plt.ylabel('Counts',fontsize = 18)
                        plt.title(f'Denoised vs Pure Signal - {channel_names[channel_idx]}', fontsize=18)
                        plt.legend(fontsize=18)
                        plt.xticks(fontsize=18)
                        plt.yticks(fontsize=18)
                    elif efield:
                        plt.plot(time[0], clean_np, label=f'Pure - {channel_names[channel_idx]}', color='blue')
                        plt.plot(time[0], denoised_np, label=f'Denoised - MSE: {mse_value:.2f}, PSNR: {psnr_value:.2f}', linestyle='--', color='orange')
                        plt.xlabel('Time (ns)',fontsize = 18)
                        plt.ylabel('Efield (µV/m)',fontsize = 18)
                        plt.title(f'Denoised vs Pure Signal - {channel_names[channel_idx]}', fontsize=18)
                        plt.legend(fontsize=18)
                        plt.xticks(fontsize=18)
                        plt.yticks(fontsize=18)

                    else:
                        sys.exit('You must select either voltage, adc, or efeild')

                    # Plot the noisy signal
                    plt.subplot(2, 1, 2)


                    if fft_mode:
                        plt.plot(frequency[0][:4096], noisy_np, label=f'Noisy signal - {channel_names[channel_idx]}, SNR = {snr:.2f}', color='red')
                        plt.title(f'FFT of the Signal - {channel_names[channel_idx]}, SNR:{snr:.2f}' ,fontsize = 18)
                        plt.yscale('log')
                        plt.xlabel('Frequency (MHz)',fontsize = 18)
                        plt.ylabel('Amplitude (mV/MHz)',fontsize = 18)
                        plt.xlim(0, 300)   # Only plot up to Nyquist frequency, converted to MHz
                        plt.legend(fontsize=18)
                        plt.xticks(fontsize=18)
                        plt.yticks(fontsize=18)  

                    elif voltage:
                        plt.plot(time[0], noisy_np, label=f'Noisy signal - {channel_names[channel_idx]}, SNR = {snr:.2f}', color='red')
                        plt.xlabel('Time (ns)',fontsize = 18)
                        plt.ylabel('Voltage (µV)',fontsize = 18)
                        plt.title(f'Noisy Signal - {channel_names[channel_idx]}, SNR:{snr:.2f}', fontsize=18)
                        plt.legend(fontsize=18)
                        plt.xticks(fontsize=18)
                        plt.yticks(fontsize=18)    
                    
                    elif efield: 
                        plt.plot(time[0], noisy_np, label=f'Noisy signal - {channel_names[channel_idx]}, SNR = {snr:.2f}', color='red')
                        plt.xlabel('Time (ns)',fontsize = 18)
                        plt.ylabel('Efield (µV/m)',fontsize = 18)
                        plt.title(f'Noisy Signal - {channel_names[channel_idx]}, SNR:{snr:.2f}', fontsize=18)
                        plt.legend(fontsize=18)
                        plt.xticks(fontsize=18)
                        plt.yticks(fontsize=18)
                    elif ADC:
                        plt.plot(time[0], noisy_np, label=f'Noisy signal - {channel_names[channel_idx]}, SNR = {snr:.2f}', color='red')
                        plt.xlabel('Time (ns)',fontsize = 18)
                        plt.ylabel('Counts',fontsize = 18)
                        plt.title(f'Noisy Signal - {channel_names[channel_idx]}, SNR:{snr:.2f}', fontsize=18)
                        plt.legend(fontsize=18)
                        plt.xticks(fontsize=18)
                        plt.yticks(fontsize=18)
                    else:
                        sys.exit('You must select either voltage, adc, or efeild')

                    # Save the figure
                    plt.tight_layout()
                    plt.savefig(os.path.join(save_folder, f'sample_{count:03d}_channel_{channel_idx}.png'))
                    plt.close()
                    
                    count += 1  # Increment the count

                    if count >= num_images:  # Stop after saving 100 images
                        break
            if count >= num_images:  # Stop after saving 100 images
                break
    print('test is completed')