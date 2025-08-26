% This script allows reproducing the examples shown in the technical report
% available in the PISCO GitHub repository at:
%     https://github.com/ralobos/PISCO
%
% The problem formulation and methods implemented by the associated software
% to this script were originally reported in:
%
% [2] R. A. Lobos, C.-C. Chan, J. P. Haldar. New Theory and Faster
%     Computations for Subspace-Based Sensitivity Map Estimation in
%     Multichannel MRI. IEEE Transactions on Medical Imaging 43:286-296, 2024.
%
% [3] R. A. Lobos, C.-C. Chan, J. P. Haldar. Extended Version of "New Theory
%     and Faster Computations for Subspace-Based Sensitivity Map Estimation in
%     Multichannel MRI", 2023, arXiv:2302.13431.
%     (https://arxiv.org/abs/2302.13431)
%
% [4] R. A. Lobos, X. Wang, R. T. L. Fung, Y. He, D. Frey, D. Gupta,
%     Z. Liu, J. A. Fessler, D. C. Noll. Spatiotemporal Maps for Dynamic MRI
%     Reconstruction, 2025, arXiv:2507.14429.
%     (https://arxiv.org/abs/2507.14429)
%
% The associated software to this script corresponds to a newer version of the
% original PISCO software, which is available at:
%   http://mr.usc.edu/download/pisco/
%
% V2.0: Rodrigo A. Lobos (rlobos@umich.edu)
% August, 2025.
%
% Placeholder: Copyright.
%
% =========================================================================


clear all
close all
clc

%% Loading data

load('./data/3D_GRE_data.mat')

[N1, N2, N3, Nc] = size(kData);

idata = fftshift(ifft(ifft(ifft(ifftshift(kData),[],1),[],2),[],3)); % data in the spatial domain

idata_sos = sqrt(sum(abs(idata).^2, 4)); % sum of squares image

slc = 34; % Slice to display

figure; 
imagesc(mdisp(abs(squeeze(idata(:, :, slc, :))))); 
axis image; 
axis tight; 
axis off; 
colormap gray; 
title(['Data in the spatial domain (all coils for one slice)']); 
clim([0 0.1]);

figure; 
imagesc(mdisp(idata_sos)); 
axis image; 
axis tight; 
axis off; 
colormap gray; 
title(['sum-of-squares data (all slices)']); 
clim([0 1]);

%% Selection of calibration data

cal_length = 32; % Length of each dimension of the calibration data

center_x = ceil(N1/2)+even_pisco(N1);
center_y = ceil(N2/2)+even_pisco(N2);
center_z = ceil(N3/2)+even_pisco(N3);

cal_index_x = center_x + [-floor(cal_length/2):floor(cal_length/2)-even_pisco(cal_length/2)];
cal_index_y = center_y + [-floor(cal_length/2):floor(cal_length/2)-even_pisco(cal_length/2)];
cal_index_z = center_z + [-floor(cal_length/2):floor(cal_length/2)-even_pisco(cal_length/2)];

kCal = kData(cal_index_x, cal_index_y, cal_index_z, :);

%% Nullspace-based algorithm parameters

dim_sens = [N1, N2, N3];               % Desired dimensions for the estimated sensitivity maps

tau = 3;                               % Kernel radius. Default: 3

threshold = 0.08;                      % Threshold for C-matrix singular values. Default: 0.05
                                       % Note: In this example we don't use the default value.

M = 30;                                % Number of iterations for Power Iteration. Default: 30

PowerIteration_flag_convergence = 1;  % Binary variable. 1 = convergence error is displayed 
                                       % for Power Iteration if the method has not converged 
                                       % for some voxels after the iterations indicated by 
                                       % the user. In this example it corresponds to an empty 
                                       % array which indicates that the default value is 
                                       % being used. Default: 1

PowerIteration_flag_auto = 0;         % Binary variable. 1 = Power Iteration is run until
                                       % convergence in case the number of iterations
                                       % indicated by the user is too small. In this example
                                       % this variable corresponds to an empty array which
                                       % indicates that the default value is being used.
                                       % Default: 0

interp_zp = 24;                        % Amount of zero-padding to create the low-resolution grid 
                                       % if FFT-interpolation is used. In this example it
                                       % corresponds to an empty array which indicates that the
                                       % default value is being used. Default: 24

gauss_win_param = 100;                  % Parameter for the Gaussian apodizing window used to 
                                       % generate the low-resolution image in the FFT-based 
                                       % interpolation approach. This is the reciprocal of the 
                                       % standard deviation of the Gaussian window. In this 
                                       % example it corresponds to an empty array which indicates 
                                       % that the default value is being used. Default: 100

sketch_dim = 500;                       % Dimension of the sketch matrix used to calculate a
                                       % basis for the nullspace of the C matrix using a sketched SVD. 
                                       % In this example it corresponds to an empty array which indicates
                                       % that the default value is being used. Default: 500

visualize_C_matrix_sv = 1;             % Binary variable. 1 = Singular values of the C matrix are displayed.
                                       % Default: 0. 
                                       % Note: In this example we set it to 1 to visualize the singular values
                                       % of the C matrix. If sketched_SVD = 1 and if the curve of the singular values flattens out,
                                       % it suggests that the sketch dimension is appropriate for the data.

%% PISCO techniques

% The following techniques are used if the corresponding binary variable is equal to 1

kernel_shape = 1;                      % Binary variable. 1 = ellipsoidal shape is adopted for 
                                       % the calculation of kernels (instead of rectangular shape).
                                       % Default: 1

FFT_nullspace_C_calculation = 1;       % Binary variable. 1 = FFT-based calculation of nullspace 
                                       % vectors of C by calculating C'*C directly (instead of 
                                       % calculating C first). Default: 1

sketched_SVD = 1;                      % Binary variable. 1 = sketched SVD is used to calculate 
                                       % a basis for the nullspace of the C matrix (instead of 
                                       % calculating the nullspace vectors directly and then the 
                                       % basis). Default: 1

PowerIteration_G_nullspace_vectors = 1; % Binary variable. 1 = Power Iteration approach is 
                                        % used to find nullspace vectors of the G matrices 
                                        % (instead of using SVD). Default: 1

FFT_interpolation = 1;                 % Binary variable. 1 = sensitivity maps are calculated on 
                                       % a small spatial grid and then interpolated to a grid with 
                                       % nominal dimensions using an FFT-approach. Default: 1

verbose = 1;                           % Binary variable. 1 = PISCO information is displayed. 
                                       % Default: 1

%% PISCO estimation

if isempty(which('PISCO_sensitivity_maps_estimation'))
    error(['The function PISCO_senseMaps_estimation.m is not found in your MATLAB path. ' ...
           'Please ensure that all required files are available and added to the path.']);
end

[senseMaps, eigenValues] = PISCO_sensitivity_maps_estimation( ...
    kCal, ...
    dim_sens, ...                          % Data and output size
    'tau', tau, ...
    'threshold', threshold, ...
    'kernel_shape', kernel_shape, ...            % Kernel and threshold parameters
    'FFT_nullspace_C_calculation', FFT_nullspace_C_calculation, ...             % FFT nullspace calculation flag
    'PowerIteration_G_nullspace_vectors', PowerIteration_G_nullspace_vectors, ...      % Power Iteration flag
    'M', M, ...
    'PowerIteration_flag_convergence', PowerIteration_flag_convergence, ...      % Power Iteration params
    'PowerIteration_flag_auto', PowerIteration_flag_auto, ...                % Power Iteration auto flag
    'FFT_interpolation', FFT_interpolation, ...
    'interp_zp', interp_zp, ...
    'gauss_win_param', gauss_win_param, ... % Interpolation params
    'sketched_SVD', sketched_SVD, ...
    'sketch_dim', sketch_dim, ...
    'visualize_C_matrix_sv', visualize_C_matrix_sv, ... % SVD/sketching params
    'verbose', verbose ...                                  % Verbosity
);

%% Support mask created from the last eigenvalues of the G matrices 

threshold_mask = 0.6;

eig_mask = zeros(N1, N2, N3);
eig_mask(find(eigenValues(:,:,:,end) < threshold_mask)) = 1;

% Optional masking step

senseMaps_masked = senseMaps.*eig_mask;

%% Estimated Sensitivity Maps 

figure; 
imagesc(mdisp(abs(squeeze(senseMaps(:, :, slc, :))))); 
axis tight; 
axis image; 
axis off;
colormap gray; 
title('Estimated sensitivity maps -- One slice');

figure; 
imagesc(mdisp(abs(squeeze(senseMaps_masked(:, :, slc, :))))); 
axis tight; 
axis image; 
axis off;
colormap gray; 
title('Masked sensitivity maps -- One slice');

figure; 
imagesc(mdisp(angle(squeeze(senseMaps_masked(:, :, slc, :))))); 
axis tight; 
axis image; 
axis off;
colormap gray; 
title('Masked sensitivity maps (Phase) -- One slice');


if PowerIteration_G_nullspace_vectors == 1 
    title_eig_values = 'Smallest eigenvalue of normalized G matrices (spatial map) -- All slices';
    figure; 
    imagesc(mdisp(abs(eigenValues))); 
    axis tight; 
    axis image; 
    axis off;
    colormap gray; 
    colorbar; 
    title(title_eig_values); 
else
    title_eig_values = 'Eigenvalues of normalized G matrices (spatial maps) -- One slice';
    figure; 
    imagesc(mdisp(abs(squeeze(eigenValues(:,:,slc,:))))); 
    axis tight; 
    axis image; 
    axis off;
    colormap gray; 
    colorbar; 
    title(title_eig_values); 
end

figure; 
imagesc(mdisp(eig_mask)); 
axis tight; 
axis image; 
axis off;
colormap gray; 
title('Support mask -- All slices');

%% Auxiliary functions

function result = even_pisco(int)
% Function that checks if an integer is even.
%
% Input parameters:
%   --int:    Integer value to be checked.
%
% Output parameters:
%   --result: Logical value. Returns 1 (true) if int is even, 0 (false) otherwise.
%
    result = not(rem(int,2));
end

function tempForDisplay = mdisp(x)
% Function used to visualize multichannel images.
% If the input corresponds to a 3D array of dimensions N1 x N2 x Nc, the
% output corresponds to a 2D array that displays Nc images of dimension
% N1 x N2.
%
% Input parameters:
%   --x:              3D array of size N1 x N2 x Nc, where Nc is the number
%                     of channels.
%
% Output parameters:
%   --tempForDisplay: 2D array that arranges the Nc images of size N1 x N2
%                     into a single displayable image.
%

    [N1,N2,Nc] = size(x);
    f = factor(Nc);if numel(f)==1;f = [1,f];end
    tempForDisplay = reshape(permute(reshape(x,[N1,N2,prod(f(1:floor(numel(f)/2))),...
        prod(f(floor(numel(f)/2)+1:end))]),[1,3,2,4]),[N1*prod(f(1:floor(numel(f)/2))),...
        N2*prod(f(floor(numel(f)/2)+1:end))]); 
end