clear all
% close all
clc

%% Loading data

addpath('/Users/rodrigolobos/Library/CloudStorage/Dropbox/Postdoc/Code/2025_PISCO/PISCO_v20/utils')

load('/Users/rodrigolobos/Library/CloudStorage/Dropbox/Postdoc/Code/2025_PISCO/PISCO_v20/data/3D_GRE_data.mat')

[N1, N2, N3, Nc] = size(kData);

idata = fftshift(ifft(ifft(ifft(ifftshift(kData),[],1),[],2),[],3)); % data in the spatial domain

idata_sos = sqrt(sum(abs(idata).^2, 4)); % sum of squares image

slc = 34; % Slice to display

figure; 
imagesc(utils.mdisp(abs(squeeze(idata(:, :, slc, :))))); 
axis image; 
axis tight; 
axis off; 
colormap gray; 
title(['Data in the spatial domain (all coils for one slice)']); 
clim([0 0.1]);

figure; 
imagesc(utils.mdisp(idata_sos)); 
axis image; 
axis tight; 
axis off; 
colormap gray; 
title(['sum-of-squares data (all slices)']); 
clim([0 1]);

%% Selection of calibration data

cal_length = 32; % Length of each dimension of the calibration data

center_x = ceil(N1/2)+utils.even_pisco(N1);
center_y = ceil(N2/2)+utils.even_pisco(N2);
center_z = ceil(N3/2)+utils.even_pisco(N3);

cal_index_x = center_x + [-floor(cal_length/2):floor(cal_length/2)-utils.even_pisco(cal_length/2)];
cal_index_y = center_y + [-floor(cal_length/2):floor(cal_length/2)-utils.even_pisco(cal_length/2)];
cal_index_z = center_z + [-floor(cal_length/2):floor(cal_length/2)-utils.even_pisco(cal_length/2)];

kCal = kData(cal_index_x, cal_index_y, cal_index_z, :);

%% C-matrix parameters

tau = 3; 
kernel_shape = 0;

%% C-matrix previous implementation

[N1, N2, N3, Nc] = size(kCal);

x = reshape(kCal,N1*N2*N3,Nc);

[in1, in2, in3] = ndgrid(-tau:tau, -tau:tau, -tau:tau);

if kernel_shape == 1
    mask = (in1.^2 + in2.^2 + in3.^2 <= tau^2);
else
    mask = true(size(in1));
end
i = find(mask);

in1 = in1(i)';
in2 = in2(i)';
in3 = in3(i)';

patchSize = numel(in1);

% Time legacy implementation
C = zeros((N1-2*tau-utils.even_pisco(N1))*(N2-2*tau-utils.even_pisco(N2))*(N3-2*tau-utils.even_pisco(N3)),patchSize*Nc,'like',x);
t_C_legacy = tic;

l = 0;

for k = tau+1+utils.even_pisco(N3):N3-tau
    for i = tau+1+utils.even_pisco(N1):N1-tau
        for j = tau+1+utils.even_pisco(N2):N2-tau
            l = l+1;
            ind = sub2ind([N1,N2,N3],i+in1,j+in2,k+in3);
            C(l,:) = utils.vect(x(ind,:));
        end
    end
end
t_C_legacy = toc(t_C_legacy);

%% C-matrix new implementation (vectorization)

   % Fully vectorized approach for 3D - create all indices at once
k_centers = tau+1+utils.even_pisco(N3):N3-tau;
i_centers = tau+1+utils.even_pisco(N1):N1-tau;
j_centers = tau+1+utils.even_pisco(N2):N2-tau;

% Create all center positions using ndgrid to match nested loop order
% Legacy loop: for k (outer), for i (middle), for j (inner) => j varies fastest
[J_centers, I_centers, K_centers] = ndgrid(j_centers, i_centers, k_centers);
centers = [I_centers(:), J_centers(:), K_centers(:)];  % [i,j,k] triplets with j fastest

% Create all patch offsets for all centers simultaneously
numCenters = size(centers, 1);
I_all = centers(:,1) + in1; % implicit expansion -> [numCenters x patchSize]
J_all = centers(:,2) + in2; % implicit expansion -> [numCenters x patchSize]
K_all = centers(:,3) + in3; % implicit expansion -> [numCenters x patchSize]

% Time new implementation
t_C_new = tic;

% Convert to linear indices and extract data
ind_all = sub2ind([N1,N2,N3], I_all, J_all, K_all);
x_patches = x(ind_all, :);  % Shape: [numCenters*patchSize, Nc]

% Reshape to [numCenters, patchSize, Nc]
x_reshaped = reshape(x_patches, [numCenters, patchSize, Nc]);

% Flatten each patch in the same way as utils.vect (column-wise)
C_new = reshape(x_reshaped, [numCenters, patchSize*Nc]);
t_C_new = toc(t_C_new);

%% Verification: C_new equals C

fprintf('\nVerification C vs C_{new} (3D)\n');
fprintf('--------------------------------\n');
fprintf('C size:      %s\n', mat2str(size(C)));
fprintf('C_new size:  %s\n', mat2str(size(C_new)));

if ~isequal(size(C), size(C_new))
    warning('Sizes differ between C and C_new.');
end

diffC = C - C_new;
maxAbsErr = max(abs(diffC(:)));
relErr = norm(diffC(:)) / max(1, norm(C(:)));
fprintf('Max abs error: %.3e\n', maxAbsErr);
fprintf('Rel. L2 error: %.3e\n', relErr);

tol = 1e-12;
if maxAbsErr <= tol || relErr <= tol
    fprintf('Status: MATCH within tolerance %.1e\n', tol);
else
    fprintf('Status: MISMATCH (tighter centering/order needed)\n');
end

%% Performance comparison

fprintf('\nPerformance (seconds)\n');
fprintf('---------------------\n');
fprintf('Legacy C:  %.6f s\n', t_C_legacy);
fprintf('Vector C:  %.6f s\n', t_C_new);
if t_C_new > 0
    fprintf('Speedup:   %.2fx\n', t_C_legacy / t_C_new);
end

%% Verifying C_fun and C_new

opts_C_matrix = struct( ...
    'tau', tau,...
    'kernel_shape', kernel_shape...
);

fn = fieldnames(opts_C_matrix);
fv = struct2cell(opts_C_matrix);
nv = [fn.'; fv.'];
nv = nv(:).';

C_fun = utils.C_matrix_3D(kCal, nv{:});

fprintf('\nVerification C_{fun} vs C_{new} (3D)\n');
fprintf('--------------------------------\n');
fprintf('C size:      %s\n', mat2str(size(C_fun)));
fprintf('C_new size:  %s\n', mat2str(size(C_new)));

if ~isequal(size(C_fun), size(C_new))
    warning('Sizes differ between C_fun and C_new.');
end

diffC = C_fun - C_new;
maxAbsErr = max(abs(diffC(:)));
relErr = norm(diffC(:)) / max(1, norm(C_fun(:)));
fprintf('Max abs error: %.3e\n', maxAbsErr);
fprintf('Rel. L2 error: %.3e\n', relErr);

tol = 1e-12;
if maxAbsErr <= tol || relErr <= tol
    fprintf('Status: MATCH within tolerance %.1e\n', tol);
else
    fprintf('Status: MISMATCH (tighter centering/order needed)\n');
end

%% ChC FFT-based implementation (old implementation)

[in1, in2, in3] = ndgrid(-tau:tau, -tau:tau, -tau:tau);

if kernel_shape == 1
    mask = (in1.^2 + in2.^2 + in3.^2 <= tau^2);
else
    mask = true(size(in1));
end
i = find(mask);

in1 = in1(i)';
in2 = in2(i)';
in3 = in3(i)';

patchSize = numel(in1);

pad = 1;

if pad == 1
    N1n = 2^(ceil(log2(N1+2*tau))); 
    N2n = 2^(ceil(log2(N2+2*tau)));
    N3n = 2^(ceil(log2(N3+2*tau)));
else
    N1n = N1;
    N2n = N2;
    N3n = N3;
end

inds = sub2ind([N1n,N2n,N3n], floor(N1n/2)+1-in1+in1', floor(N2n/2)+1-in2+in2',  floor(N3n/2)+1-in3+in3');

[n2,n1,n3] = meshgrid([-floor(N2n/2):floor(N2n/2)-utils.even_pisco(N2n/2)]/N2n, [-floor(N1n/2):floor(N1n/2)-utils.even_pisco(N1n/2)]/N1n, [-floor(N3n/2):floor(N3n/2)-utils.even_pisco(N3n/2)]/N3n);
phaseKernel = exp(complex(0,-2*pi)*(n1*(ceil(N1n/2)+tau)+n2*(ceil(N2n/2)+tau)+n3*(ceil(N3n/2)+tau)));
cphaseKernel = exp(complex(0,-2*pi)*(n1*(ceil(N1n/2))+n2*(ceil(N2n/2))+n3*(ceil(N3n/2))));

x = fft(fft(fft(kCal,N1n,1),N2n,2),N3n,3).*phaseKernel;

ChC_old = zeros(patchSize, patchSize, Nc,  Nc); 
for q = 1:Nc
    b= reshape(utils.ifft3(conj(x(:,:,:,q:Nc)).*x(:,:,:,q).*cphaseKernel),[],Nc-q+1); 
    ChC_old(:,:,q:Nc,q) = reshape(b(inds,:),patchSize,patchSize,Nc-q+1);
    ChC_old(:,:,q,q+1:Nc) = permute(conj(ChC_old(:,:,q+1:Nc,q)),[2,1,4,3]);
end
ChC_old = reshape(permute(ChC_old,[1,3,2,4]), patchSize*Nc, patchSize*Nc);

%% ChC FFT-based convolutions (new implementation analogous to 2D; no phase variables)

% Use frequency-domain correlations and spatial centering via circshift.

F = fft(fft(fft(kCal, N1n, 1), N2n, 2), N3n, 3);

tic; % timing: new implementation
ChC_new = zeros(patchSize, patchSize, Nc, Nc);
for q = 1:Nc
    A = conj(x(:,:,:,q:Nc)) .* x(:,:,:,q);             % correlations using x to mirror old numerics
    R = utils.ifft3(A);                                 % inverse 3D FFT
    R = circshift(R, [ceil(N1n/2), ceil(N2n/2), ceil(N3n/2)]); % center zero-lag
    b = reshape(R, [], Nc - q + 1);
    ChC_new(:, :, q:Nc, q) = reshape(b(inds, :), patchSize, patchSize, Nc - q + 1);
    ChC_new(:, :, q, q+1:Nc) = permute(conj(ChC_new(:, :, q+1:Nc, q)), [2, 1, 4, 3]);
end
ChC_new = reshape(permute(ChC_new, [1, 3, 2, 4]), patchSize * Nc, patchSize * Nc);
time_ChC_new = toc;

%% Verification: numerical equivalence ChC_old vs ChC_new

fprintf('\n=== 3D ChC VS ChC_{new} VERIFICATION ===\n');
fprintf('ChC_{old} size: [%d, %d]\n', size(ChC_old));
fprintf('ChC_{new} size: [%d, %d]\n', size(ChC_new));

if ~isequal(size(ChC_old), size(ChC_new))
    warning('Size mismatch between ChC_{old} and ChC_{new}.');
end

max_abs_err = max(abs(ChC_old(:) - ChC_new(:)));
rel_err = max_abs_err / max(1e-12, max(abs(ChC_old(:))));
tolerance = 1e-10;

fprintf('Max absolute error: %.3e\n', max_abs_err);
fprintf('Max relative error: %.3e\n', rel_err);
fprintf('Equal within tol (%.0e): %s\n', tolerance, mat2str(max_abs_err < tolerance));

%% Performance comparison (re-run old for timing)

tic; % timing: old implementation (phase kernels)
ChC_old_timed = zeros(patchSize, patchSize, Nc,  Nc); 
for q = 1:Nc
    b = reshape(utils.ifft3(conj(x(:,:,:,q:Nc)) .* x(:,:,:,q) .* cphaseKernel), [], Nc - q + 1);
    ChC_old_timed(:, :, q:Nc, q) = reshape(b(inds, :), patchSize, patchSize, Nc - q + 1);
    ChC_old_timed(:, :, q, q+1:Nc) = permute(conj(ChC_old_timed(:, :, q+1:Nc, q)), [2, 1, 4, 3]);
end
ChC_old_timed = reshape(permute(ChC_old_timed, [1, 3, 2, 4]), patchSize * Nc, patchSize * Nc);
time_ChC_old = toc;

fprintf('\n=== 3D ChC PERFORMANCE COMPARISON ===\n');
fprintf('Old impl time: %.6f s\n', time_ChC_old);
fprintf('New impl time: %.6f s\n', time_ChC_new);
if time_ChC_new > 0
    fprintf('Speedup (old/new): %.2fx\n', time_ChC_old / time_ChC_new);
end

%% Nullpsace vectors C-matrix

sketched_SVD = 1; % 0 = full SVD, 1 = sketched SVD
sketch_dim = 500;
visualize_C_matrix_sv = 1;

ChC = ChC_new; % C'*C matrix

threshold = 0.08;      

if sketched_SVD == 0

    [~,Sc,U] = svd(ChC,'econ');
    clear ChC
    sing = diag(Sc);
    clear Sc
    
    sing = sqrt(sing);
    sing  = sing/sing(1);

    if visualize_C_matrix_sv == 1

        % Visualize singular values of the C matrix
        figure;
        plot(sing, 'o-');
        title('Singular values of the C matrix');
        grid on;
        xlim([1 numel(sing)]);
        ylim([0 1]);
        xlabel('Index');
        ylabel('Singular value');

    end


    Nvect = find(sing >=threshold*sing(1),1,'last');
    clear sing
    U = U(:, Nvect+1:end); 

else

    %Sketching

    [~, N2c] = size(ChC);
    Sk = (1/sqrt(sketch_dim))*randn(sketch_dim, N2c) + (1/sqrt(sketch_dim))*1i*randn(sketch_dim, N2c);
    C = Sk*ChC;
    [~, sing, Vf] = svd(C, 'econ', 'vector');

    sing = sqrt(sing);
    sing  = sing/sing(1);

    if visualize_C_matrix_sv == 1

    
        figure;
        plot(sing, 'o-');
        title('Singular values of the C matrix (sketched SVD)');
        grid on;
        xlim([1 numel(sing)]);
        ylim([0 1]);
        xlabel('Index');
        ylabel('Singular value');
    end

    rank_C = find(sing >=threshold*sing(1),1,'last');
    clear sing

    U = Vf(:,1:rank_C);
    clear Vf
    
end

%% G matrices (old version)

kernel_shape = 1;
FFT_interpolation = 0;
interp_zp = 24;
FFT_nullspace_C_calculation = 1;

[N1_cal, N2_cal, N3_cal, Nc] = size(kCal);

% [in1, in2, in3] = ndgrid(-tau:tau, -tau:tau, -tau:tau);

% if kernel_shape == 1
%     mask = (in1.^2 + in2.^2 + in3.^2 <= tau^2);
% else
%     mask = true(size(in1));
% end
% i = find(mask); 

% in1 = in1(i)';
% in2 = in2(i)';
% in3 = in3(i)';

% patchSize = numel(in1);

% in1 = in1(:);
% in2 = in2(:);
% in3 = in3(:);

eind = [patchSize:-1:1]';

t_G_old = tic;

G_old = zeros(2*(2*tau+1)* 2*(2*tau+1)* 2*(2*tau+1),Nc,Nc);

if sketched_SVD == 0
    W = U*U';
else
    W = eye(size(U, 1)) - U*U';
end

U_copy = U; % preserve for G_new
clear U;
W = permute(reshape(W,patchSize,Nc,patchSize,Nc),[1,2,4,3]);

for s = 1:patchSize
    G_old(sub2ind([2*(2*tau+1),2*(2*tau+1),2*(2*tau+1)],2*tau+1+1+in1(eind)+in1(s),2*tau+1+1+in2(eind)+in2(s), 2*tau+1+1+in3(eind)+in3(s)),:,:) = ...
        G_old(sub2ind([2*(2*tau+1),2*(2*tau+1),2*(2*tau+1)],2*tau+1+1+in1(eind)+in1(s),2*tau+1+1+in2(eind)+in2(s),2*tau+1+1+in3(eind)+in3(s)),:,:)  + W(:,:,:,s);
end

clear W

if FFT_interpolation == 0
    
    N1_g = N1;
    N2_g = N2;
    N3_g = N3;
    
else
  
    if N1_cal <= N1 - interp_zp
        N1_g = N1_cal + interp_zp;
    else   
        N1_g = N1_cal;
    end

    if N2_cal <= N2 - interp_zp
        N2_g = N2_cal + interp_zp;
    else   
        N2_g = N2_cal;
    end

    if N3_cal <= N3 - interp_zp
        N3_g = N3_cal + interp_zp;
    else   
        N3_g = N3_cal;
    end
    
end

[n2, n1, n3] = meshgrid((-N2_g/2:N2_g/2-1)/N2_g, (-N1_g/2:N1_g/2-1)/N1_g, (-N3_g/2:N3_g/2-1)/N3_g);
phaseKernel = -exp(complex(0,-2*pi)*(n1*(N1_g-2*tau-1)+n2*(N2_g-2*tau-1)+n3*(N3_g-2*tau-1)));

G_old = utils.fft3(conj(reshape(G_old,2*(2*tau+1),2*(2*tau+1),2*(2*tau+1),Nc,Nc)),N1_g,N2_g,N3_g).*phaseKernel; 

G_old = fftshift(fftshift(fftshift(G_old,1),2),3);

if FFT_nullspace_C_calculation == 1
    % If the nullspace vectors of the C matrix were calculated using an FFT-based
    % approach, the G matrices are flipped in all three dimensions.
    G_old = flip(flip(flip(G_old, 2), 1), 3);
end

time_G_old = toc(t_G_old);

%% G matrices new version 
% Optimized, phase-free (uncentered) implementation analogous to 2D version
% - Vectorized spatial accumulation via accumarray on 3D grid
% - Pre-centering with spatial modulation (-1)^(i+j+k)
% - Uncentered frequency-domain phase kernel (no fftshift needed)

t_G_new = tic;

grid_size = 2 * (2 * tau + 1);

% Nullspace projector (match old path)
if sketched_SVD == 0
    Wn = U_copy * U_copy';
else
    Wn = eye(size(U_copy, 1)) - U_copy * U_copy';
end
Wn = permute(reshape(Wn, patchSize, Nc, patchSize, Nc), [1, 2, 4, 3]);

% Build 3D indices for all (p,s) pairs in one shot
offset = 2 * tau + 1 + 1;
base_r = offset + in1(eind);
base_c = offset + in2(eind);
base_d = offset + in3(eind);

row_mat = base_r(:) + in1; % [patchSize x patchSize], rows=p, cols=s
col_mat = base_c(:) + in2; % [patchSize x patchSize]
dep_mat = base_d(:) + in3; % [patchSize x patchSize]

% Precompute linear indices table [patchSize x patchSize]; column s picks rows for p
idx_tbl = sub2ind([grid_size, grid_size, grid_size], row_mat, col_mat, dep_mat);

% Accumulate like legacy: per s tile add into rows idx_tbl(:,s)
G_new = zeros(grid_size^3, Nc, Nc, 'like', Wn);
for s = 1:patchSize
    G_new(idx_tbl(:, s), :, :) = G_new(idx_tbl(:, s), :, :) + Wn(:, :, :, s);
end

clear row_mat col_mat dep_mat base_r base_c base_d idx_tbl Wn

% Frequency transform (centered), mirroring old pipeline exactly
Y = conj(reshape(G_new, grid_size, grid_size, grid_size, Nc, Nc));
s1 = N1_g - 2 * tau - 1;  s2 = N2_g - 2 * tau - 1;  s3 = N3_g - 2 * tau - 1;
[n2c_g, n1c_g, n3c_g] = meshgrid((-N2_g/2:N2_g/2-1)/N2_g, (-N1_g/2:N1_g/2-1)/N1_g, (-N3_g/2:N3_g/2-1)/N3_g);
phaseKernel_c = -exp(complex(0, -2 * pi) * (n1c_g * s1 + n2c_g * s2 + n3c_g * s3));
G_new = utils.fft3(Y, N1_g, N2_g, N3_g) .* phaseKernel_c;
G_new = fftshift(fftshift(fftshift(G_new, 1), 2), 3);

% Apply same flip policy as old path when FFT-based C nullspace was used
if FFT_nullspace_C_calculation == 1
    G_new = flip(flip(flip(G_new, 2), 1), 3);
end

time_G_new = toc(t_G_new);

%% G-matrices Verification (G_old vs G_new)

fprintf('\n=== 3D G-MATRICES VERIFICATION ===\n');
fprintf('G_old size: [%d, %d, %d, %d, %d]\n', size(G_old));
fprintf('G_new size: [%d, %d, %d, %d, %d]\n', size(G_new));

max_abs_err_G = max(abs(G_old(:) - G_new(:)));
if max(abs(G_old(:))) > 0
    rel_err_G = max_abs_err_G / max(abs(G_old(:)));
else
    rel_err_G = 0;
end
tolG = 1e-10;
fprintf('Max absolute error: %.3e\n', max_abs_err_G);
fprintf('Max relative error: %.3e\n', rel_err_G);
fprintf('Equal within tol (%.0e): %s\n', tolG, mat2str(max_abs_err_G < tolG));

%% G-matrices Performance Comparison

fprintf('\n=== 3D G-MATRICES PERFORMANCE COMPARISON ===\n');
fprintf('Old implementation time: %.6f s\n', time_G_old);
fprintf('New implementation time: %.6f s\n', time_G_new);
if time_G_new > 0
    fprintf('Speedup (old/new):    %.2fx\n', time_G_old / time_G_new);
end

%% G-matrices using the function

opts_G_matrices_3D = struct( ...
            'kernel_shape', 0, ...
            'FFT_interpolation', FFT_interpolation, ...
            'interp_zp', interp_zp, ...
            'sketched_SVD', sketched_SVD ...
        );

        fn = fieldnames(opts_G_matrices_3D);
        fv = struct2cell(opts_G_matrices_3D);
        nv = [fn.'; fv.'];
        nv = nv(:).';


G_fun = utils.G_matrices_3D(kCal, N1, N2, N3, tau, U_copy, FFT_nullspace_C_calculation, nv{:});

%% comparison G_fun vs G_new

max_abs_err_G_fun = max(abs(G_fun(:) - G_new(:)));
if max(abs(G_fun(:))) > 0
    rel_err_G_fun = max_abs_err_G_fun / max(abs(G_fun(:)));
else
    rel_err_G_fun = 0;
end
tolG_fun = 1e-10;
fprintf('Max absolute error (G_fun): %.3e\n', max_abs_err_G_fun);
fprintf('Max relative error (G_fun): %.3e\n', rel_err_G_fun);
fprintf('Equal within tol (%.0e): %s\n', tolG_fun, mat2str(max_abs_err_G_fun < tolG_fun));


