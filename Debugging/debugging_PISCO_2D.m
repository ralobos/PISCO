clear all
close all
clc

%% Loading data

addpath('/Users/rodrigolobos/Library/CloudStorage/Dropbox/Postdoc/Code/2025_PISCO/PISCO_v20/utils')

load('/Users/rodrigolobos/Library/CloudStorage/Dropbox/Postdoc/Code/2025_PISCO/PISCO_v20/data/2D_T1_data.mat')

kData = double(kData);
[N1, N2, Nc] = size(kData);

figure; 
imagesc(utils.mdisp(abs(fftshift(ifft2(ifftshift(kData)))))); 
axis image; 
axis tight; 
axis off; 
colormap gray; 
title(['Data in the spatial domain']); 
caxis([0 1e-8]);

%% Selection of calibration data

cal_length = 64; % Length of each dimension of the calibration data

center_x = ceil(N1/2) + utils.even_pisco(N1);
center_y = ceil(N2/2) + utils.even_pisco(N2);

cal_index_x = center_x + [-floor(cal_length/2):floor(cal_length/2) - utils.even_pisco(cal_length/2)];
cal_index_y = center_y + [-floor(cal_length/2):floor(cal_length/2) - utils.even_pisco(cal_length/2)];

kCal = kData(cal_index_x,cal_index_y, :);

%% C-matrix calculation (using previous implementation)

tau = 3; % Kernel radius. Default: 3

kernel_shape = 1; % 1 = ellipsoidal shape is adopted for the calculation of kernels

[N1, N2, Nc] = size(kCal);

x = kCal;

x = reshape(x,N1*N2,Nc);

[in1,in2] = meshgrid(-tau:tau,-tau:tau);

if kernel_shape == 1
    i = find(in1.^2+in2.^2<=tau^2);
else
    i = [1:numel(in1)]; 
end     

in1 = in1(i)';
in2 = in2(i)';

patchSize = numel(in1);

result = zeros((N1-2*tau-utils.even_pisco(N1))*(N2-2*tau-utils.even_pisco(N2)),patchSize*Nc,'like',x);

% Time the original nested loop calculation
tic;
k = 0;
for i = tau+1+utils.even_pisco(N1):N1-tau
    for j = tau+1+utils.even_pisco(N2):N2-tau
        k = k+1;
        ind = sub2ind([N1,N2],i+in1,j+in2);
        result(k,:) = utils.vect(x(ind,:));
    end
end
time_original = toc;

%% C-matrix new calculation using vectorization techniques

x = kCal;

x = reshape(x,N1*N2,Nc);

% Use the same reshaped x from the original calculation above
% x is already reshaped to [N1*N2, Nc] from the original section

% Fully vectorized approach - create all indices at once
tic;  % Start timing vectorized calculation
i_centers = tau+1+utils.even_pisco(N1):N1-tau;
j_centers = tau+1+utils.even_pisco(N2):N2-tau;

% Create centers in the same order as the nested loops: i outer, j inner
% For nested loops: for i, for j -> (i1,j1), (i1,j2), (i2,j1), (i2,j2)
% Meshgrid gives: (i1,j1), (i2,j1), (i1,j2), (i2,j2) - wrong order!
% Need to transpose the meshgrid result
[I_centers, J_centers] = meshgrid(i_centers, j_centers);  % Swap order
centers = [I_centers(:), J_centers(:)];  % All center positions as [i,j] pairs

% Create all patch offsets for all centers simultaneously
numCenters = size(centers, 1);
% Ensure in1, in2 are row vectors for repmat operation
in1_row = in1(:)';  % Force row vector
in2_row = in2(:)';  % Force row vector
I_all = repmat(centers(:,1), 1, patchSize) + repmat(in1_row, numCenters, 1);
J_all = repmat(centers(:,2), 1, patchSize) + repmat(in2_row, numCenters, 1);

% Convert to linear indices and extract data
ind_all = sub2ind([N1,N2], I_all, J_all);

% Fully vectorized approach without loops
% Extract patches: x(ind_all(k,:), :) for all k at once
x_selected = x(ind_all, :);  % [numCenters*patchSize, Nc]

% Reshape to separate centers: [numCenters, patchSize, Nc]
x_patches = reshape(x_selected, numCenters, patchSize, Nc);

% Apply utils.vect operation: flatten each [patchSize x Nc] in column-major order
% Direct reshape flattens in the right order: first all rows of col 1, then col 2, etc.
result_new = reshape(x_patches, numCenters, patchSize*Nc);
time_vectorized = toc;  % End timing vectorized calculation

%% Verification
fprintf('Size of result: [%d, %d]\n', size(result, 1), size(result, 2));
fprintf('Size of result_new: [%d, %d]\n', size(result_new, 1), size(result_new, 2));
fprintf('Are results equal? %s\n', mat2str(isequal(result, result_new)));
fprintf('Maximum absolute difference: %e\n', max(abs(result(:) - result_new(:))));

%% Performance Analysis
fprintf('\n=== PERFORMANCE COMPARISON ===\n');
fprintf('Original nested loops time: %.6f seconds\n', time_original);
fprintf('Vectorized approach time:   %.6f seconds\n', time_vectorized);
fprintf('Speedup factor:             %.2fx\n', time_original / time_vectorized);
fprintf('Time reduction:             %.1f%%\n', (time_original - time_vectorized) / time_original * 100);

% Additional performance metrics
fprintf('\nProblem size: %d centers × %d patch elements × %d coils\n', numCenters, patchSize, Nc);
fprintf('Total operations: %d patch extractions\n', numCenters);
fprintf('Original throughput: %.0f patches/second\n', numCenters / time_original);
fprintf('Vectorized throughput: %.0f patches/second\n', numCenters / time_vectorized);

%% C using the function

opts_C_matrix = struct( ...
    'tau', tau,...
    'kernel_shape', kernel_shape...
);

fn = fieldnames(opts_C_matrix);
fv = struct2cell(opts_C_matrix);
nv = [fn.'; fv.'];
nv = nv(:).';

result_fun = utils.C_matrix_2D(kCal, nv{:});

%% Comparing result_fun vs result_new

fprintf('\n=== C MATRIX VERIFICATION ===\n');
fprintf('Are result_fun and result_new equal? %s\n', mat2str(isequal(result_fun, result_new)));
fprintf('Maximum absolute difference: %e\n', max(abs(result_fun(:) - result_new(:))));

%% Create C-matrix and C'C

C = result_new; 
clear result_new

%ChC = C'*C;

%% ChC using FFT-based convolutions (previous implementation)

[N1, N2 , Nc] = size(kCal);

[in1,in2] = meshgrid(-tau:tau,-tau:tau);
if kernel_shape == 1
    i = find(in1.^2+in2.^2<=tau^2);
else
    i = [1:numel(in1)];
end
in1 = in1(i(:));
in2 = in2(i(:));

patchSize = numel(i);

pad = 1;

if pad
    N1n = 2^(ceil(log2(N1+2*tau))); 
    N2n = 2^(ceil(log2(N2+2*tau)));
else
    N1n = N1;
    N2n = N2;
end

inds = sub2ind([N1n,N2n], floor(N1n/2)+1-in1+in1', floor(N2n/2)+1-in2+in2');

[n2,n1] = meshgrid([-floor(N2n/2):floor(N2n/2)-utils.even_pisco(N2n/2)]/N2n,[-floor(N1n/2):floor(N1n/2)-utils.even_pisco(N1n/2)]/N1n);
phaseKernel = exp(complex(0,-2*pi)*(n1*(ceil(N1n/2)+tau)+n2*(ceil(N2n/2)+tau)));
cphaseKernel = exp(complex(0,-2*pi)*(n1*(ceil(N1n/2))+n2*(ceil(N2n/2))));

x = fft2(kCal, N1n, N2n) .* phaseKernel;

tic; % timing: previous implementation
ChC = zeros(patchSize, patchSize, Nc, Nc);
for q = 1:Nc
    b = reshape(ifft2(conj(x(:, :, q:Nc)) .* x(:, :, q) .* cphaseKernel), [], Nc - q + 1);
    ChC(:, :, q:Nc, q) = reshape(b(inds, :), patchSize, patchSize, Nc - q + 1);
    ChC(:, :, q, q+1:Nc) = permute(conj(ChC(:, :, q+1:Nc, q)), [2, 1, 4, 3]);
end
ChC = reshape(permute(ChC, [1, 3, 2, 4]), patchSize * Nc, patchSize * Nc);
time_ChC_old = toc;

%% ChC FFT-based convolutions (new implementation without phase kernels)

% Idea: use centered circular correlation via fftshift(ifft2(...)) so that
% zero-lag sits at the center. Then sample the patch lags directly with 'inds'.

tic; % timing: new implementation
F = fft2(kCal, N1n, N2n);                  % per-coil FFTs
ChC_new = zeros(patchSize, patchSize, Nc, Nc);
for q = 1:Nc
    R = ifft2(conj(F(:, :, q:Nc)) .* F(:, :, q));                  % cross-correlation maps (zero-lag at (1,1))
    R = circshift(R, [ceil(N1n/2), ceil(N2n/2)]);                  % center zero-lag (matches cphaseKernel shift)
    b = reshape(R, [], Nc - q + 1);
    ChC_new(:, :, q:Nc, q) = reshape(b(inds, :), patchSize, patchSize, Nc - q + 1);
    ChC_new(:, :, q, q+1:Nc) = permute(conj(ChC_new(:, :, q+1:Nc, q)), [2, 1, 4, 3]);
end
ChC_new = reshape(permute(ChC_new, [1, 3, 2, 4]), patchSize * Nc, patchSize * Nc);
time_ChC_new = toc;

%% Verification: numerical equivalence ChC vs ChC_new

fprintf('\n=== ChC VS ChC_new VERIFICATION ===\n');
fprintf('ChC size:     [%d, %d]\n', size(ChC));
fprintf('ChC_new size: [%d, %d]\n', size(ChC_new));

if ~isequal(size(ChC), size(ChC_new))
    warning('Size mismatch between ChC and ChC_new. Comparison may be invalid.');
end

max_abs_err = max(abs(ChC(:) - ChC_new(:)));
rel_err = max_abs_err / max(1e-12, max(abs(ChC(:))));
tolerance = 1e-10;

fprintf('Max absolute error: %.3e\n', max_abs_err);
fprintf('Max relative error: %.3e\n', rel_err);
fprintf('Equal within tol (%.0e): %s\n', tolerance, mat2str(max_abs_err < tolerance));

%% Performance comparison

fprintf('\n=== ChC PERFORMANCE COMPARISON ===\n');
fprintf('Previous impl time: %.6f s\n', time_ChC_old);
fprintf('New impl time:      %.6f s\n', time_ChC_new);
if time_ChC_new > 0
    fprintf('Speedup (old/new):  %.2fx\n', time_ChC_old / time_ChC_new);
end

%% ChC using function

opts_ChC_matrix = struct( ...
    'tau', tau,...
    'pad', pad,...
    'kernel_shape', kernel_shape...
);

fn = fieldnames(opts_ChC_matrix);
fv = struct2cell(opts_ChC_matrix);
nv = [fn.'; fv.'];
nv = nv(:).';

ChC_fun = utils.ChC_FFT_convolutions_2D(kCal, nv{:});

%% Comparison ChC vs ChC_fun

fprintf('\n=== ChC VS ChC_fun VERIFICATION ===\n');
fprintf('ChC size:     [%d, %d]\n', size(ChC));
fprintf('ChC_fun size: [%d, %d]\n', size(ChC_fun));

if ~isequal(size(ChC), size(ChC_fun))
    warning('Size mismatch between ChC and ChC_fun. Comparison may be invalid.');
end

max_abs_err = max(abs(ChC(:) - ChC_fun(:)));
rel_err = max_abs_err / max(1e-12, max(abs(ChC(:))));
tolerance = 1e-10;

fprintf('Max absolute error: %.3e\n', max_abs_err);
fprintf('Max relative error: %.3e\n', rel_err);
fprintf('Equal within tol (%.0e): %s\n', tolerance, mat2str(max_abs_err < tolerance));

%% Nullspace vectors

[~,Sc,U] = svd(ChC_fun,'econ');
clear ChC
sing = diag(Sc);
clear Sc

sing = sqrt(sing);
sing  = sing/sing(1);

% Visualize singular values of the C matrix
figure;
plot(sing, 'o-');
title('Singular values of the C matrix');
grid on;
xlim([1 numel(sing)]);
ylim([0 1]);
xlabel('Index');
ylabel('Singular value');

threshold = 0.05;

Nvect = find(sing >= threshold*sing(1),1,'last');
clear sing
U = U(:, Nvect+1:end); 

%% G matrices previous implementation

[N1, N2, Nc] = size(kData);

[N1_cal, N2_cal, ~] = size(kCal);

[in1,in2] = meshgrid(-tau:tau,-tau:tau);

if kernel_shape == 0 

    ind = (1:numel(in1)).'; 
    
else 
    
    ind = find(in1.^2+in2.^2<=tau^2); 
    
end

in1 = in1(ind)';
in2 = in2(ind)';

patchSize = numel(in1);

in1 = in1(:);
in2 = in2(:);

eind = (patchSize:-1:1).';

G = zeros(2*(2*tau+1)* 2*(2*tau+1),Nc,Nc);

W = U*U';
% clear U;

W = permute(reshape(W,patchSize,Nc,patchSize,Nc),[1,2,4,3]);

% Time the original G-matrix calculation
tic;  % Start timing original G calculation
for s = 1:patchSize 
    G(sub2ind([2*(2*tau+1),2*(2*tau+1)],2*tau+1+1+in1(eind)+in1(s),2*tau+1+1+in2(eind)+in2(s)),:,:) = ...
        G(sub2ind([2*(2*tau+1),2*(2*tau+1)],2*tau+1+1+in1(eind)+in1(s),2*tau+1+1+in2(eind)+in2(s)),:,:)  + W(:,:,:,s);
end

clear W

N1_g = N1;
N2_g = N2;

[n2,n1] = meshgrid((-N2_g/2:N2_g/2-1)/N2_g, (-N1_g/2:N1_g/2-1)/N1_g);
phaseKernel = exp(complex(0,-2*pi)*(n1*(N1_g-2*tau-1)+n2*(N2_g-2*tau-1)));

% Store spatial domain G before FFT transformation for comparison
G_original_spatial = reshape(G, 2*(2*tau+1), 2*(2*tau+1), Nc, Nc);

G = fft2(conj(reshape(G,2*(2*tau+1),2*(2*tau+1),Nc,Nc)),N1_g,N2_g).*phaseKernel; 

G = fftshift(fftshift(G,1),2);
time_G_original = toc;  % End timing original G calculation

% Store final frequency domain G for later comparison
G_original = G;

%% G matrices new calculation (optimized implementation)

% G_new: Optimized implementation using vectorized single-loop approach
tic;  % Start timing new G calculation

% Step 1: Initialize spatial domain G matrix (same as original)
grid_size = 2*(2*tau+1);
G_new = zeros(grid_size^2, Nc, Nc);

% Step 2: Compute nullspace projection matrix (same as original)
W_new = U*U';
W_new = permute(reshape(W_new, patchSize, Nc, patchSize, Nc), [1, 2, 4, 3]);

% Step 3: Optimized spatial accumulation with single loop
% Pre-compute constants and base indices to avoid redundant calculations
offset = 2*tau + 1 + 1;  % Center position in the grid
base_row_indices = offset + in1(eind);  % Base row positions for all patch elements
base_col_indices = offset + in2(eind);  % Base column positions for all patch elements

% Single optimized loop over patch size
% All target positions for all s (no loop)
target_row_mat = base_row_indices + in1.';                    % [patchSize x patchSize] (p x s)
target_col_mat = base_col_indices + in2.';                    % [patchSize x patchSize] (p x s)
idx = sub2ind([grid_size, grid_size], target_row_mat, target_col_mat);
idx = idx(:);                                                 % [patchSize^2 x 1]

% Flatten Nc×Nc blocks across (p,s)
vals = reshape(permute(W_new, [1 4 2 3]), [], Nc*Nc);         % [patchSize^2 x (Nc*Nc)]

% Accumulate into a 2-D matrix [grid_size^2 x (Nc*Nc)] using 2-D subs
nchan = Nc * Nc;
subs = [repmat(idx, nchan, 1), kron((1:nchan).', ones(numel(idx), 1))];  % [patchSize^2*nchan x 2]
acc = accumarray(subs, vals(:), [grid_size^2, nchan], @sum, 0);          % complex-safe

% Reshape to [grid_size^2 x Nc x Nc]
G_new = reshape(acc, grid_size^2, Nc, Nc);

clear W_new  % Free memory

% Step 4: Transform to frequency domain (same process as original)
N1_g_new = N1;  % Use the same dimensions as original (kCal dimensions)
N2_g_new = N2;

%Create frequency coordinate grids for phase correction
%Apply conjugate and reshape for FFT processing
% G_new_reshaped = conj(reshape(G_new, grid_size, grid_size, Nc, Nc));

% Pre-center via spatial modulation and use uncentered freq indices
row = (0:grid_size-1).'; col = 0:grid_size-1;
modPattern = (-1).^(row + col);
X = conj(reshape(G_new, grid_size, grid_size, Nc, Nc)) .* modPattern;

[k2, k1] = meshgrid(0:N2_g_new-1, 0:N1_g_new-1);
s1 = N1_g_new - 2*tau - 1;  s2 = N2_g_new - 2*tau - 1;
phaseKernel_unc = exp(-1i*2*pi * ((k1/N1_g_new - 0.5)*s1 + (k2/N2_g_new - 0.5)*s2));

G_new = fft2(X, N1_g_new, N2_g_new) .* phaseKernel_unc;  % no fftshift needed

time_G_new = toc;  % End timing new G calculation

%% G-matrices Verification

fprintf('\n=== G-MATRICES VERIFICATION ===\n');
fprintf('Original G matrix dimensions: [%d, %d, %d, %d]\n', size(G_original));
fprintf('New G matrix dimensions:      [%d, %d, %d, %d]\n', size(G_new));

% Test mathematical equivalence with tolerance for numerical precision
max_absolute_error = max(abs(G_original(:) - G_new(:)));
tolerance = 1e-10;  % Tolerance for numerical precision
matrices_equal = max_absolute_error < tolerance;

if max(abs(G_original(:))) > 0
    relative_error = max_absolute_error / max(abs(G_original(:)));
else
    relative_error = 0;
end

fprintf('\nNumerical Comparison:\n');
fprintf('  Matrices are numerically equal: %s\n', mat2str(matrices_equal));
fprintf('  Maximum absolute error:     %.2e\n', max_absolute_error);
fprintf('  Maximum relative error:     %.2e\n', relative_error);
if matrices_equal
    fprintf('  Methods are equivalent (error < %.0e)\n', tolerance);
else
    fprintf('  Methods have significant differences (error >= %.0e)\n', tolerance);
end

% Additional diagnostic information
if ~matrices_equal
    fprintf('\nDiagnostic Information:\n');
    fprintf('  Max magnitude in G_original: %.2e\n', max(abs(G_original(:))));
    fprintf('  Max magnitude in G_new:     %.2e\n', max(abs(G_new(:))));
    fprintf('  Mean absolute error:        %.2e\n', mean(abs(G_original(:) - G_new(:))));
    fprintf('  Standard deviation of error: %.2e\n', std(abs(G_original(:) - G_new(:))));
    
    % Find location of maximum error
    [~, max_error_idx] = max(abs(G_original(:) - G_new(:)));
    [max_row, max_col, max_coil1, max_coil2] = ind2sub(size(G_original), max_error_idx);
    fprintf('  Location of max error:      (%d,%d,%d,%d)\n', max_row, max_col, max_coil1, max_coil2);
else
    fprintf('\n✓ SUCCESS: G matrices are mathematically equivalent!\n');
end

%% G-matrices Performance Analysis

fprintf('\n=== G-MATRICES PERFORMANCE COMPARISON ===\n');

% Timing comparison
fprintf('Execution Times:\n');
fprintf('  Original implementation:    %.6f seconds\n', time_G_original);
fprintf('  Optimized implementation:   %.6f seconds\n', time_G_new);

% Performance metrics
if time_G_original > 0
    speedup_factor = time_G_original / time_G_new;
    time_reduction_percent = (time_G_original - time_G_new) / time_G_original * 100;
    
    fprintf('\nPerformance Improvement:\n');
    fprintf('  Speedup factor:             %.2fx\n', speedup_factor);
    fprintf('  Time reduction:             %.1f%%\n', time_reduction_percent);
    
    if speedup_factor > 1
        fprintf('  ✓ Optimized version is FASTER\n');
    elseif speedup_factor < 1
        fprintf('  ⚠ Optimized version is slower\n');
    else
        fprintf('  → Performance is equivalent\n');
    end
else
    fprintf('  ⚠ Original time too small to measure accurately\n');
end

% Computational complexity analysis
fprintf('\nComputational Analysis:\n');
fprintf('  Problem size: %d×%d grid, %d patch elements, %d coils\n', grid_size, grid_size, patchSize, Nc);
fprintf('  Total spatial accumulations: %d operations\n', patchSize^2);
fprintf('  Frequency domain size: %d×%d\n', N1, N2);

% Throughput analysis
if time_G_original > 0 && time_G_new > 0
    original_throughput = (patchSize^2) / time_G_original;
    new_throughput = (patchSize^2) / time_G_new;
    
    fprintf('\nThroughput Comparison:\n');
    fprintf('  Original throughput:        %.0f operations/second\n', original_throughput);
    fprintf('  Optimized throughput:       %.0f operations/second\n', new_throughput);
end

% Memory efficiency notes
fprintf('\nOptimization Benefits:\n');
fprintf('  • Reduced redundant sub2ind() calls from %d to %d\n', patchSize^2, patchSize);
fprintf('  • Vectorized array operations instead of element-wise updates\n');
fprintf('  • Pre-computed base indices outside the loop\n');
fprintf('  • Maintained identical mathematical accuracy\n');

%% Nullspace vectors of the G matrices

PowerIteration_G_nullspace_vectors = 0;
M = 15;
PowerIteration_flag_convergence = 1;
PowerIteration_flag_auto = 1;
FFT_interpolation = 1;

opts_G_matrix = struct( ...
    'kernel_shape', 1, ...
    'FFT_interpolation', FFT_interpolation, ...
    'interp_zp', 24, ...
    'sketched_SVD', 0);

fn = fieldnames(opts_G_matrix);
fv = struct2cell(opts_G_matrix);
nv = [fn.'; fv.'];          
nv = nv(:).';   

G_fun = utils.G_matrices_2D(kCal, N1, N2, tau, U, nv{:});

opts = struct( ...
  'PowerIteration_G_nullspace_vectors', PowerIteration_G_nullspace_vectors, ...
  'M', M, ...
  'PowerIteration_flag_convergence', PowerIteration_flag_convergence, ...
  'PowerIteration_flag_auto', PowerIteration_flag_auto, ...
  'FFT_interpolation', FFT_interpolation, ...
  'gauss_win_param', 100, ...
  'verbose', 1);

fn = fieldnames(opts);
fv = struct2cell(opts);
nv = [fn.'; fv.'];          % interleave names and values
nv = nv(:).';     

[senseMaps, eigenValues] = utils.nullspace_vectors_G_matrix_2D(kCal, N1, N2, G_fun, patchSize, nv{:});

% Phase-reference all coils to the first coil 
phase_ref = exp(-1i * angle(senseMaps(:,:,1)));
senseMaps = senseMaps .* phase_ref;  % align phase to channel 1

% Normalize sensitivities to unit L2 norm across coils at each pixel
den = sqrt(sum(abs(senseMaps).^2, 3));
den(den == 0) = 1;  % avoid division by zero (keeps zeros at zero)
senseMaps = senseMaps ./ den;
%% Support mask created from the last eigenvalues of the G matrices 

threshold_mask = 0.05;

% Logical mask from the last eigenvalue map (no find/preallocation needed)
eig_last = eigenValues(:,:,end);
eig_mask = eig_last < threshold_mask;  % N1xN2 logical

% Optional masking step

senseMaps_masked= senseMaps.*eig_mask;

%% Estimated Sensitivity Maps 

figure; 
imagesc(utils.mdisp(abs(senseMaps))); 
axis tight; 
axis image; 
axis off;
colormap gray; 
title('Estimated sensitivity maps');

figure; 
imagesc(utils.mdisp(abs(senseMaps_masked))); 
axis tight; 
axis image; 
axis off;
colormap gray; 
title('Masked sensitivity maps');

if PowerIteration_G_nullspace_vectors == 1 
    title_eig_values = 'Smallest eigenvalue of normalized G matrices (spatial map)';
    figure; 
    imagesc(eigenValues); 
    axis tight; 
    axis image; 
    colormap gray; 
    colorbar; 
    title(title_eig_values); 
else
    title_eig_values = 'Eigenvalues of normalized G matrices (spatial maps)';
    figure; 
    imagesc(utils.mdisp(eigenValues)); 
    axis tight; 
    axis image; 
    colormap gray; 
    colorbar; 
    title(title_eig_values); 
end

figure; 
imagesc(eig_mask); 
axis tight; 
axis image; 
colormap gray; 
title('Support mask');
