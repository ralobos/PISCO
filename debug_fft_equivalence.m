% Debug script to understand why G_new_fft ≠ G_new_fft_2
% Test the mathematical equivalence of explicit phase multiplication vs spatial shift

clear; clc;

% Create a simple test case
N1 = 64; N2 = 64;
tau = 5;
grid_size = 2*(2*tau+1);

% Create a simple test matrix
test_matrix = randn(grid_size, grid_size) + 1i*randn(grid_size, grid_size);

%% Method 1: Explicit phase multiplication (like G_new_fft)
N1_g = N1; N2_g = N2;

% Frequency grids (matching the original implementation)
[n2, n1] = meshgrid((-N2_g/2:N2_g/2-1)/N2_g, (-N1_g/2:N1_g/2-1)/N1_g);
phaseKernel = exp(complex(0, -2*pi) * (n1*(N1_g-2*tau-1) + n2*(N2_g-2*tau-1)));

result1 = fft2(test_matrix, N1_g, N2_g) .* phaseKernel;

%% Method 2: Spatial domain shift (like G_new_fft_2)
% Step 1: FFT
fft_temp = fft2(test_matrix, N1_g, N2_g);

% Step 2: Convert to spatial domain
spatial = ifft2(fft_temp);

% Step 3: Apply circular shift
shift1 = N1_g - 2*tau - 1;
shift2 = N2_g - 2*tau - 1;
shifted = circshift(spatial, [shift1, shift2]);

% Step 4: Back to frequency domain
result2 = fft2(shifted);

%% Compare results
fprintf('Results are equal: %s\n', mat2str(isequal(result1, result2)));
fprintf('Max absolute difference: %.2e\n', max(abs(result1(:) - result2(:))));
fprintf('Relative error: %.2e\n', max(abs(result1(:) - result2(:))) / max(abs(result1(:))));

%% Debug: Check individual components
fprintf('\nDebugging individual components:\n');
fprintf('N1_g = %d, N2_g = %d, tau = %d\n', N1_g, N2_g, tau);
fprintf('shift1 = %d, shift2 = %d\n', shift1, shift2);
fprintf('Grid size = %d x %d\n', grid_size, grid_size);
fprintf('FFT size = %d x %d\n', N1_g, N2_g);

% Check the frequency grid ranges
fprintf('\nFrequency grid ranges:\n');
fprintf('n1 range: [%.3f, %.3f]\n', min(n1(:)), max(n1(:)));
fprintf('n2 range: [%.3f, %.3f]\n', min(n2(:)), max(n2(:)));

% Check phase kernel statistics
fprintf('\nPhase kernel statistics:\n');
fprintf('Phase kernel magnitude range: [%.3f, %.3f]\n', min(abs(phaseKernel(:))), max(abs(phaseKernel(:))));
fprintf('Phase kernel phase range: [%.3f, %.3f]\n', min(angle(phaseKernel(:))), max(angle(phaseKernel(:))));

%% Test with different frequency grid definitions
fprintf('\n=== Testing alternative frequency grid definitions ===\n');

% Alternative 1: Integer coordinates (MATLAB convention)
[n2_alt1, n1_alt1] = meshgrid(0:N2_g-1, 0:N1_g-1);
phaseKernel_alt1 = exp(complex(0, -2*pi) * (n1_alt1*shift1/N1_g + n2_alt1*shift2/N2_g));
result1_alt1 = fft2(test_matrix, N1_g, N2_g) .* phaseKernel_alt1;

fprintf('Alternative 1 (0:N-1 coords) equal to spatial shift: %s\n', mat2str(isequal(result1_alt1, result2)));
fprintf('Alternative 1 max diff: %.2e\n', max(abs(result1_alt1(:) - result2(:))));

% Alternative 2: Centered but different normalization
[n2_alt2, n1_alt2] = meshgrid(-N2_g/2:N2_g/2-1, -N1_g/2:N1_g/2-1);
phaseKernel_alt2 = exp(complex(0, -2*pi) * (n1_alt2*shift1/N1_g + n2_alt2*shift2/N2_g));
result1_alt2 = fft2(test_matrix, N1_g, N2_g) .* phaseKernel_alt2;

fprintf('Alternative 2 (centered, no normalization) equal to spatial shift: %s\n', mat2str(isequal(result1_alt2, result2)));
fprintf('Alternative 2 max diff: %.2e\n', max(abs(result1_alt2(:) - result2(:))));

%% Test the exact shift theorem
fprintf('\n=== Testing exact Fourier shift theorem ===\n');

% For a shift of (s1, s2) in spatial domain, the frequency domain multiplication should be:
% exp(-2πi * (k1*s1/N1 + k2*s2/N2))
% where k1, k2 are the frequency indices

% Test with MATLAB's fftshift convention
[k2, k1] = meshgrid(0:N2_g-1, 0:N1_g-1);
% Adjust for fftshift (center the frequencies)
k1_centered = mod(k1 + N1_g/2, N1_g) - N1_g/2;
k2_centered = mod(k2 + N2_g/2, N2_g) - N2_g/2;

phaseKernel_exact = exp(complex(0, -2*pi) * (k1_centered*shift1/N1_g + k2_centered*shift2/N2_g));
result1_exact = fft2(test_matrix, N1_g, N2_g) .* phaseKernel_exact;

fprintf('Exact shift theorem equal to spatial shift: %s\n', mat2str(isequal(result1_exact, result2)));
fprintf('Exact shift theorem max diff: %.2e\n', max(abs(result1_exact(:) - result2(:))));
