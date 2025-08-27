function G = G_matrices_2D(kCal, N1, N2, tau, U, varargin)

% Function that calculates the G(x) matrices directly without calculating
% H(x) first.
%
% Input parameters:
%   --kCal:         N1_cal x N2_cal x Nc block of calibration data,
%                   where N1_cal and N2_cal are the dimensions of a
%                   rectangular block of Nyquist-sampled k-space, and
%                   Nc is the number of channels in the array.
%
%   --N1, N2:       The desired dimensions of the output sensitivity
%                   matrices.
%
%   --tau:          Parameter (in Nyquist units) that determines the
%                   size of the k-space kernel. For a rectangular
%                   kernel, the size corresponds to (2*tau+1) x
%                   (2*tau+1). For an ellipsoidal kernel, it
%                   corresponds to the radius of the associated
%                   neighborhood. Default: 3.
%
%   --U:            Matrix whose columns correspond to the nullspace
%                   vectors of the C matrix.
%
%   --kernel_shape: Binary variable. 0 = rectangular kernel, 1 = ellipsoidal
%                   kernel. Default: 1.
%
%   --FFT_interpolation: Binary variable. 0 = no interpolation is used,
%                   1 = FFT-based interpolation is used. Default: 1.
%
%   --interp_zp:    Amount of zero-padding to create the low-resolution
%                   grid if FFT-interpolation is used. The low-resolution
%                   grid has dimensions (N1_acs + interp_zp) x
%                   (N2_acs + interp_zp) x Nc. Default: 24.
%
%   --sketched_SVD: Binary variable. 1 = sketched SVD is used to calculate
%                   a basis for the nullspace of the C matrix. Default: 1.
%
% Output parameters:
%   --G:            N1 x N2 x Nc x Nc array where G[i,j,:,:]
%                   corresponds to the G matrix at the (i,j) spatial
%                   location.

p = inputParser;

p.addRequired('kCal', @(x) isnumeric(x) && ndims(x) == 3);
p.addRequired('N1', @(x) isnumeric(x) && isscalar(x));
p.addRequired('N2', @(x) isnumeric(x) && isscalar(x));
p.addRequired('tau', @(x) isnumeric(x) && isscalar(x));
p.addRequired('U', @(x) isnumeric(x) && ismatrix(x));

p.addParameter('kernel_shape', 1, @(x) isnumeric(x) && isscalar(x) && (x == 0 || x == 1));
p.addParameter('FFT_interpolation', 1, @(x) isnumeric(x) && isscalar(x) && (x == 0 || x == 1));
p.addParameter('sketched_SVD', 1, @(x) isnumeric(x) && isscalar(x) && (x == 0 || x == 1));
p.addParameter('interp_zp', 24, @(x) isnumeric(x) && isscalar(x));

if isempty(varargin) 
    parse(p, kCal, N1, N2, tau, U);
else
    parse(p, kCal, N1, N2, tau, U, varargin{:});
end

[N1_cal, N2_cal, Nc] = size(p.Results.kCal);

[in1, in2] = meshgrid(-p.Results.tau:p.Results.tau, -p.Results.tau:p.Results.tau);

if p.Results.kernel_shape == 0
    ind = (1:numel(in1)).';
else
    ind = find(in1.^2 + in2.^2 <= p.Results.tau^2);
end

in1 = in1(ind)';
in2 = in2(ind)';

patchSize = numel(in1);

in1 = in1(:);
in2 = in2(:);

eind = (patchSize:-1:1).';

grid_size = 2 * (2 * p.Results.tau + 1);

if p.Results.sketched_SVD == 0
    W = p.Results.U * p.Results.U';
else
    W = eye(size(p.Results.U, 1)) - p.Results.U * p.Results.U';
end

W = permute(reshape(W, patchSize, Nc, patchSize, Nc), [1, 2, 4, 3]);

offset = 2 * p.Results.tau + 1 + 1;
base_row_indices = offset + in1(eind);
base_col_indices = offset + in2(eind);

target_row_mat = base_row_indices + in1.';
target_col_mat = base_col_indices + in2.';
idx = sub2ind([grid_size, grid_size], target_row_mat, target_col_mat);

G = zeros(grid_size^2, Nc, Nc, 'like', W);
for s = 1:patchSize
    G(idx(:, s), :, :) = G(idx(:, s), :, :) + W(:, :, :, s);
end

clear W

if p.Results.FFT_interpolation == 0
    N1_g = p.Results.N1;
    N2_g = p.Results.N2;
else
    if N1_cal <= p.Results.N1 - p.Results.interp_zp
        N1_g = N1_cal + p.Results.interp_zp;
    else
        N1_g = N1_cal;
    end
    if N2_cal <= p.Results.N2 - p.Results.interp_zp
        N2_g = N2_cal + p.Results.interp_zp;
    else
        N2_g = N2_cal;
    end
end

row = (0:grid_size-1).'; col = 0:grid_size-1;
modPattern = (-1).^(row + col);
X = conj(reshape(G, grid_size, grid_size, Nc, Nc)) .* modPattern;

[k2, k1] = meshgrid(0:N2_g-1, 0:N1_g-1);
s1 = N1_g - 2 * p.Results.tau - 1;  s2 = N2_g - 2 * p.Results.tau - 1;
phaseKernel_unc = exp(-1i * 2 * pi * ((k1 / N1_g - 0.5) * s1 + (k2 / N2_g - 0.5) * s2));

G = fft2(X, N1_g, N2_g) .* phaseKernel_unc;

end


