function G = G_matrices_3D(kCal, N1, N2, N3, tau, U, varargin)

% Function that calculates the G(x) matrices directly without calculating
% H(x) first.
%
% Input parameters:
%   --kCal:         N1_cal x N2_cal x N3_cal x Nc block of calibration data,
%                   where N1_cal, N2_cal, and N3_cal are the dimensions of a
%                   rectangular block of Nyquist-sampled k-space, and
%                   Nc is the number of channels in the array.
%
%   --N1, N2, N3:   The desired dimensions of the output sensitivity
%                   matrices.
%
%   --tau:          Parameter (in Nyquist units) that determines the
%                   size of the k-space kernel. For a rectangular
%                   kernel, the size corresponds to (2*tau+1) x
%                   (2*tau+1) x (2*tau+1). For an ellipsoidal kernel, it
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
%                   (N2_acs + interp_zp) x (N3_acs + interp_zp) x Nc. Default: 24.
%
%   --sketched_SVD: Binary variable. 1 = sketched SVD is used to calculate
%                   a basis for the nullspace of the C matrix. Default: 1.
%
% Output parameters:
%   --G:            N1 x N2 x N3 x Nc x Nc array where G[i,j,k,:,:]
%                   corresponds to the G matrix at the (i,j,k) spatial
%                   location.

% Parse inputs
p = inputParser;

p.addRequired('kCal', @(x) isnumeric(x) && ndims(x) == 4);
p.addRequired('N1', @(x) isnumeric(x) && isscalar(x));
p.addRequired('N2', @(x) isnumeric(x) && isscalar(x));
p.addRequired('N3', @(x) isnumeric(x) && isscalar(x));
p.addRequired('tau', @(x) isnumeric(x) && isscalar(x));
p.addRequired('U', @(x) isnumeric(x) && ismatrix(x));
p.addRequired('FFT_nullspace_C_calculation', @(x) isnumeric(x) && isscalar(x) && (x == 0 || x == 1));

p.addParameter('kernel_shape', 1, @(x) isnumeric(x) && isscalar(x) && (x == 0 || x == 1));
p.addParameter('FFT_interpolation', 1, @(x) isnumeric(x) && isscalar(x) && (x == 0 || x == 1));
p.addParameter('sketched_SVD', 1, @(x) isnumeric(x) && isscalar(x) && (x == 0 || x == 1));
p.addParameter('interp_zp', 24, @(x) isnumeric(x) && isscalar(x));

if isempty(varargin)
    parse(p, kCal, N1, N2, N3, tau, U);
else
    parse(p, kCal, N1, N2, N3, tau, U, varargin{:});
end

[N1_cal, N2_cal, N3_cal, Nc] = size(p.Results.kCal);

% Kernel offsets
[in1, in2, in3] = ndgrid(-p.Results.tau:p.Results.tau, -p.Results.tau:p.Results.tau, -p.Results.tau:p.Results.tau);

if p.Results.kernel_shape == 1
    mask = (in1.^2 + in2.^2 + in3.^2 <= p.Results.tau^2);
else
    mask = true(size(in1));
end
i = find(mask);

in1 = in1(i)';
in2 = in2(i)';
in3 = in3(i)';

patchSize = numel(in1);

eind = [patchSize:-1:1]';

grid_size = 2 * (2 * p.Results.tau + 1);

if p.Results.sketched_SVD == 0
    Wn = p.Results.U * p.Results.U';
else
    Wn = eye(size(p.Results.U, 1)) - p.Results.U * p.Results.U';
end
Wn = permute(reshape(Wn, patchSize, Nc, patchSize, Nc), [1, 2, 4, 3]);

offset = 2 * p.Results.tau + 1 + 1;
base_r = offset + in1(eind);
base_c = offset + in2(eind);
base_d = offset + in3(eind);

row_mat = base_r(:) + in1;
col_mat = base_c(:) + in2;
dep_mat = base_d(:) + in3;

idx_tbl = sub2ind([grid_size, grid_size, grid_size], row_mat, col_mat, dep_mat);

G = zeros(grid_size^3, Nc, Nc, 'like', Wn);
for s = 1:patchSize
    G(idx_tbl(:, s), :, :) = G(idx_tbl(:, s), :, :) + Wn(:, :, :, s);
end

clear row_mat col_mat dep_mat base_r base_c base_d idx_tbl Wn

if p.Results.FFT_interpolation == 0

    N1_g = N1;
    N2_g = N2;
    N3_g = N3;
    
else

    if N1_cal <= N1 - p.Results.interp_zp
        N1_g = N1_cal + p.Results.interp_zp;
    else
        N1_g = N1_cal;
    end

    if N2_cal <= N2 - p.Results.interp_zp
        N2_g = N2_cal + p.Results.interp_zp;
    else   
        N2_g = N2_cal;
    end

    if N3_cal <= N3 - p.Results.interp_zp
        N3_g = N3_cal + p.Results.interp_zp;
    else
        N3_g = N3_cal;
    end
    
end

Y = conj(reshape(G, grid_size, grid_size, grid_size, Nc, Nc));
s1 = N1_g - 2 * p.Results.tau - 1;  s2 = N2_g - 2 * p.Results.tau - 1;  s3 = N3_g - 2 * p.Results.tau - 1;
[n2c_g, n1c_g, n3c_g] = meshgrid((-N2_g/2:N2_g/2-1)/N2_g, (-N1_g/2:N1_g/2-1)/N1_g, (-N3_g/2:N3_g/2-1)/N3_g);
phaseKernel_c = -exp(complex(0, -2 * pi) * (n1c_g * s1 + n2c_g * s2 + n3c_g * s3));
G = utils.fft3(Y, N1_g, N2_g, N3_g) .* phaseKernel_c;
G = fftshift(fftshift(fftshift(G, 1), 2), 3);

if p.Results.FFT_nullspace_C_calculation == 1
    G = flip(flip(flip(G, 2), 1), 3);
end

end