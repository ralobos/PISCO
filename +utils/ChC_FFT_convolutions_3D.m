function ChC = ChC_FFT_convolutions_3D(X, varargin)

% Function that directly calculates the matrix C'*C using an FFT-based
% approach.
%
% Input parameters:
%   --X:            N1 x N2 x N3 x Nc Nyquist-sampled k-space data, where N1,
%                   N2, and N3 are the data dimensions, and Nc is the number
%                   of channels in the array.
%
%   --tau:          Parameter (in Nyquist units) that determines the size of
%                   the k-space kernel. For a rectangular kernel, the size
%                   corresponds to (2*tau+1) x (2*tau+1) x (2*tau+1). For an
%                   ellipsoidal kernel, it corresponds to the radius of the
%                   associated neighborhood. Default: 3.
%
%   --pad:          Binary variable. 1 = zero-padding is employed when
%                   calculating FFTs. Default: 1.
%
%   --kernel_shape: Binary variable. 0 = rectangular kernel, 1 = ellipsoidal
%                   kernel. Default: 1.
%
% Output parameters:
%   --PhP:          Matrix C'*C calculated using the FFT-based approach.

p = inputParser;

p.addRequired('X', @(x) isnumeric(x) && ndims(x) == 4);

p.addParameter('tau', 3, @(x) isnumeric(x) && isscalar(x) && x >= 0);
p.addParameter('pad', 1, @(x) isnumeric(x) && isscalar(x) && (x == 0 || x == 1));
p.addParameter('kernel_shape', 1, @(x) isnumeric(x) && isscalar(x) && (x == 0 || x == 1));

if isempty(varargin)
    parse(p, X);
else
    parse(p, X, varargin{:});
end

[N1, N2, N3, Nc] = size(p.Results.X);

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

pad = 1;

if pad == 1
    N1n = 2^(ceil(log2(N1+2*p.Results.tau))); 
    N2n = 2^(ceil(log2(N2+2*p.Results.tau)));
    N3n = 2^(ceil(log2(N3+2*p.Results.tau)));
else
    N1n = N1;
    N2n = N2;
    N3n = N3;
end

inds = sub2ind([N1n,N2n,N3n], floor(N1n/2)+1-in1+in1', floor(N2n/2)+1-in2+in2',  floor(N3n/2)+1-in3+in3');

x = fft(fft(fft(p.Results.X, N1n, 1), N2n, 2), N3n, 3);

ChC = zeros(patchSize, patchSize, Nc, Nc);
for q = 1:Nc
    A = conj(x(:,:,:,q:Nc)) .* x(:,:,:,q);             % correlations using x to mirror old numerics
    R = utils.ifft3(A);                                 % inverse 3D FFT
    R = circshift(R, [ceil(N1n/2), ceil(N2n/2), ceil(N3n/2)]); % center zero-lag
    b = reshape(R, [], Nc - q + 1);
    ChC(:, :, q:Nc, q) = reshape(b(inds, :), patchSize, patchSize, Nc - q + 1);
    ChC(:, :, q, q+1:Nc) = permute(conj(ChC(:, :, q+1:Nc, q)), [2, 1, 4, 3]);
end
ChC = reshape(permute(ChC, [1, 3, 2, 4]), patchSize * Nc, patchSize * Nc);

end
