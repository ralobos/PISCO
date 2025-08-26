function out = ifft3_zp(x, Nz)
% Function that computes the inverse 3D Fourier transform of the input array x,
% along the first three dimensions, with optional size specification (zero-padding).
%
% Input parameters:
%   --x:   4D array of size [N1, N2, N3, Nc], where N1, N2, N3 are spatial
%          dimensions and Nc is the number of channels.
%   --Nz:  1x3 array specifying the size of each dimension for the inverse FFT.
%
% Output parameters:
%   --out: 4D array of the same size as x (with specified Nz), containing the
%          inverse 3D Fourier transform along the first three dimensions.
%
out = ifft(ifft(ifft(x, Nz(1), 1), Nz(2), 2), Nz(3), 3);
end