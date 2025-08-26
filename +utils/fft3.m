function out = fft3(x, N1, N2, N3)
% Function that computes the 3D Fourier transform of the input array x,
% along the first three dimensions, with optional size specification.
%
% Input parameters:
%   --x:   4D array of size [N1, N2, N3, Nc], where N1, N2, N3 are spatial
%          dimensions and Nc is the number of channels.
%   --N1:  Size of the first dimension for the FFT.
%   --N2:  Size of the second dimension for the FFT.
%   --N3:  Size of the third dimension for the FFT.
%
% Output parameters:
%   --out: 4D array of the same size as x (with specified N1, N2, N3),
%          containing the 3D Fourier transform along the first three
%          dimensions.
%
out = fft(fft(fft(x, N1, 1), N2, 2), N3, 3);
end