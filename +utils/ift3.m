function out = ift3(x)
% Function that computes the inverse 3D Fourier transform of the input array x,
% applying fftshift and ifftshift for proper centering.
%
% Input parameters:
%   --x:   4D array of size [N1, N2, N3, Nc], where N1, N2, N3 are spatial
%          dimensions and Nc is the number of channels.
%
% Output parameters:
%   --out: 4D array of the same size as x, containing the inverse 3D Fourier
%          transform along the first three dimensions, with the result
%          centered using fftshift.
%
out = fftshift(ifft(ifft(ifft(ifftshift(x),[],1),[],2),[],3));
end