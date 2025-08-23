function PhP = ChC_FFT_convolutions_3D(X, N1, N2, N3, Nc, tau, pad, kernel_shape)

% Function that directly calculates the matrix C'*C using an FFT-based
% approach.
%
% Input parameters:
%   --X:            N1 x N2 x N3 x Nc Nyquist-sampled k-space data, where N1,
%                   N2, and N3 are the data dimensions, and Nc is the number
%                   of channels in the array.
%
%   --N1, N2, N3:   Dimensions of the k-space data.
%
%   --Nc:           Number of channels of the k-space data.
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

x = fft(fft(fft(X,N1n,1),N2n,2),N3n,3).*phaseKernel;

PhP = zeros(patchSize, patchSize, Nc,  Nc); 
for q = 1:Nc
    b= reshape(ifft3(conj(x(:,:,:,q:Nc)).*x(:,:,:,q).*cphaseKernel),[],Nc-q+1); 
    PhP(:,:,q:Nc,q) = reshape(b(inds,:),patchSize,patchSize,Nc-q+1);
    PhP(:,:,q,q+1:Nc) = permute(conj(PhP(:,:,q+1:Nc,q)),[2,1,4,3]);
end
PhP = reshape(permute(PhP,[1,3,2,4]), patchSize*Nc, patchSize*Nc);

end
