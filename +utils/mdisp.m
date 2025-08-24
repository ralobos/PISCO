function tempForDisplay = mdisp(x)
% Function used to visualize multichannel images.
% If the input corresponds to a 3D array of dimensions N1 x N2 x Nc, the
% output corresponds to a 2D array that displays Nc images of dimension
% N1 x N2.
%
% Input parameters:
%   --x:              3D array of size N1 x N2 x Nc, where Nc is the number
%                     of channels.
%
% Output parameters:
%   --tempForDisplay: 2D array that arranges the Nc images of size N1 x N2
%                     into a single displayable image.
%

    [N1,N2,Nc] = size(x);
    f = factor(Nc);if numel(f)==1;f = [1,f];end
    tempForDisplay = reshape(permute(reshape(x,[N1,N2,prod(f(1:floor(numel(f)/2))),...
        prod(f(floor(numel(f)/2)+1:end))]),[1,3,2,4]),[N1*prod(f(1:floor(numel(f)/2))),...
        N2*prod(f(floor(numel(f)/2)+1:end))]); 
end