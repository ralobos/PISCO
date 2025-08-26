function C = C_matrix_3D(x, varargin)

% Function that calculates the C matrix.
%
% Input parameters:
%   --x:            N1 x N2 x N3 x Nc Nyquist-sampled k-space data, where N1,
%                   N2, and N3 are the data dimensions, and Nc is the number
%                   of channels in the array.
%
%   --N1, N2, N3:   Dimensions of the k-space data.
%
%   --Nc:           Number of channels of the k-space data.
%
%   --tau:          Parameter (in Nyquist units) that determines the size
%                   of the k-space kernel. For a rectangular kernel, the
%                   size corresponds to (2*tau+1) x (2*tau+1) x (2*tau+1).
%                   For an ellipsoidal kernel, it corresponds to the radius
%                   of the associated neighborhood. Default: 3.
%
%   --kernel_shape: Binary variable. 0 = rectangular kernel, 1 = ellipsoidal
%                   kernel. Default: 1.

    p = inputParser;

    p.addRequired('x', @(x) isnumeric(x) && ndims(x) == 4);

    p.addParameter('tau', 3, @(x) isnumeric(x) && isscalar(x));
    p.addParameter('kernel_shape', 1, @(x) isnumeric(x) && isscalar(x) && (x == 0 || x == 1));

    if isempty(varargin)
        parse(p, x);
    else
        parse(p, x, varargin{:});
    end

    [N1, N2, N3, Nc] = size(p.Results.x);

    x = reshape(x,N1*N2*N3,Nc);

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

    k_centers = p.Results.tau+1+utils.even_pisco(N3):N3-p.Results.tau;
    i_centers = p.Results.tau+1+utils.even_pisco(N1):N1-p.Results.tau;
    j_centers = p.Results.tau+1+utils.even_pisco(N2):N2-p.Results.tau;

    [J_centers, I_centers, K_centers] = ndgrid(j_centers, i_centers, k_centers);
    centers = [I_centers(:), J_centers(:), K_centers(:)];  

    numCenters = size(centers, 1);
    I_all = centers(:,1) + in1; 
    J_all = centers(:,2) + in2; 
    K_all = centers(:,3) + in3; 

    ind_all = sub2ind([N1,N2,N3], I_all, J_all, K_all);
    x_patches = x(ind_all, :);  

    x_reshaped = reshape(x_patches, [numCenters, patchSize, Nc]);

    C = reshape(x_reshaped, [numCenters, patchSize*Nc]);

end