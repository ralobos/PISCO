function C = C_matrix_2D(x, varargin)

% Function that calculates the C matrix.
%
% Input parameters:
%   --x:            N1 x N2 x Nc Nyquist-sampled k-space data, where N1 and
%                   N2 are the data dimensions, and Nc is the number of
%                   channels in the array.
%
%   --tau:          Parameter (in Nyquist units) that determines the size
%                   of the k-space kernel. For a rectangular kernel, the
%                   size corresponds to (2*tau+1) x (2*tau+1). For an
%                   ellipsoidal kernel, it corresponds to the radius of the
%                   associated neighborhood. Default: 3.
%
%   --kernel_shape: Binary variable. 0 = rectangular kernel, 1 = ellipsoidal
%                   kernel. Default: 1.

    p = inputParser;

    p.addRequired('x', @(x) isnumeric(x) && ndims(x) == 3);

    p.addParameter('tau', 3, @(x) isnumeric(x) && isscalar(x));
    p.addParameter('kernel_shape', 1, @(x) isnumeric(x) && isscalar(x) && (x == 0 || x == 1));

    if isempty(varargin)
        parse(p, x);
    else
        parse(p, x, varargin{:});
    end

    [N1, N2, Nc] = size(p.Results.x);

    x = reshape(x, N1 * N2, Nc);

    [in1, in2] = meshgrid(-p.Results.tau:p.Results.tau, -p.Results.tau:p.Results.tau);
    if p.Results.kernel_shape == 1
        i = find(in1.^2 + in2.^2 <= p.Results.tau^2);
    else
        i = (1:numel(in1));
    end
    in1 = in1(i(:));
    in2 = in2(i(:));

    patchSize = numel(i);

    i_centers = p.Results.tau + 1 + utils.even_pisco(N1):N1 - p.Results.tau;
    j_centers = p.Results.tau + 1 + utils.even_pisco(N2):N2 - p.Results.tau;

    [I_centers, J_centers] = meshgrid(i_centers, j_centers);
    centers = [I_centers(:), J_centers(:)];

    numCenters = size(centers, 1);

    in1_row = in1(:)';
    in2_row = in2(:)';
    I_all = centers(:,1) + in1_row; 
    J_all = centers(:,2) + in2_row; 

    ind_all = sub2ind([N1, N2], I_all, J_all);

    x_selected = x(ind_all, :);

    x_patches = reshape(x_selected, numCenters, patchSize, Nc);

    C = reshape(x_patches, numCenters, patchSize * Nc);

end