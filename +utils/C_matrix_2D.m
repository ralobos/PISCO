function result = C_matrix_2D(x, N1, N2, Nc, tau, kernel_shape)

% Function that calculates the C matrix.
%
% Input parameters:
%   --x:            N1 x N2 x Nc Nyquist-sampled k-space data, where N1 and
%                   N2 are the data dimensions, and Nc is the number of
%                   channels in the array.
%
%   --N1, N2:       Dimensions of the k-space data.
%
%   --Nc:           Number of channels of the k-space data.
%
%   --tau:          Parameter (in Nyquist units) that determines the size
%                   of the k-space kernel. For a rectangular kernel, the
%                   size corresponds to (2*tau+1) x (2*tau+1). For an
%                   ellipsoidal kernel, it corresponds to the radius of the
%                   associated neighborhood. Default: 3.
%
%   --kernel_shape: Binary variable. 0 = rectangular kernel, 1 = ellipsoidal
%                   kernel. Default: 1.

    x = reshape(x,N1*N2,Nc);

    [in1,in2] = meshgrid(-tau:tau,-tau:tau);
    
    if kernel_shape == 1
        i = find(in1.^2+in2.^2<=tau^2);
    else
        i = [1:numel(in1)]; 
    end     

    in1 = in1(i)';
    in2 = in2(i)';

    patchSize = numel(in1);

    result = zeros((N1-2*tau-utils.even_pisco(N1))*(N2-2*tau-utils.even_pisco(N2)),patchSize*Nc,'like',x);

    k = 0;
    for i = tau+1+utils.even_pisco(N1):N1-tau
        for j = tau+1+utils.even_pisco(N2):N2-tau
            k = k+1;
            ind = sub2ind([N1,N2],i+in1,j+in2);
            result(k,:) = utils.vect(x(ind,:));
        end
    end

    % % Fully vectorized approach - create all indices at once
    % i_centers = tau+1+utils.even_pisco(N1):N1-tau;
    % j_centers = tau+1+utils.even_pisco(N2):N2-tau;
    % [J_centers, I_centers] = meshgrid(j_centers, i_centers);
    % centers = [I_centers(:), J_centers(:)];  % All center positions as [i,j] pairs

    % % Create all patch offsets for all centers simultaneously
    % numCenters = size(centers, 1);
    % I_all = repmat(centers(:,1), 1, patchSize) + repmat(in1', numCenters, 1);
    % J_all = repmat(centers(:,2), 1, patchSize) + repmat(in2', numCenters, 1);

    % % Convert to linear indices and extract data
    % ind_all = sub2ind([N1,N2], I_all, J_all);
    % Extract all patches
    % x_patches = x(ind_all, :);  % Shape: [numCenters*patchSize, Nc]

    % % Reshape to [numCenters, patchSize, Nc]
    % x_reshaped = reshape(x_patches, [numCenters, patchSize, Nc]);

    % % Flatten each patch in the same way as utils.vect (column-wise)
    % result = reshape(x_reshaped, [numCenters, patchSize*Nc]);

end