function result = C_matrix_3D(x, N1, N2, N3, Nc, tau, kernel_shape)

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

    x = reshape(x,N1*N2*N3,Nc);

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

    result = zeros((N1-2*tau-utils.even_pisco(N1))*(N2-2*tau-utils.even_pisco(N2))*(N3-2*tau-utils.even_pisco(N3)),patchSize*Nc,'like',x);

    l = 0;

    for k = tau+1+utils.even_pisco(N3):N3-tau
        for i = tau+1+utils.even_pisco(N1):N1-tau
            for j = tau+1+utils.even_pisco(N2):N2-tau
                l = l+1;
                ind = sub2ind([N1,N2,N3],i+in1,j+in2,k+in3);
                result(l,:) = utils.vect(x(ind,:));
            end
        end
    end


%    % Fully vectorized approach for 3D - create all indices at once
% k_centers = tau+1+utils.even_pisco(N3):N3-tau;
% i_centers = tau+1+utils.even_pisco(N1):N1-tau;
% j_centers = tau+1+utils.even_pisco(N2):N2-tau;

% % Create all center positions using ndgrid (matches nested loop order)
% [K_centers, I_centers, J_centers] = ndgrid(k_centers, i_centers, j_centers);
% centers = [I_centers(:), J_centers(:), K_centers(:)];  % All center positions as [i,j,k] triplets

% % Create all patch offsets for all centers simultaneously
% numCenters = size(centers, 1);
% I_all = repmat(centers(:,1), 1, patchSize) + repmat(in1', numCenters, 1);
% J_all = repmat(centers(:,2), 1, patchSize) + repmat(in2', numCenters, 1);
% K_all = repmat(centers(:,3), 1, patchSize) + repmat(in3', numCenters, 1);

% % Convert to linear indices and extract data
% ind_all = sub2ind([N1,N2,N3], I_all, J_all, K_all);
% x_patches = x(ind_all, :);  % Shape: [numCenters*patchSize, Nc]

% % Reshape to [numCenters, patchSize, Nc]
% x_reshaped = reshape(x_patches, [numCenters, patchSize, Nc]);

% % Flatten each patch in the same way as utils.vect (column-wise)
% result = reshape(x_reshaped, [numCenters, patchSize*Nc]);

end