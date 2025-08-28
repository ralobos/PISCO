function [senseMaps, eigenVal] = nullspace_vectors_G_matrix_3D( ...
    kCal, N1, N2, N3, G, patchSize, varargin)

% Function that calculates the nullspace vectors for each G(x) matrix. These
% vectors correspond to sensitivity maps at the x location.
%
% Input parameters:
%   --kCal:         N1_cal x N2_cal x N3_cal x Nc block of calibration data, where
%                   N1_cal, N2_cal, and N3_cal are the dimensions of a rectangular
%                   block of Nyquist-sampled k-space, and Nc is the number of
%                   channels in the array.
%
%   --N1, N2, N3:   The desired dimensions of the output sensitivity matrices.
%
%   --G:            N1_g x N2_g x N3_g x Nc x Nc array where G(i,j,k,:,:) 
%                   corresponds to the G matrix at the (i,j,k) spatial location.
%
%   --patchSize:    Number of elements in the kernel used to calculate the
%                   nullspace vectors of the C matrix.
%
%   --PowerIteration_G_nullspace_vectors: Binary variable. 0 = nullspace
%                   vectors of the G matrices are calculated using SVD.
%                   1 = nullspace vectors of the G matrices are calculated
%                   using the Power Iteration approach. Default: 1.
%
%   --M:            Number of iterations used in the Power Iteration approach
%                   to calculate the nullspace vectors of the G matrices.
%                   Default: 30.
%
%   --PowerIteration_flag_convergence: Binary variable. 1 = convergence error
%                   is displayed for Power Iteration if the method has not
%                   converged for some voxels after the iterations indicated
%                   by the user. Default: 1.
%
%   --PowerIteration_flag_auto: Binary variable. 1 = Power Iteration is run
%                   until convergence in case the number of iterations
%                   indicated by the user is too small. Default: 0.
%
%   --FFT_interpolation: Binary variable. 0 = no interpolation is used,
%                   1 = FFT-based interpolation is used. Default: 1.
%
%   --gauss_win_param: Parameter for the Gaussian apodizing window used to
%                   generate the low-resolution image in the FFT-based
%                   interpolation approach. This is the reciprocal of the
%                   standard deviation of the Gaussian window. Default: 100.
%
%   --verbose:      Binary variable. 1 = information about the convergence
%                   of Power Iteration is displayed. Default: 1.
%
% Output parameters:
%   --senseMaps:    N1 x N2 x N3 x Nc stack corresponding to the sensitivity
%                   maps for each channel present in the calibration data.
%
%   --eigenVal:     N1 x N2 x N3 x Nc array containing the eigenvalues of G(x)
%                   for each spatial location (normalized). Can be used for
%                   creating a mask describing the image support (e.g.,
%                   mask = (eigenVal(:,:,:,end) < 0.08);). If
%                   PowerIteration_G_nullspace_vectors == 1, only the
%                   smallest eigenvalue is returned (dimensions: N1 x N2 x N3).
%                   If FFT_interpolation == 1, approximations of eigenvalues
%                   are returned.%   --FFT_interpolation: Binary variable. 0 = no interpolation is used,
%                   1 = FFT-based interpolation is used. Default: 1.

p = inputParser;

p.addRequired('kCal', @(x) isnumeric(x) && ndims(x) == 4);
p.addRequired('N1', @(x) isnumeric(x) && isscalar(x));
p.addRequired('N2', @(x) isnumeric(x) && isscalar(x));
p.addRequired('N3', @(x) isnumeric(x) && isscalar(x));
p.addRequired('G', @(x) isnumeric(x) && ndims(x) == 5);
p.addRequired('patchSize', @(x) isnumeric(x) && isscalar(x));

p.addParameter('PowerIteration_G_nullspace_vectors', 1, @(x) isnumeric(x) && isscalar(x));
p.addParameter('M', 30, @(x) isnumeric(x) && isscalar(x));
p.addParameter('PowerIteration_flag_convergence', 1, @(x) isnumeric(x) && isscalar(x));
p.addParameter('PowerIteration_flag_auto', 0, @(x) isnumeric(x) && isscalar(x));
p.addParameter('FFT_interpolation', 1, @(x) isnumeric(x) && isscalar(x));
p.addParameter('gauss_win_param', 100, @(x) isnumeric(x) && isscalar(x));
p.addParameter('verbose', 1, @(x) isnumeric(x) && isscalar(x));

if isempty(varargin)
    parse(p, kCal, N1, N2, N3, G, patchSize);
else
    parse(p, kCal, N1, N2, N3, G, patchSize, varargin{:});
end

% Normalize optional flags to scalar logicals (don't modify p.Results which is read-only)
flagAuto = logical(p.Results.PowerIteration_flag_auto);
flagConv = logical(p.Results.PowerIteration_flag_convergence);
if flagAuto
    flagConv = 0; % auto mode suppresses convergence error check
end

N1_g = size(G, 1);
N2_g = size(G, 2);
N3_g = size(G, 3);
Nc   = size(G, 4);

G = reshape(permute(p.Results.G, [4 5 1 2 3]), [Nc Nc (N1_g * N2_g * N3_g)]);

if p.Results.PowerIteration_G_nullspace_vectors == 0

    [~, eigenVal, Vpage] = pagesvd(G, 'econ', 'vector');
    eigenVal  = reshape(permute(eigenVal, [3 1 2]), [N1_g, N2_g, N3_g, Nc]);
    senseMaps = reshape(permute(Vpage(:, end, :), [3 1 2]), [N1_g N2_g N3_g Nc]);
    clear G
    eigenVal = eigenVal / p.Results.patchSize;

else

    G = G / p.Results.patchSize;
    G = permute(G, [4 5 1 2 3]);
    G = reshape(G, [Nc, Nc, N1_g * N2_g * N3_g]);
    G_null = repmat(eye(Nc), [1 1 (N1_g * N2_g * N3_g)]);
    G_null = G_null - G;
    clear G

    if ~flagConv && ~flagAuto

        S = randn(Nc, 1) + 1i * randn(Nc, 1);
        S = repmat(S, [1 1 N1_g * N2_g * N3_g]);
        S(:, 1, :) = S(:, 1, :) ./ pagenorm(S(:, 1, :));

        for m = 1:p.Results.M
            S = pagemtimes(G_null, S);
            S(:, 1, :) = S(:, 1, :) ./ pagenorm(S(:, 1, :));
            if m == p.Results.M
                E = pagemtimes(S, 'ctranspose', pagemtimes(G_null, S), 'none');
            end
        end

        clear G_null
        senseMaps = reshape(permute(squeeze(S), [2 1]), [N1_g, N2_g, N3_g, Nc]);
        eigenVal  = reshape(permute(E, [3 1 2]), [N1_g, N2_g, N3_g]);
        eigenVal  = 1 - eigenVal;

    end

    if flagConv || flagAuto

        S  = randn(Nc, 1) + 1i * randn(Nc, 1);
        S2 = randn(Nc, 1) + 1i * randn(Nc, 1);
        S(:, 1, :)  = S(:, 1, :)  ./ pagenorm(S(:, 1, :));
        S2(:, 1, :) = S2(:, 1, :) ./ pagenorm(S2(:, 1, :));

        for m = 1:p.Results.M
            S  = pagemtimes(G_null, S);
            S2 = pagemtimes(G_null, S2);
            S(:, 1, :) = S(:, 1, :) ./ pagenorm(S(:, 1, :));
            inner_prod = pagemtimes(S(:, 1, :), 'ctranspose', S2(:, 1, :), 'none');
            S2(:, 1, :) = S2(:, 1, :) - inner_prod .* S(:, 1, :);
            S2(:, 1, :) = S2(:, 1, :) ./ pagenorm(S2(:, 1, :));
            if m == p.Results.M
                E  = pagemtimes(S,  'ctranspose', pagemtimes(G_null, S),  'none');
                E2 = pagemtimes(S2, 'ctranspose', pagemtimes(G_null, S2), 'none');
            end
        end

        eigen1 = reshape(permute(E,  [3 1 2]), [N1_g, N2_g, N3_g]);
        eigen2 = reshape(permute(E2, [3 1 2]), [N1_g, N2_g, N3_g]);
        

    if p.Results.FFT_interpolation == 0

            eigenVal = 1 - eigen1;

            threshold_mask = 0.075;
            
            support_mask = zeros(size(eigenVal));
            support_mask(find(eigenVal < threshold_mask)) = 1;

            ratioEig   = (eigen2 ./ eigen1) .^ p.Results.M;
            ratio_small = support_mask .* ratioEig;
        
            th_ratio = 0.008;
        
            ratio_small(find(ratio_small <= th_ratio)) = 0;
            ratio_small(find(ratio_small >  th_ratio)) = 1;
        
            flag_convergence_PI = sum(ratio_small(:)) > 0;
            
            
            if flag_convergence_PI == 1 && flagConv
                error(['Power Iteration might have not converged for some voxels within the support after the ' int2str(p.Results.M) ...
                    ' iterations indicated by the user. Increasing the number of iterations is recommended. You can ignore this error by setting PowerIteration_flag_convergence = 0. ' ...
                    'The number of needed iterations for convergence can be found automatically by setting PowerIteration_flag_auto = 1. '])
            end

            if flag_convergence_PI == 0
                if p.Results.verbose == 1
                     disp(['Most likely Power Iteration has converged for all the voxels within the support after the ' int2str(p.Results.M) ' iterations indicated by the user.'])
                end
            end
            
    if flagAuto == 1 && flag_convergence_PI == 1
            if p.Results.verbose == 1
        warning('off', 'backtrace')
        warning(['Power Iteration might have not converged for some voxels within the support after the ' int2str(p.Results.M) ...
            ' iterations indicated by the user. The number of iterations for the convergence of Power Iteration will be found automatically. You can turn off this option by setting PowerIteration_flag_auto = 0. '])
            end
        M_auto = p.Results.M + 1;
        while (flag_convergence_PI == 1)
                S = pagemtimes(G_null, S);
                S2 = pagemtimes(G_null, S2);
        S(:, 1, :) = S(:, 1, :) ./ pagenorm(S(:, 1, :));
        inner_prod = pagemtimes(S(:, 1, :), 'ctranspose', S2(:, 1, :), 'none');
        S2(:, 1, :) = S2(:, 1, :) - inner_prod .* S(:, 1, :);
        S2(:, 1, :) = S2(:, 1, :) ./ pagenorm(S2(:, 1, :));
        E  = pagemtimes(S,  'ctranspose', pagemtimes(G_null, S),  'none');
        E2 = pagemtimes(S2, 'ctranspose', pagemtimes(G_null, S2), 'none');

                eigen1 = reshape(permute(E, [3 1 2]), [N1_g, N2_g, N3_g]);
                eigen2 = reshape(permute(E2, [3 1 2]), [N1_g, N2_g, N3_g]);

                eigenVal = 1 - eigen1;
                
                support_mask = zeros(size(eigenVal));
                support_mask(find(eigenVal < threshold_mask)) = 1;
        ratioEig   = (eigen2 ./ eigen1) .^ M_auto;
        ratio_small = support_mask .* ratioEig;
        ratio_small(find(ratio_small <= th_ratio)) = 0;
        ratio_small(find(ratio_small >  th_ratio)) = 1;
                flag_convergence_PI = sum(ratio_small(:)) > 0;

                M_auto = M_auto + 1;
            end

            if p.Results.verbose == 1
        disp(['Most likely Power Iteration has converged for all the voxels within the support. ' int2str(M_auto) ' iterations were needed.'])
            end

        end

    eigenVal = abs(eigenVal);

        end

    senseMaps = reshape(permute(squeeze(S), [2 1]), [N1_g, N2_g, N3_g, Nc]);

    end

    


end

% ==== FFT-based interpolation ====

if p.Results.FFT_interpolation == 1

    [N1_cal, N2_cal, N3_cal, ~] = size(kCal);

    w_sm  = (0.54 - 0.46 * cos(2 * pi * ((0:(N1_g - 1)) / (N1_g - 1))))';
    w_sm2 = (0.54 - 0.46 * cos(2 * pi * ((0:(N2_g - 1)) / (N2_g - 1))))';
    w_sm3 = (0.54 - 0.46 * cos(2 * pi * ((0:(N3_g - 1)) / (N3_g - 1))))';
    w_sm_2d = w_sm * w_sm2';

    w_sm = bsxfun(@times, w_sm_2d, reshape(w_sm3, [1 1 N3_g]));
    w_sm_sM = repmat(w_sm, [1 1 1 Nc]);

    if p.Results.PowerIteration_G_nullspace_vectors == 1 && (flagConv || flagAuto)

        auxVal = 1 - eigen1;

        eigenVal = abs(fftshift(fftshift(fftshift( ...
            utils.ifft3_zp(utils.ft3(auxVal) .* w_sm, [p.Results.N1 p.Results.N2 p.Results.N3]), 1), 2), 3));

        eigenVal = eigenVal/max(eigenVal(:));

        threshold_mask = 0.075;
        
        support_mask = zeros(size(eigenVal));
        support_mask(find(eigenVal < threshold_mask)) = 1;
    
        eigen1_us = abs(fftshift(fftshift(fftshift( ...
            utils.ifft3_zp(utils.ft3(eigen1) .* w_sm, [p.Results.N1 p.Results.N2 p.Results.N3]), 1), 2), 3));
        eigen2_us = abs(fftshift(fftshift(fftshift( ...
            utils.ifft3_zp(utils.ft3(eigen2) .* w_sm, [p.Results.N1 p.Results.N2 p.Results.N3]), 1), 2), 3));

        ratioEig = (eigen2_us./eigen1_us).^p.Results.M;
        ratio_small = support_mask.*ratioEig;
    
        th_ratio = 0.008;
    
        ratio_small(ratio_small <= th_ratio) = 0;
        ratio_small(ratio_small > th_ratio) = 1;
    
        flag_convergence_PI = any(ratio_small(:));
    
        if flag_convergence_PI == 1 && flagConv
            error(['Power Iteration might have not converged for some voxels within the support after the ' int2str(p.Results.M) ...
                ' iterations indicated by the user. Increasing the number of iterations is recommended. You can ignore this error by setting PowerIteration_flag_convergence = 0. ' ...
                'The number of needed iterations for convergence can be found automatically by setting PowerIteration_flag_auto = 1. '])
        end

        if flag_convergence_PI == 0
            if p.Results.verbose == 1
                 disp(['Most likely Power Iteration has converged for all the voxels within the support after the ' int2str(p.Results.M) ' iterations indicated by the user.'])
            end
        end

    if flagAuto && flag_convergence_PI == 1
        if p.Results.verbose == 1
            warning('off', 'backtrace')
            warning(['Power Iteration might have not converged for some voxels within the support after the ' int2str(p.Results.M) ...
                ' iterations indicated by the user. The number of iterations for the convergence of Power Iteration will be found automatically. You can turn off this option by setting PowerIteration_flag_auto = 0.'])
        end
        M_auto = p.Results.M+1;
        while (flag_convergence_PI == 1)
            S = pagemtimes(G_null, S);
            S2 = pagemtimes(G_null, S2);
            
            S(:, 1, :) = S(:, 1, :) ./ pagenorm(S(:, 1, :));
            inner_prod = pagemtimes(S(:, 1, :), 'ctranspose', S2(:, 1, :), 'none');
            S2(:, 1, :) = S2(:, 1, :) - inner_prod .* S(:, 1, :);
            S2(:, 1, :) = S2(:, 1, :) ./ pagenorm(S2(:, 1, :));
            E  = pagemtimes(S,  'ctranspose', pagemtimes(G_null, S),  'none');
            E2 = pagemtimes(S2, 'ctranspose', pagemtimes(G_null, S2), 'none');

            eigen1 = reshape(permute(E, [3 1 2]), [N1_g, N2_g, N3_g]);
            eigen2 = reshape(permute(E2, [3 1 2]), [N1_g, N2_g, N3_g]);

            auxVal = 1 - eigen1;

            eigenVal = abs(fftshift(fftshift(fftshift( ...
                utils.ifft3_zp(utils.ft3(auxVal) .* w_sm, [p.Results.N1 p.Results.N2 p.Results.N3]), 1), 2), 3));

            eigenVal = eigenVal/max(eigenVal(:));
            
            support_mask = zeros(size(eigenVal));
            support_mask(find(eigenVal < threshold_mask)) = 1;

            eigen1_us = abs(fftshift(fftshift(fftshift( ...
                utils.ifft3_zp(utils.ft3(eigen1) .* w_sm, [p.Results.N1 p.Results.N2 p.Results.N3]), 1), 2), 3));
            eigen2_us = abs(fftshift(fftshift(fftshift( ...
                utils.ifft3_zp(utils.ft3(eigen2) .* w_sm, [p.Results.N1 p.Results.N2 p.Results.N3]), 1), 2), 3));

            ratioEig   = (eigen2_us ./ eigen1_us) .^ M_auto;
            ratio_small = support_mask .* ratioEig;
        
            ratio_small(find(ratio_small <= th_ratio)) = 0;
            ratio_small(find(ratio_small >  th_ratio)) = 1;
        
            flag_convergence_PI = sum(ratio_small(:)) > 0;

            M_auto = M_auto + 1;
        end

        if p.Results.verbose == 1
            disp(['Most likely Power Iteration has converged for all the voxels within the support. ' int2str(M_auto) ' iterations were needed.'])
        end

    end

    end

    if p.Results.PowerIteration_G_nullspace_vectors == 1 && ~flagConv && ~flagAuto
        eigenVal = abs(fftshift(fftshift(fftshift( ...
            utils.ifft3_zp(utils.ft3(eigenVal) .* w_sm, [p.Results.N1 p.Results.N2 p.Results.N3]), 1), 2), 3));
        eigenVal = eigenVal / max(eigenVal(:));
    end

    if p.Results.PowerIteration_G_nullspace_vectors == 0
        eigenVal = abs(fftshift(fftshift(fftshift( ...
            utils.ifft3_zp(utils.ft3(eigenVal) .* w_sm, [p.Results.N1 p.Results.N2 p.Results.N3]), 1), 2), 3));
        eigenVal = eigenVal / max(eigenVal(:));
    end

    apodizing_window_2D = gausswin(N1_g, p.Results.gauss_win_param) * gausswin(N2_g, p.Results.gauss_win_param)';
    apodizing_window = bsxfun(@times, apodizing_window_2D, reshape(gausswin(N3_g, p.Results.gauss_win_param), [1 1 N3_g]));

    imLowRes_cal = zeros(N1_g, N2_g, N3_g, Nc);
    imLowRes_cal( ...
        ceil(N1_g/2) + utils.even_pisco(N1_g/2) + (-floor(N1_cal/2):floor(N1_cal/2) - utils.even_pisco(N1_cal/2)), ...
        ceil(N2_g/2) + utils.even_pisco(N2_g/2) + (-floor(N2_cal/2):floor(N2_cal/2) - utils.even_pisco(N2_cal/2)), ...
        ceil(N3_g/2) + utils.even_pisco(N3_g/2) + (-floor(N3_cal/2):floor(N3_cal/2) - utils.even_pisco(N3_cal/2)), :) = kCal;
    imLowRes_cal = utils.ift3(imLowRes_cal .* apodizing_window);

    cim = sum(conj(senseMaps) .* imLowRes_cal, 4) ./ sum(abs(senseMaps) .^ 2, 4);
    phase_norm = exp(1i * angle(cim));
    senseMaps = phase_norm .* senseMaps;

    senseMaps = fftshift(fftshift(fftshift( ...
        utils.ifft3_zp(utils.ft3(senseMaps) .* w_sm_sM, [p.Results.N1 p.Results.N2 p.Results.N3]), 1), 2), 3);

end

end
