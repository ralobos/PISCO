
function [senseMaps, eigenValues] = PISCO_sensitivity_maps_estimation(kCal, dim_sens, varargin)

    % Input parameters:
    %   --kCal:                            2D case: N1_cal x N2_cal x Nc block of calibration data, where 
    %                                               N1_cal and N2_cal are the dimensions of a rectangular 
    %                                               block of Nyquist-sampled k-space, and Nc is the number of 
    %                                               channels in the array.
    %                                      3D case: N1_cal x N2_cal x N3_cal x Nc block of calibration 
    %                                               data, where N1_cal, N2_cal, and N3_cal are the dimensions 
    %                                               of a rectangular block of Nyquist-sampled k-space.
    %
    %   --dim_sens:                        2D case: 1x2 array with the desired dimensions of the output 
    %                                               sensitivity matrices.
    %                                      3D case: 1x3 array with the desired dimensions of the output 
    %                                               sensitivity matrices.
    %
    %   --tau:                             2D case: Parameter (in Nyquist units) that determines the size of 
    %                                               the k-space kernel. For a rectangular kernel, the size is 
    %                                               (2*tau+1) x (2*tau+1). For an ellipsoidal kernel, it is 
    %                                               the radius of the associated neighborhood. Default: 3.
    %                                      3D case: Parameter (in Nyquist units) that determines the size of
    %                                               the k-space kernel. For a rectangular kernel, the size is
    %                                               (2*tau+1) x (2*tau+1) x (2*tau+1). For an ellipsoidal kernel, it is
    %                                               the radius of the associated neighborhood. Default: 3.
    %
    %   --threshold:                       Specifies how small a singular value needs to be (relative 
    %                                      to the maximum singular value) before its associated 
    %                                      singular vector is considered to be in the nullspace of 
    %                                      the C-matrix. Default: 0.05.
    %
    %   --kernel_shape:                    Binary variable. 0 = rectangular kernel, 1 = ellipsoidal 
    %                                      kernel. Default: 1.
    %
    %   --FFT_nullspace_C_calculation:     Binary variable. 0 = nullspace vectors 
    %                                      of C are calculated from C'*C by calculating C first. 
    %                                      1 = nullspace vectors of C are calculated from C'*C 
    %                                      directly using an FFT-based approach. Default: 1.
    %
    %   --PowerIteration_G_nullspace_vectors: Binary variable. 0 = nullspace 
    %                                         vectors of the G matrices are calculated using SVD. 
    %                                         1 = nullspace vectors of the G matrices are calculated 
    %                                         using a Power Iteration approach. Default: 1.
    %
    %   --M:                               Number of iterations used in the Power Iteration approach 
    %                                      to calculate the nullspace vectors of the G matrices. 
    %                                      Default: 30.
    %
    %   --PowerIteration_flag_convergence: Binary variable. 1 = display a 
    %                                      convergence error for Power Iteration if the method has 
    %                                      not converged for some voxels after the specified 
    %                                      iterations. Default: 1.
    %
    %   --PowerIteration_flag_auto:        Binary variable. 1 = Power Iteration is run 
    %                                      until convergence if the number of iterations is too 
    %                                      small. Default: 0.
    %
    %   --FFT_interpolation:               Binary variable. 0 = no interpolation. 1 = 
    %                                      FFT-based interpolation is used. Default: 1.
    %
    %   --interp_zp:                       Amount of zero-padding to create the low-resolution grid 
    %                                      if FFT-interpolation is used. 
    %                                      2D case: The grid has dimensions 
    %                                               (N1_cal + interp_zp) x (N2_cal + interp_zp) x Nc. 
    %                                      3D case: The grid has dimensions
    %                                               (N1_cal + interp_zp) x (N2_cal + interp_zp) x (N3_cal + interp_zp) x Nc.
    %                                      Default: 24.
    %
    %   --gauss_win_param:                 Parameter for the Gaussian apodizing window used to 
    %                                      generate the low-resolution image in the FFT-based 
    %                                      interpolation approach. This is the reciprocal of the 
    %                                      standard deviation of the Gaussian window. Default: 100.
    %
    %   --sketched_SVD:                    Binary variable. 1 = sketched SVD is used to calculate 
    %                                      a basis for the nullspace of the C matrix. Default: 1.
    %
    %   --sketch_dim:                      Dimension of the sketch matrix used to calculate a basis 
    %                                      for the nullspace of the C matrix using a sketched SVD. 
    %                                      Only used if sketched_SVD is enabled. Default: 500.
    %
    %   --visualize_C_matrix_sv:           Binary variable. 1 = Singular values of the C matrix are displayed.
    %                                      Default: 0. 
    %                                      Note: If sketched_SVD = 1 and if the curve of the singular values flattens out,
    %                                      it suggests that the sketch dimension is appropriate for the data.
    %
    %   --verbose:                         Binary variable. 1 = display PISCO information, including 
    %                                      which techniques are employed and computation times for 
    %                                      each step. Default: 1.
    %
    % Output parameters:
    %   --senseMaps:                       2D case: dim_sens(1) x dim_sens(2) x Nc stack corresponding to the 
    %                                               sensitivity maps for each channel present in the 
    %                                               calibration data.
    %                                      3D case: dim_sens(1) x dim_sens(2) x dim_sens(3) x Nc stack    
    %                                               corresponding to the sensitivity maps for each channel
    %                                               present in the calibration data. The maps are normalized   
    %                                               to have unit norm in the sense that the sum of squares
    %                                               of each map equals one.
    %
    %   --eigenValues:                     2D case: dim_sens(1) x dim_sens(2) x Nc array containing the 
    %                                               eigenvalues of G(x) for each spatial location (normalized 
    %                                               by the kernel size). 
    %                                      3D case: dim_sens(1) x dim_sens(2) x dim_sens(3) x Nc array
    %                                               containing the eigenvalues of G(x) for each spatial location
    %                                               (normalized by the kernel size). Can be used for creating a mask
    %                                               describing the image support (e.g., mask = 
    %                                               (eigenValues(:,:,end) < 0.08);). If
    %                                               PowerIteration_G_nullspace_vectors == 1, only the smallest
    %                                               eigenvalue is returned (dimensions: dim_sens(1) x
    %                                               dim_sens(2)). If FFT_interpolation == 1, approximations
    %                                               of eigenvalues are returned.

    % Set default values for optional parameters

    p = inputParser;

    addRequired(p, 'kCal', @(x) isnumeric(x) && ndims(x) == 3);
    addRequired(p, 'dim_sens', @(x) isnumeric(x) && isvector(x) && length(x) == 2);

    addParameter(p, 'tau', 3, @(x) isnumeric(x) && isscalar(x));
    addParameter(p, 'threshold', 0.05, @(x) isnumeric(x) && isscalar(x));
    addParameter(p, 'kernel_shape', 1, @(x) isnumeric(x) && isscalar(x));
    addParameter(p, 'FFT_nullspace_C_calculation', 1, @(x) isnumeric(x) && isscalar(x));
    addParameter(p, 'PowerIteration_G_nullspace_vectors', 1, @(x) isnumeric(x) && isscalar(x));
    addParameter(p, 'M', 30, @(x) isnumeric(x) && isscalar(x));
    addParameter(p, 'PowerIteration_flag_convergence', 1, @(x) isnumeric(x) && isscalar(x));
    addParameter(p, 'PowerIteration_flag_auto', 0, @(x) isnumeric(x) && isscalar(x));
    addParameter(p, 'FFT_interpolation', 1, @(x) isnumeric(x) && isscalar(x));
    addParameter(p, 'interp_zp', 24, @(x) isnumeric(x) && isscalar(x));
    addParameter(p, 'gauss_win_param', 100, @(x) isnumeric(x) && isscalar(x));
    addParameter(p, 'sketched_SVD', 1, @(x) isnumeric(x) && isscalar(x));
    addParameter(p, 'sketch_dim', 500, @(x) isnumeric(x) && isscalar(x));
    addParameter(p, 'visualize_C_matrix_sv', 0, @(x) isnumeric(x) && isscalar(x));
    addParameter(p, 'verbose', 1, @(x) isnumeric(x) && isscalar(x));

    if isempty(varargin)
        parse(p, kCal, dim_sens);
    else
        parse(p, kCal, dim_sens, varargin{:});
    end
   
    if p.Results.verbose == 1

        if p.Results.kernel_shape ==0
            kernel_shape_q = 'Rectangular';
        else
            kernel_shape_q = 'Ellipsoidal';
        end

        if p.Results.FFT_nullspace_C_calculation == 0
            FFT_nullspace_C_calculation_q = 'No';
        else
            FFT_nullspace_C_calculation_q = 'Yes';
        end

        if p.Results.FFT_interpolation == 0
            FFT_interpolation_q = 'No';
        else
            FFT_interpolation_q = 'Yes';
        end

        if p.Results.PowerIteration_G_nullspace_vectors == 0
            PowerIteration_nullspace_vectors_q = 'No';
        else
            PowerIteration_nullspace_vectors_q = 'Yes';
        end

        if p.Results.sketched_SVD == 0
            sketched_SVD_q = 'No';
        else
            sketched_SVD_q = 'Yes';
        end
        
        disp('Selected PISCO techniques:')
        disp('=======================')
        disp(['Kernel shape : ' kernel_shape_q])
        disp(['FFT-based calculation of nullspace vectors of C : ' FFT_nullspace_C_calculation_q])
        disp(['Sketched SVD for nullspace vectors of C : ' sketched_SVD_q])
        disp(['FFT-based interpolation : ' FFT_interpolation_q])
        disp(['PowerIteration-based nullspace estimation for G matrices : ' PowerIteration_nullspace_vectors_q])
        disp('=======================')

    end

    if numel(size(kCal)) > 3
        flag_3D = 1; % 3D data
    else
        flag_3D = 0; % 2D data
    end

    t_null = tic;

    % ==== Nullspace-based algorithm Steps (1) and (2)  ====

    % Calculation of nullspace vectors of C 

    t_null_vecs = tic;

    if flag_3D == 0

        opts_nullspace_C_matrix_2D = struct( ...
            'tau', p.Results.tau,...
            'threshold', p.Results.threshold,...
            'kernel_shape', p.Results.kernel_shape,...
            'FFT_nullspace_C_calculation', p.Results.FFT_nullspace_C_calculation,...
            'sketched_SVD', p.Results.sketched_SVD,...
            'sketch_dim', p.Results.sketch_dim,...
            'visualize_C_matrix_sv', p.Results.visualize_C_matrix_sv...
             );

        fn = fieldnames(opts_nullspace_C_matrix_2D);
        fv = struct2cell(opts_nullspace_C_matrix_2D);
        nv = [fn.'; fv.'];
        nv = nv(:).';

        U = utils.nullspace_vectors_C_matrix_2D(kCal, nv{:});
    else
        U = nullspace_vectors_C_matrix_3D(kCal, p.Results.tau, p.Results.threshold, p.Results.kernel_shape, p.Results.FFT_nullspace_C_calculation, p.Results.sketched_SVD, p.Results.sketch_dim, p.Results.visualize_C_matrix_sv);
    end

    t_null_vecs = toc(t_null_vecs);

    if p.Results.verbose == 1

        if p.Results.FFT_nullspace_C_calculation == 0
            aux_word = 'Calculating C first';
        else
            aux_word = 'FFT-based direct calculation of ChC';
        end

        if p.Results.sketched_SVD == 0
            aux_word = [aux_word ', using regular SVD'];
        else
            aux_word = [aux_word ', using sketched SVD'];
        end
        
        disp('=======================')
        disp('PISCO computation times (secs):')
        disp('=======================')
        disp(['Time nullspace vectors of C (' aux_word ') : ' num2str(t_null_vecs)]) 
        disp('=======================')

    end

    % ==== Nullspace-based algorithm Step (3)  ====

    % Direct computation of G matrices 

    t_G_matrices = tic;

    if flag_3D == 0

        opts_G_matrices_2D = struct( ...
            'kernel_shape', p.Results.kernel_shape, ...
            'FFT_interpolation', p.Results.FFT_interpolation, ...
            'interp_zp', p.Results.interp_zp, ...
            'sketched_SVD', p.Results.sketched_SVD ...
        );

        fn = fieldnames(opts_G_matrices_2D);
        fv = struct2cell(opts_G_matrices_2D);
        nv = [fn.'; fv.'];
        nv = nv(:).';

        G = utils.G_matrices_2D(kCal, dim_sens(1), dim_sens(2), p.Results.tau, U, nv{:});
    else
        G = G_matrices_3D(kCal, dim_sens(1), dim_sens(2), dim_sens(3), p.Results.tau, U, p.Results.kernel_shape, p.Results.FFT_nullspace_C_calculation, p.Results.FFT_interpolation, p.Results.interp_zp, p.Results.sketched_SVD);
    end

    t_G_matrices = toc(t_G_matrices);

    if flag_3D == 0
        Nc = size(kCal,3);
        patchSize = size(U,1)/Nc;
        clear U
    else
        Nc = size(kCal,4);
        patchSize = size(U,1)/Nc;
        clear U
    end

    if p.Results.verbose == 1
        disp(['Time G matrices (direct calculation): ' num2str(t_G_matrices )]) 
        disp('=======================')
    end

    % ==== Nullspace-based algorithm Step (4)  ====

    % Calculation of nullspace vectors of the G matrices

    t_null_G = tic;

    if flag_3D == 0

        opts_G_nullspace_vectors_2D = struct( ...
            'PowerIteration_G_nullspace_vectors', p.Results.PowerIteration_G_nullspace_vectors, ...
            'M', p.Results.M, ...
            'PowerIteration_flag_convergence', p.Results.PowerIteration_flag_convergence, ...
            'PowerIteration_flag_auto', p.Results.PowerIteration_flag_auto, ...
            'FFT_interpolation', p.Results.FFT_interpolation, ...
            'gauss_win_param', p.Results.gauss_win_param, ...
            'verbose', p.Results.verbose);

        fn = fieldnames(opts_G_nullspace_vectors_2D);
        fv = struct2cell(opts_G_nullspace_vectors_2D);
        nv = [fn.'; fv.'];
        nv = nv(:).';

        [senseMaps, eigenValues] = utils.nullspace_vectors_G_matrix_2D(kCal, ...
        dim_sens(1), dim_sens(2), G, patchSize, nv{:});

    else
        [senseMaps, eigenValues] = nullspace_vectors_G_matrix_3D(kCal, dim_sens(1), dim_sens(2), dim_sens(3), ...
            G, patchSize, p.Results.PowerIteration_G_nullspace_vectors, p.Results.M, p.Results.PowerIteration_flag_convergence, p.Results.PowerIteration_flag_auto,...
            p.Results.FFT_interpolation, p.Results.gauss_win_param, p.Results.verbose);
    end

    t_null_G = toc(t_null_G);

    clear G

    if p.Results.verbose == 1

        if p.Results.PowerIteration_G_nullspace_vectors == 0
            aux_word = 'Using SVD';
        else
            aux_word = 'Using Power Iteration';
        end

        disp(['Time nullspace vector G matrices (' aux_word ') : ' num2str(t_null_G)]) 
        disp('=======================')
        
    end

    % ==== Nullspace-based algorithm Step (5)  ====

    %  Normalization 

    if flag_3D == 0
        
        senseMaps = senseMaps.*repmat((exp(-complex(0,1)*angle(senseMaps(:,:,1)))), [1 1 Nc]); %Final maps after phase referencing w.r.t the first channel

        senseMaps = senseMaps./repmat(sqrt(sum(abs(senseMaps).^2,3)), [1 1 Nc]);

    else

        senseMaps = senseMaps.*repmat((exp(-complex(0,1)*angle(senseMaps(:,:,:,1)))), [1 1 1 Nc]); 

        senseMaps = senseMaps./repmat(sqrt(sum(abs(senseMaps).^2,4)), [1 1 1 Nc]);

    end

    if p.Results.verbose == 1
        disp(['Total time: ' num2str(toc(t_null))]) 
        disp('=======================')
    end

end

%% Extra functions

function U = nullspace_vectors_C_matrix_3D(kCal, tau, threshold, kernel_shape, FFT_nullspace_C_calculation, sketched_SVD, sketch_dim, visualize_C_matrix_sv)

% Function that returns the nullspace vectors of the C matrix.
%
% Input parameters:
%   --kCal:                      N1_cal x N2_cal x N3_cal x Nc block of calibration data, where
%                                N1_cal, N2_cal, and N3_cal are the dimensions of a rectangular
%                                block of Nyquist-sampled k-space, and Nc is the number of
%                                channels in the array.
%
%   --tau:                       Parameter (in Nyquist units) that determines the size of
%                                the k-space kernel. For a rectangular kernel, the size
%                                corresponds to (2*tau+1) x (2*tau+1) x (2*tau+1). For an
%                                ellipsoidal kernel, it corresponds to the radius of the
%                                associated neighborhood. Default: 3.
%
%   --threshold:                 Specifies how small a singular value needs to be
%                                (relative to the maximum singular value) before its
%                                associated singular vector is considered to be in the
%                                nullspace of the C-matrix. Default: 0.05.
%
%   --kernel_shape:              Binary variable. 0 = rectangular kernel, 1 = ellipsoidal
%                                kernel. Default: 1.
%
%   --FFT_nullspace_C_calculation: Binary variable. 0 = nullspace vectors of
%                                the C matrix are calculated from C'*C by calculating C
%                                first. 1 = nullspace vectors of the C matrix are calculated
%                                from C'*C, which is calculated directly using an FFT-based
%                                approach. Default: 1.
%
%   --sketched_SVD:              Binary variable. 1 = sketched SVD is used to calculate
%                                a basis for the nullspace of the C matrix. Default: 1.
%
%   --sketch_dim:                Dimension of the sketch matrix used to calculate a basis
%                                for the nullspace of the C matrix using a sketched SVD.
%                                Only used if sketched_SVD is enabled. Default: 500.
%
%   --visualize_C_matrix_sv:     Binary variable. 1 = Singular values of the C matrix are displayed.
%                                Default: 0. 
%                                Note: If sketched_SVD = 1 and if the curve of the singular values flattens out,
%                                it suggests that the sketch dimension is appropriate for the data.
%
% Output parameters:
%   --U:                         Matrix whose columns correspond to the nullspace vectors
%                                of the C matrix.

if nargin < 2 || not(isnumeric(tau)) || not(numel(tau))
    tau = 3;
end

if nargin < 3 || not(isnumeric(threshold)) || not(numel(threshold))
    threshold = 0.05;
end

if nargin < 4 || not(isnumeric(kernel_shape)) || not(numel(kernel_shape))
    kernel_shape = 1;
end

if nargin < 5 || not(isnumeric(FFT_nullspace_C_calculation)) || not(numel(FFT_nullspace_C_calculation))
    FFT_nullspace_C_calculation = 1;
end

if FFT_nullspace_C_calculation == 0
    
    C = utils.C_matrix_3D(kCal(:), size(kCal,1), size(kCal,2), size(kCal,3), size(kCal,4), tau, kernel_shape);
       
    ChC = C'*C;
    clear C
    
else
    
    ChC = utils.ChC_FFT_convolutions_3D(kCal, size(kCal,1), size(kCal,2), size(kCal,3), size(kCal,4), tau, 1, kernel_shape);
      
end

if nargin < 6 || not(isnumeric(sketched_SVD)) || not(numel(sketched_SVD))
    sketched_SVD = 1;
end

if nargin < 7 || not(isnumeric(sketch_dim)) || not(numel(sketch_dim))
    sketch_dim = 500;
end

if nargin < 8 || not(isnumeric(visualize_C_matrix_sv)) || not(numel(visualize_C_matrix_sv))
    visualize_C_matrix_sv = 0;
end

if sketched_SVD == 0

    [~,Sc,U] = svd(ChC,'econ');
    clear ChC
    sing = diag(Sc);
    clear Sc
    
    sing = sqrt(sing);
    sing  = sing/sing(1);

    if visualize_C_matrix_sv == 1

        % Visualize singular values of the C matrix
        figure;
        plot(sing, 'o-');
        title('Singular values of the C matrix');
        grid on;
        xlim([1 numel(sing)]);
        ylim([0 1]);
        xlabel('Index');
        ylabel('Singular value');

    end


    Nvect = find(sing >=threshold*sing(1),1,'last');
    clear sing
    U = U(:, Nvect+1:end); 

else

    %Sketching

    [~, N2c] = size(ChC);
    Sk = (1/sqrt(sketch_dim))*randn(sketch_dim, N2c) + (1/sqrt(sketch_dim))*1i*randn(sketch_dim, N2c);
    C = Sk*ChC;
    [~, sing, Vf] = svd(C, 'econ', 'vector');

    sing = sqrt(sing);
    sing  = sing/sing(1);

    if visualize_C_matrix_sv == 1

    
        figure;
        plot(sing, 'o-');
        title('Singular values of the C matrix (sketched SVD)');
        grid on;
        xlim([1 numel(sing)]);
        ylim([0 1]);
        xlabel('Index');
        ylabel('Singular value');
    end

    rank_C = find(sing >=threshold*sing(1),1,'last');
    clear sing

    U = Vf(:,1:rank_C);
    clear Vf
    
end
end

function G = G_matrices_2D(kCal, N1, N2, tau, U, kernel_shape, FFT_interpolation, interp_zp, sketched_SVD)

% Function that calculates the G(x) matrices directly without calculating
% H(x) first.
%
% Input parameters:
%   --kCal:         N1_cal x N2_cal x Nc block of calibration data,
%                   where N1_cal and N2_cal are the dimensions of a
%                   rectangular block of Nyquist-sampled k-space, and
%                   Nc is the number of channels in the array.
%
%   --N1, N2:       The desired dimensions of the output sensitivity
%                   matrices.
%
%   --tau:          Parameter (in Nyquist units) that determines the
%                   size of the k-space kernel. For a rectangular
%                   kernel, the size corresponds to (2*tau+1) x
%                   (2*tau+1). For an ellipsoidal kernel, it
%                   corresponds to the radius of the associated
%                   neighborhood. Default: 3.
%
%   --U:            Matrix whose columns correspond to the nullspace
%                   vectors of the C matrix.
%
%   --kernel_shape: Binary variable. 0 = rectangular kernel, 1 = ellipsoidal
%                   kernel. Default: 1.
%
%   --FFT_interpolation: Binary variable. 0 = no interpolation is used,
%                   1 = FFT-based interpolation is used. Default: 1.
%
%   --interp_zp:    Amount of zero-padding to create the low-resolution
%                   grid if FFT-interpolation is used. The low-resolution
%                   grid has dimensions (N1_acs + interp_zp) x
%                   (N2_acs + interp_zp) x Nc. Default: 24.
%
%   --sketched_SVD: Binary variable. 1 = sketched SVD is used to calculate
%                   a basis for the nullspace of the C matrix. Default: 1.
%
% Output parameters:
%   --G:            N1 x N2 x Nc x Nc array where G[i,j,:,:]
%                   corresponds to the G matrix at the (i,j) spatial
%                   location.

if nargin < 4 || not(isnumeric(tau)) || not(numel(tau))
    tau = 3;
end

if nargin < 6 || not(isnumeric(kernel_shape)) || not(numel(kernel_shape))
    kernel_shape = 1;
end

if nargin < 7 || not(isnumeric(FFT_interpolation)) || not(numel(FFT_interpolation))
    FFT_interpolation = 1;
end

if nargin < 8 || not(isnumeric(interp_zp)) || not(numel(interp_zp))
    interp_zp = 24;
end

if nargin < 9 || not(isnumeric(sketched_SVD)) || not(numel(sketched_SVD))
    sketched_SVD = 1;
end

[N1_cal, N2_cal, Nc] = size(kCal);

[in1,in2] = meshgrid(-tau:tau,-tau:tau);

if kernel_shape == 0 

    ind = [1:numel(in1)]'; 
    
else 
    
    ind = find(in1.^2+in2.^2<=tau^2); 
    
end

in1 = in1(ind)';
in2 = in2(ind)';

patchSize = numel(in1);

in1 = in1(:);
in2 = in2(:);

eind = [patchSize:-1:1]';

G = zeros(2*(2*tau+1)* 2*(2*tau+1),Nc,Nc);

if sketched_SVD == 0
    W = U*U';
else
    W = eye(size(U, 1)) - U*U';
end

clear U;
W = permute(reshape(W,patchSize,Nc,patchSize,Nc),[1,2,4,3]);

for s = 1:patchSize 
    G(sub2ind([2*(2*tau+1),2*(2*tau+1)],2*tau+1+1+in1(eind)+in1(s),2*tau+1+1+in2(eind)+in2(s)),:,:) = ...
        G(sub2ind([2*(2*tau+1),2*(2*tau+1)],2*tau+1+1+in1(eind)+in1(s),2*tau+1+1+in2(eind)+in2(s)),:,:)  + W(:,:,:,s);
end

clear W

if FFT_interpolation == 0
    
    N1_g = N1;
    N2_g = N2;
    
else
  
    if N1_cal <= N1 - interp_zp
        N1_g = N1_cal + interp_zp;
    else   
        N1_g = N1_cal;
    end

    if N2_cal <= N2 - interp_zp
        N2_g = N2_cal + interp_zp;
    else   
        N2_g = N2_cal;
    end
    
end

[n2,n1] = meshgrid([-N2_g/2:N2_g/2-1]/N2_g,[-N1_g/2:N1_g/2-1]/N1_g);
phaseKernel = exp(complex(0,-2*pi)*(n1*(N1_g-2*tau-1)+n2*(N2_g-2*tau-1)));

G = fft2(conj(reshape(G,2*(2*tau+1),2*(2*tau+1),Nc,Nc)),N1_g,N2_g).*phaseKernel; 

G = fftshift(fftshift(G,1),2);

end

function G = G_matrices_3D(kCal, N1, N2, N3, tau, U, kernel_shape, FFT_nullspace_C_calculation, FFT_interpolation, interp_zp, sketched_SVD)

% Function that calculates the G(x) matrices directly without calculating
% H(x) first.
%
% Input parameters:
%   --kCal:                        N1_cal x N2_cal x N3_cal x Nc block of calibration data,
%                                  where N1_cal, N2_cal, and N3_cal are the dimensions of a
%                                  rectangular block of Nyquist-sampled k-space, and Nc is the
%                                  number of channels in the array.
%
%   --N1, N2, N3:                  The desired dimensions of the output sensitivity
%                                  matrices.
%
%   --tau:                         Parameter (in Nyquist units) that determines the
%                                  size of the k-space kernel. For a rectangular
%                                  kernel, the size corresponds to (2*tau+1) x
%                                  (2*tau+1) x (2*tau+1). For an ellipsoidal kernel, it
%                                  corresponds to the radius of the associated
%                                  neighborhood. Default: 3.
%
%   --U:                           Matrix whose columns correspond to the nullspace
%                                  vectors of the C matrix.
%
%   --kernel_shape:                Binary variable. 0 = rectangular kernel, 1 = ellipsoidal
%                                  kernel. Default: 1.
%
%   --FFT_nullspace_C_calculation: Binary variable. 1 = FFT-based calculation of nullspace
%                                  vectors of C by calculating C'*C directly (instead of
%                                  calculating C first). Default: 1.
%
%   --FFT_interpolation:           Binary variable. 0 = no interpolation is used,
%                                  1 = FFT-based interpolation is used. Default: 1.
%
%   --interp_zp:                   Amount of zero-padding to create the low-resolution
%                                  grid if FFT-interpolation is used. The low-resolution
%                                  grid has dimensions (N1_cal + interp_zp) x
%                                  (N2_cal + interp_zp) x (N3_cal + interp_zp) x Nc. Default: 24.
%
%   --sketched_SVD:                Binary variable. 1 = sketched SVD is used to calculate
%                                  a basis for the nullspace of the C matrix. Default: 1.
%
% Output parameters:
%   --G:                           N1 x N2 x N3 x Nc x Nc array where G(i,j,k,:,:) 
%                                  corresponds to the G matrix at the (i,j,k) spatial
%                                  location.

if nargin < 5 || not(isnumeric(tau)) || not(numel(tau))
    tau = 3;
end

if nargin < 7 || not(isnumeric(kernel_shape)) || not(numel(kernel_shape))
    kernel_shape = 1;
end

if nargin < 8 || not(isnumeric(FFT_nullspace_C_calculation)) || not(numel(FFT_nullspace_C_calculation))
    FFT_nullspace_C_calculation = 1;
end

if nargin < 9 || not(isnumeric(FFT_interpolation)) || not(numel(FFT_interpolation))
    FFT_interpolation = 1;
end

if nargin < 10 || not(isnumeric(interp_zp)) || not(numel(interp_zp))
    interp_zp = 24;
end

if nargin < 11 || not(isnumeric(sketched_SVD)) || not(numel(sketched_SVD))
    sketched_SVD = 1;
end

[N1_cal, N2_cal, N3_cal, Nc] = size(kCal);

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

in1 = in1(:);
in2 = in2(:);
in3 = in3(:);

eind = [patchSize:-1:1]';

G = zeros(2*(2*tau+1)* 2*(2*tau+1)* 2*(2*tau+1),Nc,Nc);

if sketched_SVD == 0
    W = U*U';
else
    W = eye(size(U, 1)) - U*U';
end

clear U;
W = permute(reshape(W,patchSize,Nc,patchSize,Nc),[1,2,4,3]);

for s = 1:patchSize 
    G(sub2ind([2*(2*tau+1),2*(2*tau+1),2*(2*tau+1)],2*tau+1+1+in1(eind)+in1(s),2*tau+1+1+in2(eind)+in2(s), 2*tau+1+1+in3(eind)+in3(s)),:,:) = ...
        G(sub2ind([2*(2*tau+1),2*(2*tau+1),2*(2*tau+1)],2*tau+1+1+in1(eind)+in1(s),2*tau+1+1+in2(eind)+in2(s),2*tau+1+1+in3(eind)+in3(s)),:,:)  + W(:,:,:,s);
end

clear W

if FFT_interpolation == 0
    
    N1_g = N1;
    N2_g = N2;
    N3_g = N3;
    
else
  
    if N1_cal <= N1 - interp_zp
        N1_g = N1_cal + interp_zp;
    else   
        N1_g = N1_cal;
    end

    if N2_cal <= N2 - interp_zp
        N2_g = N2_cal + interp_zp;
    else   
        N2_g = N2_cal;
    end

    if N3_cal <= N3 - interp_zp
        N3_g = N3_cal + interp_zp;
    else   
        N3_g = N3_cal;
    end
    
end

[n2,n1,n3] = meshgrid([-N2_g/2:N2_g/2-1]/N2_g,[-N1_g/2:N1_g/2-1]/N1_g, [-N3_g/2:N3_g/2-1]/N3_g);
phaseKernel = -exp(complex(0,-2*pi)*(n1*(N1_g-2*tau-1)+n2*(N2_g-2*tau-1)+n3*(N3_g-2*tau-1)));

G = fft3(conj(reshape(G,2*(2*tau+1),2*(2*tau+1),2*(2*tau+1),Nc,Nc)),N1_g,N2_g,N3_g).*phaseKernel; 

G = fftshift(fftshift(fftshift(G,1),2),3);

if FFT_nullspace_C_calculation == 1
    % If the nullspace vectors of the C matrix were calculated using an FFT-based
    % approach, the G matrices are flipped in all three dimensions.
    G = flip(flip(flip(G, 2), 1), 3);
end 

% ==== FFT-based interpolation ====

if FFT_interpolation == 1

    [N1_cal, N2_cal, ~] = size(kCal);
    
    w_sm = [0.54 - 0.46*cos(2*pi*([0:(N1_g-1)]/(N1_g-1)))].';
    w_sm2 = [0.54 - 0.46*cos(2*pi*([0:(N2_g-1)]/(N2_g-1)))].';
    w_sm = w_sm*w_sm2';
    w_sm = repmat(w_sm, [1 1 Nc]); 

    if PowerIteration_G_nullspace_vectors == 1 && (PowerIteration_flag_convergence == 1 || PowerIteration_flag_auto == 1) 

        auxVal = 1 - eigen1;

        eigenVal = abs(fftshift(fftshift(ifft2(fftshift(fft2(ifftshift(auxVal))).*w_sm(:,:,end), N1, N2), 1), 2)); 

        eigenVal = eigenVal/max(eigenVal(:));

        threshold_mask = 0.075;
        
        support_mask = zeros(size(eigenVal));
        support_mask(find(eigenVal < threshold_mask)) = 1;
    
        eigen1_us = abs(fftshift(fftshift(ifft2(fftshift(fft2(ifftshift(eigen1))).*w_sm(:,:,end), N1, N2), 1), 2)); 
    
        eigen2_us = abs(fftshift(fftshift(ifft2(fftshift(fft2(ifftshift(eigen2))).*w_sm(:,:,end), N1, N2), 1), 2)); 
    
        ratioEig = (eigen2_us./eigen1_us).^M;
        ratio_small = support_mask.*ratioEig;
    
        th_ratio = 0.008;
    
        ratio_small(find(ratio_small <= th_ratio)) = 0;
        ratio_small(find(ratio_small > th_ratio)) = 1;
    
        flag_convergence_PI = sum(ratio_small(:)) > 0;
    
        if flag_convergence_PI == 1 && PowerIteration_flag_convergence == 1
            error(['Power Iteration might have not converged for some voxels within the support after the ' int2str(M)...
                ' iterations indicated by the user. Increasing the number of iterations is recommended. You can ignore this error by setting PowerIteration_flag_convergence = 0. '...
                'The number of needed iterations for convergence can be found automatically by setting PowerIteration_flag_auto = 1. '])   
        end

        if flag_convergence_PI == 0
            if verbose == 1
                 disp(['Most likely Power Iteration has converged for all the voxels within the support after the ' int2str(M) ' iterations indicated by the user.'])
            end
        end

    if PowerIteration_flag_auto == 1 && flag_convergence_PI == 1
        if verbose == 1
            warning('off','backtrace')
            warning(['Power Iteration might have not converged for some voxels within the support after the ' int2str(M)...
                    ' iterations indicated by the user. The number of iterations for the convergence of Power Iteration will be found automatically. You can turn off this option by setting PowerIteration_flag_auto = 0.'])
        end
        M_auto = M+1;
        while(flag_convergence_PI == 1)
            S = pagemtimes(G_null, S);
            S2 = pagemtimes(G_null, S2);
            
            S(:,1,:) = S(:,1,:)./pagenorm(S(:,1,:));
    
            inner_prod = pagemtimes(S(:,1,:), 'ctranspose', S2(:,1,:), 'none');
    
            S2(:,1,:) = S2(:,1,:) - inner_prod.*S(:,1,:);

            S2(:,1,:) = S2(:,1,:)./pagenorm(S2(:,1,:));
    
            E = pagemtimes(S, 'ctranspose', pagemtimes(G_null, S), 'none');
            E2 = pagemtimes(S2, 'ctranspose', pagemtimes(G_null, S2), 'none');

            eigen1 = reshape(permute(E, [3 1 2]), [N1_g, N2_g]);
            eigen2 = reshape(permute(E2, [3 1 2]), [N1_g, N2_g]);

            auxVal = 1 - eigen1;

            eigenVal = abs(fftshift(fftshift(ifft2(fftshift(fft2(ifftshift(auxVal))).*w_sm(:,:,end), N1, N2), 1), 2)); 

            eigenVal = eigenVal/max(eigenVal(:));
            
            support_mask = zeros(size(eigenVal));
            support_mask(find(eigenVal < threshold_mask)) = 1;

            eigen1_us = abs(fftshift(fftshift(ifft2(fftshift(fft2(ifftshift(eigen1))).*w_sm(:,:,end), N1, N2), 1), 2)); 
    
            eigen2_us = abs(fftshift(fftshift(ifft2(fftshift(fft2(ifftshift(eigen2))).*w_sm(:,:,end), N1, N2), 1), 2));     
    
            ratioEig = (eigen2_us./eigen1_us).^M_auto;
            ratio_small = support_mask.*ratioEig;
        
            ratio_small(find(ratio_small <= th_ratio)) = 0;
            ratio_small(find(ratio_small > th_ratio)) = 1;
        
            flag_convergence_PI = sum(ratio_small(:)) > 0;

            M_auto = M_auto + 1;
        end

        if verbose == 1
            disp(['Most likely Power Iteration has converged for all the voxels within the support. ' int2str(M_auto) ' iterations were needed.'] )
        end

    end

    end

    if PowerIteration_G_nullspace_vectors == 1 && PowerIteration_flag_convergence == 0 && PowerIteration_flag_auto == 0

        eigenVal = abs(fftshift(fftshift(ifft2(fftshift(fft2(ifftshift(eigenVal))).*w_sm(:,:,end), N1, N2), 1), 2)); 

        eigenVal = eigenVal/max(eigenVal(:));

    end

    if PowerIteration_G_nullspace_vectors == 0

        eigenVal = abs(fftshift(fftshift(ifft2(fftshift(fft2(ifftshift(eigenVal))).*w_sm(:,:,end), N1, N2), 1), 2)); 

        eigenVal = eigenVal/max(eigenVal(:));

    end
    
    apodizing_window = gausswin(N1_g,gauss_win_param)*gausswin(N2_g,gauss_win_param)';
    
    imLowRes_cal = zeros(N1_g,N2_g,Nc);
    imLowRes_cal(ceil(N1_g/2)+even_pisco(N1_g/2)+[-floor(N1_cal/2):floor(N1_cal/2)-even_pisco(N1_cal/2)],ceil(N2_g/2)+even_pisco(N2_g/2)+[-floor(N2_cal/2):floor(N2_cal/2)-even_pisco(N2_cal/2)],:) = kCal;
    imLowRes_cal = fftshift(ifft2(ifftshift(imLowRes_cal.*apodizing_window)));    

    cim = sum(conj(senseMaps).*imLowRes_cal,3)./sum(abs(senseMaps).^2,3); 

    senseMaps = senseMaps.*repmat((exp(complex(0,1)*angle(cim))), [1 1 Nc]); 

    senseMaps = fftshift(fftshift(ifft2(fftshift(fft2(ifftshift(senseMaps))).*w_sm, N1, N2), 1), 2); 

    

end


end

function [senseMaps, eigenVal] = nullspace_vectors_G_matrix_3D(kCal, N1, N2, N3, ...
    G, patchSize, PowerIteration_G_nullspace_vectors, M, PowerIteration_flag_convergence, PowerIteration_flag_auto, ...
    FFT_interpolation, gauss_win_param, verbose)

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

if nargin < 6 || not(isnumeric(PowerIteration_G_nullspace_vectors)) || not(numel(PowerIteration_G_nullspace_vectors))
    PowerIteration_G_nullspace_vectors = 1;
end

if nargin < 7 || not(isnumeric(M)) || not(numel(M))
    M = 30;
end

if nargin < 8 || not(isnumeric(PowerIteration_flag_convergence)) || not(numel(PowerIteration_flag_convergence))
    PowerIteration_flag_convergence = 1;
end

if nargin < 9 || not(isnumeric(PowerIteration_flag_auto)) || not(numel(PowerIteration_flag_auto))
    PowerIteration_flag_auto = 0;
end

if nargin < 10 || not(isnumeric(FFT_interpolation)) || not(numel(FFT_interpolation))
    FFT_interpolation = 1;
end

if nargin < 11 || not(isnumeric(gauss_win_param)) || not(numel(gauss_win_param))
    gauss_win_param = 100;
end

if nargin < 12 || not(isnumeric(verbose)) || not(numel(verbose))
    verbose = 1;
end

if PowerIteration_flag_auto == 1
    PowerIteration_flag_convergence = 0;
end

N1_g = size(G,1);
N2_g = size(G,2);
N3_g = size(G,3);
Nc = size(G,4);

G = reshape(permute(G, [4 5 1 2 3]), [Nc Nc (N1_g*N2_g*N3_g)]);

if PowerIteration_G_nullspace_vectors == 0
    
    [~, eigenVal, Vpage] = pagesvd(G, 'econ', 'vector');

    eigenVal = reshape(permute(eigenVal, [3 1 2]), [N1_g, N2_g, N3_g, Nc]);

    senseMaps = reshape(permute(Vpage(:, end, :), [3 1 2]), [N1_g N2_g N3_g Nc]);
    
    clear G
    
    eigenVal = eigenVal/patchSize;
    
else
    
    G = G/patchSize;
    
    G = permute(G, [4 5 1 2 3]);
    G = reshape(G, [Nc, Nc, N1_g*N2_g*N3_g]);

    G_null = repmat(eye(Nc), [1 1 (N1_g*N2_g*N3_g)]); 

    G_null = G_null - G;

    clear G

    if PowerIteration_flag_convergence == 0 && PowerIteration_flag_auto == 0

        S = randn(Nc,1) + 1i*randn(Nc,1);
        S = repmat(S, [1 1 N1_g*N2_g*N3_g]);
        S(:,1,:) = S(:,1,:)./pagenorm(S(:,1,:));

        for m = 1:M
            S = pagemtimes(G_null, S);
    
            S(:,1,:) = S(:,1,:)./pagenorm(S(:,1,:));
    
            if m == M
                E = pagemtimes(S, 'ctranspose', pagemtimes(G_null, S), 'none');
            end   
        end
        
        clear G_null 

        senseMaps = reshape(permute(squeeze(S), [2 1]) , [N1_g, N2_g, N3_g, Nc]);
        
        eigenVal = reshape(permute(E, [3 1 2]), [N1_g, N2_g, N3_g]);

        eigenVal = 1 - eigenVal;

    end

    if PowerIteration_flag_convergence == 1 || PowerIteration_flag_auto == 1

        S = randn(Nc,1) + 1i*randn(Nc,1);;
        S2 = randn(Nc,1) + 1i*randn(Nc,1);
        
        S(:,1,:) = S(:,1,:)./pagenorm(S(:,1,:)); 
        S2(:,1,:) = S2(:,1,:)./pagenorm(S2(:,1,:));       
        
        for m = 1:M
            S = pagemtimes(G_null, S);
            S2 = pagemtimes(G_null, S2);
            
            S(:,1,:) = S(:,1,:)./pagenorm(S(:,1,:));
    
            inner_prod = pagemtimes(S(:,1,:), 'ctranspose', S2(:,1,:), 'none');
    
            S2(:,1,:) = S2(:,1,:) - inner_prod.*S(:,1,:);
   
            S2(:,1,:) = S2(:,1,:)./pagenorm(S2(:,1,:));
    
            if m == M

                E = pagemtimes(S, 'ctranspose', pagemtimes(G_null, S), 'none');
                E2 = pagemtimes(S2, 'ctranspose', pagemtimes(G_null, S2), 'none');
            end
        end
    
        eigen1 = reshape(permute(E, [3 1 2]), [N1_g, N2_g, N3_g]);
        eigen2 = reshape(permute(E2, [3 1 2]), [N1_g, N2_g, N3_g]);
        

        if  FFT_interpolation == 0

            eigenVal = 1 - eigen1;

            threshold_mask = 0.075;
            
            support_mask = zeros(size(eigenVal));
            support_mask(find(eigenVal < threshold_mask)) = 1;
    
            ratioEig = (eigen2./eigen1).^M;
            ratio_small = support_mask.*ratioEig;
        
            th_ratio = 0.008;
        
            ratio_small(find(ratio_small <= th_ratio)) = 0;
            ratio_small(find(ratio_small > th_ratio)) = 1;
        
            flag_convergence_PI = sum(ratio_small(:)) > 0;
            
            
            if flag_convergence_PI == 1 && PowerIteration_flag_convergence == 1
                error(['Power Iteration might have not converged for some voxels within the support after the ' int2str(M)...
                ' iterations indicated by the user. Increasing the number of iterations is recommended. You can ignore this error by setting PowerIteration_flag_convergence = 0. '...
                'The number of needed iterations for convergence can be found automatically by setting PowerIteration_flag_auto = 1. '])   
            end

            if flag_convergence_PI == 0
                if verbose == 1
                     disp(['Most likely Power Iteration has converged for all the voxels within the support after the ' int2str(M) ' iterations indicated by the user.'])
                end
            end
            
        

        if PowerIteration_flag_auto == 1 && flag_convergence_PI == 1
            if verbose == 1
                warning('off','backtrace')
                warning(['Power Iteration might have not converged for some voxels within the support after the ' int2str(M)...
                                ' iterations indicated by the user. The number of iterations for the convergence of Power Iteration will be found automatically. You can turn off this option by setting PowerIteration_flag_auto = 0. ']) 
            end
            M_auto = M+1;
            while(flag_convergence_PI == 1)
                S = pagemtimes(G_null, S);
                S2 = pagemtimes(G_null, S2);
                
                S(:,1,:) = S(:,1,:)./pagenorm(S(:,1,:));
        
                inner_prod = pagemtimes(S(:,1,:), 'ctranspose', S2(:,1,:), 'none');
        
                S2(:,1,:) = S2(:,1,:) - inner_prod.*S(:,1,:);
    
                S2(:,1,:) = S2(:,1,:)./pagenorm(S2(:,1,:));
        
                E = pagemtimes(S, 'ctranspose', pagemtimes(G_null, S), 'none');
                E2 = pagemtimes(S2, 'ctranspose', pagemtimes(G_null, S2), 'none');

                eigen1 = reshape(permute(E, [3 1 2]), [N1_g, N2_g, N3_g]);
                eigen2 = reshape(permute(E2, [3 1 2]), [N1_g, N2_g, N3_g]);

                eigenVal = 1 - eigen1;
                
                support_mask = zeros(size(eigenVal));
                support_mask(find(eigenVal < threshold_mask)) = 1;
        
                ratioEig = (eigen2./eigen1).^M_auto;
                ratio_small = support_mask.*ratioEig;
            
                ratio_small(find(ratio_small <= th_ratio)) = 0;
                ratio_small(find(ratio_small > th_ratio)) = 1;
            
                flag_convergence_PI = sum(ratio_small(:)) > 0;

                M_auto = M_auto + 1;
            end
            
            if verbose == 1
                disp(['Most likely Power Iteration has converged for all the voxels within the support. ' int2str(M_auto) ' iterations were needed.'] )
            end

        end

        eigenVal = abs(eigenVal);

        end

     senseMaps = reshape(permute(squeeze(S), [2 1]) , [N1_g, N2_g, N3_g, Nc]);

     

    end

    


end

% ==== FFT-based interpolation ====

if FFT_interpolation == 1

    [N1_cal, N2_cal, N3_cal, ~] = size(kCal);
    
    w_sm = [0.54 - 0.46*cos(2*pi*([0:(N1_g-1)]/(N1_g-1)))].';
    w_sm2 = [0.54 - 0.46*cos(2*pi*([0:(N2_g-1)]/(N2_g-1)))].';
    w_sm3 = [0.54 - 0.46*cos(2*pi*([0:(N3_g-1)]/(N3_g-1)))].';
    w_sm_2d = w_sm*w_sm2';

    w_sm = bsxfun(@times, w_sm_2d, reshape(w_sm3, [1 1 N3_g]));

    w_sm_sM = repmat(w_sm, [1 1 1 Nc]); 

    if PowerIteration_G_nullspace_vectors == 1 && (PowerIteration_flag_convergence == 1 || PowerIteration_flag_auto == 1) 

        auxVal = 1 - eigen1;

        eigenVal = abs(fftshift(fftshift(fftshift(ifft3_zp(ft3(auxVal).*w_sm, [N1 N2 N3]), 1), 2), 3));

        eigenVal = eigenVal/max(eigenVal(:));

        threshold_mask = 0.075;
        
        support_mask = zeros(size(eigenVal));
        support_mask(find(eigenVal < threshold_mask)) = 1;
    
        eigen1_us = abs(fftshift(fftshift(fftshift(ifft3_zp(ft3(eigen1).*w_sm, [N1 N2 N3]), 1), 2), 3));
    
        eigen2_us = abs(fftshift(fftshift(fftshift(ifft3_zp(ft3(eigen2).*w_sm, [N1 N2 N3]), 1), 2), 3)); 
    
        ratioEig = (eigen2_us./eigen1_us).^M;
        ratio_small = support_mask.*ratioEig;
    
        th_ratio = 0.008;
    
        ratio_small(find(ratio_small <= th_ratio)) = 0;
        ratio_small(find(ratio_small > th_ratio)) = 1;
    
        flag_convergence_PI = sum(ratio_small(:)) > 0;
    
        if flag_convergence_PI == 1 && PowerIteration_flag_convergence == 1
            error(['Power Iteration might have not converged for some voxels within the support after the ' int2str(M)...
                ' iterations indicated by the user. Increasing the number of iterations is recommended. You can ignore this error by setting PowerIteration_flag_convergence = 0. '...
                'The number of needed iterations for convergence can be found automatically by setting PowerIteration_flag_auto = 1. '])   
        end

        if flag_convergence_PI == 0
            if verbose == 1
                 disp(['Most likely Power Iteration has converged for all the voxels within the support after the ' int2str(M) ' iterations indicated by the user.'])
            end
        end

    if PowerIteration_flag_auto == 1 && flag_convergence_PI == 1
        if verbose == 1
            warning('off','backtrace')
            warning(['Power Iteration might have not converged for some voxels within the support after the ' int2str(M)...
                    ' iterations indicated by the user. The number of iterations for the convergence of Power Iteration will be found automatically. You can turn off this option by setting PowerIteration_flag_auto = 0.'])
        end
        M_auto = M+1;
        while(flag_convergence_PI == 1)
            S = pagemtimes(G_null, S);
            S2 = pagemtimes(G_null, S2);
            
            S(:,1,:) = S(:,1,:)./pagenorm(S(:,1,:));
    
            inner_prod = pagemtimes(S(:,1,:), 'ctranspose', S2(:,1,:), 'none');
    
            S2(:,1,:) = S2(:,1,:) - inner_prod.*S(:,1,:);

            S2(:,1,:) = S2(:,1,:)./pagenorm(S2(:,1,:));
    
            E = pagemtimes(S, 'ctranspose', pagemtimes(G_null, S), 'none');
            E2 = pagemtimes(S2, 'ctranspose', pagemtimes(G_null, S2), 'none');

            eigen1 = reshape(permute(E, [3 1 2]), [N1_g, N2_g, N3_g]);
            eigen2 = reshape(permute(E2, [3 1 2]), [N1_g, N2_g, N3_g]);

            auxVal = 1 - eigen1;

            eigenVal = abs(fftshift(fftshift(fftshift(ifft3_zp(ft3(auxVal).*w_sm, [N1 N2 N3]), 1), 2), 3));

            eigenVal = eigenVal/max(eigenVal(:));
            
            support_mask = zeros(size(eigenVal));
            support_mask(find(eigenVal < threshold_mask)) = 1;

            eigen1_us = abs(fftshift(fftshift(fftshift(ifft3_zp(ft3(eigen1).*w_sm, [N1 N2 N3]), 1), 2), 3));
    
            eigen2_us = abs(fftshift(fftshift(fftshift(ifft3_zp(ft3(eigen2).*w_sm, [N1 N2 N3]), 1), 2), 3));   
    
            ratioEig = (eigen2_us./eigen1_us).^M_auto;
            ratio_small = support_mask.*ratioEig;
        
            ratio_small(find(ratio_small <= th_ratio)) = 0;
            ratio_small(find(ratio_small > th_ratio)) = 1;
        
            flag_convergence_PI = sum(ratio_small(:)) > 0;

            M_auto = M_auto + 1;
        end

        if verbose == 1
            disp(['Most likely Power Iteration has converged for all the voxels within the support. ' int2str(M_auto) ' iterations were needed.'] )
        end

    end

    end

    if PowerIteration_G_nullspace_vectors == 1 && PowerIteration_flag_convergence == 0 && PowerIteration_flag_auto == 0

        eigenVal = abs(fftshift(fftshift(fftshift(ifft3_zp(ft3(eigenVal).*w_sm, [N1 N2 N3]), 1), 2), 3));

        eigenVal = eigenVal/max(eigenVal(:));

    end

    if PowerIteration_G_nullspace_vectors == 0

        eigenVal = abs(fftshift(fftshift(fftshift(ifft3_zp(ft3(eigenVal).*w_sm, [N1 N2 N3]), 1), 2), 3));

        eigenVal = eigenVal/max(eigenVal(:));

    end
    
    apodizing_window_2D = gausswin(N1_g,gauss_win_param)*gausswin(N2_g,gauss_win_param)';

    apodizing_window = bsxfun(@times, apodizing_window_2D, reshape(gausswin(N3_g,gauss_win_param), [1 1 N3_g]));

    imLowRes_cal = zeros(N1_g,N2_g,N3_g,Nc);
    imLowRes_cal(ceil(N1_g/2)+even_pisco(N1_g/2)+[-floor(N1_cal/2):floor(N1_cal/2)-even_pisco(N1_cal/2)],ceil(N2_g/2)+even_pisco(N2_g/2)+[-floor(N2_cal/2):floor(N2_cal/2)-even_pisco(N2_cal/2)],...
        ceil(N3_g/2)+even_pisco(N3_g/2)+[-floor(N3_cal/2):floor(N3_cal/2)-even_pisco(N3_cal/2)],:) = kCal;
    imLowRes_cal = ift3(imLowRes_cal.*apodizing_window); 

    cim = sum(conj(senseMaps).*imLowRes_cal,4)./sum(abs(senseMaps).^2,4) ;  

    phase_norm = exp(complex(0,1)* angle(cim));

    senseMaps = phase_norm.*senseMaps;

    senseMaps = fftshift(fftshift(fftshift(ifft3_zp(ft3(senseMaps).*w_sm_sM, [N1 N2 N3]), 1), 2), 3); 

end

end

function out = ft3(x)
% Function that computes the 3D Fourier transform of the input array x,
% applying fftshift and ifftshift for proper centering.
%
% Input parameters:
%   --x:   4D array of size [N1, N2, N3, Nc], where N1, N2, N3 are spatial
%          dimensions and Nc is the number of channels.
%
% Output parameters:
%   --out: 4D array of the same size as x, containing the 3D Fourier
%          transform along the first three dimensions, with the result
%          centered using fftshift.
%
out = fftshift(fft(fft(fft(ifftshift(x),[],1),[],2),[],3));
end

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

function out = ifft3(x)
% Function that computes the inverse 3D Fourier transform of the input array x,
% along the first three dimensions.
%
% Input parameters:
%   --x:   4D array of size [N1, N2, N3, Nc], where N1, N2, N3 are spatial
%          dimensions and Nc is the number of channels.
%
% Output parameters:
%   --out: 4D array of the same size as x, containing the inverse 3D Fourier
%          transform along the first three dimensions.
%
out = ifft(ifft(ifft(x, [], 1), [], 2), [], 3);
end

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




