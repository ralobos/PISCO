function [senseMaps, eigenValues] = PISCO_sensitivity_map_estimation(kCal, dim_sens, varargin)

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

    % V2.0: Rodrigo A. Lobos (rlobos@umich.edu)
    % August, 2025.

    % Set default values for optional parameters
    p = inputParser;

    addRequired(p, 'kCal', @(x) isnumeric(x) && (ndims(x) == 3 || ndims(x) == 4));
    addRequired(p, 'dim_sens', @(x) isnumeric(x) && isvector(x) && (length(x) == 2 || length(x) == 3));

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
        if p.Results.kernel_shape == 0
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
        flag_3D = 1;  % 3D data
    else
        flag_3D = 0;  % 2D data
    end

    t_null = tic;

    % ==== Nullspace-based algorithm Steps (1) and (2)  ====
    % Calculation of nullspace vectors of C
    t_null_vecs = tic;

    opts_nullspace_C_matrix = struct( ...
        'tau',                        p.Results.tau, ...
        'threshold',                  p.Results.threshold, ...
        'kernel_shape',               p.Results.kernel_shape, ...
        'FFT_nullspace_C_calculation', p.Results.FFT_nullspace_C_calculation, ...
        'sketched_SVD',               p.Results.sketched_SVD, ...
        'sketch_dim',                 p.Results.sketch_dim, ...
        'visualize_C_matrix_sv',      p.Results.visualize_C_matrix_sv ...
    );

    fn = fieldnames(opts_nullspace_C_matrix);
    fv = struct2cell(opts_nullspace_C_matrix);
    nv = [fn.'; fv.'];
    nv = nv(:).';

    if flag_3D == 0
        U = utils.nullspace_vectors_C_matrix_2D(kCal, nv{:});
    else
        U = utils.nullspace_vectors_C_matrix_3D(kCal, nv{:});
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

    opts_G_matrices = struct( ...
        'kernel_shape',  p.Results.kernel_shape, ...
        'FFT_interpolation', p.Results.FFT_interpolation, ...
        'interp_zp',     p.Results.interp_zp, ...
        'sketched_SVD',  p.Results.sketched_SVD ...
    );

    fn = fieldnames(opts_G_matrices);
    fv = struct2cell(opts_G_matrices);
    nv = [fn.'; fv.'];
    nv = nv(:).';

    if flag_3D == 0
        G = utils.G_matrices_2D( ...
            kCal, dim_sens(1), dim_sens(2), p.Results.tau, U, nv{:} ...
        );
    else
        G = utils.G_matrices_3D( ...
            kCal, dim_sens(1), dim_sens(2), dim_sens(3), p.Results.tau, U, ...
            p.Results.FFT_nullspace_C_calculation, nv{:} ...
        );
    end

    t_G_matrices = toc(t_G_matrices);

    if flag_3D == 0
        Nc = size(kCal, 3);
        patchSize = size(U, 1) / Nc;
        clear U
    else
        Nc = size(kCal, 4);
        patchSize = size(U, 1) / Nc;
        clear U
    end

    if p.Results.verbose == 1
        disp(['Time G matrices (direct calculation): ' num2str(t_G_matrices)])
        disp('=======================')
    end

    % ==== Nullspace-based algorithm Step (4)  ====
    % Calculation of nullspace vectors of the G matrices
    t_null_G = tic;

    opts_G_nullspace_vectors = struct( ...
        'PowerIteration_G_nullspace_vectors', p.Results.PowerIteration_G_nullspace_vectors, ...
        'M',                                  p.Results.M, ...
        'PowerIteration_flag_convergence',    p.Results.PowerIteration_flag_convergence, ...
        'PowerIteration_flag_auto',           p.Results.PowerIteration_flag_auto, ...
        'FFT_interpolation',                  p.Results.FFT_interpolation, ...
        'gauss_win_param',                    p.Results.gauss_win_param, ...
        'verbose',                            p.Results.verbose ...
    );

    fn = fieldnames(opts_G_nullspace_vectors);
    fv = struct2cell(opts_G_nullspace_vectors);
    nv = [fn.'; fv.'];
    nv = nv(:).';

    if flag_3D == 0
        [senseMaps, eigenValues] = utils.nullspace_vectors_G_matrix_2D( ...
            kCal, dim_sens(1), dim_sens(2), G, patchSize, nv{:} ...
        );
    else
        [senseMaps, eigenValues] = utils.nullspace_vectors_G_matrix_3D( ...
            kCal, dim_sens(1), dim_sens(2), dim_sens(3), G, patchSize, nv{:} ...
        );
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
    % Normalization
    if flag_3D == 0
        % Phase-reference all coils to the first coil 
        phase_ref = exp(-1i * angle(senseMaps(:, :, 1)));
        senseMaps = senseMaps .* phase_ref;

        % Normalize sensitivities to unit L2 norm across coils at each pixel
        den = sqrt(sum(abs(senseMaps).^2, 3));
        den(den == 0) = 1;
        senseMaps = senseMaps ./ den;
    else
        % Phase-reference all coils to the first coil 
        phase_ref = exp(-1i * angle(senseMaps(:, :, :, 1)));
        senseMaps = senseMaps .* phase_ref;

        % Normalize sensitivities to unit L2 norm across coils at each pixel
        den = sqrt(sum(abs(senseMaps).^2, 4));
        den(den == 0) = 1;
        senseMaps = senseMaps ./ den;
    end

    if p.Results.verbose == 1
        disp(['Total time: ' num2str(toc(t_null))])
        disp('=======================')
    end

end