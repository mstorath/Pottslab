function out = iPotts2DADMM(f, gamma, A, dataterm, varargin)
%iPotts2DADMM Minimization strategy fot the inverse 2D Potts problem

% written by M. Storath
% $Date: 2014-09-12 12:35:45 +0200 (Fr, 12 Sep 2014) $	$Revision: 117 $

% parse options

backproj = A'*f;
[m,n,~] = size(backproj);
sseq = @(k) k^2.01 * 1e-7;

% parse options
ip = inputParser;
addParamValue(ip, 'muSeq', sseq ); % by default quadratic + eps progression
addParamValue(ip, 'nuSeq', @(k) 0 ); 
addParamValue(ip, 'tol', 1e-3);
addParamValue(ip, 'isotropic', 2); % knight moves neighborhood by default
addParamValue(ip, 'multiThreading', true);
addParamValue(ip, 'verbose', false);
addParamValue(ip, 'weights', ones(m,n));
addParamValue(ip, 'maxIter', 50000);
addParamValue(ip, 'LSEmaxIter', 1000);
addParamValue(ip, 'LSEtol', []);
addParamValue(ip, 'groundTruth', []);

parse(ip, varargin{:});
par = ip.Results;

assert(par.tol > 0, 'Stopping tolerance must be > 0.');
assert((dataterm == 2), 'Currently only L2 data terms supported.');

% counts total number of iterations
nIter = 0;

% options for iterative linear solver
LSEtol.maxit = par.LSEmaxIter;
LSEtol.tol = par.LSEtol;

% initialize variables

switch par.isotropic
    case 0
        % anisotropic finite differences
        nhood = { [1,0] };
        omega = { 1 };
    case 1
        % finite differences with diagonals
        nhood = { [1,0], [1,1] };
        omega = { sqrt(2) - 1,...
            1 - sqrt(2)/2 };
    case 2
        % finite differences with knight moves
        nhood = { [1,0], [1,1], [2,1], [1,2] };
        omega = { sqrt(5) - 2,...
            sqrt(5) - 3/2 * sqrt(2),...
            0.5 * (1 + sqrt(2) - sqrt(5)),...
            0.5 * (1 + sqrt(2) - sqrt(5)) };
        
    otherwise
        error('Value of isotropic must be 0, 1, or 2')
end

S = numel(nhood) * 2;

u = cell(S, 1);
lam = cell(S, 1);
rho = cell(S, S);
for s = 1:S
    u{s} = zeros(size(backproj));
    lam{s} = zeros(size(backproj));
    for t = s+1:S
        rho{s,t} = zeros(size(backproj));
    end
end
v = zeros(size(backproj));


% init datastructures -> Java
proc1 = javaObject('pottslab.PLProcessor');
proc1.setMultiThreaded(par.multiThreading);
bufferImg1 = javaObject('pottslab.PLImage', u{1});
proc1.set(bufferImg1, par.weights);

proc2 = javaObject('pottslab.PLProcessor');
proc2.setMultiThreaded(par.multiThreading);
bufferImg2 = javaObject('pottslab.PLImage', rot90(u{2}));
proc2.set(bufferImg2, par.weights);

for k=1:par.maxIter
    
    % set coupling parameter
    mu = par.muSeq(k);
    nu = par.nuSeq(k);
    
    q = zeros(size(backproj));
    for a = 1:S/2
        % apply univariate Potts to nhood directions
        s = 2*a -1;
        % prepare data
        aux = 0;
        for r = 1:s-1
            aux = aux + nu * u{r} + rho{r,s};
        end
        for t = s+1:S
            aux = aux + nu * u{t} - rho{s,t};
        end
        bufferImg1.set( (mu * v + lam{s} + aux) / (mu + nu * (S - 1)) );
        proc1.setGamma(2 * gamma * omega{a} / (mu + nu * (S - 1)));
        proc1.set(bufferImg1, par.weights);
        proc1.applyToDirection(nhood{a});
        
        u{s} = plimage2double(bufferImg1);
        
        % set data for Tikhonov problem
        q = q + u{s} - lam{s}/mu;
        
        % apply same to orthogonal directions
        s = 2*a;
        % prepare data
        aux = 0;
        for r = 1:s-1
            aux = aux + nu * u{r} + rho{r,s};
        end
        for t = s+1:S
            aux = aux + nu * u{t} - rho{s,t};
        end
        bufferImg2.set( rot90( (mu * v + lam{s} + aux) / (mu + nu * (S - 1))  ));
        proc2.setGamma(2 * gamma * omega{a} / (mu + nu * (S - 1)));
        proc2.set(bufferImg2, rot90(par.weights));
        proc2.applyToDirection(nhood{a});
        u{s} = rot90( plimage2double(bufferImg2), -1 );
        % set data for Tikhonov problem
        q = q + u{s} - lam{s}/mu;
    end
    
    % offset for Tikhonov problem
    LSEtol.u0 = q/S;
    v = minL2Tikhonov(f, mu/2 * S, A, LSEtol);
    
    % init h for next Tikhonov iteration ("warm start")
    LSEtol.init = v;
    
    % update Lagrange multipliers
    for s = 1:S
        lam{s} = lam{s} + mu * (v - u{s});
        for t = (s+1):S
            rho{s,t} = rho{s,t} + nu * (u{s} - u{t});
        end
    end
    
    %
    nIter = nIter + 1;
    uScale =  (norm(u{1}(:), 2) + norm(u{2}(:), 2));
    relError = norm(u{1}(:) - u{2}(:),2)/ uScale;
    %relError = norm(u{end}(:) - u{end-1}(:),2)/norm(u{end}(:) + u{end-1}(:), 2);
    
    if ((relError < par.tol) || (nIter > par.maxIter) || (uScale == 0)) && (nIter > 1)
        break;
    end
    
    
    if par.verbose && (mod(nIter, par.verbose) ==0)
        T = S/2 + 1;
        out = 0;
        for s = 1:S
            out = out + u{s};
        end
        out = out/S;
        subplot(2,T, T)
        imshow(f, [])
        title('Data')
        subplot(2, T, 2*T)
        imshow(out);
        title('Average of all u')
        
        for s= 1:S/2
            subplot(2, T, s)
            imshow(u{s});
            title(sprintf('u %i', s) )
            subplot(2, T, T + s )
            imshow(u{s + S/2});
            title(sprintf('u %i', s + S/2) )
        end
        
        colormap gray;
        drawnow;
        fprintf('Mu: %f, ', mu);
        fprintf('Rel. error: %f', relError);
        if ~isempty(par.groundTruth)
            fprintf(', PSNR: %f', plpsnr(par.groundTruth, out));
        end
        fprintf('\n');
    end
end

out = zeros(size(backproj));
for s = 1:S
    out = out + u{s};
end
out = out/S;

end
