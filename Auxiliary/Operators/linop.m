classdef linop
%linop A class for linear operators
    
% written by M. Storath
% $Date: 2015-10-15 15:07:43 +0200 (Do, 15. Okt 2015) $	$Revision: 132 $

    
    properties
        % function handle for evaluation A * x
        eval;
        % function handle for evaluation A' * x
        ctrans;
        % A'A
        normalOp;
        % true if A'A positive definite
        posdef;
        % true if A'A positive definite
        lseSolver;
    end
    
    methods
        % constructor
        function A = linop(eval, ctrans, varargin)
            % if it is a matrix
            if ~isa(eval, 'function_handle') 
                A.eval = @(x) eval * x;
                A.ctrans = @(x) eval' * x;
                M = eval' * eval;
                A.normalOp = @(x) M * x;
            else
                A.eval = eval;
                A.ctrans = ctrans;
                A.normalOp = @(x) A.ctrans(A.eval(x));
            end
            ip = inputParser;
            addParamValue(ip, 'posdef', false);
            addParamValue(ip, 'lseSolver', []);
            parse(ip, varargin{:});
            par = ip.Results;
            A.posdef = par.posdef;
            A.lseSolver = par.lseSolver;
        end
        
        % ctranspose (')
        function C = ctranspose(A)
            C =  linop(A.ctrans, A.eval);
        end
        
        % mtimes (*)
        function C = mtimes( A, B )
            if isa(B, 'linop')
                C = linop( @(x) A.eval(B.eval(x)), @(x) B.ctrans(A.ctrans(x)));
            else
                C = A.eval(B);
            end
        end
        
        % size
        function s = size(A)
            %warning('Size is deprecated for class linop');
            s = [1 1];
        end
        
    end
end

