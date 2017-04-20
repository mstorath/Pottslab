classdef radonop < linop
    %radonop A class for Radon transform
    
    % written by M. Storath
    % $Date: 2014-09-02 14:56:23 +0200 (Di, 02. Sep 2014) $	$Revision: 105 $
    
    properties
        % solution method for prox
        useFBP;
    end
    
    methods
        % constructor
        function A = radonop(theta, imgSize)
            
            eval = @(x) radon(x, theta);
            ctrans = @(x) iradon(x, theta, 'linear', 'None', 1, imgSize);
            
            A = A@linop(eval, ctrans);
            A.posdef = true;
            A.useFBP = false;
        end
    end
end

