%  BYPASSDLGRADIENTS Bypass gradient for non-differentiable operations
%    Y = bypassdlgradients(FUN, X, varargin) evaluates FUN(X, varargin)
%    while overriding the derivitive calculation used during backward
%    propogation to an identity function instead.
% 
%    Examples:
%      a = dlarray([1.0 2.5]); % point at which to evaluate gradient
% 
%      % non-differentiable gradient calculation
%      function [y,grad] = objectiveAndGradient(x)
%          y = round(x(1) + x(2));
%          grad = dlgradient(y,x);
%      end
%      [val,grad] = dlfeval(@objectiveAndGradient,a);
%      % val is dlarray(4)
%      % grad is dlarray([0 0])
% 
%      % non-differentiable gradient calculation with a straight-through
%      % estimator for the 'round' function
%      function [y,grad] = steObjectiveAndGradient(x)
%          y = BYPASSDLGRADIENTS(@round, x(1) + x(2) );
%          grad = dlgradient(y,x);
%      end
%      [val,grad] = dlfeval(@steObjectiveAndGradient,a);
%      % val is dlarray(4)
%      % grad is dlarray([1 1])
% 
%    See also: DLARRAY, DLACCELERATE, EXTRACTDATA
%