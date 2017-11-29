function p = kNNDensity(x, k)
% Compute the density based on kNN of the points in x.

nPoints = length(x);

[~, dists] = knnsearch(x,x,'k',k+1);
p = k./(2*nPoints*dists(:,end));

% p = 1./dists(:,end);
% [xSort,ind] = sort(x);
% p = p/trapz(xSort,p(ind));

end