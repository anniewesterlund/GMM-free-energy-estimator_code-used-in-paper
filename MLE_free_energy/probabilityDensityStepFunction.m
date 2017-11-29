function p = probabilityDensityStepFunction(x, edges, amplitudes)

p = 0;

x = x';

binWidth = edges(2)-edges(1);
edges = edges-binWidth/2;
p = zeros(size(x));

for i = 1:length(edges)
    
    points_within_interval = x >= edges(i) & x < edges(i) + binWidth;
    p(points_within_interval) = amplitudes(i)/binWidth;
    
end

end